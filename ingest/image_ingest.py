"""
Image Ingestion Module for Visual RAG
Handles image processing, OCR, object detection, and KG node creation
"""

import os
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import pytesseract
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_index.core.schema import BaseNode, TextNode, ImageNode
from llama_index.core.graph_stores import SimpleGraphStore
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageIngestor:
    """
    Handles image ingestion for visual RAG pipeline
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 caption_model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the image ingestor
        
        Args:
            embedding_model_name: Model name for generating embeddings
            caption_model_name: Model name for image captioning
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize captioning model
        self.caption_processor = BlipProcessor.from_pretrained(caption_model_name)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name)
        
        # Store for processed images metadata
        self.image_metadata = {}
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"OCR extraction failed for {image_path}: {e}")
            return ""
    
    def generate_image_caption(self, image_path: str) -> str:
        """
        Generate caption for the image using BLIP model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption for the image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.caption_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self.caption_model.generate(**inputs, max_length=50)
                caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            print(f"Caption generation failed for {image_path}: {e}")
            return ""
    
    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Simple object detection using contour detection
        For more advanced detection, integrate YOLO or similar models
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected objects with bounding boxes
        """
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple contour detection
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for i, contour in enumerate(contours[:10]):  # Limit to top 10 objects
                if cv2.contourArea(contour) > 500:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "id": f"obj_{i}",
                        "bbox": [x, y, w, h],
                        "area": cv2.contourArea(contour),
                        "type": "detected_object"
                    })
            
            return objects
        except Exception as e:
            print(f"Object detection failed for {image_path}: {e}")
            return []
    
    def create_image_nodes(self, image_path: str, 
                          include_ocr: bool = True,
                          include_caption: bool = True,
                          include_objects: bool = True) -> List[BaseNode]:
        """
        Create knowledge graph nodes for an image
        
        Args:
            image_path: Path to the image file
            include_ocr: Whether to include OCR text
            include_caption: Whether to include image caption
            include_objects: Whether to include detected objects
            
        Returns:
            List of nodes representing the image and its components
        """
        nodes = []
        image_id = str(uuid.uuid4())
        
        # Main image node
        image_node = TextNode(
            text=f"Image file: {os.path.basename(image_path)}",
            id_=f"image_{image_id}",
            metadata={
                "type": "image",
                "image_path": image_path,
                "image_id": image_id,
                "file_name": os.path.basename(image_path)
            }
        )
        nodes.append(image_node)
        
        # OCR text node
        if include_ocr:
            ocr_text = self.extract_text_from_image(image_path)
            if ocr_text:
                ocr_node = TextNode(
                    text=f"OCR Text from {os.path.basename(image_path)}: {ocr_text}",
                    id_=f"ocr_{image_id}",
                    metadata={
                        "type": "ocr_text",
                        "image_id": image_id,
                        "source_image": image_path
                    }
                )
                nodes.append(ocr_node)
        
        # Caption node
        if include_caption:
            caption = self.generate_image_caption(image_path)
            if caption:
                caption_node = TextNode(
                    text=f"Image caption for {os.path.basename(image_path)}: {caption}",
                    id_=f"caption_{image_id}",
                    metadata={
                        "type": "image_caption",
                        "image_id": image_id,
                        "source_image": image_path
                    }
                )
                nodes.append(caption_node)
        
        # Object nodes
        if include_objects:
            objects = self.detect_objects(image_path)
            for obj in objects:
                obj_node = TextNode(
                    text=f"Detected object in {os.path.basename(image_path)}: {obj['type']} at position {obj['bbox']}",
                    id_=f"object_{image_id}_{obj['id']}",
                    metadata={
                        "type": "detected_object",
                        "image_id": image_id,
                        "source_image": image_path,
                        "bbox": obj["bbox"],
                        "area": obj["area"]
                    }
                )
                nodes.append(obj_node)
        
        # Store metadata
        self.image_metadata[image_id] = {
            "path": image_path,
            "nodes": [node.id_ for node in nodes],
            "file_name": os.path.basename(image_path)
        }
        
        return nodes
    
    def process_image_directory(self, directory_path: str) -> List[BaseNode]:
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            List of all nodes created from images in the directory
        """
        all_nodes = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(directory_path, filename)
                print(f"Processing image: {filename}")
                nodes = self.create_image_nodes(image_path)
                all_nodes.extend(nodes)
        
        print(f"Created {len(all_nodes)} nodes from {len(self.image_metadata)} images")
        return all_nodes
    
    def get_image_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for all processed images
        
        Returns:
            Dictionary containing metadata for all processed images
        """
        return self.image_metadata


def main():
    """
    Example usage of the ImageIngestor
    """
    ingestor = ImageIngestor()
    
    # Process a single image
    # nodes = ingestor.create_image_nodes("path/to/image.jpg")
    
    # Process a directory of images
    # nodes = ingestor.process_image_directory("path/to/image/directory")
    
    print("ImageIngestor initialized successfully!")


if __name__ == "__main__":
    main()