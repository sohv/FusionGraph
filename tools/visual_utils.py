"""
Visual Utilities for Image Processing and Visualization
"""

import os
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


class VisualUtils:
    """
    Utility functions for visual processing and annotation
    """
    
    @staticmethod
    def create_thumbnail(image_path: str, size: Tuple[int, int] = (200, 200)) -> str:
        """
        Create thumbnail of an image
        
        Args:
            image_path: Path to the source image
            size: Thumbnail size as (width, height)
            
        Returns:
            Base64 encoded thumbnail image
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return f"data:image/jpeg;base64,{img_str}"
                
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")
            return ""
    
    @staticmethod
    def annotate_image_with_objects(image_path: str, 
                                   objects: List[Dict[str, Any]],
                                   output_path: Optional[str] = None) -> str:
        """
        Annotate image with detected objects and bounding boxes
        
        Args:
            image_path: Path to the source image
            objects: List of detected objects with bbox information
            output_path: Optional path to save annotated image
            
        Returns:
            Path to annotated image or base64 encoded image
        """
        try:
            # Open image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Draw bounding boxes and labels
            for i, obj in enumerate(objects):
                if 'bbox' in obj:
                    x, y, w, h = obj['bbox']
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{obj.get('type', 'object')}_{i}"
                    cv2.putText(image, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if output_path:
                cv2.imwrite(output_path, image)
                return output_path
            else:
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', image)
                img_str = base64.b64encode(buffer).decode()
                return f"data:image/jpeg;base64,{img_str}"
                
        except Exception as e:
            print(f"Error annotating image {image_path}: {e}")
            return ""
    
    @staticmethod
    def create_image_grid(image_paths: List[str], 
                         grid_size: Tuple[int, int],
                         thumbnail_size: Tuple[int, int] = (150, 150)) -> str:
        """
        Create a grid of images
        
        Args:
            image_paths: List of paths to images
            grid_size: Grid dimensions as (rows, cols)
            thumbnail_size: Size of each thumbnail
            
        Returns:
            Base64 encoded grid image
        """
        try:
            rows, cols = grid_size
            thumb_w, thumb_h = thumbnail_size
            
            # Create blank canvas
            canvas_w = cols * thumb_w
            canvas_h = rows * thumb_h
            canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
            
            # Place images in grid
            for i, img_path in enumerate(image_paths[:rows * cols]):
                if not os.path.exists(img_path):
                    continue
                
                row = i // cols
                col = i % cols
                
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB and resize
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                        
                        # Calculate position
                        x = col * thumb_w + (thumb_w - img.width) // 2
                        y = row * thumb_h + (thumb_h - img.height) // 2
                        
                        canvas.paste(img, (x, y))
                        
                        # Add filename label
                        draw = ImageDraw.Draw(canvas)
                        filename = os.path.basename(img_path)
                        try:
                            font = ImageFont.truetype("arial.ttf", 10)
                        except:
                            font = ImageFont.load_default()
                        
                        text_y = y + img.height + 2
                        draw.text((x, text_y), filename[:15], fill=(0, 0, 0), font=font)
                        
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    continue
            
            # Convert to base64
            buffer = io.BytesIO()
            canvas.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            print(f"Error creating image grid: {e}")
            return ""
    
    @staticmethod
    def highlight_text_regions(image_path: str, 
                              text_regions: List[Dict[str, Any]]) -> str:
        """
        Highlight text regions detected by OCR
        
        Args:
            image_path: Path to the source image
            text_regions: List of text regions with bounding boxes
            
        Returns:
            Base64 encoded image with highlighted text regions
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Create overlay
            overlay = image.copy()
            
            for region in text_regions:
                if 'bbox' in region:
                    x, y, w, h = region['bbox']
                    
                    # Draw semi-transparent rectangle
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), -1)
                    
                    # Add text if available
                    if 'text' in region and region['text'].strip():
                        text = region['text'][:20] + "..." if len(region['text']) > 20 else region['text']
                        cv2.putText(overlay, text, (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Blend overlay with original image
            alpha = 0.3
            result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', result)
            img_str = base64.b64encode(buffer).decode()
            return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            print(f"Error highlighting text regions in {image_path}: {e}")
            return ""
    
    @staticmethod
    def create_comparison_view(original_path: str, 
                              processed_path: str,
                              title1: str = "Original",
                              title2: str = "Processed") -> str:
        """
        Create side-by-side comparison of two images
        
        Args:
            original_path: Path to original image
            processed_path: Path to processed image
            title1: Title for first image
            title2: Title for second image
            
        Returns:
            Base64 encoded comparison image
        """
        try:
            # Load images
            img1 = Image.open(original_path)
            img2 = Image.open(processed_path)
            
            # Convert to RGB
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Resize to same height
            target_height = min(img1.height, img2.height, 300)
            aspect1 = img1.width / img1.height
            aspect2 = img2.width / img2.height
            
            new_width1 = int(target_height * aspect1)
            new_width2 = int(target_height * aspect2)
            
            img1 = img1.resize((new_width1, target_height), Image.Resampling.LANCZOS)
            img2 = img2.resize((new_width2, target_height), Image.Resampling.LANCZOS)
            
            # Create canvas
            canvas_width = new_width1 + new_width2 + 20  # 20px gap
            canvas_height = target_height + 40  # 40px for titles
            canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
            
            # Paste images
            canvas.paste(img1, (10, 30))
            canvas.paste(img2, (new_width1 + 20, 30))
            
            # Add titles
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 5), title1, fill=(0, 0, 0), font=font)
            draw.text((new_width1 + 20, 5), title2, fill=(0, 0, 0), font=font)
            
            # Convert to base64
            buffer = io.BytesIO()
            canvas.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            print(f"Error creating comparison view: {e}")
            return ""
    
    @staticmethod
    def get_image_info(image_path: str) -> Dict[str, Any]:
        """
        Get basic information about an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'filename': os.path.basename(image_path),
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'has_transparency': 'transparency' in img.info,
                    'file_size': os.path.getsize(image_path),
                    'file_size_mb': round(os.path.getsize(image_path) / (1024 * 1024), 2)
                }
        except Exception as e:
            return {
                'filename': os.path.basename(image_path),
                'error': str(e)
            }


def main():
    """
    Example usage of VisualUtils
    """
    print("VisualUtils module loaded successfully!")
    print("Available functions:")
    print("- create_thumbnail()")
    print("- annotate_image_with_objects()")
    print("- create_image_grid()")
    print("- highlight_text_regions()")
    print("- create_comparison_view()")
    print("- get_image_info()")


if __name__ == "__main__":
    main()