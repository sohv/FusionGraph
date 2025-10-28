"""
Visual RAG Pipeline with Faiss Integration and Explainability
Combines text and image retrieval with knowledge graph reasoning
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from llama_index.core import KnowledgeGraphIndex, StorageContext
from llama_index.core.schema import BaseNode, QueryBundle
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.graph_stores import SimpleGraphStore

from ingest.image_ingest import ImageIngestor
from retrieval.faiss_retriever import FaissRetriever, RetrievalResult
from explainability.engine import ExplainabilityEngine, QueryExplanation


@dataclass
class VisualRAGResult:
    """
    Enhanced result from Visual RAG query with explainability
    """
    answer: str
    text_sources: List[Dict[str, Any]]
    image_sources: List[Dict[str, Any]]
    graph_context: Dict[str, Any]
    confidence_score: float
    provenance: Dict[str, List[str]]
    explanation: Optional[QueryExplanation] = None


class VisualRAGPipeline:
    """
    Enhanced Visual RAG Pipeline with Faiss retrieval and explainability
    """
    
    def __init__(self, 
                 knowledge_graph_index: KnowledgeGraphIndex,
                 image_ingestor: Optional[ImageIngestor] = None,
                 similarity_threshold: float = 0.7,
                 use_faiss: bool = True):
        """
        Initialize Visual RAG Pipeline
        
        Args:
            knowledge_graph_index: Existing KG index from text documents
            image_ingestor: Image processing component
            similarity_threshold: Minimum similarity for retrieval
            use_faiss: Whether to use Faiss for faster retrieval
        """
        self.kg_index = knowledge_graph_index
        self.image_ingestor = image_ingestor or ImageIngestor()
        self.similarity_threshold = similarity_threshold
        self.use_faiss = use_faiss
        
        # Get the underlying graph store for direct graph operations
        self.graph_store = knowledge_graph_index.storage_context.graph_store
        
        # Track image nodes separately
        self.image_node_mapping = {}
        
        # Initialize Faiss retriever if requested
        self.faiss_retriever = None
        if use_faiss:
            try:
                self.faiss_retriever = FaissRetriever()
                print("✅ Faiss retriever initialized")
            except ImportError:
                print("⚠️ Faiss not available, falling back to default retrieval")
                self.use_faiss = False
        
        # Initialize explainability engine
        self.explainability_engine = ExplainabilityEngine()
    
    def add_images_to_kg(self, image_paths: Union[str, List[str]]) -> int:
        """
        Add images to the existing knowledge graph
        
        Args:
            image_paths: Single image path or list of image paths
            
        Returns:
            Number of nodes added to the knowledge graph
        """
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                # Process directory
                image_nodes = self.image_ingestor.process_image_directory(image_paths)
            else:
                # Process single image
                image_nodes = self.image_ingestor.create_image_nodes(image_paths)
        else:
            # Process list of image paths
            image_nodes = []
            for path in image_paths:
                nodes = self.image_ingestor.create_image_nodes(path)
                image_nodes.extend(nodes)
        
        # Add nodes to the knowledge graph index
        for node in image_nodes:
            self.kg_index.insert_nodes([node])
            
            # Track image nodes
            if node.metadata.get('type') == 'image':
                image_id = node.metadata.get('image_id')
                self.image_node_mapping[image_id] = {
                    'node_id': node.id_,
                    'image_path': node.metadata.get('image_path'),
                    'file_name': node.metadata.get('file_name')
                }
        
        print(f"Added {len(image_nodes)} nodes to knowledge graph")
        return len(image_nodes)
    
    def index_documents_in_faiss(self, documents_dir: str = "./documents/text") -> None:
        """
        Index text documents in Faiss for faster retrieval
        
        Args:
            documents_dir: Directory containing text documents
        """
        if not self.use_faiss or not self.faiss_retriever:
            print("⚠️ Faiss retriever not available")
            return
        
        documents = []
        
        # Load documents from directory
        if os.path.exists(documents_dir):
            for filename in os.listdir(documents_dir):
                if filename.endswith(('.txt', '.md')):
                    filepath = os.path.join(documents_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        documents.append({
                            'text': content,
                            'source_id': filename,
                            'source_name': filename,
                            'metadata': {
                                'file_path': filepath,
                                'file_size': len(content)
                            }
                        })
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
        
        if documents:
            print(f"Indexing {len(documents)} documents in Faiss...")
            self.faiss_retriever.add_documents(documents)
            self.faiss_retriever.save_index()
            print(f"✅ Indexed {len(documents)} documents")
        else:
            print("No documents found to index")
    
    def query_with_visual_context(self, 
                                 query: str,
                                 include_images: bool = True,
                                 max_text_results: int = 5,
                                 max_image_results: int = 3,
                                 include_explanation: bool = True) -> VisualRAGResult:
        """
        Execute a query with both text and visual context using enhanced retrieval
        
        Args:
            query: Natural language query
            include_images: Whether to include image sources in results
            max_text_results: Maximum number of text sources to retrieve
            max_image_results: Maximum number of image sources to retrieve
            include_explanation: Whether to generate explainability information
            
        Returns:
            VisualRAGResult containing multimodal response with explanations
        """
        retrieval_results = []
        retrieval_metadata = None
        
        # Use Faiss retrieval if available, otherwise fall back to default
        if self.use_faiss and self.faiss_retriever:
            try:
                retrieval_results, retrieval_metadata = self.faiss_retriever.retrieve(
                    query, 
                    top_k=max_text_results,
                    return_explanations=include_explanation
                )
                print(f"🔍 Faiss retrieval found {len(retrieval_results)} results")
            except Exception as e:
                print(f"⚠️ Faiss retrieval failed: {e}, falling back to default")
                self.use_faiss = False
        # Standard text-based KG query
        query_engine = self.kg_index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=max_text_results
        )
        
        # Enhanced query template for visual RAG
        enhanced_query = f"""
        <|system|>
        You are an AI assistant that can understand both text and visual information.
        Consider the following context from both documents and images when answering.
        If the context mentions images, captions, or visual elements, include that information in your response.
        Provide specific references to sources when possible.
        <|/system|>
        
        <|user|>
        Question: {query}
        
        Please provide a comprehensive answer using both textual and visual information available.
        <|/user|>
        """
        
        # Execute the query
        response = query_engine.query(enhanced_query)
        
        # Extract sources and categorize them
        text_sources = []
        image_sources = []
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for source_node in response.source_nodes:
                source_info = {
                    'id': source_node.id_,
                    'text': source_node.text,
                    'score': getattr(source_node, 'score', 0.0),
                    'metadata': source_node.metadata
                }
                
                node_type = source_node.metadata.get('type', 'text')
                if node_type in ['image', 'ocr_text', 'image_caption', 'detected_object']:
                    image_sources.append(source_info)
                else:
                    text_sources.append(source_info)
        
        # Get graph context (simplified)
        graph_context = self._extract_graph_context(query, response)
        
        # Calculate confidence score based on source relevance
        confidence_score = self._calculate_confidence_score(text_sources, image_sources)
        
        # Build provenance information
        provenance = {
            'text_node_ids': [src['id'] for src in text_sources],
            'image_node_ids': [src['id'] for src in image_sources],
            'graph_nodes': list(graph_context.get('nodes', {}).keys()) if graph_context else []
        }
        
        return VisualRAGResult(
            answer=str(response),
            text_sources=text_sources[:max_text_results],
            image_sources=image_sources[:max_image_results] if include_images else [],
            graph_context=graph_context,
            confidence_score=confidence_score,
            provenance=provenance
        )
    
    def _extract_graph_context(self, query: str, response: Response) -> Dict[str, Any]:
        """
        Extract relevant graph context from the response
        
        Args:
            query: Original query
            response: Query response
            
        Returns:
            Dictionary containing graph context information
        """
        try:
            # Get the networkx graph from the KG index
            graph = self.kg_index.get_networkx_graph()
            
            # Extract nodes and edges related to the response
            context_nodes = {}
            context_edges = []
            
            # Simple approach: include nodes mentioned in source nodes
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for source_node in response.source_nodes:
                    node_id = source_node.id_
                    if node_id in graph.nodes:
                        context_nodes[node_id] = {
                            'id': node_id,
                            'text': source_node.text[:100] + "..." if len(source_node.text) > 100 else source_node.text,
                            'type': source_node.metadata.get('type', 'text')
                        }
                        
                        # Add connected edges
                        for neighbor in graph.neighbors(node_id):
                            if neighbor in [sn.id_ for sn in response.source_nodes]:
                                context_edges.append({
                                    'source': node_id,
                                    'target': neighbor,
                                    'relation': graph.edges[node_id, neighbor].get('relation', 'related_to')
                                })
            
            return {
                'nodes': context_nodes,
                'edges': context_edges,
                'total_graph_nodes': len(graph.nodes),
                'total_graph_edges': len(graph.edges)
            }
        except Exception as e:
            print(f"Error extracting graph context: {e}")
            return {'nodes': {}, 'edges': [], 'error': str(e)}
    
    def _calculate_confidence_score(self, text_sources: List[Dict], image_sources: List[Dict]) -> float:
        """
        Calculate confidence score based on source quality and relevance
        
        Args:
            text_sources: List of text source information
            image_sources: List of image source information
            
        Returns:
            Confidence score between 0 and 1
        """
        total_sources = len(text_sources) + len(image_sources)
        if total_sources == 0:
            return 0.0
        
        # Base score from number of sources
        source_score = min(total_sources / 5.0, 1.0)  # Max out at 5 sources
        
        # Adjust for source quality (based on similarity scores)
        quality_scores = []
        for source in text_sources + image_sources:
            score = source.get('score', 0.5)
            quality_scores.append(score)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        
        # Bonus for multimodal sources
        multimodal_bonus = 0.1 if len(image_sources) > 0 and len(text_sources) > 0 else 0.0
        
        final_score = (source_score * 0.6 + avg_quality * 0.4 + multimodal_bonus)
        return min(final_score, 1.0)
    
    def get_image_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for all processed images
        
        Returns:
            Dictionary containing image metadata and node mappings
        """
        return {
            'image_ingestor_metadata': self.image_ingestor.get_image_metadata(),
            'image_node_mapping': self.image_node_mapping,
            'total_images': len(self.image_node_mapping)
        }
    
    def visualize_multimodal_result(self, result: VisualRAGResult) -> Dict[str, Any]:
        """
        Prepare data for visualizing multimodal results
        
        Args:
            result: VisualRAGResult from query
            
        Returns:
            Dictionary with visualization data
        """
        viz_data = {
            'answer': result.answer,
            'confidence': result.confidence_score,
            'source_breakdown': {
                'text_sources': len(result.text_sources),
                'image_sources': len(result.image_sources)
            },
            'graph_visualization': {
                'nodes': result.graph_context.get('nodes', {}),
                'edges': result.graph_context.get('edges', [])
            },
            'image_previews': []
        }
        
        # Add image preview information
        for img_source in result.image_sources:
            if 'image_path' in img_source.get('metadata', {}):
                viz_data['image_previews'].append({
                    'path': img_source['metadata']['image_path'],
                    'type': img_source['metadata'].get('type', 'unknown'),
                    'score': img_source.get('score', 0.0)
                })
        
        return viz_data


def main():
    """
    Example usage of Visual RAG Pipeline
    """
    print("Visual RAG Pipeline module loaded successfully!")
    print("To use this module:")
    print("1. Initialize with an existing KnowledgeGraphIndex")
    print("2. Add images using add_images_to_kg()")
    print("3. Query with query_with_visual_context()")


if __name__ == "__main__":
    main()