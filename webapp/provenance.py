"""
Provenance Extraction Module
Extracts and manages provenance information for RAG responses
"""

import json
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.schema import BaseNode
from llama_index.core.base.response.schema import Response


@dataclass
class ProvenanceNode:
    """
    Represents a node in the provenance graph
    """
    id: str
    text: str
    node_type: str
    source_file: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ProvenanceEdge:
    """
    Represents an edge in the provenance graph
    """
    source: str
    target: str
    relation: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ProvenanceTrace:
    """
    Complete provenance trace for a query response
    """
    query: str
    answer: str
    nodes: List[ProvenanceNode]
    edges: List[ProvenanceEdge]
    reasoning_path: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


class ProvenanceExtractor:
    """
    Extracts provenance information from RAG query responses
    """
    
    def __init__(self, knowledge_graph_index: KnowledgeGraphIndex):
        """
        Initialize provenance extractor
        
        Args:
            knowledge_graph_index: The knowledge graph index to extract from
        """
        self.kg_index = knowledge_graph_index
        self.graph_store = knowledge_graph_index.storage_context.graph_store
        
    def extract_subgraph(self, 
                        source_nodes: List[BaseNode], 
                        max_hops: int = 2,
                        max_nodes: int = 50) -> Tuple[Dict[str, ProvenanceNode], List[ProvenanceEdge]]:
        """
        Extract relevant subgraph around source nodes
        
        Args:
            source_nodes: Source nodes from the query response
            max_hops: Maximum number of hops to explore
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Tuple of (provenance_nodes_dict, provenance_edges)
        """
        try:
            # Get the full networkx graph
            full_graph = self.kg_index.get_networkx_graph()
            
            # Start with source node IDs
            source_node_ids = {node.id_ for node in source_nodes}
            
            # Perform BFS to find connected nodes within max_hops
            subgraph_nodes = set(source_node_ids)
            current_layer = source_node_ids
            
            for hop in range(max_hops):
                next_layer = set()
                for node_id in current_layer:
                    if node_id in full_graph:
                        neighbors = set(full_graph.neighbors(node_id))
                        next_layer.update(neighbors)
                
                subgraph_nodes.update(next_layer)
                current_layer = next_layer
                
                # Stop if we have enough nodes
                if len(subgraph_nodes) >= max_nodes:
                    break
            
            # Limit to max_nodes
            if len(subgraph_nodes) > max_nodes:
                # Prioritize source nodes and their immediate neighbors
                priority_nodes = set(source_node_ids)
                for node_id in source_node_ids:
                    if node_id in full_graph:
                        priority_nodes.update(full_graph.neighbors(node_id))
                
                remaining_slots = max_nodes - len(priority_nodes)
                other_nodes = subgraph_nodes - priority_nodes
                subgraph_nodes = priority_nodes.union(list(other_nodes)[:remaining_slots])
            
            # Create provenance nodes
            provenance_nodes = {}
            source_node_map = {node.id_: node for node in source_nodes}
            
            for node_id in subgraph_nodes:
                if node_id in full_graph.nodes:
                    graph_node_data = full_graph.nodes[node_id]
                    
                    # Get additional info if this is a source node
                    if node_id in source_node_map:
                        source_node = source_node_map[node_id]
                        text = source_node.text[:200] + "..." if len(source_node.text) > 200 else source_node.text
                        node_type = source_node.metadata.get('type', 'text')
                        metadata = source_node.metadata
                        confidence = getattr(source_node, 'score', 0.8)
                    else:
                        text = str(graph_node_data.get('text', node_id))[:100] + "..."
                        node_type = 'graph_node'
                        metadata = graph_node_data
                        confidence = 0.5
                    
                    provenance_nodes[node_id] = ProvenanceNode(
                        id=node_id,
                        text=text,
                        node_type=node_type,
                        confidence=confidence,
                        metadata=metadata
                    )
            
            # Create provenance edges
            provenance_edges = []
            for source_id, target_id in full_graph.edges():
                if source_id in subgraph_nodes and target_id in subgraph_nodes:
                    edge_data = full_graph.edges[source_id, target_id]
                    relation = edge_data.get('relation', 'connected_to')
                    
                    provenance_edges.append(ProvenanceEdge(
                        source=source_id,
                        target=target_id,
                        relation=relation,
                        confidence=0.7,  # Default confidence for graph edges
                        metadata=edge_data
                    ))
            
            return provenance_nodes, provenance_edges
            
        except Exception as e:
            print(f"Error extracting subgraph: {e}")
            return {}, []
    
    def create_reasoning_path(self, 
                            source_nodes: List[BaseNode],
                            answer: str) -> List[str]:
        """
        Create a reasoning path showing how the answer was derived
        
        Args:
            source_nodes: Source nodes used in the response
            answer: The generated answer
            
        Returns:
            List of reasoning steps
        """
        reasoning_steps = []
        
        # Step 1: Query processing
        reasoning_steps.append("1. Query processed and embedded for semantic search")
        
        # Step 2: Source retrieval
        text_sources = []
        image_sources = []
        
        for node in source_nodes:
            node_type = node.metadata.get('type', 'text')
            if node_type in ['image', 'ocr_text', 'image_caption', 'detected_object']:
                image_sources.append(node)
            else:
                text_sources.append(node)
        
        if text_sources:
            reasoning_steps.append(f"2. Retrieved {len(text_sources)} relevant text passages")
        
        if image_sources:
            reasoning_steps.append(f"3. Retrieved {len(image_sources)} relevant image-based sources")
        
        # Step 3: Knowledge graph traversal
        reasoning_steps.append("4. Explored knowledge graph connections between sources")
        
        # Step 4: Answer generation
        reasoning_steps.append("5. Generated answer using retrieved context and graph relations")
        
        return reasoning_steps
    
    def extract_full_provenance(self, 
                               query: str,
                               response: Response,
                               confidence_score: float = 0.0) -> ProvenanceTrace:
        """
        Extract complete provenance trace for a query response
        
        Args:
            query: Original query
            response: Query response object
            confidence_score: Overall confidence score
            
        Returns:
            Complete provenance trace
        """
        # Extract source nodes
        source_nodes = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            source_nodes = response.source_nodes
        
        # Extract subgraph
        provenance_nodes, provenance_edges = self.extract_subgraph(source_nodes)
        
        # Create reasoning path
        reasoning_path = self.create_reasoning_path(source_nodes, str(response))
        
        # Compile metadata
        metadata = {
            'total_source_nodes': len(source_nodes),
            'subgraph_size': len(provenance_nodes),
            'edge_count': len(provenance_edges),
            'reasoning_steps': len(reasoning_path),
            'query_timestamp': datetime.now().isoformat()
        }
        
        return ProvenanceTrace(
            query=query,
            answer=str(response),
            nodes=list(provenance_nodes.values()),
            edges=provenance_edges,
            reasoning_path=reasoning_path,
            confidence_score=confidence_score,
            metadata=metadata
        )
    
    def export_provenance_json(self, trace: ProvenanceTrace) -> str:
        """
        Export provenance trace to JSON format
        
        Args:
            trace: Provenance trace to export
            
        Returns:
            JSON string representation
        """
        def serialize_node(node: ProvenanceNode) -> Dict[str, Any]:
            return {
                'id': node.id,
                'text': node.text,
                'node_type': node.node_type,
                'source_file': node.source_file,
                'confidence': node.confidence,
                'metadata': node.metadata or {}
            }
        
        def serialize_edge(edge: ProvenanceEdge) -> Dict[str, Any]:
            return {
                'source': edge.source,
                'target': edge.target,
                'relation': edge.relation,
                'confidence': edge.confidence,
                'metadata': edge.metadata or {}
            }
        
        export_data = {
            'query': trace.query,
            'answer': trace.answer,
            'nodes': [serialize_node(node) for node in trace.nodes],
            'edges': [serialize_edge(edge) for edge in trace.edges],
            'reasoning_path': trace.reasoning_path,
            'confidence_score': trace.confidence_score,
            'metadata': trace.metadata
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def visualize_provenance_graph(self, trace: ProvenanceTrace) -> Dict[str, Any]:
        """
        Prepare provenance data for graph visualization
        
        Args:
            trace: Provenance trace to visualize
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        viz_nodes = []
        for node in trace.nodes:
            viz_nodes.append({
                'id': node.id,
                'label': node.text[:30] + "..." if len(node.text) > 30 else node.text,
                'type': node.node_type,
                'confidence': node.confidence,
                'size': 10 + int(node.confidence * 20)  # Size based on confidence
            })
        
        viz_edges = []
        for edge in trace.edges:
            viz_edges.append({
                'source': edge.source,
                'target': edge.target,
                'label': edge.relation,
                'confidence': edge.confidence
            })
        
        return {
            'nodes': viz_nodes,
            'edges': viz_edges,
            'reasoning_path': trace.reasoning_path,
            'metadata': trace.metadata
        }


def main():
    """
    Example usage of ProvenanceExtractor
    """
    print("ProvenanceExtractor module loaded successfully!")
    print("Usage: Initialize with a KnowledgeGraphIndex and call extract_full_provenance()")


if __name__ == "__main__":
    main()