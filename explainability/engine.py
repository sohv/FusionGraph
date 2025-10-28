"""
Enhanced explainability module for Visual RAG with Faiss integration
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from retrieval.faiss_retriever import RetrievalResult, ExplainabilityMetadata


@dataclass
class QueryExplanation:
    """Complete explanation for a Visual RAG query response"""
    answer: str
    confidence: Dict[str, Any]
    provenance: List[Dict[str, Any]]
    retrieval_explanation: Optional[ExplainabilityMetadata]
    cot_trace: Optional[List[str]]
    query_metadata: Dict[str, Any]


class ExplainabilityEngine:
    """
    Enhanced explainability engine that provides comprehensive explanations
    for Visual RAG responses including retrieval provenance and confidence scoring
    """
    
    def __init__(self):
        self.explanation_history = []
    
    def create_explanation(self,
                          query: str,
                          answer: str,
                          retrieval_results: List[RetrievalResult],
                          retrieval_metadata: Optional[ExplainabilityMetadata],
                          llm_response_metadata: Optional[Dict[str, Any]] = None,
                          image_sources: Optional[List[Dict[str, Any]]] = None) -> QueryExplanation:
        """
        Create comprehensive explanation for a query response
        
        Args:
            query: Original user query
            answer: Generated answer
            retrieval_results: Results from Faiss retriever
            retrieval_metadata: Explainability metadata from retriever
            llm_response_metadata: Metadata from LLM generation
            image_sources: Image-based evidence if any
            
        Returns:
            Complete query explanation
        """
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            retrieval_results, 
            retrieval_metadata, 
            llm_response_metadata
        )
        
        # Build provenance information
        provenance = self._build_provenance(
            retrieval_results, 
            image_sources or []
        )
        
        # Generate chain-of-thought trace
        cot_trace = self._generate_cot_trace(
            query, 
            retrieval_results, 
            answer
        )
        
        # Query metadata
        query_metadata = {
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'answer_length': len(answer),
            'num_text_sources': len(retrieval_results),
            'num_image_sources': len(image_sources) if image_sources else 0,
            'retrieval_method': 'faiss_hnsw'
        }
        
        explanation = QueryExplanation(
            answer=answer,
            confidence=confidence,
            provenance=provenance,
            retrieval_explanation=retrieval_metadata,
            cot_trace=cot_trace,
            query_metadata=query_metadata
        )
        
        # Store for future analysis
        self.explanation_history.append(asdict(explanation))
        
        return explanation
    
    def _calculate_confidence(self,
                            retrieval_results: List[RetrievalResult],
                            retrieval_metadata: Optional[ExplainabilityMetadata],
                            llm_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate multi-factor confidence score
        """
        if not retrieval_results:
            return {
                'score': 0.0,
                'label': 'very_low',
                'factors': {
                    'retrieval_quality': 0.0,
                    'source_diversity': 0.0,
                    'llm_confidence': 0.0
                },
                'explanation': 'No relevant sources found'
            }
        
        # Retrieval quality (0-1)
        avg_similarity = sum(r.score for r in retrieval_results) / len(retrieval_results)
        retrieval_quality = min(1.0, avg_similarity * 1.2)  # Slight boost for good similarity
        
        # Source diversity (0-1) - more unique sources = higher confidence
        unique_sources = len(set(r.source_id for r in retrieval_results))
        max_possible_sources = min(5, len(retrieval_results))  # Cap at 5 for practical purposes
        source_diversity = unique_sources / max_possible_sources if max_possible_sources > 0 else 0
        
        # LLM confidence (placeholder - could be enhanced with logprobs)
        llm_confidence = llm_metadata.get('confidence', 0.7) if llm_metadata else 0.7
        
        # Weighted combination
        overall_score = (
            0.4 * retrieval_quality + 
            0.3 * source_diversity + 
            0.3 * llm_confidence
        )
        
        # Determine label
        if overall_score >= 0.8:
            label = 'high'
        elif overall_score >= 0.6:
            label = 'medium'
        elif overall_score >= 0.4:
            label = 'low'
        else:
            label = 'very_low'
        
        explanation = (f"Avg similarity: {avg_similarity:.2f}, "
                      f"Sources: {unique_sources}, "
                      f"LLM confidence: {llm_confidence:.2f}")
        
        return {
            'score': round(overall_score, 3),
            'label': label,
            'factors': {
                'retrieval_quality': round(retrieval_quality, 3),
                'source_diversity': round(source_diversity, 3),
                'llm_confidence': round(llm_confidence, 3)
            },
            'explanation': explanation
        }
    
    def _build_provenance(self,
                         retrieval_results: List[RetrievalResult],
                         image_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build detailed provenance information
        """
        provenance = []
        
        # Add text sources
        for i, result in enumerate(retrieval_results):
            # Truncate text for display
            display_text = result.text[:200] + "..." if len(result.text) > 200 else result.text
            
            provenance.append({
                'rank': i + 1,
                'type': 'text',
                'source_id': result.source_id,
                'source_name': result.source_name,
                'similarity_score': round(result.score, 3),
                'text_snippet': display_text,
                'chunk_offsets': result.chunk_offsets,
                'chunk_id': result.chunk_id,
                'embedding_model': result.embedding_model
            })
        
        # Add image sources
        for i, img_source in enumerate(image_sources):
            provenance.append({
                'rank': len(retrieval_results) + i + 1,
                'type': 'image',
                'source_id': img_source.get('source_id', 'unknown'),
                'source_name': img_source.get('source_name', 'Unknown Image'),
                'similarity_score': img_source.get('score', 0.0),
                'image_path': img_source.get('path', ''),
                'ocr_text': img_source.get('ocr_text', '')[:200] + "..." if len(img_source.get('ocr_text', '')) > 200 else img_source.get('ocr_text', ''),
                'bounding_boxes': img_source.get('bounding_boxes', [])
            })
        
        return provenance
    
    def _generate_cot_trace(self,
                           query: str,
                           retrieval_results: List[RetrievalResult],
                           answer: str) -> List[str]:
        """
        Generate a simple chain-of-thought trace
        """
        trace = []
        
        trace.append(f"1. Received query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        if retrieval_results:
            trace.append(f"2. Found {len(retrieval_results)} relevant text chunks from {len(set(r.source_id for r in retrieval_results))} sources")
            
            best_match = max(retrieval_results, key=lambda x: x.score)
            trace.append(f"3. Best matching source: '{best_match.source_name}' (similarity: {best_match.score:.3f})")
            
            trace.append(f"4. Generated answer using retrieved context and knowledge graph connections")
        else:
            trace.append("2. No relevant sources found in knowledge base")
            trace.append("3. Generated answer using only pre-trained knowledge")
        
        trace.append(f"5. Final answer length: {len(answer)} characters")
        
        return trace
    
    def get_explanation_summary(self, explanation: QueryExplanation) -> str:
        """
        Generate a human-readable summary of the explanation
        """
        confidence_emoji = {
            'high': '🟢',
            'medium': '🟡', 
            'low': '🟠',
            'very_low': '🔴'
        }
        
        emoji = confidence_emoji.get(explanation.confidence['label'], '⚪')
        
        summary_parts = [
            f"{emoji} Confidence: {explanation.confidence['label']} ({explanation.confidence['score']:.2f})",
            f"📚 Sources: {len([p for p in explanation.provenance if p['type'] == 'text'])} text, {len([p for p in explanation.provenance if p['type'] == 'image'])} image",
        ]
        
        if explanation.retrieval_explanation:
            summary_parts.append(f"🔍 Search: {explanation.retrieval_explanation.similarity_method}")
        
        return " | ".join(summary_parts)
    
    def export_explanations(self, filepath: str) -> None:
        """Export explanation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.explanation_history, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about explanations generated"""
        if not self.explanation_history:
            return {'total_explanations': 0}
        
        confidence_scores = [exp['confidence']['score'] for exp in self.explanation_history]
        
        return {
            'total_explanations': len(self.explanation_history),
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'confidence_distribution': {
                label: len([exp for exp in self.explanation_history if exp['confidence']['label'] == label])
                for label in ['very_low', 'low', 'medium', 'high']
            }
        }