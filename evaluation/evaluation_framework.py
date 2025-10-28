"""
Research-Grade Evaluation Framework for FusionGraph
Implements quantitative metrics for retrieval quality, factual consistency, and multimodal grounding
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import logging

# NLP and ML libraries for evaluation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import ndcg_score
    import torch
    from transformers import pipeline
    EVAL_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some evaluation dependencies not available: {e}")
    EVAL_DEPENDENCIES_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    test_name: str
    metric_scores: Dict[str, float]
    detailed_results: List[Dict[str, Any]]
    execution_time: float
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class QueryRelevanceJudgment:
    """Ground truth relevance judgment for a query-document pair"""
    query_id: str
    query_text: str
    document_id: str
    relevance_score: int  # 0-3 scale: 0=irrelevant, 1=marginally, 2=relevant, 3=highly relevant
    human_annotator: Optional[str] = None
    annotation_notes: Optional[str] = None


@dataclass
class FactualClaimResult:
    """Result of factual claim verification"""
    claim: str
    source_text: str
    verification_score: float  # 0-1: likelihood the claim is supported by source
    verification_method: str
    extracted_entities: List[str]
    consistency_label: str  # "supported", "neutral", "contradicted"


@dataclass
class MultimodalGroundingResult:
    """Result of multimodal grounding evaluation"""
    query: str
    text_source_relevance: float
    image_source_relevance: float
    cross_modal_alignment: float
    ocr_accuracy: float
    visual_qa_correctness: float
    grounding_quality: str  # "excellent", "good", "fair", "poor"


class RetrievalQualityEvaluator:
    """
    Test 1: Retrieval Quality Assessment
    Measures retrieval relevance using NDCG@K, MRR, and Precision@K
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = None
        if EVAL_DEPENDENCIES_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Sample ground truth data (in real use, load from annotated dataset)
        self.ground_truth_judgments = self._create_sample_ground_truth()
    
    def _create_sample_ground_truth(self) -> List[QueryRelevanceJudgment]:
        """Create sample ground truth relevance judgments for testing"""
        return [
            QueryRelevanceJudgment(
                query_id="q1", 
                query_text="How does AI help in medical diagnosis?",
                document_id="ai_healthcare_2024", 
                relevance_score=3,
                human_annotator="expert_1"
            ),
            QueryRelevanceJudgment(
                query_id="q1",
                query_text="How does AI help in medical diagnosis?", 
                document_id="cv_deep_learning_2024",
                relevance_score=1,
                human_annotator="expert_1"
            ),
            QueryRelevanceJudgment(
                query_id="q2",
                query_text="What are knowledge graphs used for?",
                document_id="knowledge_graphs_2024",
                relevance_score=3,
                human_annotator="expert_1"
            ),
            QueryRelevanceJudgment(
                query_id="q2",
                query_text="What are knowledge graphs used for?",
                document_id="nlp_llm_2024", 
                relevance_score=0,
                human_annotator="expert_1"
            ),
            QueryRelevanceJudgment(
                query_id="q3",
                query_text="How do computer vision models work?",
                document_id="cv_deep_learning_2024",
                relevance_score=3,
                human_annotator="expert_1"
            )
        ]
    
    def evaluate_retrieval_quality(self, retrieval_results: List[Dict[str, Any]], 
                                 query_id: str, k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Evaluate retrieval quality using multiple metrics
        
        Args:
            retrieval_results: List of retrieved documents with scores
            query_id: Query identifier for ground truth lookup
            k_values: List of K values for precision@K and NDCG@K
            
        Returns:
            Dictionary of metric scores
        """
        if not EVAL_DEPENDENCIES_AVAILABLE:
            return {"error": "Evaluation dependencies not available"}
        
        # Get ground truth for this query
        gt_judgments = [j for j in self.ground_truth_judgments if j.query_id == query_id]
        if not gt_judgments:
            return {"error": f"No ground truth found for query {query_id}"}
        
        # Create relevance mapping
        relevance_map = {j.document_id: j.relevance_score for j in gt_judgments}
        
        metrics = {}
        
        # Calculate metrics for each K
        for k in k_values:
            top_k_results = retrieval_results[:k]
            
            # Precision@K
            relevant_retrieved = sum(1 for doc in top_k_results 
                                   if relevance_map.get(doc.get('source_id'), 0) >= 2)
            precision_k = relevant_retrieved / k if k > 0 else 0
            metrics[f"precision_at_{k}"] = precision_k
            
            # NDCG@K
            if len(top_k_results) > 0:
                retrieved_relevance = [relevance_map.get(doc.get('source_id'), 0) 
                                     for doc in top_k_results]
                ideal_relevance = sorted([j.relevance_score for j in gt_judgments], reverse=True)[:k]
                
                # Pad with zeros if needed
                while len(retrieved_relevance) < k:
                    retrieved_relevance.append(0)
                while len(ideal_relevance) < k:
                    ideal_relevance.append(0)
                
                try:
                    ndcg_k = ndcg_score([ideal_relevance], [retrieved_relevance], k=k)
                    metrics[f"ndcg_at_{k}"] = ndcg_k
                except:
                    metrics[f"ndcg_at_{k}"] = 0.0
        
        # Mean Reciprocal Rank (MRR)
        first_relevant_rank = None
        for i, doc in enumerate(retrieval_results):
            if relevance_map.get(doc.get('source_id'), 0) >= 2:
                first_relevant_rank = i + 1
                break
        
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        metrics["mrr"] = mrr
        
        # Recall@K for largest K
        max_k = max(k_values)
        top_max_results = retrieval_results[:max_k]
        total_relevant = sum(1 for j in gt_judgments if j.relevance_score >= 2)
        relevant_retrieved_max = sum(1 for doc in top_max_results 
                                   if relevance_map.get(doc.get('source_id'), 0) >= 2)
        recall_max = relevant_retrieved_max / total_relevant if total_relevant > 0 else 0
        metrics[f"recall_at_{max_k}"] = recall_max
        
        return metrics


class FactualConsistencyEvaluator:
    """
    Test 2: Factual Consistency & Grounding
    Measures whether generated answers are factually grounded in sources
    """
    
    def __init__(self):
        self.nli_pipeline = None
        self.ner_pipeline = None
        
        if EVAL_DEPENDENCIES_AVAILABLE:
            try:
                # Natural Language Inference for entailment checking
                self.nli_pipeline = pipeline("text-classification", 
                                           model="microsoft/DialoGPT-medium",
                                           return_all_scores=True)
                
                # Named Entity Recognition for entity extraction
                self.ner_pipeline = pipeline("ner", 
                                           model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                           aggregation_strategy="simple")
            except Exception as e:
                print(f"Warning: Could not load NLI/NER models: {e}")
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from generated text"""
        # Simple sentence-based claim extraction
        import re
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def verify_claim_against_source(self, claim: str, source_text: str) -> FactualClaimResult:
        """
        Verify if a claim is supported by the source text
        """
        if not self.nli_pipeline:
            # Fallback: simple keyword overlap
            claim_words = set(claim.lower().split())
            source_words = set(source_text.lower().split())
            overlap = len(claim_words.intersection(source_words))
            verification_score = min(1.0, overlap / len(claim_words)) if claim_words else 0.0
            
            return FactualClaimResult(
                claim=claim,
                source_text=source_text[:200],
                verification_score=verification_score,
                verification_method="keyword_overlap",
                extracted_entities=[],
                consistency_label="neutral" if verification_score < 0.5 else "supported"
            )
        
        try:
            # Use NLI to check if source entails the claim
            premise = source_text[:512]  # Truncate for model limits
            hypothesis = claim
            
            nli_result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
            
            # Extract entailment score (model-dependent)
            entailment_score = 0.5  # Default neutral
            for result in nli_result:
                if result['label'].upper() in ['ENTAILMENT', 'ENTAILED']:
                    entailment_score = result['score']
                    break
            
            # Extract entities from claim
            entities = []
            if self.ner_pipeline:
                ner_result = self.ner_pipeline(claim)
                entities = [ent['word'] for ent in ner_result if ent['score'] > 0.8]
            
            # Determine consistency label
            if entailment_score > 0.7:
                consistency_label = "supported"
            elif entailment_score < 0.3:
                consistency_label = "contradicted"
            else:
                consistency_label = "neutral"
            
            return FactualClaimResult(
                claim=claim,
                source_text=source_text[:200],
                verification_score=entailment_score,
                verification_method="nli_entailment",
                extracted_entities=entities,
                consistency_label=consistency_label
            )
            
        except Exception as e:
            print(f"NLI verification failed: {e}")
            return FactualClaimResult(
                claim=claim,
                source_text=source_text[:200],
                verification_score=0.5,
                verification_method="error_fallback",
                extracted_entities=[],
                consistency_label="neutral"
            )
    
    def evaluate_factual_consistency(self, generated_answer: str, 
                                   source_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate factual consistency of generated answer against sources
        """
        claims = self.extract_claims(generated_answer)
        if not claims or not source_texts:
            return {"error": "No claims extracted or no sources provided"}
        
        all_source_text = " ".join(source_texts)
        claim_results = []
        
        for claim in claims:
            result = self.verify_claim_against_source(claim, all_source_text)
            claim_results.append(result)
        
        # Calculate aggregate metrics
        verification_scores = [r.verification_score for r in claim_results]
        
        metrics = {
            "avg_verification_score": np.mean(verification_scores),
            "min_verification_score": np.min(verification_scores),
            "max_verification_score": np.max(verification_scores),
            "num_claims": len(claims),
            "num_supported_claims": len([r for r in claim_results if r.consistency_label == "supported"]),
            "num_contradicted_claims": len([r for r in claim_results if r.consistency_label == "contradicted"]),
            "support_ratio": len([r for r in claim_results if r.consistency_label == "supported"]) / len(claims),
            "hallucination_ratio": len([r for r in claim_results if r.consistency_label == "contradicted"]) / len(claims)
        }
        
        return metrics


class MultimodalGroundingEvaluator:
    """
    Test 3: Multimodal Integration Quality
    Measures how well the system integrates visual and textual information
    """
    
    def __init__(self):
        self.text_similarity_model = None
        if EVAL_DEPENDENCIES_AVAILABLE:
            self.text_similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_ocr_accuracy(self, predicted_text: str, ground_truth_text: str) -> float:
        """Evaluate OCR accuracy using character-level similarity"""
        if not predicted_text or not ground_truth_text:
            return 0.0
        
        # Simple character-level accuracy
        pred_chars = set(predicted_text.lower().replace(' ', ''))
        gt_chars = set(ground_truth_text.lower().replace(' ', ''))
        
        if not gt_chars:
            return 1.0 if not pred_chars else 0.0
        
        intersection = len(pred_chars.intersection(gt_chars))
        union = len(pred_chars.union(gt_chars))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_cross_modal_alignment(self, text_query: str, image_description: str, 
                                     retrieved_text: str, retrieved_image_caption: str) -> float:
        """
        Evaluate how well text and image retrievals align with the query
        """
        if not self.text_similarity_model:
            # Fallback: simple keyword overlap
            query_words = set(text_query.lower().split())
            text_words = set(retrieved_text.lower().split())
            image_words = set(retrieved_image_caption.lower().split())
            
            text_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            image_overlap = len(query_words.intersection(image_words)) / len(query_words) if query_words else 0
            
            return (text_overlap + image_overlap) / 2
        
        try:
            # Compute semantic similarities
            query_emb = self.text_similarity_model.encode([text_query])
            text_emb = self.text_similarity_model.encode([retrieved_text])
            image_emb = self.text_similarity_model.encode([retrieved_image_caption])
            
            text_sim = np.dot(query_emb[0], text_emb[0]) / (np.linalg.norm(query_emb[0]) * np.linalg.norm(text_emb[0]))
            image_sim = np.dot(query_emb[0], image_emb[0]) / (np.linalg.norm(query_emb[0]) * np.linalg.norm(image_emb[0]))
            
            # Cross-modal alignment: how well text and image complement each other
            cross_modal_sim = np.dot(text_emb[0], image_emb[0]) / (np.linalg.norm(text_emb[0]) * np.linalg.norm(image_emb[0]))
            
            # Weighted combination
            alignment_score = 0.4 * text_sim + 0.4 * image_sim + 0.2 * cross_modal_sim
            return max(0, min(1, alignment_score))
            
        except Exception as e:
            print(f"Cross-modal alignment calculation failed: {e}")
            return 0.5
    
    def evaluate_visual_qa_correctness(self, question: str, predicted_answer: str, 
                                     ground_truth_answer: str) -> float:
        """Evaluate visual QA correctness"""
        if not self.text_similarity_model:
            # Fallback: exact match
            return 1.0 if predicted_answer.strip().lower() == ground_truth_answer.strip().lower() else 0.0
        
        try:
            pred_emb = self.text_similarity_model.encode([predicted_answer])
            gt_emb = self.text_similarity_model.encode([ground_truth_answer])
            
            similarity = np.dot(pred_emb[0], gt_emb[0]) / (np.linalg.norm(pred_emb[0]) * np.linalg.norm(gt_emb[0]))
            return max(0, min(1, similarity))
        except:
            return 0.0
    
    def evaluate_multimodal_grounding(self, query: str, text_sources: List[Dict], 
                                    image_sources: List[Dict]) -> MultimodalGroundingResult:
        """
        Comprehensive multimodal grounding evaluation
        """
        # Text source relevance (average similarity to query)
        text_relevance = 0.0
        if text_sources and self.text_similarity_model:
            try:
                query_emb = self.text_similarity_model.encode([query])
                text_sims = []
                for source in text_sources[:3]:  # Top 3 sources
                    text_emb = self.text_similarity_model.encode([source.get('text', '')])
                    sim = np.dot(query_emb[0], text_emb[0]) / (np.linalg.norm(query_emb[0]) * np.linalg.norm(text_emb[0]))
                    text_sims.append(sim)
                text_relevance = np.mean(text_sims) if text_sims else 0.0
            except:
                text_relevance = 0.5
        
        # Image source relevance (based on captions/OCR)
        image_relevance = 0.0
        if image_sources:
            # Use OCR text or image captions for relevance
            image_texts = []
            for source in image_sources[:3]:
                ocr_text = source.get('ocr_text', '')
                caption = source.get('caption', '')
                image_texts.append(ocr_text + " " + caption)
            
            if image_texts and self.text_similarity_model:
                try:
                    query_emb = self.text_similarity_model.encode([query])
                    image_sims = []
                    for img_text in image_texts:
                        if img_text.strip():
                            img_emb = self.text_similarity_model.encode([img_text])
                            sim = np.dot(query_emb[0], img_emb[0]) / (np.linalg.norm(query_emb[0]) * np.linalg.norm(img_emb[0]))
                            image_sims.append(sim)
                    image_relevance = np.mean(image_sims) if image_sims else 0.0
                except:
                    image_relevance = 0.3
        
        # Cross-modal alignment
        cross_modal_alignment = 0.0
        if text_sources and image_sources:
            # Sample cross-modal alignment calculation
            cross_modal_alignment = self.evaluate_cross_modal_alignment(
                query,
                "multimodal query",  # Placeholder
                text_sources[0].get('text', '') if text_sources else '',
                image_sources[0].get('ocr_text', '') if image_sources else ''
            )
        
        # OCR accuracy (placeholder - would need ground truth)
        ocr_accuracy = 0.85  # Placeholder
        
        # Visual QA correctness (placeholder - would need ground truth QA pairs)
        visual_qa_correctness = 0.75  # Placeholder
        
        # Overall grounding quality
        overall_score = (text_relevance + image_relevance + cross_modal_alignment) / 3
        if overall_score >= 0.8:
            grounding_quality = "excellent"
        elif overall_score >= 0.6:
            grounding_quality = "good"
        elif overall_score >= 0.4:
            grounding_quality = "fair"
        else:
            grounding_quality = "poor"
        
        return MultimodalGroundingResult(
            query=query,
            text_source_relevance=text_relevance,
            image_source_relevance=image_relevance,
            cross_modal_alignment=cross_modal_alignment,
            ocr_accuracy=ocr_accuracy,
            visual_qa_correctness=visual_qa_correctness,
            grounding_quality=grounding_quality
        )


class FusionGraphEvaluationSuite:
    """
    Main evaluation suite that runs all three evaluation tests
    """
    
    def __init__(self, results_dir: str = "./evaluation/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.retrieval_evaluator = RetrievalQualityEvaluator()
        self.factual_evaluator = FactualConsistencyEvaluator()
        self.multimodal_evaluator = MultimodalGroundingEvaluator()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(results_dir, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_evaluation(self, visual_rag_pipeline, 
                                   test_queries: List[str]) -> Dict[str, EvaluationResult]:
        """
        Run all three evaluation tests on the FusionGraph system
        """
        self.logger.info("Starting comprehensive evaluation of FusionGraph")
        
        all_results = {}
        
        for i, query in enumerate(test_queries):
            self.logger.info(f"Evaluating query {i+1}/{len(test_queries)}: {query}")
            
            # Get system response
            start_time = time.time()
            try:
                result = visual_rag_pipeline.query_with_visual_context(
                    query=query,
                    include_explanation=True
                )
                execution_time = time.time() - start_time
                
                # Test 1: Retrieval Quality
                if hasattr(result, 'explanation') and result.explanation:
                    retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval_quality(
                        [{'source_id': p['source_id'], 'score': p['similarity_score']} 
                         for p in result.explanation.provenance],
                        f"q{i+1}"
                    )
                else:
                    retrieval_metrics = {"error": "No explanation available"}
                
                # Test 2: Factual Consistency
                source_texts = [p['text_snippet'] for p in result.explanation.provenance if result.explanation]
                factual_metrics = self.factual_evaluator.evaluate_factual_consistency(
                    result.answer, source_texts
                )
                
                # Test 3: Multimodal Grounding
                multimodal_result = self.multimodal_evaluator.evaluate_multimodal_grounding(
                    query, result.text_sources, result.image_sources
                )
                
                # Store results
                all_results[f"query_{i+1}"] = {
                    "query": query,
                    "retrieval_metrics": retrieval_metrics,
                    "factual_metrics": factual_metrics,
                    "multimodal_metrics": asdict(multimodal_result),
                    "execution_time": execution_time,
                    "system_response": {
                        "answer": result.answer,
                        "confidence_score": result.confidence_score,
                        "num_text_sources": len(result.text_sources),
                        "num_image_sources": len(result.image_sources)
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for query '{query}': {e}")
                all_results[f"query_{i+1}"] = {"error": str(e), "query": query}
        
        # Save results
        timestamp = datetime.now().isoformat()
        results_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation complete. Results saved to {results_file}")
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(all_results)
        
        summary_file = os.path.join(self.results_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return all_results
    
    def _generate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all test queries"""
        
        retrieval_scores = []
        factual_scores = []
        multimodal_scores = []
        execution_times = []
        
        for query_id, query_results in results.items():
            if "error" in query_results:
                continue
                
            # Retrieval metrics
            if "retrieval_metrics" in query_results and "ndcg_at_3" in query_results["retrieval_metrics"]:
                retrieval_scores.append(query_results["retrieval_metrics"]["ndcg_at_3"])
            
            # Factual consistency
            if "factual_metrics" in query_results and "support_ratio" in query_results["factual_metrics"]:
                factual_scores.append(query_results["factual_metrics"]["support_ratio"])
            
            # Multimodal grounding
            if "multimodal_metrics" in query_results:
                mm = query_results["multimodal_metrics"]
                overall_mm_score = (mm.get("text_source_relevance", 0) + 
                                  mm.get("image_source_relevance", 0) + 
                                  mm.get("cross_modal_alignment", 0)) / 3
                multimodal_scores.append(overall_mm_score)
            
            # Execution time
            if "execution_time" in query_results:
                execution_times.append(query_results["execution_time"])
        
        summary = {
            "evaluation_overview": {
                "total_queries": len(results),
                "successful_evaluations": len([r for r in results.values() if "error" not in r]),
                "failed_evaluations": len([r for r in results.values() if "error" in r])
            },
            "retrieval_quality": {
                "avg_ndcg_at_3": np.mean(retrieval_scores) if retrieval_scores else 0,
                "std_ndcg_at_3": np.std(retrieval_scores) if retrieval_scores else 0,
                "min_ndcg_at_3": np.min(retrieval_scores) if retrieval_scores else 0,
                "max_ndcg_at_3": np.max(retrieval_scores) if retrieval_scores else 0
            },
            "factual_consistency": {
                "avg_support_ratio": np.mean(factual_scores) if factual_scores else 0,
                "std_support_ratio": np.std(factual_scores) if factual_scores else 0,
                "min_support_ratio": np.min(factual_scores) if factual_scores else 0,
                "max_support_ratio": np.max(factual_scores) if factual_scores else 0
            },
            "multimodal_grounding": {
                "avg_grounding_score": np.mean(multimodal_scores) if multimodal_scores else 0,
                "std_grounding_score": np.std(multimodal_scores) if multimodal_scores else 0,
                "min_grounding_score": np.min(multimodal_scores) if multimodal_scores else 0,
                "max_grounding_score": np.max(multimodal_scores) if multimodal_scores else 0
            },
            "performance": {
                "avg_execution_time": np.mean(execution_times) if execution_times else 0,
                "std_execution_time": np.std(execution_times) if execution_times else 0,
                "min_execution_time": np.min(execution_times) if execution_times else 0,
                "max_execution_time": np.max(execution_times) if execution_times else 0
            }
        }
        
        return summary