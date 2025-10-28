import os
import sys
import tempfile
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from retrieval.faiss_retriever import FaissRetriever, RetrievalResult
    from explainability.engine import ExplainabilityEngine
    FAISS_AVAILABLE = True
except ImportError as e:
    print(f"Faiss not available: {e}")
    FAISS_AVAILABLE = False


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="Faiss not installed")
class TestFaissIntegration:
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.retriever = FaissRetriever(
            index_path=os.path.join(self.temp_dir, "test_index"),
            metadata_path=os.path.join(self.temp_dir, "test_metadata.json")
        )
        self.explainability_engine = ExplainabilityEngine()
    
    def test_document_indexing(self):
        """Test adding documents to Faiss index"""
        documents = [
            {
                'text': 'Artificial intelligence is transforming healthcare by enabling faster diagnosis and personalized treatment plans.',
                'source_id': 'ai_healthcare_1',
                'source_name': 'AI in Healthcare Study',
                'metadata': {'domain': 'healthcare', 'year': 2024}
            },
            {
                'text': 'Machine learning algorithms can analyze medical images to detect anomalies with high accuracy.',
                'source_id': 'ml_medical_imaging_1', 
                'source_name': 'Medical Imaging ML Research',
                'metadata': {'domain': 'medical_imaging', 'year': 2024}
            }
        ]
        
        self.retriever.add_documents(documents)
        
        # Verify indexing
        assert self.retriever.index.ntotal == 2  # Should have 2 chunks
        assert len(self.retriever.chunk_metadata) >= 2
    
    def test_retrieval_with_explanations(self):
        """Test retrieval with explainability metadata"""
        # First index some documents
        documents = [
            {
                'text': 'Deep learning models are revolutionizing computer vision tasks by achieving human-level performance in image recognition.',
                'source_id': 'deep_learning_cv_1',
                'source_name': 'Deep Learning in Computer Vision',
                'metadata': {'domain': 'computer_vision'}
            },
            {
                'text': 'Natural language processing has made significant advances with transformer architectures like BERT and GPT.',
                'source_id': 'nlp_transformers_1',
                'source_name': 'NLP Transformer Models',
                'metadata': {'domain': 'nlp'}
            }
        ]
        
        self.retriever.add_documents(documents)
        
        # Test retrieval
        query = "How do deep learning models work in computer vision?"
        results, metadata = self.retriever.retrieve(query, top_k=2, return_explanations=True)
        
        # Verify results
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert results[0].score > 0
        assert metadata is not None
        assert metadata.similarity_method == "cosine (L2 normalized)"
        assert metadata.confidence_score > 0
    
    def test_explainability_engine(self):
        """Test explainability engine with mock retrieval results"""
        # Create mock retrieval results
        mock_results = [
            RetrievalResult(
                chunk_id="test_chunk_1",
                text="AI models can process data efficiently",
                score=0.85,
                source_id="ai_doc_1",
                source_name="AI Fundamentals",
                chunk_offsets=(0, 100),
                embedding_model="test_model",
                retrieval_time="2024-10-29T10:00:00",
                metadata={}
            )
        ]
        
        # Create explanation
        explanation = self.explainability_engine.create_explanation(
            query="What are AI models?",
            answer="AI models are computational systems that can learn from data.",
            retrieval_results=mock_results,
            retrieval_metadata=None
        )
        
        # Verify explanation
        assert explanation.answer == "AI models are computational systems that can learn from data."
        assert explanation.confidence['score'] > 0
        assert len(explanation.provenance) == 1
        assert explanation.provenance[0]['source_name'] == "AI Fundamentals"
        assert len(explanation.cot_trace) > 0
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # High quality results
        high_quality_results = [
            RetrievalResult(
                chunk_id="hq_chunk_1", text="High quality match", score=0.95,
                source_id="source_1", source_name="Source 1", chunk_offsets=(0, 50),
                embedding_model="test", retrieval_time="2024-10-29T10:00:00", metadata={}
            ),
            RetrievalResult(
                chunk_id="hq_chunk_2", text="Another high quality match", score=0.90,
                source_id="source_2", source_name="Source 2", chunk_offsets=(0, 50),
                embedding_model="test", retrieval_time="2024-10-29T10:00:00", metadata={}
            )
        ]
        
        explanation = self.explainability_engine.create_explanation(
            query="Test query",
            answer="Test answer",
            retrieval_results=high_quality_results,
            retrieval_metadata=None
        )
        
        # Should have high confidence with diverse, high-quality sources
        assert explanation.confidence['score'] > 0.7
        assert explanation.confidence['label'] in ['high', 'medium']
        
        # Low quality results
        low_quality_results = [
            RetrievalResult(
                chunk_id="lq_chunk_1", text="Low quality match", score=0.3,
                source_id="source_1", source_name="Source 1", chunk_offsets=(0, 50),
                embedding_model="test", retrieval_time="2024-10-29T10:00:00", metadata={}
            )
        ]
        
        explanation_low = self.explainability_engine.create_explanation(
            query="Test query",
            answer="Test answer", 
            retrieval_results=low_quality_results,
            retrieval_metadata=None
        )
        
        # Should have lower confidence
        assert explanation_low.confidence['score'] < explanation.confidence['score']


if __name__ == "__main__":
    # Run basic functionality test if Faiss is available
    if FAISS_AVAILABLE:
        print("Testing Faiss integration...")
        
        test = TestFaissIntegration()
        test.setup_method()
        
        try:
            test.test_document_indexing()
            print("✅ Document indexing test passed")
            
            test.test_retrieval_with_explanations() 
            print("✅ Retrieval with explanations test passed")
            
            test.test_explainability_engine()
            print("✅ Explainability engine test passed")
            
            test.test_confidence_calculation()
            print("✅ Confidence calculation test passed")
            
            print("🎉 All integration tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("⚠️ Faiss not available - install with: pip install faiss-cpu")