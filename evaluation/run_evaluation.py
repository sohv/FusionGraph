#!/usr/bin/env python3
"""
FusionGraph Evaluation Runner
Run comprehensive research-grade evaluation tests
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation_framework import FusionGraphEvaluationSuite


def create_sample_test_queries():
    """Create sample test queries for evaluation"""
    return [
        "How does artificial intelligence improve medical diagnosis accuracy?",
        "What are the main applications of computer vision in healthcare?", 
        "How do knowledge graphs enhance information retrieval systems?",
        "What role do large language models play in natural language processing?",
        "How can machine learning be applied to analyze medical images?",
        "What are the benefits of using transformers in NLP tasks?",
        "How do neural networks process visual information?",
        "What techniques are used for multimodal AI systems?"
    ]


def create_test_documents_if_needed():
    """Create sample documents for testing if they don't exist"""
    docs_dir = "./documents/text"
    os.makedirs(docs_dir, exist_ok=True)
    
    sample_docs = {
        "ai_healthcare_2024.txt": """Artificial intelligence is revolutionizing healthcare by enabling faster and more accurate medical diagnosis. Machine learning algorithms can analyze medical images, electronic health records, and genomic data to identify patterns that human doctors might miss. AI-powered diagnostic tools have shown remarkable success in detecting diseases like cancer, diabetic retinopathy, and cardiovascular conditions at early stages. The integration of AI in healthcare has led to personalized treatment plans, predictive analytics for patient outcomes, and automated analysis of complex medical data. Studies have demonstrated that AI systems can achieve diagnostic accuracy comparable to or exceeding that of specialist physicians in certain domains.""",
        
        "cv_deep_learning_2024.txt": """Computer vision technology has advanced significantly with deep learning models achieving human-level performance in image recognition tasks. Convolutional neural networks (CNNs) and transformer architectures like Vision Transformers (ViTs) are being used extensively for medical imaging, autonomous vehicles, surveillance systems, and industrial automation. These models can process and analyze visual information at unprecedented speed and accuracy. In healthcare, computer vision applications include automated analysis of X-rays, MRI scans, CT scans, and pathology images. The technology enables real-time detection of anomalies, automated measurement of anatomical structures, and assistance in surgical procedures through augmented reality systems.""",
        
        "nlp_llm_2024.txt": """Natural language processing has been transformed by large language models like GPT, BERT, Claude, and their variants. These models can understand context, generate human-like text, and perform complex reasoning tasks across multiple domains including legal, medical, and scientific texts. LLMs have enabled breakthrough applications in conversational AI, automatic summarization, machine translation, code generation, and question answering systems. The transformer architecture has become the foundation for most modern NLP systems, enabling models to process long sequences of text and capture complex semantic relationships. Recent advances include multimodal models that can process both text and images simultaneously.""",
        
        "knowledge_graphs_2024.txt": """Knowledge graphs provide structured representations of information that enable better reasoning and inference capabilities in AI systems. They connect entities through relationships and can be used for recommendation systems, question answering, semantic search applications, and knowledge discovery. Knowledge graphs integrate information from multiple sources and create a unified view of knowledge that can be queried and reasoned over. In enterprise applications, knowledge graphs help in data integration, compliance monitoring, risk assessment, and decision support systems. The combination of knowledge graphs with large language models has created powerful hybrid systems that can leverage both structured knowledge and natural language understanding."""
    }
    
    for filename, content in sample_docs.items():
        filepath = os.path.join(docs_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created sample document: {filepath}")


def run_evaluation_suite():
    """Run the comprehensive evaluation suite"""
    
    # Ensure test documents exist
    create_test_documents_if_needed()
    
    # Import FusionGraph components
    try:
        from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, Settings, StorageContext
        from llama_index.core.graph_stores import SimpleGraphStore
        from llama_index.llms.huggingface import HuggingFaceLLM
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from llama_index.embeddings.langchain import LangchainEmbedding
        from pipeline.visual_rag import VisualRAGPipeline
        
        print("✅ Successfully imported FusionGraph components")
        
    except ImportError as e:
        print(f"❌ Failed to import FusionGraph components: {e}")
        return
    
    print("🚀 Initializing FusionGraph Evaluation Suite")
    print("=" * 60)
    
    # Initialize evaluation suite
    eval_suite = FusionGraphEvaluationSuite()
    
    # Create test queries
    test_queries = create_sample_test_queries()
    print(f"📝 Created {len(test_queries)} test queries")
    
    # Initialize a minimal FusionGraph system for testing
    print("🔧 Setting up test FusionGraph system...")
    
    try:
        # Setup embedding model
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Setup LLM (placeholder - would use actual model in practice)
        llm = HuggingFaceLLM(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={"use_auth_token": False},
            generate_kwargs={"temperature": 0.1, "max_new_tokens": 256}
        )
        
        # Configure global settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        # Load documents
        documents = SimpleDirectoryReader("./documents/text").load_data()
        print(f"📚 Loaded {len(documents)} documents")
        
        # Create knowledge graph index
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        
        # This would normally take a while with a real LLM
        print("⚠️  Note: Using simplified setup for demo purposes")
        print("   In production, this would build a full knowledge graph")
        
        # Create a mock knowledge graph index for testing
        kg_index = KnowledgeGraphIndex.from_documents(
            documents[:2],  # Use only first 2 docs for speed
            storage_context=storage_context,
            show_progress=True
        )
        
        # Initialize Visual RAG pipeline
        visual_rag = VisualRAGPipeline(kg_index, use_faiss=True)
        
        # Index documents in Faiss if available
        try:
            visual_rag.index_documents_in_faiss("./documents/text")
            print("✅ Faiss indexing completed")
        except Exception as e:
            print(f"⚠️  Faiss indexing failed (continuing anyway): {e}")
        
        print("🧪 Running comprehensive evaluation...")
        
        # Run evaluation on subset of queries for demo
        test_subset = test_queries[:3]  # Use first 3 queries for speed
        
        results = eval_suite.run_comprehensive_evaluation(visual_rag, test_subset)
        
        print("\n📊 Evaluation Results Summary:")
        print("=" * 60)
        
        # Display high-level results
        for query_id, result in results.items():
            if "error" in result:
                print(f"❌ {query_id}: {result['error']}")
                continue
                
            print(f"\n🔍 {query_id}: {result['query'][:60]}...")
            
            # Retrieval quality
            if "retrieval_metrics" in result and "ndcg_at_3" in result["retrieval_metrics"]:
                ndcg = result["retrieval_metrics"]["ndcg_at_3"]
                print(f"   📈 Retrieval Quality (NDCG@3): {ndcg:.3f}")
            
            # Factual consistency
            if "factual_metrics" in result and "support_ratio" in result["factual_metrics"]:
                support_ratio = result["factual_metrics"]["support_ratio"]
                print(f"   ✅ Factual Support Ratio: {support_ratio:.3f}")
            
            # Multimodal grounding
            if "multimodal_metrics" in result:
                mm = result["multimodal_metrics"]
                grounding_quality = mm.get("grounding_quality", "unknown")
                print(f"   🎭 Multimodal Grounding: {grounding_quality}")
            
            # Performance
            exec_time = result.get("execution_time", 0)
            print(f"   ⏱️  Execution Time: {exec_time:.2f}s")
        
        # Check if summary files were created
        results_dir = "./evaluation/results"
        if os.path.exists(results_dir):
            recent_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')])
            if recent_files:
                print(f"\n💾 Detailed results saved to: {results_dir}/{recent_files[-1]}")
        
        print("\n🎉 Evaluation completed successfully!")
        print("\n📋 Research-Grade Metrics Implemented:")
        print("   1. ✅ Retrieval Quality: NDCG@K, MRR, Precision@K, Recall@K")
        print("   2. ✅ Factual Consistency: NLI-based verification, claim support ratios")
        print("   3. ✅ Multimodal Grounding: Cross-modal alignment, OCR accuracy, VQA")
        print("   4. ✅ Performance: Execution time, throughput metrics")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run FusionGraph evaluation suite")
    parser.add_argument("--queries", type=str, help="Path to custom queries JSON file")
    parser.add_argument("--output", type=str, default="./evaluation/results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🧪 FusionGraph Research-Grade Evaluation Suite")
    print("=" * 50)
    
    if not os.path.exists("./documents/text"):
        print("📁 Creating sample documents for evaluation...")
        create_test_documents_if_needed()
    
    # Run the evaluation
    results = run_evaluation_suite()
    
    if results:
        print("\n✨ Evaluation completed successfully!")
        print("🔬 Your FusionGraph system now has research-grade evaluation metrics!")
    else:
        print("\n❌ Evaluation failed. Check the logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())