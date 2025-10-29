"""
Executable benchmark runner for FusionGraph vs Baselines
Implements the three baseline systems and runs comparative evaluation
"""

import sys
import os
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from benchmark.benchmark_framework import BenchmarkRunner, BaselineSystemImplementation


def run_baseline_1_plain_rag():
    """
    Baseline 1: Plain RAG Implementation
    - Only vector-based retrieval using LlamaIndex
    - No knowledge graph construction
    - No image processing capabilities
    - Uses same embedding model as FusionGraph for fair comparison
    """
    print("\n🔧 Testing Baseline 1: Plain RAG")
    print("Configuration: LlamaIndex VectorStore only, no KG, no images")
    
    try:
        baseline = BaselineSystemImplementation.create_plain_rag_baseline()
        
        # Test with sample query
        test_query = "What are the main applications of artificial intelligence?"
        print(f"Test query: {test_query}")
        
        result = baseline.query(test_query)
        
        print(f"✅ Plain RAG baseline working")
        print(f"Answer length: {len(result['answer'])} chars")
        print(f"Sources found: {len(result['text_sources'])}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Plain RAG baseline failed: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False, None


def run_baseline_2_text_kg():
    """
    Baseline 2: Text-only KG Implementation
    - Knowledge graph construction enabled
    - No image processing capabilities
    - Same LLM and embedding models as FusionGraph
    """
    print("\n🔧 Testing Baseline 2: Text-only KG")
    print("Configuration: Knowledge Graph + text retrieval, no images")
    
    try:
        baseline = BaselineSystemImplementation.create_text_only_kg_baseline()
        
        # Test with sample query
        test_query = "How do machine learning algorithms improve over time?"
        print(f"Test query: {test_query}")
        
        result = baseline.query(test_query)
        
        print(f"✅ Text-only KG baseline working")
        print(f"Answer length: {len(result['answer'])} chars")
        print(f"Sources found: {len(result['text_sources'])}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Text-only KG baseline failed: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False, None


def test_fusiongraph_system():
    """
    Test the full FusionGraph system for comparison
    """
    print("\n🔧 Testing FusionGraph System")
    print("Configuration: Full features with KG + multimodal + Faiss")
    
    try:
        # Import FusionGraph components
        from pipeline.visual_rag import VisualRAGPipeline
        from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, Settings, StorageContext
        from llama_index.core.graph_stores import SimpleGraphStore
        from llama_index.llms.openai import OpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from llama_index.embeddings.langchain import LangchainEmbedding
        
        # Setup LLM
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Setup embedding model
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        Settings.embed_model = embed_model
        
        # Load documents
        documents = SimpleDirectoryReader("./documents/text").load_data()
        
        # Create knowledge graph index
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Initialize Visual RAG
        pipeline = VisualRAGPipeline(kg_index, use_faiss=True)
        
        # Test with sample query
        test_query = "What are the differences between supervised and unsupervised learning?"
        print(f"Test query: {test_query}")
        
        start_time = time.time()
        result = pipeline.query_with_visual_context(
            query=test_query,
            include_explanation=True
        )
        execution_time = time.time() - start_time
        
        print(f"✅ FusionGraph system working")
        print(f"Answer length: {len(result.answer)} chars")
        print(f"Text sources: {len(result.text_sources)}")
        print(f"Image sources: {len(result.image_sources)}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Execution time: {execution_time:.2f}s")
        
        return True, {
            'answer': result.answer,
            'text_sources': result.text_sources,
            'image_sources': result.image_sources,
            'confidence_score': result.confidence_score,
            'execution_time': execution_time,
            'explanation': result.explanation,
            'system_type': 'fusiongraph'
        }
        
    except Exception as e:
        print(f"❌ FusionGraph system failed: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False, None


def run_comparative_benchmark():
    """
    Run all systems and compare results
    """
    print("🚀 Running Comparative Benchmark")
    print("=" * 60)
    
    # Define test queries covering different aspects
    test_queries = [
        {
            "query": "What are the main applications of artificial intelligence?",
            "type": "factual_retrieval",
            "expected_advantage": "KG should provide better structured knowledge"
        },
        {
            "query": "How do machine learning algorithms improve over time?", 
            "type": "process_explanation",
            "expected_advantage": "KG relationships should help explain processes"
        },
        {
            "query": "What are the differences between supervised and unsupervised learning?",
            "type": "comparative_analysis", 
            "expected_advantage": "KG should excel at comparative relationships"
        }
    ]
    
    results = {}
    
    # Test each system
    for query_info in test_queries:
        query = query_info["query"]
        print(f"\n{'='*60}")
        print(f"📝 Query: {query}")
        print(f"Type: {query_info['type']}")
        print(f"Expected Advantage: {query_info['expected_advantage']}")
        print(f"{'='*60}")
        
        query_results = {}
        
        # Test Plain RAG
        try:
            baseline_1 = BaselineSystemImplementation.create_plain_rag_baseline()
            result_1 = baseline_1.query(query)
            query_results['plain_rag'] = result_1
            print(f"✅ Plain RAG: {len(result_1['answer'])} chars, {result_1['execution_time']:.2f}s")
        except Exception as e:
            print(f"❌ Plain RAG failed: {e}")
            query_results['plain_rag'] = None
        
        # Test Text-only KG (Note: Might be slow due to KG construction)
        print("⏳ Text-only KG might take longer due to KG construction...")
        try:
            baseline_2 = BaselineSystemImplementation.create_text_only_kg_baseline()
            result_2 = baseline_2.query(query)
            query_results['text_only_kg'] = result_2
            print(f"✅ Text-only KG: {len(result_2['answer'])} chars, {result_2['execution_time']:.2f}s")
        except Exception as e:
            print(f"❌ Text-only KG failed: {e}")
            query_results['text_only_kg'] = None
        
        # Test FusionGraph
        print("⏳ FusionGraph with full features...")
        success, result_3 = test_fusiongraph_system()
        if success:
            query_results['fusiongraph'] = result_3
            print(f"✅ FusionGraph: {len(result_3['answer'])} chars, {result_3['execution_time']:.2f}s")
        else:
            query_results['fusiongraph'] = None
        
        results[query] = query_results
    
    # Generate comparison summary
    print(f"\n{'='*60}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    systems = ['plain_rag', 'text_only_kg', 'fusiongraph']
    system_names = {
        'plain_rag': 'Plain RAG (No KG, No Images)',
        'text_only_kg': 'Text-only KG (No Images)', 
        'fusiongraph': 'FusionGraph (Full Features)'
    }
    
    for system in systems:
        print(f"\n🔧 {system_names[system]}:")
        
        total_time = 0
        total_queries = 0
        avg_answer_length = 0
        success_count = 0
        
        for query, query_results in results.items():
            if query_results.get(system):
                result = query_results[system]
                total_time += result['execution_time']
                total_queries += 1
                avg_answer_length += len(result['answer'])
                success_count += 1
                print(f"  ✅ Q{total_queries}: {result['execution_time']:.2f}s, {len(result['answer'])} chars")
            else:
                print(f"  ❌ Q{total_queries + 1}: Failed")
                total_queries += 1
        
        if success_count > 0:
            print(f"  📊 Success Rate: {success_count}/{total_queries} ({100*success_count/total_queries:.1f}%)")
            print(f"  📊 Avg Time: {total_time/success_count:.2f}s")
            print(f"  📊 Avg Answer Length: {avg_answer_length/success_count:.0f} chars")
        else:
            print(f"  📊 Success Rate: 0/{total_queries} (0%)")
    
    # Key insights based on expected behavior
    print(f"\n{'='*60}")
    print("🎯 EXPECTED KEY INSIGHTS")
    print(f"{'='*60}")
    print("1. Knowledge Graph Value:")
    print("   • Text-only KG should show better factual accuracy than Plain RAG")
    print("   • KG excels at relationship queries and comparative analysis")
    
    print("\n2. Multimodal Value:")
    print("   • FusionGraph can process visual information (diagrams, charts)")
    print("   • Enhanced context from images improves answer completeness")
    
    print("\n3. Performance Optimization:")
    print("   • Faiss indexing significantly reduces retrieval latency")
    print("   • Knowledge graph construction has one-time setup cost")
    
    print("\n4. Use Case Recommendations:")
    print("   • Plain RAG: Resource-constrained environments")
    print("   • Text-only KG: High-accuracy text-only domains")
    print("   • FusionGraph: Complex multimodal knowledge tasks")
    
    return results


def verify_dataset():
    """Verify the benchmark dataset exists and is properly structured"""
    print("📊 Verifying Benchmark Dataset")
    print("=" * 50)
    
    docs_path = Path("./documents")
    text_path = docs_path / "text"
    
    if not docs_path.exists():
        print(f"❌ Documents directory not found: {docs_path}")
        return False
    
    if not text_path.exists():
        print(f"❌ Text documents directory not found: {text_path}")
        return False
    
    # Find PDF files
    pdf_files = list(docs_path.rglob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  • {pdf.name} ({size_mb:.1f} MB)")
    
    # Find text files
    text_files = list(text_path.glob("*"))
    print(f"📝 Text directory contains {len(text_files)} items")
    
    if len(pdf_files) == 0:
        print("⚠️  No PDF files found. Benchmarks may not work properly.")
        return False
    
    print("✅ Dataset verification complete")
    return True


if __name__ == "__main__":
    print("🚀 FusionGraph Benchmark Runner")
    print("Testing three baseline systems vs FusionGraph")
    
    # Verify dataset first
    if not verify_dataset():
        print("❌ Dataset verification failed. Please check document structure.")
        sys.exit(1)
    
    # Run the comparative benchmark
    try:
        results = run_comparative_benchmark()
        print("\n✅ Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")