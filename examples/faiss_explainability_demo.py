import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print(" FusionGraph - Faiss Integration with Explainability Demo")
    print("=" * 60)
    
    # Check if Faiss is available
    try:
        import faiss
        print(" Faiss is available")
    except ImportError:
        print(" Faiss not installed. Install with: pip install faiss-cpu")
        return
    
    try:
        from retrieval.faiss_retriever import FaissRetriever
        from explainability.engine import ExplainabilityEngine
        print(" Custom modules imported successfully")
    except ImportError as e:
        print(f" Module import error: {e}")
        return
    
    # Initialize components
    print("\n Initializing components...")
    retriever = FaissRetriever()
    explainability_engine = ExplainabilityEngine()
    
    # Sample documents
    sample_documents = [
        {
            'text': 'Artificial intelligence is revolutionizing healthcare by enabling faster diagnosis, personalized treatment plans, and predictive analytics. Machine learning algorithms can analyze medical images, electronic health records, and genomic data to identify patterns that human doctors might miss.',
            'source_id': 'ai_healthcare_2024',
            'source_name': 'AI in Healthcare: 2024 Trends',
            'metadata': {'domain': 'healthcare', 'year': 2024, 'type': 'research_paper'}
        },
        {
            'text': 'Computer vision technology has advanced significantly with deep learning models achieving human-level performance in image recognition tasks. Convolutional neural networks (CNNs) and transformer architectures are being used for medical imaging, autonomous vehicles, and surveillance systems.',
            'source_id': 'cv_deep_learning_2024',
            'source_name': 'Computer Vision Deep Learning Advances',
            'metadata': {'domain': 'computer_vision', 'year': 2024, 'type': 'technical_report'}
        },
        {
            'text': 'Natural language processing has been transformed by large language models like GPT, BERT, and their variants. These models can understand context, generate human-like text, and perform complex reasoning tasks across multiple domains including legal, medical, and scientific texts.',
            'source_id': 'nlp_llm_2024',
            'source_name': 'Large Language Models in NLP',
            'metadata': {'domain': 'nlp', 'year': 2024, 'type': 'survey_paper'}
        },
        {
            'text': 'Knowledge graphs provide structured representations of information that enable better reasoning and inference. They connect entities through relationships and can be used for recommendation systems, question answering, and semantic search applications.',
            'source_id': 'knowledge_graphs_2024',
            'source_name': 'Knowledge Graphs for AI Applications',
            'metadata': {'domain': 'knowledge_representation', 'year': 2024, 'type': 'tutorial'}
        }
    ]
    
    print(f" Indexing {len(sample_documents)} sample documents...")
    retriever.add_documents(sample_documents)
    
    # Save index
    retriever.save_index()
    print(" Index saved to storage/")
    
    # Test queries
    test_queries = [
        "How does AI help in medical diagnosis?",
        "What are the latest advances in computer vision?",
        "How do knowledge graphs work?",
        "What can large language models do?"
    ]

    print(f"\n Testing {len(test_queries)} queries with explainability...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n Query {i}: {query}")
        print("-" * 40)
        
        # Retrieve with explanations
        results, metadata = retriever.retrieve(query, top_k=3, return_explanations=True)
        
        if results:
            print(f"Found {len(results)} relevant results")
            
            # Create explanation
            mock_answer = f"Based on the retrieved information, {query.lower().replace('?', '')} involves multiple aspects covered in our knowledge base."
            
            explanation = explainability_engine.create_explanation(
                query=query,
                answer=mock_answer,
                retrieval_results=results,
                retrieval_metadata=metadata
            )
            
            # Display explanation summary
            summary = explainability_engine.get_explanation_summary(explanation)
            print(f" {summary}")
            
            # Show top result
            top_result = results[0]
            print(f" Best match: {top_result.source_name} (score: {top_result.score:.3f})")
            print(f" Snippet: {top_result.text[:150]}...")
            
            # Show confidence breakdown
            confidence = explanation.confidence
            print(f" Confidence: {confidence['label']} ({confidence['score']:.2f})")
            print(f" Explanation: {confidence['explanation']}")
            
        else:
            print("No relevant results found")
    
    # Display retriever statistics
    print(f"\n Retriever Statistics:")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Display explainability statistics
    print(f"\n Explainability Statistics:")
    exp_stats = explainability_engine.get_stats()
    for key, value in exp_stats.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n Demo completed successfully!")


if __name__ == "__main__":
    main()