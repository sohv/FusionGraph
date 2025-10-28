import streamlit as st
import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, Settings, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from pipeline.visual_rag import VisualRAGPipeline, VisualRAGResult
from ingest.image_ingest import ImageIngestor
from webapp.provenance import ProvenanceExtractor
from webapp.feedback_sink import FeedbackCollector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class VisualRAGApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        st.set_page_config(
            page_title="Visual RAG with Knowledge Graph",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .source-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'kg_index' not in st.session_state:
            st.session_state.kg_index = None
        if 'visual_rag_pipeline' not in st.session_state:
            st.session_state.visual_rag_pipeline = None
        if 'provenance_extractor' not in st.session_state:
            st.session_state.provenance_extractor = None
        if 'feedback_collector' not in st.session_state:
            st.session_state.feedback_collector = FeedbackCollector()
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'last_result' not in st.session_state:
            st.session_state.last_result = None
    
    def setup_sidebar(self):
        """setup the sidebar with configuration options"""
        st.sidebar.header(" Configuration")
        
        # model configuration
        st.sidebar.subheader("Model Settings")
        
        # get HuggingFace token from environment or user input
        default_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN', '')
        hf_token = st.sidebar.text_input(
            "HuggingFace Token", 
            value=default_token,
            type="password",
            help="Required for HuggingFace models. Set HUGGINGFACEHUB_API_TOKEN env var to auto-fill."
        )
        
        llm_model = st.sidebar.selectbox(
            "Language Model",
            ["HuggingFaceH4/zephyr-7b-beta", "microsoft/DialoGPT-medium"],
            help="Select the language model for responses"
        )
        
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["thenlper/gte-large", "sentence-transformers/all-MiniLM-L6-v2"],
            help="Select the embedding model for similarity search"
        )
        
        # Data ingestion
        st.sidebar.subheader("Data Sources")
        
        # Document directory selection
        doc_directory = st.sidebar.text_input(
            "Document Directory",
            value="./documents/text",
            help="Path to directory containing text documents (PDF, TXT, DOCX)"
        )
        
        # Image directory selection
        img_directory = st.sidebar.text_input(
            "Image Directory", 
            value="./documents/images",
            help="Path to directory containing images (PNG, JPG, etc.)"
        )
        
        # Initialize system button
        if st.sidebar.button(" Initialize System", type="primary"):
            self.initialize_system(hf_token, llm_model, embedding_model, 
                                 doc_directory, img_directory)
        
        st.sidebar.subheader("System Status")
        if st.session_state.kg_index is not None:
            st.sidebar.success(" Knowledge Graph Loaded")
        else:
            st.sidebar.warning(" Knowledge Graph Not Loaded")
            
        if st.session_state.visual_rag_pipeline is not None:
            st.sidebar.success(" Visual RAG Pipeline Ready")
        else:
            st.sidebar.warning(" Visual RAG Pipeline Not Ready")
    
    def validate_hf_token(self, token: str) -> bool:
        if not token:
            return False
        if not token.startswith('hf_'):
            st.warning(" Token should start with 'hf_' but we'll try anyway...")
            return True
        
        return True

    def initialize_system(self, hf_token: str, llm_model: str, embedding_model: str,
                         doc_directory: str, img_directory: str):
        try:
            with st.spinner("Initializing system..."):
                if not hf_token:
                    st.error("HuggingFace token is required!")
                    return
                
                try:
                    st.info(f" Loading model: {llm_model}")
                    llm = HuggingFaceLLM(
                        model_name=llm_model,
                        model_kwargs={"use_auth_token": hf_token}
                    )
                    st.success(" LLM loaded successfully!")
                except Exception as e:
                    st.error(f" Failed to load LLM: {str(e)}")
                    st.info("This might be due to an invalid token or network issues.")
                    return
                
                # Setup embedding model
                embed_model = LangchainEmbedding(
                    HuggingFaceEmbeddings(model_name=embedding_model)
                )
                
                # Configure settings
                Settings.llm = llm
                Settings.chunk_size = 512
                
                # Load documents
                if os.path.exists(doc_directory):
                    documents = SimpleDirectoryReader(doc_directory).load_data()
                    st.success(f"Loaded {len(documents)} documents")
                else:
                    st.error(f"Document directory not found: {doc_directory}")
                    return
                
                # Setup storage context
                graph_store = SimpleGraphStore()
                storage_context = StorageContext.from_defaults(graph_store=graph_store)
                
                # Create Knowledge Graph Index
                st.session_state.kg_index = KnowledgeGraphIndex.from_documents(
                    documents=documents,
                    max_triplets_per_chunk=3,
                    storage_context=storage_context,
                    embed_model=embed_model,
                    include_embeddings=True
                )
                
                # Initialize Visual RAG Pipeline
                st.session_state.visual_rag_pipeline = VisualRAGPipeline(
                    st.session_state.kg_index
                )
                
                # Add images if directory exists
                if os.path.exists(img_directory):
                    num_nodes = st.session_state.visual_rag_pipeline.add_images_to_kg(img_directory)
                    st.success(f"Added {num_nodes} image nodes to knowledge graph")
                
                # Initialize provenance extractor
                st.session_state.provenance_extractor = ProvenanceExtractor(
                    st.session_state.kg_index
                )
                
                st.success(" System initialized successfully!")
                
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
    
    def main_interface(self):
        """Main query interface"""
        st.markdown('<h1 class="main-header">🧠 Visual RAG with Knowledge Graph</h1>', 
                   unsafe_allow_html=True)
        
        if st.session_state.visual_rag_pipeline is None:
            st.warning("Please initialize the system using the sidebar configuration.")
            return
        
        # Query input
        st.subheader(" Ask a Question")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is artificial intelligence? How many PhD researchers are in India?",
                help="Ask questions about your documents and images"
            )
        
        with col2:
            include_images = st.checkbox("Include Images", value=True)
            max_results = st.slider("Max Results", 1, 10, 5)
        
        if st.button(" Search", type="primary") and query:
            self.process_query(query, include_images, max_results)
        
        # Display results
        if st.session_state.last_result:
            self.display_results(st.session_state.last_result)
    
    def process_query(self, query: str, include_images: bool, max_results: int):
        """Process the user query and store results"""
        try:
            with st.spinner("Processing your query..."):
                result = st.session_state.visual_rag_pipeline.query_with_visual_context(
                    query=query,
                    include_images=include_images,
                    max_text_results=max_results,
                    max_image_results=3,
                    include_explanation=True  # Enable explainability
                )
                
                st.session_state.last_result = result
                st.session_state.query_history.append({
                    'query': query,
                    'timestamp': datetime.now(),
                    'confidence': result.confidence_score,
                    'explanation': result.explanation
                })
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    
    def display_results(self, result):
        """Display query results with enhanced explainability"""
        st.subheader("🎯 Results")
        
        # Answer section with explanation summary
        with st.container():
            st.markdown("### 💬 Answer")
            st.write(result.answer)
            
            # Enhanced confidence display
            st.markdown("### 📊 Confidence & Quality")
            confidence_score = result.confidence_score
            
            # Color-coded confidence
            if confidence_score >= 0.8:
                confidence_color = "🟢"
                confidence_text = "High"
            elif confidence_score >= 0.6:
                confidence_color = "🟡"
                confidence_text = "Medium"
            elif confidence_score >= 0.4:
                confidence_color = "🟠"
                confidence_text = "Low"
            else:
                confidence_color = "🔴"
                confidence_text = "Very Low"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{confidence_color} {confidence_text}", f"{confidence_score:.1%}")
            with col2:
                # Show explanation summary if available
                if result.explanation:
                    summary = st.session_state.visual_rag_pipeline.explainability_engine.get_explanation_summary(result.explanation)
                    st.info(summary)
        
        # Explainability Panel
        if result.explanation:
            with st.expander("🔍 **Explanation & Provenance**", expanded=False):
                explanation = result.explanation
                
                # Confidence factors breakdown
                st.markdown("#### Confidence Factors")
                confidence_data = explanation.confidence
                
                factor_cols = st.columns(3)
                factors = confidence_data.get('factors', {})
                
                with factor_cols[0]:
                    st.metric("Retrieval Quality", f"{factors.get('retrieval_quality', 0):.2f}")
                with factor_cols[1]:
                    st.metric("Source Diversity", f"{factors.get('source_diversity', 0):.2f}")
                with factor_cols[2]:
                    st.metric("LLM Confidence", f"{factors.get('llm_confidence', 0):.2f}")
                
                st.info(f"💡 {confidence_data.get('explanation', 'No detailed explanation available')}")
                
                # Provenance sources
                st.markdown("#### 📚 Source Provenance")
                provenance = explanation.provenance
                
                for i, source in enumerate(provenance[:5]):  # Show top 5 sources
                    with st.container():
                        source_type_emoji = "📄" if source['type'] == 'text' else "🖼️"
                        
                        st.markdown(f"**{source_type_emoji} Source {source['rank']}: {source['source_name']}**")
                        st.markdown(f"*Similarity Score: {source['similarity_score']:.3f}*")
                        
                        if source['type'] == 'text':
                            st.text_area(
                                f"Text snippet",
                                source['text_snippet'],
                                height=100,
                                key=f"source_text_{i}",
                                disabled=True
                            )
                        elif source['type'] == 'image':
                            st.write(f"**OCR Text:** {source.get('ocr_text', 'No OCR text available')}")
                        
                        st.markdown("---")
                
                # Chain of thought trace
                if explanation.cot_trace:
                    st.markdown("#### 🧠 Reasoning Trace")
                    for step in explanation.cot_trace:
                        st.markdown(f"• {step}")
                
                # Retrieval explanation
                if explanation.retrieval_explanation:
                    st.markdown("#### ⚙️ Retrieval Details")
                    ret_exp = explanation.retrieval_explanation
                    
                    detail_cols = st.columns(4)
                    with detail_cols[0]:
                        st.metric("Query Norm", f"{ret_exp.query_embedding_norm:.3f}")
                    with detail_cols[1]:
                        st.metric("Top-K Requested", ret_exp.top_k_requested)
                    with detail_cols[2]:
                        st.metric("Total Candidates", ret_exp.total_candidates)
                    with detail_cols[3]:
                        st.metric("Similarity Method", ret_exp.similarity_method)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Text Sources", len(result.text_sources))
        with col2:
            st.metric("Image Sources", len(result.image_sources))
        with col3:
            st.metric("Graph Nodes", len(result.graph_context.get('nodes', {})))
        
        # Tabbed results display
        tab1, tab2, tab3, tab4 = st.tabs([" Text Sources", " Image Sources", 
                                         " Knowledge Graph", " Provenance"])
        
        with tab1:
            self.display_text_sources(result.text_sources)
        
        with tab2:
            self.display_image_sources(result.image_sources)
        
        with tab3:
            self.display_knowledge_graph(result.graph_context)
        
        with tab4:
            self.display_provenance(result)
    
    def display_text_sources(self, text_sources: List[Dict[str, Any]]):
        if not text_sources:
            st.info("No text sources found for this query.")
            return
        
        for i, source in enumerate(text_sources):
            with st.expander(f" Source {i+1} (Score: {source.get('score', 0):.3f})"):
                st.write(source['text'])
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button(f" Helpful", key=f"text_helpful_{i}"):
                        st.session_state.feedback_collector.add_feedback(
                            source['id'], 'helpful', 'text_source'
                        )
                        st.success("Feedback recorded!")
                
                with col2:
                    if st.button(f" Not Helpful", key=f"text_not_helpful_{i}"):
                        st.session_state.feedback_collector.add_feedback(
                            source['id'], 'not_helpful', 'text_source'
                        )
                        st.success("Feedback recorded!")
    
    def display_image_sources(self, image_sources: List[Dict[str, Any]]):
        """Display image sources with feedback options"""
        if not image_sources:
            st.info("No image sources found for this query.")
            return
        
        for i, source in enumerate(image_sources):
            with st.expander(f" Image Source {i+1} (Score: {source.get('score', 0):.3f})"):
                st.write(f"**Type:** {source['metadata'].get('type', 'Unknown')}")
                st.write(f"**Text:** {source['text']}")
                
                # display image if available
                image_path = source['metadata'].get('image_path') or source['metadata'].get('source_image')
                if image_path and os.path.exists(image_path):
                    st.image(image_path, caption=f"Source: {os.path.basename(image_path)}", width=400)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f" Helpful", key=f"img_helpful_{i}"):
                        st.session_state.feedback_collector.add_feedback(
                            source['id'], 'helpful', 'image_source'
                        )
                        st.success("Feedback recorded!")
                
                with col2:
                    if st.button(f" Not Helpful", key=f"img_not_helpful_{i}"):
                        st.session_state.feedback_collector.add_feedback(
                            source['id'], 'not_helpful', 'image_source'
                        )
                        st.success("Feedback recorded!")
    
    def display_knowledge_graph(self, graph_context: Dict[str, Any]):
        if not graph_context.get('nodes'):
            st.info("No graph context available for this query.")
            return
        
        try:
            G = nx.Graph()
            
            # Add nodes
            for node_id, node_data in graph_context['nodes'].items():
                G.add_node(node_id, **node_data)
            
            # Add edges
            for edge in graph_context.get('edges', []):
                G.add_edge(edge['source'], edge['target'], relation=edge.get('relation', ''))
            
            # Create layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Prepare data for Plotly
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node[:20]}...")
                # Color by node type
                node_type = G.nodes[node].get('type', 'text')
                if node_type == 'image':
                    node_color.append('red')
                elif node_type in ['ocr_text', 'image_caption']:
                    node_color.append('orange')
                else:
                    node_color.append('blue')
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=10, color=node_color),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                showlegend=False
            ))
            
            fig.update_layout(
                title="Knowledge Graph Context",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Blue: Text nodes, Red: Image nodes, Orange: Image-derived content",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Graph statistics
            st.subheader(" Graph Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes in Context", len(graph_context['nodes']))
            with col2:
                st.metric("Edges in Context", len(graph_context.get('edges', [])))
            with col3:
                total_nodes = graph_context.get('total_graph_nodes', 0)
                st.metric("Total Graph Nodes", total_nodes)
            
        except Exception as e:
            st.error(f"Error creating graph visualization: {str(e)}")
            st.json(graph_context)
    
    def display_provenance(self, result):
        """Display provenance information"""
        st.subheader(" Answer Provenance")
        
        # Provenance summary
        provenance = result.provenance
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Text Nodes Used", len(provenance.get('text_node_ids', [])))
        with col2:
            st.metric("Image Nodes Used", len(provenance.get('image_node_ids', [])))
        
        # Detailed provenance
        with st.expander(" Detailed Provenance"):
            st.json(provenance)
        
        # Query history
        if st.session_state.query_history:
            st.subheader(" Query History")
            history_df = pd.DataFrame(st.session_state.query_history)
            st.dataframe(history_df, use_container_width=True)
    
    def run(self):
        """Run the Streamlit application"""
        self.setup_sidebar()
        self.main_interface()


def main():
    """Main function to run the Streamlit app"""
    app = VisualRAGApp()
    app.run()


if __name__ == "__main__":
    main()