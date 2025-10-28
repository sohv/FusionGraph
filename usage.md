# Visual RAG + Interactive Web UI

This extends the original RAG-knowledgegraph system with two major enhancements:

1. **Multimodal Visual RAG**: Image processing, OCR, captioning, and visual question answering
2. **Interactive Web UI**: Streamlit-based interface with knowledge graph visualization and feedback collection

## What's New

### Visual RAG Capabilities
- **Image Ingestion**: Automatic processing of images with OCR, captioning, and object detection
- **Multimodal Knowledge Graph**: Integration of text documents with visual content
- **Visual Question Answering**: Queries that leverage both textual and visual information
- **Enhanced Provenance**: Detailed tracking of both text and image sources

### Interactive Web Interface
- **Real-time Visualization**: Interactive knowledge graph exploration
- **User Feedback System**: Continuous improvement through user input
- **Confidence Scoring**: Transparent confidence metrics for all responses
- **Multimodal Results**: Side-by-side display of text and image sources

## Architecture Overview

```
📁 Project Structure
├── 📄 requirements.txt          # Updated dependencies
├── 📄 rag_with_kgraph_llamaindex.ipynb  # Original notebook
├── 📁 documents/               # Document and image storage
│   ├── 📁 sample/              # Sample PDF documents
│   ├── 📁 sample_ai/           # AI-related documents
│   └── 🖼️ *.png, *.jpg         # Images for processing
├── 📁 storage/                 # Persistent storage
├── 📁 ingest/                  # NEW: Image processing
│   └── 📄 image_ingest.py      # Image ingestion pipeline
├── 📁 pipeline/                # NEW: Enhanced RAG pipeline
│   └── 📄 visual_rag.py        # Multimodal RAG implementation
├── 📁 webapp/                  # NEW: Web interface
│   ├── 📄 app.py              # Streamlit application
│   ├── 📄 provenance.py       # Provenance tracking
│   └── 📄 feedback_sink.py    # User feedback collection
├── 📁 tools/                   # NEW: Utilities
│   └── 📄 visual_utils.py      # Image processing utilities
├── 📁 notebooks/               # NEW: Demo notebooks
│   └── 📄 visual_rag_demo.ipynb  # Comprehensive demo
└── 📁 tests/                   # NEW: Unit tests
    ├── 📄 test_image_ingest.py
    └── 📄 test_visual_rag.py
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd RAG-knowledgegraph

# Install dependencies
pip install -r requirements.txt

# For OCR support (optional - install system dependencies)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract
```

### 2. Configuration

Set up your HuggingFace token:
```python
HF_TOKEN = 'your_huggingface_token_here'
```

### 3. Run the Demo Notebook

```bash
jupyter lab notebooks/visual_rag_demo.ipynb
```

### 4. Launch the Web Interface

```bash
streamlit run webapp/app.py
```

Then open your browser to `http://localhost:8501`

## Usage Examples

### Visual RAG Pipeline

```python
from pipeline.visual_rag import VisualRAGPipeline
from ingest.image_ingest import ImageIngestor

# Initialize with existing knowledge graph
visual_rag = VisualRAGPipeline(kg_index)

# Add images to the knowledge graph
num_nodes = visual_rag.add_images_to_kg("./documents/")

# Query with multimodal context
result = visual_rag.query_with_visual_context(
    query="What is artificial intelligence and how is it represented visually?",
    include_images=True,
    max_text_results=5,
    max_image_results=3
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence_score:.2%}")
print(f"Text sources: {len(result.text_sources)}")
print(f"Image sources: {len(result.image_sources)}")
```

### Image Processing

```python
from ingest.image_ingest import ImageIngestor

# Initialize image processor
ingestor = ImageIngestor()

# Process a single image
nodes = ingestor.create_image_nodes(
    "path/to/image.jpg",
    include_ocr=True,
    include_caption=True,
    include_objects=True
)

# Process a directory of images
all_nodes = ingestor.process_image_directory("./images/")
```

### Web Interface Features

The Streamlit web interface provides:

- **Configuration Panel**: Model selection and parameter tuning
- **Interactive Querying**: Natural language query interface
- **Real-time Visualization**: Knowledge graph exploration
- **Feedback Collection**: User rating and improvement system
- **Performance Analytics**: Confidence scores and source analysis

## Configuration Options

### Model Configuration

```python
# Language Model
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# Embedding Model
EMBEDDING_MODEL = "thenlper/gte-large"

# Image Captioning Model
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
```

### Pipeline Parameters

```python
# Visual RAG settings
pipeline = VisualRAGPipeline(
    knowledge_graph_index=kg_index,
    similarity_threshold=0.7  # Minimum similarity for retrieval
)

# Query parameters
result = pipeline.query_with_visual_context(
    query="Your question here",
    include_images=True,        # Include image sources
    max_text_results=5,         # Maximum text sources
    max_image_results=3         # Maximum image sources
)
```

## Performance Metrics

The system provides detailed performance metrics:

- **Confidence Scores**: Query-level confidence based on source quality
- **Source Attribution**: Breakdown of text vs. image contributions
- **Graph Coverage**: Percentage of knowledge graph utilized
- **Response Time**: Query processing and retrieval speed
- **User Feedback**: Aggregated user satisfaction ratings

## Example Queries

Try these sample queries to explore the multimodal capabilities:

1. **Basic Text + Image**: "What is artificial intelligence and how is it represented visually?"
2. **Image-Focused**: "Show me visual information about machine learning algorithms"
3. **OCR-Based**: "What text is visible in the uploaded images?"
4. **Object Detection**: "What objects or diagrams are present in the AI documents?"
5. **Comparative**: "Compare textual definitions with visual representations of neural networks"

## Testing

Run the test suite to validate functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python tests/test_image_ingest.py
python tests/test_visual_rag.py

# Run with integration tests (requires models)
RUN_INTEGRATION_TESTS=1 python tests/test_image_ingest.py
```

## Advanced Usage

### Custom Image Processing

```python
from ingest.image_ingest import ImageIngestor

# Custom initialization with different models
ingestor = ImageIngestor(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    caption_model_name="Salesforce/blip-image-captioning-large"
)

# Selective processing
nodes = ingestor.create_image_nodes(
    image_path="document.jpg",
    include_ocr=True,      # Extract text
    include_caption=False, # Skip captioning
    include_objects=True   # Detect objects
)
```

### Provenance Analysis

```python
from webapp.provenance import ProvenanceExtractor

# Initialize provenance extractor
extractor = ProvenanceExtractor(kg_index)

# Extract full provenance trace
trace = extractor.extract_full_provenance(
    query="Your query",
    response=response_object,
    confidence_score=0.85
)

# Export provenance data
json_data = extractor.export_provenance_json(trace)
```

### Feedback Collection

```python
from webapp.feedback_sink import FeedbackCollector

# Initialize feedback collector
collector = FeedbackCollector()

# Add user feedback
feedback_id = collector.add_feedback(
    target_id="source_node_123",
    feedback_type="helpful",
    target_type="image_source",
    user_id="user_456",
    comment="Very relevant image"
)

# Analyze feedback trends
summary = collector.get_feedback_summary(days=30)
```