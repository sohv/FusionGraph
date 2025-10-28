# RAG System with Knowledge Graph and LlamaIndex

## Introduction
This project aims to build a Retrieval-Augmented Generation (RAG) system that combines the power of **Knowledge Graphs** and **LlamaIndex** to provide advanced document retrieval and natural language generation capabilities. The goal is to develop a system that can efficiently retrieve relevant information from a structured knowledge base and use this information to generate high-quality, context-aware responses.

## Key Features
- **Knowledge Graph Integration**: The project leverages a structured Knowledge Graph that enables better organization, storage, and retrieval of interconnected data.

- **LlamaIndex**: Using LlamaIndex, this system can index large amounts of unstructured data (e.g., documents) and combine it with structured data from the knowledge graph to enhance search and retrieval capabilities.

- **Retrieval-Augmented Generation**: By augmenting language models with real-time document retrieval, the system generates responses that are both contextually relevant and informed by up-to-date knowledge.

### Purpose
This system can be applied to a wide range of use cases, such as:

- **Question Answering Systems**: Use natural language queries to retrieve answers from a knowledge base.
- **Virtual Assistants**: Provide contextually aware responses by combining structured knowledge with language generation.
- **Document Summarization**: Retrieve key information from large document corpora and generate concise summaries.

The integration of Knowledge Graphs ensures that relationships between different data points are captured effectively, allowing the system to answer complex queries with more accuracy and depth than traditional methods.

### Tech Stack

- **LlamaIndex** : LlamaIndex is an orchestration framework that simplifies the integration of private data with public data for building applications using Large Language Models through tools for data ingestion, indexing, and querying. Here, we have used the llama-index version 0.10.33
- **Embedding Model** : Embedding Model is required to convert the text into numerical representation of a piece of information for the text provided. Here, we have used **thenlper/gte-large** model.
- **LLM** : We have used Zephyr 7B beta model.
- **PyVis** : PyVis is used for for visualizing graph structures.

## Getting Started

### Basic Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/sohv/RAG-knowledgegraph.git
   cd RAG-knowledgegraph

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Configure HuggingFace Token**:
   - Get your token from [HuggingFace](https://huggingface.co/settings/tokens)
   - Set it as environment variable: `export HF_TOKEN=your_token_here`
   - Or configure it directly in the notebook/web interface

### Quick Start Options

#### Option 1: Interactive Web Interface (Recommended)
```bash
# Launch the enhanced web interface
streamlit run webapp/app.py
```
- Open your browser to `http://localhost:8501`
- Configure your HuggingFace token in the sidebar
- Upload documents and images
- Start querying with natural language!

#### Option 2: Demo Notebook
```bash
# Launch Jupyter notebook with comprehensive demo
jupyter lab notebooks/visual_rag_demo.ipynb
```

#### Option 3: Original Notebook
```bash
# Run the original implementation
jupyter notebook rag_with_kgraph_llamaindex.ipynb
```

#### Option 4: Automated Setup
```bash
# Run the startup script for guided setup
python start.py
```

### Enhanced Features (New!)
- **🖼️ Visual RAG**: Process images with OCR, captioning, and object detection
- **🌐 Interactive Web UI**: Real-time visualization and feedback collection
- **🔍 Enhanced Provenance**: Detailed source tracking for transparency
- **💬 Feedback System**: Improve results through user interactions

### Documents for Testing
The documents are arranged in following structure:
```bash
documents/
│
├── sample/
│   ├── sample.pdf
│   └── sample1.pdf
│
└── sample_ai/
    ├── AI Trend story.pdf
    └── A_Brief_Introduction_To_AI.pdf
```

### Running Tests
```bash
# Quick validation
python run_tests.py --quick

# Full test suite
python run_tests.py

# Integration tests (requires models)
python run_tests.py --integration
```

## Tips & Troubleshooting
1. **Knowledge Graph visualization**

   The knowledge graph generated and uploaded in this repository is in .html format.After generating the knowledge graph for your document, save the file and open it in a browser. For the best experience, we recommend using Mozilla Firefox or Google Chrome.

   Below is a sample knowledge graph for a document stored in this repository:

   ![Sample Knowledge Graph](documents/knowledgegraph.png)

3. **HuggingFace Token**

   The HuggingFace token was made public in the course of the project as Google Colab's **Secret key** option was not working properly for reasons unknown. Don't ever expose your API key to the public and in the repository. For details on securing your API key while working on Colab, refer to this document: https://docs.google.com/document/d/1D4TP8RCTySyWouqyA8VaDgbhfh1xtqeUqEuUZa1ifRA/edit?usp=sharing

## Contributing

We welcome contributions to improve the system! If you'd like to contribute, please follow these steps:

1. **Fork the repository**

2. **Create a new branch**: Once you have forked the repository, create a new branch for your changes:
   
   ```bash
   git checkout -b your-feature-name

3. **Submit a pull request** with your changes.


For any issues related to the code, please raise them in the **Issues** section of the repository.


   
