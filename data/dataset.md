# FusionGraph Synthetic Multimodal Dataset

This directory contains a synthetic multimodal dataset for testing and demonstrating FusionGraph's capabilities.

## Dataset Overview

**2-Domain Corpus:**
- **AI Research Domain**: Neural Networks foundations and applications
- **Technical Documentation Domain**: Quantum Computing comprehensive guide

**Total Content:**
- 2 documents (6 pages total)
- 6 images/diagrams
- 2,016 total words
- 80 extracted entities (estimated)

## Directory Structure

```
data/
├── pdfs/                           # Original source documents
│   ├── AI_Neural_Networks.pdf      # 3 pages, 2.1MB
│   └── Quantum_Computing_Guide.pdf # 3 pages, 1.8MB
├── text/                           # Extracted page-level text
│   ├── AI_Neural_Networks/
│   │   ├── page_01.txt            # Introduction (245 words)
│   │   ├── page_02.txt            # Network types (312 words)
│   │   └── page_03.txt            # Training methods (387 words)
│   └── Quantum_Computing_Guide/
│       ├── page_01.txt            # Introduction (298 words)
│       ├── page_02.txt            # Algorithms & hardware (351 words)
│       └── page_03.txt            # ML & future apps (423 words)
├── images/                         # Extracted diagrams and charts
│   ├── AI_Neural_Networks/
│   │   ├── page_01_img_1.png      # Feedforward network diagram
│   │   ├── page_02_img_1.png      # RNN architecture diagram
│   │   └── page_03_img_1.png      # Training curves chart
│   └── Quantum_Computing_Guide/
│       ├── page_01_img_1.png      # Quantum circuit diagram
│       ├── page_02_img_1.png      # Hardware comparison chart
│       └── page_03_img_1.png      # Roadmap timeline
├── metadata/                       # Document manifests
│   ├── AI_Neural_Networks.json    # Page/image mappings
│   └── Quantum_Computing_Guide.json
└── ground_truth/                   # Evaluation queries and relevance
    ├── queries.json               # 10 test queries across complexity levels
    └── qrels.tsv                  # Relevance judgments (TREC format)
```

## Document Details

### AI_Neural_Networks.pdf
- **Domain**: AI Research
- **Topics**: Neural network architectures, training, optimization
- **Images**: Network diagrams, training curves
- **Key Concepts**: CNNs, RNNs, backpropagation, regularization

### Quantum_Computing_Guide.pdf
- **Domain**: Technical Documentation  
- **Topics**: Quantum principles, algorithms, hardware, applications
- **Images**: Circuit diagrams, hardware comparisons, roadmaps
- **Key Concepts**: Qubits, superposition, Shor's algorithm, quantum ML

## Ground Truth Queries

10 evaluation queries covering:
- **Simple retrieval** (Q001, Q005, Q006): Direct fact lookup
- **Medium complexity** (Q002, Q003, Q007, Q009): Process explanation
- **Complex synthesis** (Q004, Q008, Q010): Cross-document reasoning
- **Multimodal** (Q005, Q007, Q008): Requiring image context

## Usage in FusionGraph

1. **Text Chunking**: Split page text into ~200-300 word chunks with provenance
2. **Embedding Generation**: Use sentence-transformers for text, CLIP for images
3. **Knowledge Graph**: Extract entities and relationships from text content
4. **Faiss Indexing**: Create HNSW index for fast retrieval
5. **Evaluation**: Run queries against ground truth for NDCG@K, MRR metrics

## Expected Performance Targets

- **Retrieval Quality**: NDCG@3 ≥ 0.75, MRR ≥ 0.60
- **Factual Grounding**: Support ratio ≥ 0.80
- **Multimodal Integration**: Cross-modal alignment ≥ 0.65
- **Response Time**: <2 seconds per query with Faiss optimization

## Extending the Dataset

To add more documents:
1. Place PDFs in `pdfs/`
2. Extract text to `text/<doc_id>/`
3. Extract images to `images/<doc_id>/`
4. Create metadata JSON in `metadata/`
5. Add evaluation queries to `ground_truth/`

## Limitations

- **Synthetic content**: Generated for demonstration purposes
- **Small scale**: Only 2 documents (expand for production use)
- **Image placeholders**: Text descriptions instead of actual images
- **Limited domains**: Add more domains for robust evaluation

This dataset provides a foundation for testing FusionGraph's multimodal RAG capabilities while maintaining a manageable size for development and demonstration.