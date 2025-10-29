# FusionGraph Benchmark Framework

This directory contains the comprehensive benchmark framework for evaluating FusionGraph against simple baselines.

## Dataset Description

**FusionGraph-AI-Research Dataset:**
- **Domain**: Artificial Intelligence Research & Technical Documentation
- **Size**: 4 PDF documents across AI/ML topics
- **Content**: Research papers, technical guides, sample documentation
- **Organization**: Structured in `documents/text/`, `documents/images/`, `documents/examples/`
- **Total Size**: ~2-5 MB of text content
- **Knowledge Graph**: ~100 entities, ~160 relationships (estimated)

### Document Inventory:
1. `A_Brief_Introduction_To_AI.pdf` - Foundational AI concepts
2. `AI Trend story.pdf` - AI industry trends and developments  
3. `sample.pdf` - Technical documentation samples
4. `sample1.pdf` - Additional sample content

## Baseline Definitions

### Baseline 1: Plain RAG
- **Configuration**: LlamaIndex VectorStoreIndex only
- **Features**: Vector-based retrieval, no knowledge graph, no images
- **Purpose**: Isolate the value-add of knowledge graphs
- **Expected Performance**: Fast but lower accuracy on complex queries

### Baseline 2: Text-only KG  
- **Configuration**: KnowledgeGraphIndex with text documents only
- **Features**: Knowledge graph construction, no image processing
- **Purpose**: Isolate the value-add of multimodal capabilities
- **Expected Performance**: Better factual accuracy, slower than Plain RAG

### Baseline 3: No-Faiss Baseline
- **Configuration**: Full FusionGraph features without Faiss optimization
- **Features**: KG + multimodal, standard LlamaIndex retrieval
- **Purpose**: Isolate the performance impact of Faiss optimization
- **Expected Performance**: Full accuracy but slower retrieval

## Benchmark Methodology

### Query Categories:
1. **Factual Retrieval**: Direct information lookup
2. **Process Explanation**: How-to and procedural knowledge  
3. **Comparative Analysis**: Differences and relationships
4. **Conceptual Relationships**: Complex interconnected concepts
5. **Cross-document Synthesis**: Information spanning multiple sources
6. **Multimodal Queries**: Requiring visual context

### Evaluation Metrics:
- **Retrieval Quality**: NDCG@K, MRR, Precision@K
- **Factual Consistency**: NLI-based verification, hallucination detection
- **Execution Performance**: Latency, throughput, scalability
- **Answer Quality**: Length, completeness, source diversity

## Usage

### Quick Start:
```bash
# Run complete benchmark suite
python benchmark/run_benchmark.py

# Run with verbose output
python benchmark/run_benchmark.py --verbose

# Run specific baseline only
python benchmark/run_benchmark.py --baseline plain_rag
```

### Advanced Usage:
```python
from benchmark.benchmark_framework import BenchmarkRunner

# Initialize benchmark runner
runner = BenchmarkRunner()

# Phase 1: Analyze dataset
runner.analyze_dataset_and_generate_queries()

# Phase 2: Run comparisons  
runner.run_baseline_comparisons()

# Phase 3: Generate analysis
insights = runner.generate_comparative_analysis()
```

## Expected Results

### Performance Ranking:
- **Retrieval Quality**: FusionGraph > Text-only KG > Plain RAG
- **Factual Consistency**: FusionGraph ≈ Text-only KG > Plain RAG
- **Execution Speed**: FusionGraph (Faiss) > No-Faiss > Others
- **Multimodal Capability**: FusionGraph > Others (N/A)

### Value Propositions:
1. **Knowledge Graph**: Improves retrieval precision and factual grounding
2. **Multimodal Processing**: Enables richer context from images and diagrams
3. **Faiss Optimization**: Significantly reduces retrieval latency (5-10x speedup)

### Use Case Recommendations:
- **High Accuracy Needs**: FusionGraph with full KG + multimodal
- **Speed Critical**: FusionGraph with Faiss optimization
- **Resource Constrained**: Plain RAG baseline
- **Text-only Domains**: Text-only KG baseline

## Files

- `benchmark_framework.py`: Core benchmark infrastructure
- `run_benchmark.py`: Executable benchmark runner
- `README.md`: This documentation
- `results/`: Benchmark results and analysis (generated)

## Research Applications

This benchmark framework enables:
1. **Academic Evaluation**: Quantitative comparison for research papers
2. **System Optimization**: Performance profiling and bottleneck identification  
3. **Feature Validation**: Evidence-based justification of design choices
4. **Deployment Planning**: Informed system selection based on requirements

## Integration with Evaluation Framework

The benchmark framework integrates with the existing evaluation framework:
- Uses same metrics (NDCG@K, MRR, NLI scores)
- Leverages evaluation infrastructure for consistent measurements
- Extends with comparative analysis and baseline implementations
- Maintains research-grade rigor and reproducibility

## Limitations

1. **Dataset Size**: Limited to 4 documents (expandable)
2. **Baseline Sophistication**: Simple baselines (can add more advanced ones)
3. **Evaluation Scope**: Focused on RAG performance (not end-to-end applications)
4. **Resource Requirements**: Knowledge graph construction can be memory intensive

## Future Extensions

1. **Larger Datasets**: Scale to 100+ documents across domains
2. **Advanced Baselines**: Include GPT-4, RAG-Fusion, etc.
3. **Real-time Benchmarks**: Streaming evaluation and online learning
4. **Domain-specific Tests**: Specialized benchmarks for medical, legal, etc.
5. **Human Evaluation**: User studies and expert assessments