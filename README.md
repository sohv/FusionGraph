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
1. **Clone the repository**:
   ```bash
   git clone https://github.com/sohv/RAG-knowledgegraph.git

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Documents for testing the system**:
   The documents are arranged in following structure:
   ```markdown
  - `documents/`
    - `sample/`
      - `sample.pdf`
      - `sample1.pdf`
    - `sample_ai/`
      - `AI Trend story.pdf`
      - `A_Brief_Introduction_To_AI.pdf`

4. **Run the application** : Follow the instructions in the code to generate the knowledge graph successfully and visualize it.
