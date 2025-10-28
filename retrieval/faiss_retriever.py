import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with explainability metadata"""
    chunk_id: str
    text: str
    score: float
    source_id: str
    source_name: str
    chunk_offsets: Tuple[int, int]  # start, end positions in original document
    embedding_model: str
    retrieval_time: str
    metadata: Dict[str, Any]


@dataclass
class ExplainabilityMetadata:
    """Metadata for explaining retrieval decisions"""
    query_embedding_norm: float
    top_k_requested: int
    total_candidates: int
    similarity_method: str
    confidence_score: float
    confidence_explanation: str


class FaissRetriever:
    """
    Faiss-based retriever with built-in explainability features
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "./storage/faiss_index",
                 metadata_path: str = "./storage/retrieval_metadata.json"):
        """
        Initialize Faiss retriever
        
        Args:
            embedding_model_name: Name of the embedding model
            index_path: Path to store Faiss index
            metadata_path: Path to store chunk metadata
        """
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Faiss index (HNSW for speed/accuracy balance)
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 = M parameter
        self.index.hnsw.efConstruction = 128
        self.index.hnsw.efSearch = 64
        
        # Metadata storage
        self.chunk_metadata = {}
        self.id_to_vector_idx = {}  # chunk_id -> vector index mapping
        self.vector_idx_to_id = {}  # vector index -> chunk_id mapping
        self._next_vector_idx = 0
        
        # Load existing index if available
        self.load_index()
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]], 
                     chunk_size: int = 256, 
                     chunk_overlap: int = 50) -> None:
        """
        Add documents to the index with chunking and metadata preservation
        
        Args:
            documents: List of documents with 'text', 'source_id', 'source_name', etc.
            chunk_size: Size of text chunks in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
        """
        all_chunks = []
        all_embeddings = []
        
        for doc in documents:
            chunks = self._chunk_document(doc, chunk_size, chunk_overlap)
            
            for chunk_data in chunks:
                chunk_id = f"{doc['source_id']}_{chunk_data['chunk_idx']}"
                
                # Store metadata
                self.chunk_metadata[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': chunk_data['text'],
                    'source_id': doc['source_id'],
                    'source_name': doc.get('source_name', 'Unknown'),
                    'chunk_offsets': chunk_data['offsets'],
                    'embedding_model': self.embedding_model_name,
                    'indexed_time': datetime.now().isoformat(),
                    'metadata': doc.get('metadata', {})
                }
                
                # Map chunk_id to vector index
                self.id_to_vector_idx[chunk_id] = self._next_vector_idx
                self.vector_idx_to_id[self._next_vector_idx] = chunk_id
                self._next_vector_idx += 1
                
                all_chunks.append(chunk_data['text'])
        
        # Compute embeddings in batch
        if all_chunks:
            embeddings = self.embedding_model.encode(all_chunks, convert_to_tensor=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            print(f"Added {len(all_chunks)} chunks to index. Total: {self.index.ntotal}")
    
    def retrieve(self, 
                query: str, 
                top_k: int = 5, 
                return_explanations: bool = True) -> Tuple[List[RetrievalResult], Optional[ExplainabilityMetadata]]:
        """
        Retrieve relevant chunks with explainability metadata
        
        Args:
            query: Search query
            top_k: Number of results to return
            return_explanations: Whether to include explainability metadata
            
        Returns:
            Tuple of (retrieval_results, explainability_metadata)
        """
        if self.index.ntotal == 0:
            return [], None
        
        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize query embedding
        query_norm = float(np.linalg.norm(query_embedding))
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Build results
        results = []
        current_time = datetime.now().isoformat()
        
        for score, vector_idx in zip(scores[0], indices[0]):
            if vector_idx == -1:  # Faiss returns -1 for missing results
                continue
                
            chunk_id = self.vector_idx_to_id.get(vector_idx)
            if chunk_id and chunk_id in self.chunk_metadata:
                metadata = self.chunk_metadata[chunk_id]
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    text=metadata['text'],
                    score=float(score),
                    source_id=metadata['source_id'],
                    source_name=metadata['source_name'],
                    chunk_offsets=metadata['chunk_offsets'],
                    embedding_model=metadata['embedding_model'],
                    retrieval_time=current_time,
                    metadata=metadata['metadata']
                )
                results.append(result)
        
        # Generate explainability metadata
        explainability = None
        if return_explanations and results:
            avg_score = np.mean([r.score for r in results])
            
            # Simple confidence based on average similarity and number of results
            confidence = min(1.0, avg_score * (len(results) / top_k))
            confidence_explanation = f"Average similarity: {avg_score:.3f}, Found {len(results)}/{top_k} results"
            
            explainability = ExplainabilityMetadata(
                query_embedding_norm=query_norm,
                top_k_requested=top_k,
                total_candidates=self.index.ntotal,
                similarity_method="cosine (L2 normalized)",
                confidence_score=confidence,
                confidence_explanation=confidence_explanation
            )
        
        return results, explainability
    
    def _chunk_document(self, doc: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Simple text chunking with overlap tracking
        """
        text = doc['text']
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate approximate character offsets
            start_offset = len(' '.join(words[:i])) + (1 if i > 0 else 0)
            end_offset = start_offset + len(chunk_text)
            
            chunks.append({
                'text': chunk_text,
                'chunk_idx': len(chunks),
                'offsets': (start_offset, end_offset)
            })
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def save_index(self) -> None:
        """Save index and metadata to disk"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save Faiss index
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # Save metadata
        save_data = {
            'chunk_metadata': self.chunk_metadata,
            'id_to_vector_idx': self.id_to_vector_idx,
            'vector_idx_to_id': self.vector_idx_to_id,
            'next_vector_idx': self._next_vector_idx,
            'embedding_model_name': self.embedding_model_name,
            'embedding_dim': self.embedding_dim
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_index(self) -> bool:
        """Load index and metadata from disk"""
        try:
            # Load Faiss index
            if os.path.exists(f"{self.index_path}.faiss"):
                self.index = faiss.read_index(f"{self.index_path}.faiss")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    save_data = json.load(f)
                
                self.chunk_metadata = save_data['chunk_metadata']
                self.id_to_vector_idx = save_data['id_to_vector_idx']
                self.vector_idx_to_id = {int(k): v for k, v in save_data['vector_idx_to_id'].items()}
                self._next_vector_idx = save_data['next_vector_idx']
                
                return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'total_chunks': self.index.ntotal,
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'unique_sources': len(set(meta['source_id'] for meta in self.chunk_metadata.values()))
        }