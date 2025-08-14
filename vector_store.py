import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from openai import OpenAI
import json
from datetime import datetime

class PersistentFAISSVectorStore:
    def __init__(self, embedding_model: str = "text-embedding-3-small", 
                 dimension: int = 1536, storage_path: str = "vector_storage"):
        """
        Initialize Persistent FAISS Vector Store with OpenAI embeddings
        
        Args:
            embedding_model: OpenAI embedding model name
            dimension: Embedding dimension
            storage_path: Path to store vector database
        """
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.storage_path = storage_path
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # File paths
        self.index_file = os.path.join(storage_path, "faiss_index.bin")
        self.documents_file = os.path.join(storage_path, "documents.pkl")
        self.metadata_file = os.path.join(storage_path, "metadata.pkl")
        self.config_file = os.path.join(storage_path, "config.json")
        
        # Initialize or load existing index
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Try to load existing data
        if self.load_existing_data():
            print(f"Loaded existing vector store with {len(self.documents)} documents")
        else:
            self.reset_index()
            print("Initialized new vector store")
    
    def load_existing_data(self) -> bool:
        """Load existing vector store data if available"""
        try:
            if (os.path.exists(self.index_file) and 
                os.path.exists(self.documents_file) and 
                os.path.exists(self.metadata_file)):
                
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                # Load documents and metadata
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                return True
        except Exception as e:
            print(f"Error loading existing data: {e}")
        
        return False
    
    def reset_index(self):
        """Reset the FAISS index"""
        # Using Inner Product for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(self.dimension, dtype=np.float32)
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[np.ndarray]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                
                batch_embeddings = []
                for data in response.data:
                    embedding = np.array(data.embedding, dtype=np.float32)
                    # Normalize for cosine similarity
                    embedding = embedding / np.linalg.norm(embedding)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                print(f"Processed embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"Error in embedding batch {i//batch_size + 1}: {e}")
                # Add zero embeddings for failed batch
                for _ in batch:
                    embeddings.append(np.zeros(self.dimension, dtype=np.float32))
        
        return embeddings
    
    def add_documents(self, chunks: List[Dict]) -> bool:
        """Add document chunks to the vector store"""
        if not chunks:
            return False
        
        print(f"Adding {len(chunks)} documents to vector store...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Get embeddings in batches
        embeddings = self.get_embeddings_batch(texts)
        
        # Convert to numpy array
        embeddings_array = np.vstack(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend([chunk['metadata'] for chunk in chunks])
        
        # Save immediately after adding
        self.save_to_disk()
        
        print(f"Successfully added {len(chunks)} documents. Total documents: {len(self.documents)}")
        return True
    
    def search(self, query: str, k: int = 5, language_filter: str = None) -> List[Dict]:
        """Search for similar documents with optional language filtering"""
        if self.index.ntotal == 0:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search with more results for filtering
        search_k = min(k * 3, self.index.ntotal)  # Get more results for filtering
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                doc_metadata = self.metadata[idx]
                
                # Apply language filter if specified
                if language_filter:
                    doc_language = doc_metadata.get('primary_language', 'unknown')
                    if language_filter != doc_language and doc_language != 'mixed':
                        continue
                
                result = {
                    'text': self.documents[idx],
                    'metadata': doc_metadata,
                    'score': float(score),
                    'rank': len(results) + 1
                }
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        return results
    
    def save_to_disk(self):
        """Save the vector store to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save documents and metadata
            with open(self.documents_file, "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.metadata, f)
            
            # Save configuration
            config = {
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "total_documents": len(self.documents),
                "last_updated": datetime.now().isoformat()
            }
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"Vector store saved to {self.storage_path}")
            
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0,
            "storage_path": self.storage_path
        }
    
    def get_documents_by_file(self, filename: str) -> List[Dict]:
        """Get all chunks from a specific document"""
        results = []
        for i, doc_metadata in enumerate(self.metadata):
            if doc_metadata.get('filename') == filename:
                results.append({
                    'text': self.documents[i],
                    'metadata': doc_metadata,
                    'index': i
                })
        return results
    
    def delete_documents_by_file(self, filename: str) -> int:
        """Delete all chunks from a specific document (not implemented for FAISS)"""
        # Note: FAISS doesn't support deletion easily
        # This would require rebuilding the entire index
        print("Document deletion not implemented - would require index rebuild")
        return 0
    
    def clear_all_data(self):
        """Clear all data and reset"""
        self.reset_index()
        
        # Remove files
        for file_path in [self.index_file, self.documents_file, self.metadata_file, self.config_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        print("All vector store data cleared")