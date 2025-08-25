"""Vector store abstraction layer supporting FAISS and Pinecone."""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

def get_secret_or_env(key: str, default: str = None) -> str:
    """Get value from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets') and st.secrets:
            return st.secrets.get(key, os.getenv(key, default))
        else:
            # Fallback to environment variables
            return os.getenv(key, default)
    except Exception:
        # Fallback to environment variables
        return os.getenv(key, default)

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def remove_document(self, document_name: str) -> Dict[str, Any]:
        """Remove all vectors for a specific document."""
        pass
    
    @abstractmethod
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of documents with their chunk counts."""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all vectors from the store."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, storage_dir: str, dimension: int = 384):
        self.storage_dir = Path(storage_dir)
        self.dimension = dimension
        self.index_path = self.storage_dir / "faiss_index.bin"
        self.metadata_path = self.storage_dir / "metadata.json"
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS
        try:
            import faiss
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.metadata = []
            self._load_existing_data()
            logger.info(f"✅ FAISS vector store initialized with {len(self.metadata)} vectors")
        except ImportError:
            logger.error("❌ FAISS not available")
            raise ImportError("FAISS is required for this vector store backend")
    
    def _load_existing_data(self):
        """Load existing FAISS index and metadata if they exist."""
        try:
            import faiss
            import json
            
            if self.index_path.exists() and self.metadata_path.exists():
                # Load index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                logger.info(f"✅ Loaded existing FAISS index with {len(self.metadata)} vectors")
            else:
                logger.info("📁 No existing FAISS index found, starting fresh")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing FAISS data: {e}")
            self.metadata = []
    
    def _save_data(self):
        """Save FAISS index and metadata to disk."""
        try:
            import faiss
            import json
            
            # Save index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            
            logger.info(f"✅ Saved FAISS index with {len(self.metadata)} vectors")
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS data: {e}")
            raise
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to FAISS index."""
        try:
            if len(vectors) != len(metadata):
                raise ValueError("Number of vectors must match number of metadata entries")
            
            # Add vectors to index
            self.index.add(vectors.astype('float32'))
            
            # Add metadata
            self.metadata.extend(metadata)
            
            # Save to disk
            self._save_data()
            
            logger.info(f"✅ Added {len(vectors)} vectors to FAISS index")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add vectors to FAISS: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search for similar vectors in FAISS."""
        try:
            # Search the index
            scores, indices = self.index.search(query_vector.astype('float32').reshape(1, -1), top_k)
            
            # Get metadata for returned indices
            results_metadata = []
            for idx in indices[0]:
                if idx < len(self.metadata):
                    results_metadata.append(self.metadata[idx])
                else:
                    results_metadata.append({})
            
            return scores[0], results_metadata
        except Exception as e:
            logger.error(f"❌ FAISS search failed: {e}")
            return np.array([]), []
    
    def remove_document(self, document_name: str) -> Dict[str, Any]:
        """Remove all vectors for a specific document from FAISS."""
        try:
            # Find indices to remove
            indices_to_remove = []
            for i, meta in enumerate(self.metadata):
                if meta.get('source_document') == document_name:
                    indices_to_remove.append(i)
            
            if not indices_to_remove:
                return {'success': False, 'message': f"No vectors found for document: {document_name}"}
            
            # Create new index and metadata without the removed vectors
            import faiss
            new_index = faiss.IndexFlatIP(self.dimension)
            new_metadata = []
            
            for i, meta in enumerate(self.metadata):
                if i not in indices_to_remove:
                    # Get the vector from the old index
                    vector = self.index.reconstruct(i).reshape(1, -1)
                    new_index.add(vector)
                    new_metadata.append(meta)
            
            # Replace old index and metadata
            self.index = new_index
            self.metadata = new_metadata
            
            # Save updated data
            self._save_data()
            
            logger.info(f"✅ Removed {len(indices_to_remove)} vectors for document: {document_name}")
            return {'success': True, 'message': f"Removed {len(indices_to_remove)} vectors for document: {document_name}"}
        except Exception as e:
            logger.error(f"❌ Failed to remove document from FAISS: {e}")
            return {'success': False, 'message': f"Failed to remove document: {e}"}
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of documents with their chunk counts."""
        try:
            doc_counts = {}
            for meta in self.metadata:
                doc_name = meta.get('source_document', 'Unknown')
                if doc_name not in doc_counts:
                    doc_counts[doc_name] = {
                        'name': doc_name,
                        'chunk_count': 0,
                        'document_type': meta.get('document_type', 'Unknown')
                    }
                doc_counts[doc_name]['chunk_count'] += 1
            
            return list(doc_counts.values())
        except Exception as e:
            logger.error(f"❌ Failed to get document list from FAISS: {e}")
            return []
    
    def clear_all(self) -> bool:
        """Clear all vectors from FAISS."""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            
            # Remove saved files
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            
            logger.info("✅ Cleared all vectors from FAISS")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear FAISS: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS vector store."""
        try:
            return {
                'backend': 'FAISS',
                'total_vectors': len(self.metadata),
                'index_size': self.index.ntotal,
                'dimension': self.dimension,
                'storage_path': str(self.storage_dir)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get FAISS stats: {e}")
            return {'backend': 'FAISS', 'error': str(e)}


class PineconeVectorStore(VectorStore):
    """Pinecone-based vector store implementation."""
    
    def __init__(self, storage_dir: str, dimension: int = 384, index_name: str = "hoa-bot"):
        self.storage_dir = Path(storage_dir)
        self.dimension = dimension
        self.index_name = index_name
        self.metadata_path = self.storage_dir / "pinecone_metadata.json"
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Pinecone
        try:
            import pinecone
            from pinecone import Pinecone
            
            # Debug logging
            logger.info(f"🔍 Pinecone Debug - Index name: {self.index_name}")
            logger.info(f"🔍 Pinecone Debug - Storage dir: {self.storage_dir}")
            
            # Get API key from environment or secrets
            api_key = get_secret_or_env('PINECONE_API_KEY')
            logger.info(f"🔍 Pinecone Debug - API key: {'Set' if api_key else 'Not set'}")
            
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable or secret is required for Pinecone backend")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=api_key)
            
            # Check if index exists, create if not
            existing_indexes = pc.list_indexes()
            if self.index_name not in [idx.name for idx in existing_indexes]:
                logger.info(f"🔧 Creating Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric='cosine',
                    spec={
                        'serverless': {
                            'cloud': 'aws',
                            'region': 'us-east-1'
                        }
                    }
                )
                # Wait for index to be ready
                import time
                time.sleep(10)
            
            self.index = pc.Index(self.index_name)
            self.metadata = []
            self._load_existing_metadata()
            
            logger.info(f"✅ Pinecone vector store initialized with {len(self.metadata)} vectors")
        except ImportError:
            logger.error("❌ Pinecone not available")
            raise ImportError("Pinecone is required for this vector store backend")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {e}")
            raise
    
    def _load_existing_metadata(self):
        """Load existing metadata from disk."""
        try:
            import json
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Loaded existing Pinecone metadata with {len(self.metadata)} entries")
            else:
                logger.info("📁 No existing Pinecone metadata found, starting fresh")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing Pinecone metadata: {e}")
            self.metadata = []
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            import json
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.info(f"✅ Saved Pinecone metadata with {len(self.metadata)} entries")
        except Exception as e:
            logger.error(f"❌ Failed to save Pinecone metadata: {e}")
            raise
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to Pinecone index."""
        try:
            if len(vectors) != len(metadata):
                raise ValueError("Number of vectors must match number of metadata entries")
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                vector_id = f"chunk_{len(self.metadata) + i}"
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': vector.tolist(),
                    'metadata': meta
                })
            
            # Upsert vectors to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            
            # Add metadata to local storage
            self.metadata.extend(metadata)
            self._save_metadata()
            
            logger.info(f"✅ Added {len(vectors)} vectors to Pinecone index")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add vectors to Pinecone: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search for similar vectors in Pinecone."""
        try:
            # Search Pinecone index
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract scores and metadata
            scores = np.array([match.score for match in results.matches])
            metadata = [match.metadata for match in results.matches]
            
            return scores, metadata
        except Exception as e:
            logger.error(f"❌ Pinecone search failed: {e}")
            return np.array([]), []
    
    def remove_document(self, document_name: str) -> Dict[str, Any]:
        """Remove all vectors for a specific document from Pinecone."""
        try:
            # Find vector IDs to remove
            vector_ids_to_remove = []
            for i, meta in enumerate(self.metadata):
                if meta.get('source_document') == document_name:
                    vector_ids_to_remove.append(f"chunk_{i}")
            
            if not vector_ids_to_remove:
                return {'success': False, 'message': f"No vectors found for document: {document_name}"}
            
            # Delete vectors from Pinecone
            self.index.delete(ids=vector_ids_to_remove)
            
            # Update local metadata
            self.metadata = [meta for meta in self.metadata if meta.get('source_document') != document_name]
            self._save_metadata()
            
            logger.info(f"✅ Removed {len(vector_ids_to_remove)} vectors for document: {document_name}")
            return {'success': True, 'message': f"Removed {len(vector_ids_to_remove)} vectors for document: {document_name}"}
        except Exception as e:
            logger.error(f"❌ Failed to remove document from Pinecone: {e}")
            return {'success': False, 'message': f"Failed to remove document: {e}"}
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of documents with their chunk counts."""
        try:
            doc_counts = {}
            
            # If local metadata is empty but we have vectors in Pinecone, try to fetch from Pinecone
            if not self.metadata:
                try:
                    # Get index stats to see if there are vectors
                    index_stats = self.index.describe_index_stats()
                    if index_stats.total_vector_count > 0:
                        logger.info(f"📁 Local metadata empty but found {index_stats.total_vector_count} vectors in Pinecone")
                        # Try to fetch metadata from Pinecone by querying with a dummy vector
                        return self._recover_document_list_from_pinecone()
                except Exception as e:
                    logger.warning(f"⚠️ Could not check Pinecone index stats: {e}")
                    return []
            
            # Use local metadata to build document list
            for meta in self.metadata:
                doc_name = meta.get('source_document', 'Unknown')
                if doc_name not in doc_counts:
                    doc_counts[doc_name] = {
                        'name': doc_name,
                        'chunk_count': 0,
                        'document_type': meta.get('document_type', 'Unknown')
                    }
                doc_counts[doc_name]['chunk_count'] += 1
            
            return list(doc_counts.values())
        except Exception as e:
            logger.error(f"❌ Failed to get document list from Pinecone: {e}")
            return []
    
    def _recover_document_list_from_pinecone(self) -> List[Dict[str, Any]]:
        """Recover document list by querying Pinecone for metadata."""
        try:
            # Create a dummy query vector (all zeros) to fetch some vectors with metadata
            dummy_vector = [0.0] * self.dimension
            
            # Query Pinecone to get some vectors with metadata
            results = self.index.query(
                vector=dummy_vector,
                top_k=1000,  # Get as many as possible
                include_metadata=True
            )
            
            if not results.matches:
                logger.warning("⚠️ No vectors found in Pinecone query")
                return []
            
            # Extract document information from metadata
            doc_counts = {}
            for match in results.matches:
                if match.metadata:
                    doc_name = match.metadata.get('source_document', 'Unknown')
                    doc_type = match.metadata.get('document_type', 'Unknown')
                    
                    if doc_name not in doc_counts:
                        doc_counts[doc_name] = {
                            'name': doc_name,
                            'chunk_count': 0,
                            'document_type': doc_type
                        }
                    doc_counts[doc_name]['chunk_count'] += 1
            
            logger.info(f"✅ Recovered {len(doc_counts)} documents from Pinecone metadata")
            return list(doc_counts.values())
            
        except Exception as e:
            logger.error(f"❌ Failed to recover document list from Pinecone: {e}")
            return []
    
    def clear_all(self) -> bool:
        """Clear all vectors from Pinecone."""
        try:
            # Delete all vectors from Pinecone
            self.index.delete(delete_all=True)
            
            # Clear local metadata
            self.metadata = []
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            
            logger.info("✅ Cleared all vectors from Pinecone")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear Pinecone: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone vector store."""
        try:
            # Use local metadata count for immediate accuracy
            total_vectors = len(self.metadata)
            
            # Try to get index stats for additional info
            try:
                index_stats = self.index.describe_index_stats()
                index_vector_count = index_stats.total_vector_count
            except Exception:
                index_vector_count = total_vectors
            
            return {
                'backend': 'Pinecone',
                'total_vectors': total_vectors,
                'index_vector_count': index_vector_count,
                'dimension': self.dimension,
                'index_name': self.index_name,
                'local_metadata_count': len(self.metadata)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get Pinecone stats: {e}")
            return {'backend': 'Pinecone', 'error': str(e)}


def create_vector_store(backend: str = "faiss", storage_dir: str = "data", **kwargs) -> VectorStore:
    """Factory function to create a vector store instance."""
    
    backend = backend.lower()
    dimension = kwargs.get('dimension', 384)
    
    if backend == "faiss":
        return FAISSVectorStore(storage_dir, dimension)
    elif backend == "pinecone":
        index_name = kwargs.get('index_name', 'hoa-bot')
        return PineconeVectorStore(storage_dir=storage_dir, dimension=dimension, index_name=index_name)
    else:
        raise ValueError(f"Unsupported vector store backend: {backend}. Supported backends: faiss, pinecone")
