"""Configuration settings for the HOA Bot application."""

import os
from typing import Dict, Any

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    # Default vector store backend
    'default_backend': os.getenv('VECTOR_STORE_BACKEND', 'faiss'),  # 'faiss' or 'pinecone'
    
    # FAISS Configuration
    'faiss': {
        'metric': 'cosine',  # Similarity metric
    },
    
    # Pinecone Configuration
    'pinecone': {
        'index_name': os.getenv('PINECONE_INDEX_NAME', 'hoa-bot'),
        'metric': 'cosine',  # Similarity metric
        'environment': os.getenv('PINECONE_ENVIRONMENT', 'us-east1-aws'),  # Pinecone environment
    }
}

def get_vector_store_config() -> Dict[str, Any]:
    """Get vector store configuration based on environment variables."""
    backend = VECTOR_STORE_CONFIG['default_backend']
    
    # Debug logging
    print(f"🔍 Config Debug - VECTOR_STORE_BACKEND: {os.getenv('VECTOR_STORE_BACKEND', 'Not set')}")
    print(f"🔍 Config Debug - PINECONE_API_KEY: {'Set' if os.getenv('PINECONE_API_KEY') else 'Not set'}")
    print(f"🔍 Config Debug - PINECONE_INDEX_NAME: {os.getenv('PINECONE_INDEX_NAME', 'Not set')}")
    print(f"🔍 Config Debug - PINECONE_ENVIRONMENT: {os.getenv('PINECONE_ENVIRONMENT', 'Not set')}")
    print(f"🔍 Config Debug - Selected backend: {backend}")
    
    if backend == 'pinecone':
        # Validate Pinecone configuration
        if not os.getenv('PINECONE_API_KEY'):
            print("⚠️ PINECONE_API_KEY not found, falling back to FAISS")
            backend = 'faiss'
    
    config = {
        'backend': backend,
        'config': VECTOR_STORE_CONFIG.get(backend, {})
    }
    
    return config

def get_rag_chatbot_config() -> Dict[str, Any]:
    """Get RAG chatbot configuration."""
    vector_config = get_vector_store_config()
    
    return {
        'vector_store_backend': vector_config['backend'],
        'vector_store_config': vector_config['config'],
        'enable_reflection': os.getenv('ENABLE_REFLECTION', 'true').lower() == 'true'
    }
