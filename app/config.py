"""Configuration settings for the HOA Bot application."""

import os
import streamlit as st
from typing import Dict, Any

def get_secret_or_env(key: str, default: str = None) -> str:
    """Get value from Streamlit secrets or environment variables."""
    try:
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets') and st.secrets:
            return st.secrets.get(key, os.getenv(key, default))
        else:
            # Fallback to environment variables
            return os.getenv(key, default)
    except Exception:
        # Fallback to environment variables
        return os.getenv(key, default)

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    # Default vector store backend
    'default_backend': get_secret_or_env('VECTOR_STORE_BACKEND', 'faiss'),  # 'faiss' or 'pinecone'
    
    # FAISS Configuration
    'faiss': {
        'metric': 'cosine',  # Similarity metric
    },
    
    # Pinecone Configuration
    'pinecone': {
        'index_name': get_secret_or_env('PINECONE_INDEX_NAME', 'hoa-bot'),
        'metric': 'cosine',  # Similarity metric
        'environment': get_secret_or_env('PINECONE_ENVIRONMENT', 'us-east1-aws'),  # Pinecone environment
    }
}

def get_vector_store_config() -> Dict[str, Any]:
    """Get vector store configuration based on environment variables."""
    backend = VECTOR_STORE_CONFIG['default_backend']
    
    # Debug logging
    print(f"🔍 Config Debug - VECTOR_STORE_BACKEND: {get_secret_or_env('VECTOR_STORE_BACKEND', 'Not set')}")
    print(f"🔍 Config Debug - PINECONE_API_KEY: {'Set' if get_secret_or_env('PINECONE_API_KEY') else 'Not set'}")
    print(f"🔍 Config Debug - PINECONE_INDEX_NAME: {get_secret_or_env('PINECONE_INDEX_NAME', 'Not set')}")
    print(f"🔍 Config Debug - PINECONE_ENVIRONMENT: {get_secret_or_env('PINECONE_ENVIRONMENT', 'Not set')}")
    print(f"🔍 Config Debug - Selected backend: {backend}")
    
    if backend == 'pinecone':
        # Validate Pinecone configuration
        if not get_secret_or_env('PINECONE_API_KEY'):
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
