"""
Model caching system for sentence transformers to avoid repeated downloads.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
CACHE_DIR = Path.home() / '.cache' / 'huggingface' / 'transformers'
MODEL_CACHE_DIR = CACHE_DIR / 'sentence-transformers' / MODEL_NAME

def ensure_model_cache_dir():
    """Ensure the model cache directory exists."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_CACHE_DIR

def is_model_cached() -> bool:
    """Check if the model is already cached locally."""
    try:
        # Check if the model directory exists and has the necessary files
        if not MODEL_CACHE_DIR.exists():
            return False
        
        # Check for essential model files
        required_files = [
            'config.json',
            'pytorch_model.bin',
            'sentence_bert_config.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.txt'
        ]
        
        for file_name in required_files:
            file_path = MODEL_CACHE_DIR / file_name
            if not file_path.exists():
                logger.info(f"Missing required model file: {file_name}")
                return False
        
        logger.info(f"✅ Model {MODEL_NAME} is cached at {MODEL_CACHE_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking model cache: {e}")
        return False

def download_model_with_retry(max_retries: int = 3, retry_delay: int = 5) -> Optional[SentenceTransformer]:
    """Download the model with retry logic and better error handling."""
    
    # Set environment variables to avoid warnings and improve caching
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)
    
    # Disable Hugging Face Hub warnings
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Downloading model {MODEL_NAME} (attempt {attempt + 1}/{max_retries})")
            
            # Create cache directory
            ensure_model_cache_dir()
            
            # Download model with specific settings
            model = SentenceTransformer(
                MODEL_NAME,
                device='cpu',
                cache_folder=str(CACHE_DIR)
            )
            
            # Test the model to ensure it's working
            test_text = "This is a test sentence."
            embedding = model.encode(test_text)
            
            if embedding is not None and len(embedding) > 0:
                logger.info(f"✅ Model {MODEL_NAME} downloaded and tested successfully")
                return model
            else:
                raise Exception("Model test failed - no embedding generated")
                
        except Exception as e:
            logger.error(f"❌ Download attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"❌ Failed to download model after {max_retries} attempts")
                return None
    
    return None

def load_cached_model() -> Optional[SentenceTransformer]:
    """Load the model from cache if available."""
    try:
        if is_model_cached():
            logger.info(f"📁 Loading cached model {MODEL_NAME}")
            
            # Set environment variables
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HOME"] = str(CACHE_DIR)
            os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
            
            # Load from cache
            model = SentenceTransformer(
                MODEL_NAME,
                device='cpu',
                cache_folder=str(CACHE_DIR)
            )
            
            # Test the model
            test_text = "This is a test sentence."
            embedding = model.encode(test_text)
            
            if embedding is not None and len(embedding) > 0:
                logger.info(f"✅ Cached model {MODEL_NAME} loaded successfully")
                return model
            else:
                logger.warning("⚠️ Cached model test failed, will try to re-download")
                return None
                
    except Exception as e:
        logger.error(f"❌ Error loading cached model: {e}")
        return None
    
    return None

def get_sentence_transformer() -> Optional[SentenceTransformer]:
    """Get the sentence transformer model, loading from cache or downloading as needed."""
    
    # Try to load from cache first
    model = load_cached_model()
    if model is not None:
        return model
    
    # If not cached, download it
    logger.info(f"📥 Model {MODEL_NAME} not found in cache, downloading...")
    model = download_model_with_retry()
    
    if model is not None:
        logger.info(f"✅ Model {MODEL_NAME} ready for use")
        return model
    else:
        logger.error(f"❌ Failed to load model {MODEL_NAME}")
        return None

def clear_model_cache():
    """Clear the model cache (useful for troubleshooting)."""
    try:
        if MODEL_CACHE_DIR.exists():
            import shutil
            shutil.rmtree(MODEL_CACHE_DIR)
            logger.info(f"🧹 Cleared model cache: {MODEL_CACHE_DIR}")
        else:
            logger.info("📁 No model cache to clear")
    except Exception as e:
        logger.error(f"❌ Error clearing model cache: {e}")

def get_model_info() -> dict:
    """Get information about the model cache status."""
    return {
        'model_name': MODEL_NAME,
        'cache_dir': str(MODEL_CACHE_DIR),
        'is_cached': is_model_cached(),
        'cache_exists': MODEL_CACHE_DIR.exists(),
        'cache_size_mb': get_cache_size_mb() if MODEL_CACHE_DIR.exists() else 0
    }

def get_cache_size_mb() -> float:
    """Get the size of the model cache in MB."""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(MODEL_CACHE_DIR):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return round(total_size / (1024 * 1024), 2)
    except Exception as e:
        logger.error(f"❌ Error calculating cache size: {e}")
        return 0.0
