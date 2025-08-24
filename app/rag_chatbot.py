"""
RAG Chatbot for HOA Document Analysis
Uses FAISS vector store and Perplexity API for document-aware conversations.
"""

import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import requests
import time
from pathlib import Path
from app.utils import truncate_text

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAGChatbot:
    """RAG-powered chatbot using FAISS vector store and Perplexity API."""
    
    def __init__(self, storage_dir: str = "data", property_id: str = None, enable_reflection: bool = True):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar"
        self.timeout = 10
        self.max_retries = 3
        self.rate_limit_delay = 1.0
        
        # Reflection configuration
        self.enable_reflection = enable_reflection
        
        # Property-specific storage
        self.property_id = property_id
        if property_id:
            self.storage_dir = Path(storage_dir) / "properties" / property_id
        else:
            self.storage_dir = Path(storage_dir)
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "faiss_index.bin"
        self.chunks_file = self.storage_dir / "chunks.json"
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # Initialize sentence transformer for embeddings with device handling
        import torch
        
        # Set environment variables to avoid meta tensor issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        try:
            # Try to initialize with explicit CPU device
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        except Exception as e:
            try:
                # Fallback: initialize without device and manually move to CPU
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Try different methods to move to CPU
                try:
                    self.embedding_model.to_empty(device='cpu')
                except:
                    try:
                        self.embedding_model.to('cpu')
                    except:
                        # Last resort: try to force CPU
                        for param in self.embedding_model.parameters():
                            param.data = param.data.cpu()
            except Exception as inner_e:
                print(f"Warning: Could not initialize SentenceTransformer properly: {inner_e}")
                # Create a minimal fallback - this will be slow but should work
                self.embedding_model = None
        
        # FAISS index for document storage
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Document storage
        self.documents = []
        self.document_metadata = []
        

        
        # Load existing data if available
        self._load_persistent_data()
        
    def _load_persistent_data(self):
        """Load persistent data from disk."""
        try:
            # Load FAISS index
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load chunks
            if self.chunks_file.exists():
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"✅ Loaded {len(self.documents)} document chunks")
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(self.document_metadata)} chunks")
                
        except Exception as e:
            print(f"⚠️ Error loading persistent data: {e}")
            # Reset to empty state if loading fails
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.document_metadata = []
    
    def _save_persistent_data(self):
        """Save data to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save chunks
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Saved {len(self.documents)} chunks and metadata to disk")
            
        except Exception as e:
            print(f"❌ Error saving persistent data: {e}")
    
    def clear_all_data(self) -> Dict[str, Any]:
        """Clear all stored data from memory and disk."""
        try:
            # Clear memory
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.document_metadata = []
            
            # Remove files from disk
            files_removed = []
            for file_path in [self.index_file, self.chunks_file, self.metadata_file]:
                if file_path.exists():
                    file_path.unlink()
                    files_removed.append(file_path.name)
            
            # Remove storage directory if empty
            if self.storage_dir.exists() and not any(self.storage_dir.iterdir()):
                self.storage_dir.rmdir()
            
            return {
                'success': True,
                'message': f"Cleared all data. Removed files: {', '.join(files_removed)}",
                'files_removed': files_removed
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error clearing data: {str(e)}",
                'files_removed': []
            }
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        total_size = 0
        file_info = {}
        
        for file_path in [self.index_file, self.chunks_file, self.metadata_file]:
            if file_path.exists():
                size = file_path.stat().st_size
                total_size += size
                file_info[file_path.name] = {
                    'size_bytes': size,
                    'size_mb': round(size / (1024 * 1024), 2)
                }
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files': file_info,
            'chunks_in_memory': len(self.documents),
            'vectors_in_index': self.index.ntotal
        }
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store."""
        if not documents:
            return
            
        # Extract text chunks from documents
        chunks = []
        metadata = []
        
        for doc in documents:
            if 'text' in doc and doc['text']:
                # Split text into chunks (token-based approach)
                text_chunks = self._split_text_into_chunks(doc['text'], max_tokens=400, overlap=50)
                
                for i, chunk in enumerate(text_chunks):
                    chunks.append(chunk)
                    metadata.append({
                        'document_name': doc.get('name', 'Unknown'),
                        'document_type': doc.get('type', 'Unknown'),
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'source_document': doc.get('name', 'Unknown')
                    })
        
        if chunks:
            # Check if embedding model is available
            if self.embedding_model is None:
                return {
                    'success': False,
                    'message': 'Embedding model not available. Please restart the application.',
                    'chunks_added': 0
                }
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            
            # Add to FAISS index
            if self.index.ntotal == 0:
                self.index.add(embeddings.astype('float32'))
            else:
                self.index.add(embeddings.astype('float32'))
            
            # Store documents and metadata
            self.documents.extend(chunks)
            self.document_metadata.extend(metadata)
            
            # Save to disk
            self._save_persistent_data()
    
    def remove_document(self, document_name: str) -> Dict[str, Any]:
        """
        Remove a specific document and all its chunks from the RAG datastore.
        
        Args:
            document_name: Name of the document to remove
            
        Returns:
            Dictionary with success status and details
        """
        try:
            if not self.documents or not self.document_metadata:
                return {
                    'success': False,
                    'message': 'No documents in storage to remove',
                    'chunks_removed': 0
                }
            
            # Find indices of chunks belonging to this document
            chunks_to_remove = []
            for i, metadata in enumerate(self.document_metadata):
                if metadata.get('source_document') == document_name:
                    chunks_to_remove.append(i)
            
            if not chunks_to_remove:
                return {
                    'success': False,
                    'message': f'Document "{document_name}" not found in storage',
                    'chunks_removed': 0
                }
            
            # Remove chunks in reverse order to maintain correct indices
            chunks_to_remove.reverse()
            
            # Remove from documents and metadata
            for index in chunks_to_remove:
                if index < len(self.documents):
                    self.documents.pop(index)
                if index < len(self.document_metadata):
                    self.document_metadata.pop(index)
            
            # Rebuild FAISS index with remaining documents
            if self.documents:
                # Check if embedding model is available
                if self.embedding_model is None:
                    return {
                        'success': False,
                        'message': 'Embedding model not available. Please restart the application.',
                        'chunks_removed': 0
                    }
                
                # Generate new embeddings for remaining documents
                embeddings = self.embedding_model.encode(self.documents, show_progress_bar=False)
                
                # Create new FAISS index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings.astype('float32'))
            else:
                # If no documents left, create empty index
                self.index = faiss.IndexFlatIP(self.dimension)
            
            # Save updated data to disk
            self._save_persistent_data()
            
            return {
                'success': True,
                'message': f'Successfully removed document "{document_name}" and {len(chunks_to_remove)} chunks',
                'chunks_removed': len(chunks_to_remove),
                'remaining_chunks': len(self.documents)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error removing document: {str(e)}',
                'chunks_removed': 0
            }
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all documents in the RAG datastore with their chunk counts.
        
        Returns:
            List of dictionaries containing document information
        """
        if not self.document_metadata:
            return []
        
        # Group chunks by document
        document_counts = {}
        for metadata in self.document_metadata:
            doc_name = metadata.get('source_document', 'Unknown')
            if doc_name not in document_counts:
                document_counts[doc_name] = {
                    'name': doc_name,
                    'type': metadata.get('document_type', 'Unknown'),
                    'chunk_count': 0
                }
            document_counts[doc_name]['chunk_count'] += 1
        
        return list(document_counts.values())
            
    def _split_text_into_chunks(self, text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks based on token count.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk (using whitespace split)
            overlap: Number of overlapping tokens between consecutive chunks
            
        Returns:
            List of chunked strings
        """
        if not text.strip():
            return []
        
        # Split text into tokens using whitespace
        tokens = text.split()
        
        if len(tokens) <= max_tokens:
            return [text.strip()]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + max_tokens
            
            # Try to find a good break point
            if end < len(tokens):
                # First, try to break at paragraph boundaries
                break_point = self._find_paragraph_break(tokens, start, end)
                
                # If no paragraph break, try sentence boundaries
                if break_point == end:
                    break_point = self._find_sentence_break(tokens, start, end)
                
                end = break_point
            
            # Create chunk from tokens
            chunk_tokens = tokens[start:end]
            chunk_text = ' '.join(chunk_tokens).strip()
            
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= len(tokens):
                break
        
        return chunks
    
    def chunk_document_with_metadata(self, text: str, max_tokens: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata indicating original text position.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between consecutive chunks
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text.strip():
            return []
        
        # Split text into tokens using whitespace
        tokens = text.split()
        
        if len(tokens) <= max_tokens:
            return [{
                'text': text.strip(),
                'start_token': 0,
                'end_token': len(tokens),
                'start_char': 0,
                'end_char': len(text)
            }]
        
        chunks_with_metadata = []
        start = 0
        
        while start < len(tokens):
            end = start + max_tokens
            
            # Try to find a good break point
            if end < len(tokens):
                # First, try to break at paragraph boundaries
                break_point = self._find_paragraph_break(tokens, start, end)
                
                # If no paragraph break, try sentence boundaries
                if break_point == end:
                    break_point = self._find_sentence_break(tokens, start, end)
                
                end = break_point
            
            # Create chunk from tokens
            chunk_tokens = tokens[start:end]
            chunk_text = ' '.join(chunk_tokens).strip()
            
            if chunk_text:
                # Calculate character positions
                start_char = len(' '.join(tokens[:start])) + (1 if start > 0 else 0)
                end_char = start_char + len(chunk_text)
                
                chunks_with_metadata.append({
                    'text': chunk_text,
                    'start_token': start,
                    'end_token': end,
                    'start_char': start_char,
                    'end_char': end_char
                })
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= len(tokens):
                break
        
        return chunks_with_metadata
    
    def _find_paragraph_break(self, tokens: List[str], start: int, end: int) -> int:
        """Find the best paragraph break point within the token range."""
        # Look for double newlines or paragraph markers
        for i in range(end - 1, start, -1):
            if i < len(tokens):
                token = tokens[i]
                # Check for paragraph breaks (double newlines, section markers, etc.)
                if '\n\n' in token or token.strip() in ['', '\n', '\r\n']:
                    return i + 1
                # Check for common paragraph/section markers
                if token.strip().startswith(('§', 'Section', 'Article', 'Chapter')):
                    return i
        return end
    
    def _find_sentence_break(self, tokens: List[str], start: int, end: int) -> int:
        """Find the best sentence break point within the token range."""
        # Look for sentence endings
        for i in range(end - 1, start, -1):
            if i < len(tokens):
                token = tokens[i]
                # Check for sentence endings
                if token.strip().endswith(('.', '!', '?')):
                    return i + 1
                # Check for common sentence endings with quotes
                if token.strip().endswith(('."', '!"', '?"')):
                    return i + 1
        return end
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query with enhanced search strategy."""
        if not self.documents or self.index.ntotal == 0:
            return []
        
        # Check if embedding model is available
        if self.embedding_model is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index with more results to filter by document type
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 3)
        
        # Separate chunks by document type
        other_chunks = []
        bylaws_chunks = []
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                chunk_data = {
                    'content': self.documents[idx],
                    'metadata': self.document_metadata[idx],
                    'similarity_score': float(score)
                }
                
                # Categorize by document type
                doc_type = self.document_metadata[idx].get('document_type', 'Unknown')
                if doc_type == 'HOA Bylaws':
                    bylaws_chunks.append(chunk_data)
                else:
                    other_chunks.append(chunk_data)
        
        # Enhanced search strategy: prioritize "Other" documents first, then "HOA Bylaws"
        relevant_chunks = []
        
        # Add "Other" document chunks first (up to top_k//2)
        other_limit = max(1, top_k // 2)
        relevant_chunks.extend(other_chunks[:other_limit])
        
        # Add "HOA Bylaws" chunks to fill remaining slots
        bylaws_limit = top_k - len(relevant_chunks)
        relevant_chunks.extend(bylaws_chunks[:bylaws_limit])
        
        # If we don't have enough chunks, add more from either category
        if len(relevant_chunks) < top_k:
            remaining_slots = top_k - len(relevant_chunks)
            if len(other_chunks) > other_limit:
                relevant_chunks.extend(other_chunks[other_limit:other_limit + remaining_slots])
            elif len(bylaws_chunks) > bylaws_limit:
                relevant_chunks.extend(bylaws_chunks[bylaws_limit:bylaws_limit + remaining_slots])
        
        return relevant_chunks
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Chat with the RAG-powered chatbot with reflection for improved responses."""
        if not self.api_key:
            return {
                'error': 'Perplexity API not enabled',
                'response': 'Please set your PERPLEXITY_API_KEY to enable the chatbot.',
                'sources': [],
                'success': False
            }
        
        # Retrieve relevant context
        relevant_chunks = self.retrieve_relevant_context(user_message, top_k=3)
        
        # Build context from relevant chunks
        context = self._build_context_from_chunks(relevant_chunks)
        
        # Step 1: Generate initial response
        initial_response = self._generate_initial_response(user_message, context)
        
        # Step 2: Reflect and improve the response (if enabled)
        if self.enable_reflection:
            improved_response = self._reflect_and_improve(user_message, context, initial_response, relevant_chunks)
        else:
            improved_response = initial_response
        
        # Extract sources from relevant chunks
        sources = [chunk['metadata']['source_document'] for chunk in relevant_chunks]
        
        return {
            'response': improved_response,
            'sources': list(set(sources)),  # Remove duplicates
            'relevant_chunks': relevant_chunks,
            'success': True
        }
    
    def _generate_initial_response(self, user_message: str, context: str) -> str:
        """Generate initial response to the user's question."""
        system_prompt = """You are HOA Bot, a specialized legal research assistant focused on identifying conflicts and discrepancies between HOA rules, governing documents, and Florida condominium law.

Your primary mission is to detect and surface conflicts between:
1. New rules or policies being implemented
2. Existing HOA bylaws and governing documents  
3. Florida condominium law (Chapter 718, Florida Statutes)

When analyzing documents and answering questions:

**CONFLICT DETECTION FOCUS:**
- Identify when new rules reference "founding documents" or "bylaws" that don't actually exist in the uploaded documents
- Flag discrepancies between stated rules and what's actually documented
- Highlight conflicts between HOA policies and Florida condominium law
- Surface missing documentation that rules claim to be based on
- Identify unenforceable rules that violate Florida law

**ANALYSIS REQUIREMENTS:**
1. Cross-reference all claims against uploaded HOA documents
2. Search Florida condominium laws (Chapter 718) for legal requirements
3. Identify specific conflicts with citations to both documents and statutes
4. Flag when rules reference non-existent bylaws or documents
5. Provide factual information about Florida condominium law requirements
6. Do not provide legal advice - only factual analysis and conflict identification
7. Always cite specific sections from documents and Florida statutes

**RESPONSE STRUCTURE:**
- Clearly identify any conflicts or discrepancies found
- List specific rules that reference non-existent documents
- Highlight violations of Florida condominium law
- Provide relevant Florida statute citations
- Note missing documentation that rules claim to be based on

Always be thorough, accurate, and professional in your conflict analysis. You are HOA Bot, the Conflict Detection Assistant."""

        # Create user prompt with context
        if context:
            user_prompt = f"""Context from uploaded HOA documents:
{context}

User question: {user_message}

**CONFLICT ANALYSIS REQUESTED:**
Please analyze this question with a focus on identifying conflicts and discrepancies. Specifically:

1. **Cross-reference claims**: Check if any rules or policies reference documents that don't exist in the uploaded materials
2. **Identify missing documentation**: Flag when rules claim to be based on "founding documents" or "bylaws" that aren't present
3. **Detect legal conflicts**: Compare against Florida condominium law (Chapter 718) for violations
4. **Surface discrepancies**: Highlight differences between stated rules and actual documentation

Search Florida condominium laws (Chapter 718, Florida Statutes) for additional context and identify any conflicts with the uploaded documents."""
        else:
            user_prompt = f"""User question: {user_message}

**CONFLICT ANALYSIS REQUESTED:**
Since no relevant documents are available, focus on:
1. Identifying what documentation would be needed to validate the claims in the question
2. Searching Florida condominium laws (Chapter 718, Florida Statutes) for relevant legal requirements
3. Highlighting potential conflicts that would need to be resolved with proper documentation

Please provide factual information about Florida condominium law requirements and note what documentation is missing."""
        
        # Make API call
        response = self._make_api_request(system_prompt, user_prompt)
        return response.get('content', '')
    
    def _reflect_and_improve(self, user_message: str, context: str, initial_response: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Reflect on the initial response and improve it using a multi-step reflection process."""
        # Step 1: Analyze the response
        analysis = self._analyze_response(user_message, context, initial_response, relevant_chunks)
        
        # Step 2: Generate improvements based on analysis
        improved_response = self._generate_improvements(user_message, context, initial_response, analysis, relevant_chunks)
        
        return improved_response
    
    def _analyze_response(self, user_message: str, context: str, response: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Analyze the response for potential improvements."""
        analysis_prompt = """You are an expert legal research assistant specializing in conflict detection. Analyze the following response to identify areas for improvement.

ANALYSIS CRITERIA:
1. **Conflict Detection**: Did the response properly identify conflicts between rules and documentation?
2. **Missing Documentation**: Were claims about "founding documents" or "bylaws" properly flagged when not found?
3. **Legal Compliance**: Were Florida condominium law conflicts properly identified?
4. **Accuracy**: Are all claims factually correct and supported by evidence?
5. **Completeness**: Does the response fully address the user's question with conflict analysis?
6. **Context Utilization**: Is all relevant document context properly used for cross-referencing?
7. **Legal Accuracy**: Are Florida condominium law references accurate and current?
8. **Citations**: Are sources properly cited and attributed?
9. **Clarity**: Is the response clear, well-structured, and easy to understand?
10. **Professionalism**: Is the tone appropriate and authoritative?
11. **Gaps**: What information is missing or could be added for better conflict analysis?

Provide a detailed analysis focusing on conflict detection and areas that need improvement."""

        analysis_context = f"""USER QUESTION: {user_message}

AVAILABLE DOCUMENT CONTEXT:
{context}

RESPONSE TO ANALYZE:
{response}

RELEVANT SOURCES:
{chr(10).join([f"- {chunk['metadata']['source_document']} (relevance: {chunk['similarity_score']:.2f})" for chunk in relevant_chunks])}

Please provide a detailed analysis of the response."""

        analysis_response = self._make_api_request(analysis_prompt, analysis_context)
        return analysis_response.get('content', '')
    
    def _generate_improvements(self, user_message: str, context: str, original_response: str, analysis: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate an improved response based on the analysis."""
        improvement_prompt = """You are an expert legal research assistant specializing in conflict detection. Based on the analysis provided, generate an improved response to the user's question.

IMPROVEMENT GUIDELINES:
- Address all issues identified in the analysis, especially conflict detection gaps
- Add missing information from the provided context
- Clarify any ambiguous statements
- Strengthen citations and references
- Improve logical flow and structure
- Ensure all claims are supported by evidence
- Make the response more comprehensive and helpful
- Maintain a professional, authoritative tone
- Do not provide legal advice - only factual information
- Search Florida condominium laws (Chapter 718) for additional context
- Prioritize conflict identification and discrepancy analysis
- Flag missing documentation that rules claim to be based on
- Highlight violations of Florida condominium law

Provide a significantly improved version that addresses the analysis findings with enhanced conflict detection."""

        improvement_context = f"""ORIGINAL USER QUESTION: {user_message}

AVAILABLE DOCUMENT CONTEXT:
{context}

ORIGINAL RESPONSE:
{original_response}

ANALYSIS FINDINGS:
{analysis}

RELEVANT SOURCES:
{chr(10).join([f"- {chunk['metadata']['source_document']} (relevance: {chunk['similarity_score']:.2f})" for chunk in relevant_chunks])}

Please provide an improved response that addresses the analysis findings."""

        improvement_response = self._make_api_request(improvement_prompt, improvement_context)
        return improvement_response.get('content', original_response)
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from relevant chunks."""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            content = chunk['content']
            score = chunk['similarity_score']
            
            context_parts.append(f"Document {i}: {metadata['source_document']} (Relevance: {score:.2f})")
            context_parts.append(f"Content: {content}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _make_api_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make API request to Perplexity."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.3,
            'top_p': 0.9
        }
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    return {'content': content}
                elif response.status_code == 429:
                    # Rate limited
                    time.sleep(self.rate_limit_delay * 2)
                    continue
                elif response.status_code >= 500:
                    # Server error
                    time.sleep(self.rate_limit_delay)
                    continue
                else:
                    return {'content': f"API Error: {response.status_code} - {response.text}"}
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                    continue
                return {'content': "Request timeout. Please try again."}
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                    continue
                return {'content': f"Request error: {str(e)}"}
        
        return {'content': "Max retries exceeded. Please try again."}
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of uploaded documents."""
        if not self.documents:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'document_types': [],
                'document_names': []
            }
        
        # Group by document
        doc_groups = {}
        for metadata in self.document_metadata:
            doc_name = metadata['source_document']
            if doc_name not in doc_groups:
                doc_groups[doc_name] = {
                    'name': doc_name,
                    'type': metadata['document_type'],
                    'chunks': metadata['total_chunks']
                }
        
        return {
            'total_documents': len(doc_groups),
            'total_chunks': len(self.documents),
            'document_types': list(set([doc['type'] for doc in doc_groups.values()])),
            'document_names': [doc['name'] for doc in doc_groups.values()],
            'documents': list(doc_groups.values())
        }


def create_rag_chatbot(property_id: str = None, enable_reflection: bool = True) -> Optional[RAGChatbot]:
    """Create and return a RAGChatbot instance if API is enabled."""
    if os.getenv('PERPLEXITY_API_KEY'):
        return RAGChatbot(property_id=property_id, enable_reflection=enable_reflection)
    return None
