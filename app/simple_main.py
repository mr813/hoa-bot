"""
Simplified HOA Document Chatbot
Focuses on document upload with OCR and RAG-powered chatbot.
"""

import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import app modules
try:
    from app.parsing import parse_pdf, validate_pdf_file
    from app.rag_chatbot import create_rag_chatbot
    from app.utils import clean_text, truncate_text
except (ImportError, OSError):
    # Fallback to relative imports if absolute imports fail
    from .parsing import parse_pdf, validate_pdf_file
    from .rag_chatbot import create_rag_chatbot
    from .utils import clean_text, truncate_text


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="HOA Bot - Document Assistant",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #333;
        line-height: 1.5;
    }
    .user-message {
        background-color: #f0f8ff;
        border-left: 4px solid #0066cc;
        color: #333;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        color: #333;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
        background-color: #f1f1f1;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏠 HOA Bot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">Document Assistant</h3>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your HOA documents (PDF format)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload your Declaration, Bylaws, Rules, or other HOA governing documents"
        )
        
        # API key status
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if api_key:
            st.success("✅ Perplexity API configured")
        else:
            st.error("❌ Perplexity API key not found")
            st.info("Set PERPLEXITY_API_KEY in your .env file to enable the chatbot")
        
        # Document processing status
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = []
        
        if 'rag_chatbot' not in st.session_state:
            st.session_state.rag_chatbot = create_rag_chatbot()
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if file was already processed
                if uploaded_file.name not in [doc['name'] for doc in st.session_state.documents_processed]:
                    try:
                        # Create temporary file for validation
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Validate file
                        is_valid, validation_message = validate_pdf_file(tmp_file_path)
                        if not is_valid:
                            st.error(f"Invalid PDF file: {uploaded_file.name} - {validation_message}")
                            # Clean up temp file
                            os.unlink(tmp_file_path)
                            continue
                        
                        # Create progress bar and status for this file
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(progress, message):
                            progress_bar.progress(progress / 100)
                            status_text.text(f"📄 {uploaded_file.name}: {message}")
                        
                        # Parse the PDF with progress tracking (using existing temp file)
                        document = parse_pdf(tmp_file_path, uploaded_file.name, update_progress)
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        if document and document.get_all_text():
                            # Add to processed documents
                            doc_data = {
                                'name': uploaded_file.name,
                                'type': 'Unknown',  # Could add classification later
                                'text': document.get_all_text(),
                                'pages': len(document.pages)
                            }
                            
                            st.session_state.documents_processed.append(doc_data)
                            
                            # Add to RAG chatbot
                            if st.session_state.rag_chatbot:
                                st.session_state.rag_chatbot.add_documents([doc_data])
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"✅ Processed: {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to extract text from: {uploaded_file.name}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Document summary
        if st.session_state.documents_processed:
            st.header("📊 Document Summary")
            if st.session_state.rag_chatbot:
                summary = st.session_state.rag_chatbot.get_document_summary()
                st.write(f"**Total Documents:** {summary['total_documents']}")
                st.write(f"**Total Chunks:** {summary['total_chunks']}")
                
                if summary['document_names']:
                    st.write("**Documents:**")
                    for doc in summary['documents']:
                        st.write(f"• {doc['name']} ({doc['chunks']} chunks)")
        
        # Storage management section
        st.header("💾 Storage Management")
        
        if st.session_state.rag_chatbot:
            # Show storage info
            storage_info = st.session_state.rag_chatbot.get_storage_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Storage", f"{storage_info['total_size_mb']} MB")
                st.metric("Chunks in Memory", storage_info['chunks_in_memory'])
            
            with col2:
                st.metric("Vectors in Index", storage_info['vectors_in_index'])
                st.metric("Files on Disk", len(storage_info['files']))
            
            # Show file details
            if storage_info['files']:
                st.write("**Storage Files:**")
                for filename, info in storage_info['files'].items():
                    st.write(f"• {filename}: {info['size_mb']} MB")
            
            # Clear data button
            st.write("---")
            if st.button("🗑️ Clear All Data", type="secondary", help="Remove all documents and chunks from memory and disk"):
                if st.session_state.rag_chatbot:
                    result = st.session_state.rag_chatbot.clear_all_data()
                    if result['success']:
                        st.success(result['message'])
                        # Clear session state
                        st.session_state.documents_processed = []
                        st.rerun()
                    else:
                        st.error(result['message'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with HOA Bot")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong><br>
                    {message['content']}
                    <div class="source-info">
                        Sources: {', '.join(message.get('sources', [])) if message.get('sources') else 'No specific sources'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_area(
            "Ask a question about your HOA documents or Florida condominium law:",
            height=100,
            placeholder="e.g., What are the rental restrictions in my documents? What does Florida law say about HOA board elections?"
        )
        
        if st.button("Send", type="primary"):
            if user_input.strip():
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input.strip()
                })
                
                # Get chatbot response
                if st.session_state.rag_chatbot:
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_chatbot.chat(user_input.strip())
                    
                    if response['success']:
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response['response'],
                            'sources': response['sources']
                        })
                    else:
                        st.error(f"Error: {response.get('error', 'Unknown error')}")
                else:
                    st.error("Chatbot not available. Please check your API configuration.")
                
                # Rerun to update the display
                st.rerun()
    
    with col2:
        st.header("ℹ️ Help")
        
        st.markdown("""
        **How to use HOA Bot:**
        1. Upload your HOA documents (PDF format)
        2. Wait for processing to complete
        3. Ask questions about your documents or Florida law
        
        **Example questions:**
        - What are the rental restrictions?
        - How are board members elected?
        - What are the assessment collection procedures?
        - What does Florida law say about HOA meetings?
        
        **HOA Bot Features:**
        - ✅ Document OCR and text extraction
        - ✅ RAG-powered responses using your documents
        - ✅ Florida condominium law knowledge
        - ✅ Source citations from your documents
        
        **Note:** HOA Bot provides factual information only and does not constitute legal advice.
        """)
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
