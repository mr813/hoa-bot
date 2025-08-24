"""
HOA Bot - Florida Condominium Compliance Assistant
Streamlit Community Cloud Deployment Version
"""

import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import geocoder
import time

# Load environment variables
load_dotenv()

# Add project root to Python path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import app modules
try:
    from app.parsing import parse_pdf, validate_pdf_file
    from app.rag_chatbot import create_rag_chatbot
    from app.user_management import create_user_manager, ensure_property_storage
    from app.utils import clean_text, truncate_text
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def main():
    """Main Streamlit application with authentication."""
    
    # Page configuration
    st.set_page_config(
        page_title="HOA Bot - Secure Document Assistant",
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
    .property-card {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .property-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .property-card.selected {
        border-color: #1f77b4;
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
    }
    .property-info {
        margin-bottom: 1rem;
    }
    .property-address {
        font-weight: bold;
        color: #1f77b4;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .property-type {
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    if 'show_map' not in st.session_state:
        st.session_state.show_map = {}
    
    # Main header
    st.markdown('<h1 class="main-header">🏠 HOA Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Florida Condominium Compliance Assistant</p>', unsafe_allow_html=True)
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        show_authentication_page()
    else:
        show_main_application()

def show_authentication_page():
    """Display authentication page with login and registration."""
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.header("Login to HOA Bot")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            if username and password:
                user_manager = create_user_manager()
                if user_manager.authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
    
    with tab2:
        st.header("Create New Account")
        
        new_username = st.text_input("Username", key="reg_username")
        email = st.text_input("Email", key="reg_email")
        
        if st.button("Register", type="primary"):
            if new_username and email:
                user_manager = create_user_manager()
                result = user_manager.register_user(new_username, email)
                if result['success']:
                    st.success(f"Account created! Check your email ({email}) for login credentials.")
                else:
                    st.error(result['message'])
            else:
                st.warning("Please enter both username and email")

def show_main_application():
    """Display main application after authentication."""
    
    # Sidebar for navigation
    st.sidebar.title(f"Welcome, {st.session_state.current_user}!")
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.selected_property = None
        st.session_state.chat_history = []
        st.session_state.documents_processed = []
        st.rerun()
    
    st.sidebar.divider()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Properties", "📄 Documents", "💬 Chat", "⚙️ Settings"])
    
    with tab1:
        show_properties_page()
    
    with tab2:
        show_documents_page()
    
    with tab3:
        show_chat_page()
    
    with tab4:
        show_settings_page()

def show_properties_page():
    """Display properties management page."""
    
    st.header("🏠 Property Management")
    
    # Add new property
    with st.expander("➕ Add New Property", expanded=False):
        with st.form("add_property_form"):
            property_nickname = st.text_input("Property Nickname (e.g., Beach House)")
            property_address = st.text_area("Full Address")
            property_type = st.selectbox("Property Type", ["Condo", "House"])
            
            if st.form_submit_button("Add Property", type="primary"):
                if property_nickname and property_address:
                    user_manager = create_user_manager()
                    result = user_manager.add_property(
                        st.session_state.current_user,
                        property_nickname,
                        property_address,
                        property_type
                    )
                    if result['success']:
                        st.success("Property added successfully!")
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.warning("Please fill in all fields")
    
    # Display existing properties
    st.subheader("Your Properties")
    
    user_manager = create_user_manager()
    properties = user_manager.get_user_properties(st.session_state.current_user)
    
    if not properties:
        st.info("No properties added yet. Add your first property above!")
        return
    
    for property_id, property_data in properties.items():
        with st.container():
            st.markdown(f"""
            <div class="property-card {'selected' if st.session_state.selected_property == property_id else ''}">
                <div class="property-info">
                    <div class="property-address">🏠 {property_data['nickname']}</div>
                    <div>📍 {property_data['address']}</div>
                    <div class="property-type">Type: {property_data['type']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                if st.button(f"Select", key=f"select_{property_id}"):
                    st.session_state.selected_property = property_id
                    st.rerun()
            
            with col2:
                if st.button(f"🗺️ View Map", key=f"map_{property_id}"):
                    st.session_state.show_map[property_id] = not st.session_state.show_map.get(property_id, False)
                    st.rerun()
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{property_id}"):
                    user_manager.delete_property(st.session_state.current_user, property_id)
                    if st.session_state.selected_property == property_id:
                        st.session_state.selected_property = None
                    st.rerun()
            
            # Show map if requested
            if st.session_state.show_map.get(property_id, False):
                try:
                    # Geocoding
                    g = geocoder.osm(property_data['address'])
                    if g.ok:
                        lat, lng = g.lat, g.lng
                    else:
                        # Fallback to ArcGIS
                        g = geocoder.arcgis(property_data['address'])
                        if g.ok:
                            lat, lng = g.lat, g.lng
                        else:
                            # Manual entry fallback
                            st.warning("Could not geocode address automatically. Please enter coordinates:")
                            lat = st.number_input("Latitude", value=27.6648, key=f"lat_{property_id}")
                            lng = st.number_input("Longitude", value=-82.5158, key=f"lng_{property_id}")
                    
                    # Create map
                    m = folium.Map(location=[lat, lng], zoom_start=15)
                    folium.Marker([lat, lng], popup=property_data['nickname']).add_to(m)
                    
                    st_folium(m, width=700, height=400)
                    
                    if st.button("Close Map", key=f"close_map_{property_id}"):
                        st.session_state.show_map[property_id] = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error displaying map: {e}")
            
            st.divider()

def show_documents_page():
    """Display document upload and management page."""
    
    st.header("📄 Document Management")
    
    if not st.session_state.selected_property:
        st.warning("Please select a property first from the Properties tab.")
        return
    
    # Get property info
    user_manager = create_user_manager()
    property_data = user_manager.get_property(st.session_state.current_user, st.session_state.selected_property)
    
    st.subheader(f"Documents for: {property_data['nickname']}")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload HOA Documents (PDF)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your HOA bylaws, rules, and other governing documents"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")
            
            # Document type selection
            doc_type = st.selectbox(
                "Document type",
                ["HOA Bylaws", "Other"],
                key=f"type_{uploaded_file.name}"
            )
            
            if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                try:
                    st.info(f"🔄 Starting processing for: {uploaded_file.name}")
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    st.info(f"📁 Temporary file created: {tmp_file_path}")
                    st.info(f"📊 File size: {len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
                    
                    # Validate PDF first
                    is_valid, validation_message = validate_pdf_file(tmp_file_path)
                    if not is_valid:
                        st.error(f"PDF validation failed: {validation_message}")
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
                        continue
                    
                    st.success(f"✅ PDF validated: {validation_message}")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current_page, total_pages):
                        progress = current_page / total_pages
                        progress_bar.progress(progress)
                        status_text.text(f"Processing page {current_page} of {total_pages}")
                    
                    # Parse PDF
                    try:
                        st.info(f"📖 Starting PDF parsing...")
                        document = parse_pdf(tmp_file_path, uploaded_file.name, progress_callback)
                        text_content = document.get_all_text()
                        
                        st.info(f"✅ PDF parsing completed")
                        st.info(f"📊 Extracted text length: {len(text_content)} characters")
                        st.info(f"📄 Pages processed: {len(document.pages)}")
                        st.info(f"🔍 OCR used: {document.ocr_used}")
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        st.info(f"🗑️ Temporary file cleaned up")
                        
                        if text_content:
                        # Create RAG chatbot for this property
                        st.info(f"🤖 Creating RAG chatbot for property: {st.session_state.selected_property}")
                        rag_chatbot = create_rag_chatbot(st.session_state.selected_property)
                        
                        # Add document to RAG system
                        st.info(f"📚 Adding document to RAG system...")
                        rag_chatbot.add_documents([{
                            'text': text_content,
                            'source_document': uploaded_file.name,
                            'document_type': doc_type
                        }])
                        st.info(f"✅ Document added to RAG system")
                        
                        # Update session state
                        doc_data = {
                            'name': uploaded_file.name,
                            'type': doc_type,
                            'pages': len(text_content.split('\n\n')),  # Rough page count
                            'property_id': st.session_state.selected_property
                        }
                        
                        if doc_data not in st.session_state.documents_processed:
                            st.session_state.documents_processed.append(doc_data)
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to extract text from {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        st.error(f"📋 Error details: {type(e).__name__}")
                        import traceback
                        st.error(f"📋 Full traceback: {traceback.format_exc()}")
                        # Clean up temporary file if it still exists
                        try:
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                                st.info(f"🗑️ Cleaned up temporary file after error")
                        except Exception as cleanup_error:
                            st.warning(f"⚠️ Failed to clean up temporary file: {cleanup_error}")
    
    # Document summary
    if st.session_state.documents_processed:
        st.subheader("📋 Document Summary")
        
        # Count documents by type
        bylaws_count = sum(1 for doc in st.session_state.documents_processed if doc['type'] == 'HOA Bylaws')
        other_count = sum(1 for doc in st.session_state.documents_processed if doc['type'] == 'Other')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(st.session_state.documents_processed))
        with col2:
            st.metric("HOA Bylaws", bylaws_count)
        with col3:
            st.metric("Other Documents", other_count)
        
        # List documents
        for doc in st.session_state.documents_processed:
            icon = "📋" if doc['type'] == 'HOA Bylaws' else "📄"
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"{icon} {doc['name']} ({doc['pages']} pages, {doc['type']})")
            
            with col2:
                if st.button("🗑️ Remove", key=f"remove_{doc['name']}"):
                    # Remove from RAG system
                    rag_chatbot = create_rag_chatbot(st.session_state.selected_property)
                    rag_chatbot.remove_document(doc['name'])
                    
                    # Remove from session state
                    st.session_state.documents_processed.remove(doc)
                    st.rerun()

def show_chat_page():
    """Display RAG chatbot interface."""
    
    st.header("💬 HOA Compliance Chat")
    
    if not st.session_state.selected_property:
        st.warning("Please select a property first from the Properties tab.")
        return
    
    # Get property info
    user_manager = create_user_manager()
    property_data = user_manager.get_property(st.session_state.current_user, st.session_state.selected_property)
    
    st.subheader(f"Chat for: {property_data['nickname']}")
    
    # Check if documents are available
    rag_chatbot = create_rag_chatbot(st.session_state.selected_property)
    storage_info = rag_chatbot.get_storage_info()
    
    if storage_info.get('chunks_in_memory', 0) == 0:
        st.info("No documents uploaded yet. Please upload documents in the Documents tab first.")
        return
    
    # Reflection toggle
    enable_reflection = st.checkbox("🧠 Enable Reflection", value=False, 
                                  help="Enable multi-step reflection for improved responses")
    
    if enable_reflection:
        st.info("Reflection mode enabled - responses will be analyzed and improved for better accuracy.")
    
    # Chat interface
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>HOA Bot:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # User input
    user_input = st.chat_input("Ask about HOA compliance, conflicts, or Florida law...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from RAG chatbot
        with st.spinner("🤖 Analyzing your question..."):
            try:
                response = rag_chatbot.chat(user_input, enable_reflection=enable_reflection)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")

def show_settings_page():
    """Display settings and account management."""
    
    st.header("⚙️ Settings")
    
    # User information
    st.subheader("Account Information")
    st.write(f"**Username:** {st.session_state.current_user}")
    
    # Change password
    with st.expander("🔒 Change Password"):
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Change Password"):
                if new_password == confirm_password:
                    user_manager = create_user_manager()
                    result = user_manager.change_password(
                        st.session_state.current_user,
                        current_password,
                        new_password
                    )
                    if result['success']:
                        st.success("Password changed successfully!")
                    else:
                        st.error(result['message'])
                else:
                    st.error("New passwords do not match")
    
    # Storage management
    st.subheader("📊 Storage Management")
    
    if st.session_state.selected_property:
        rag_chatbot = create_rag_chatbot(st.session_state.selected_property)
        storage_info = rag_chatbot.get_storage_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", storage_info.get('chunks_in_memory', 0))
        with col2:
            st.metric("Vectors in Index", storage_info.get('vectors_in_index', 0))
        with col3:
            st.metric("Documents", len(storage_info.get('document_metadata', [])))
        
        # Clear all data
        if st.button("🗑️ Clear All Data", type="secondary"):
            rag_chatbot.clear_all_data()
            st.session_state.documents_processed = []
            st.session_state.chat_history = []
            st.success("All data cleared successfully!")
            st.rerun()
        
        # Document management
        if storage_info.get('document_metadata'):
            st.subheader("📄 Document Management")
            for metadata in storage_info['document_metadata']:
                doc_name = metadata.get('source_document', 'Unknown')
                chunk_count = metadata.get('chunk_count', 0)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc_name} ({chunk_count} chunks)")
                with col2:
                    if st.button("🗑️ Remove", key=f"settings_remove_{doc_name}"):
                        rag_chatbot.remove_document(doc_name)
                        st.success(f"Removed {doc_name}")
                        st.rerun()

if __name__ == "__main__":
    main()
