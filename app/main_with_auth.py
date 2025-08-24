"""
HOA Bot with Authentication and Property Management
Complete application with user management, property handling, and RAG chatbot.
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

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import app modules - use direct imports for Streamlit Cloud compatibility
from app.parsing import parse_pdf, validate_pdf_file
from app.rag_chatbot import create_rag_chatbot
from app.config import get_rag_chatbot_config
from app.user_management import create_user_manager, ensure_property_storage
from app.utils import clean_text, truncate_text


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
        border-color: #2196F3;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        box-shadow: 0 6px 12px rgba(33, 150, 243, 0.3);
    }
    .property-info {
        color: #2c3e50;
        font-weight: 500;
    }
    .property-address {
        color: #34495e;
        font-size: 0.95rem;
        margin: 0.5rem 0;
    }
    .property-type {
        color: #7f8c8d;
        font-size: 0.9rem;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize user manager
    user_manager = create_user_manager()
    
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
    
    # Header
    st.markdown('<h1 class="main-header">🏠 HOA Bot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">Secure Document Assistant</h3>', unsafe_allow_html=True)
    
    # Authentication Section
    if not st.session_state.authenticated:
        show_authentication_page(user_manager)
    else:
        show_main_application(user_manager)


def show_authentication_page(user_manager):
    """Display authentication and registration page."""
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.header("Login to HOA Bot")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    success, message = user_manager.authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        st.header("Create New Account")
        
        with st.form("register_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if name and email:
                    success, message, username = user_manager.register_user(name, email)
                    if success:
                        st.success(message)
                        if username:
                            st.info(f"Your username is: {username}")
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both name and email")


def show_main_application(user_manager):
    """Display the main application after authentication."""
    
    # User info and logout
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_info = user_manager.get_user_info(st.session_state.current_user)
        if user_info:
            st.write(f"**Welcome, {user_info['name']}!**")
    with col2:
        st.write(f"**Username:** {st.session_state.current_user}")
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.selected_property = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = []
            st.rerun()
    
    st.write("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Properties", "📁 Documents", "💬 Chat", "⚙️ Settings"])
    
    with tab1:
        show_properties_page(user_manager)
    
    with tab2:
        show_documents_page(user_manager)
    
    with tab3:
        show_chat_page(user_manager)
    
    with tab4:
        show_settings_page(user_manager)


def show_properties_page(user_manager):
    """Display property management page."""
    
    st.header("🏠 Property Management")
    
    # Add new property
    with st.expander("➕ Add New Property", expanded=False):
        with st.form("add_property_form"):
            address = st.text_input("Property Address")
            nickname = st.text_input("Property Nickname (e.g., 'Beach House', 'Downtown Condo')")
            property_type = st.selectbox("Property Type", ["condo", "house"])
            
            submit_button = st.form_submit_button("Add Property")
            
            if submit_button:
                if address and nickname:
                    success, message = user_manager.add_property(
                        st.session_state.current_user, 
                        address, 
                        nickname, 
                        property_type
                    )
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both address and nickname")
    
    # Display user properties
    properties = user_manager.get_user_properties(st.session_state.current_user)
    
    if not properties:
        st.info("No properties added yet. Add your first property above!")
    else:
        st.subheader("Your Properties")
        
        for property_data in properties:
            with st.container():
                # Property card with improved styling
                st.markdown(f"""
                <div class="property-card {'selected' if st.session_state.selected_property == property_data['id'] else ''}">
                    <div class="property-info">
                        <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">🏠 {property_data['nickname']}</h3>
                        <div class="property-address">
                            <strong>📍 Address:</strong> {property_data['address']}
                        </div>
                        <div class="property-type">
                            <strong>🏘️ Type:</strong> {property_data['property_type'].title()}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if st.button(f"🗺️ View Map", key=f"map_{property_data['id']}", type="secondary"):
                        st.session_state.selected_property = property_data['id']
                        st.session_state.show_map = True
                        st.rerun()
                
                with col2:
                    if st.button(f"✅ Select", key=f"select_{property_data['id']}", type="primary"):
                        st.session_state.selected_property = property_data['id']
                        st.session_state.chat_history = []
                        st.session_state.documents_processed = []
                        st.rerun()
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{property_data['id']}", type="secondary"):
                        success, message = user_manager.delete_property(
                            st.session_state.current_user, 
                            property_data['id']
                        )
                        if success:
                            if st.session_state.selected_property == property_data['id']:
                                st.session_state.selected_property = None
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                # Show map if this property is selected and map is requested
                if (st.session_state.selected_property == property_data['id'] and 
                    st.session_state.get('show_map', False)):
                    
                    st.markdown("---")
                    st.subheader(f"🗺️ Map: {property_data['nickname']}")
                    
                    # Geocoding and map display
                    lat, lng = None, None
                    geocoding_success = False
                    
                    # Method 1: Try OpenStreetMap (Nominatim)
                    try:
                        with st.spinner("📍 Geocoding address with OpenStreetMap..."):
                            time.sleep(1)  # Rate limiting delay
                            g = geocoder.osm(property_data['address'])
                            if g.ok:
                                lat, lng = g.lat, g.lng
                                geocoding_success = True
                                st.success("✅ Address geocoded successfully using OpenStreetMap")
                    except Exception as e:
                        st.warning(f"OpenStreetMap geocoding failed: {str(e)}")
                    
                    # Method 2: Try ArcGIS if OpenStreetMap failed
                    if not geocoding_success:
                        try:
                            with st.spinner("📍 Geocoding address with ArcGIS..."):
                                time.sleep(1)  # Rate limiting delay
                                g = geocoder.arcgis(property_data['address'])
                                if g.ok:
                                    lat, lng = g.lat, g.lng
                                    geocoding_success = True
                                    st.success("✅ Address geocoded successfully using ArcGIS")
                        except Exception as e:
                            st.warning(f"ArcGIS geocoding failed: {str(e)}")
                    
                    # Display map if geocoding was successful
                    if geocoding_success and lat is not None and lng is not None:
                        try:
                            # Create map
                            m = folium.Map(location=[lat, lng], zoom_start=16)
                            
                            # Add marker
                            folium.Marker(
                                [lat, lng],
                                popup=f"<b>{property_data['nickname']}</b><br>{property_data['address']}<br>Type: {property_data['property_type'].title()}",
                                icon=folium.Icon(color='red', icon='home')
                            ).add_to(m)
                            
                            # Display map
                            st_folium(m, width=700, height=400)
                            
                            # Show coordinates
                            st.caption(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
                            
                        except Exception as e:
                            st.error(f"Error displaying map: {str(e)}")
                    else:
                        st.error("❌ Could not geocode the address with any available service.")
                        st.info("**Address Format Tips:**")
                        st.info("• Use standard format: '123 Main St, City, State ZIP'")
                        st.info("• Include city and state for better accuracy")
                        st.info("• Example: '19810 Gulf Blvd #8, Indian Shores, FL 33785'")
                        
                        # Show manual coordinate input option
                        with st.expander("🔧 Manual Coordinate Entry"):
                            st.write("If geocoding fails, you can manually enter coordinates:")
                            col1, col2 = st.columns(2)
                            with col1:
                                manual_lat = st.number_input("Latitude", value=27.8506, format="%.6f", key=f"lat_{property_data['id']}")
                            with col2:
                                manual_lng = st.number_input("Longitude", value=-82.8512, format="%.6f", key=f"lng_{property_data['id']}")
                            
                            if st.button("Show Map with Manual Coordinates", key=f"manual_{property_data['id']}"):
                                try:
                                    m = folium.Map(location=[manual_lat, manual_lng], zoom_start=16)
                                    folium.Marker(
                                        [manual_lat, manual_lng],
                                        popup=f"<b>{property_data['nickname']}</b><br>{property_data['address']}<br>Type: {property_data['property_type'].title()}<br><i>(Manual coordinates)</i>",
                                        icon=folium.Icon(color='orange', icon='home')
                                    ).add_to(m)
                                    st_folium(m, width=700, height=400)
                                    st.caption(f"📍 Manual Coordinates: {manual_lat:.6f}, {manual_lng:.6f}")
                                except Exception as e:
                                    st.error(f"Error displaying map: {str(e)}")
                    
                    # Close map button
                    if st.button("❌ Close Map", key=f"close_map_{property_data['id']}"):
                        st.session_state.show_map = False
                        st.rerun()
                    
                    st.markdown("---")
        
        # Show selected property info
        if st.session_state.selected_property:
            selected_prop = user_manager.get_property(st.session_state.selected_property)
            if selected_prop:
                st.success(f"✅ Selected: {selected_prop['nickname']} ({selected_prop['address']})")


def show_documents_page(user_manager):
    """Display document management page."""
    
    st.header("📁 Document Management")
    
    if not st.session_state.selected_property:
        st.warning("Please select a property first to manage documents.")
        return
    
    selected_prop = user_manager.get_property(st.session_state.selected_property)
    if not selected_prop:
        st.error("Selected property not found.")
        return
    
    st.subheader(f"Documents for: {selected_prop['nickname']}")
    
    # Check for existing documents in RAG chatbot storage
    rag_config = get_rag_chatbot_config()
    rag_chatbot = create_rag_chatbot(
        property_id=st.session_state.selected_property,
        vector_store_backend=rag_config['vector_store_backend'],
        vector_store_config=rag_config['vector_store_config']
    )
    storage_info = rag_chatbot.get_storage_info() if rag_chatbot else {}
    
    # Debug: Show storage info
    st.write(f"🔍 Debug - Storage Info: {storage_info}")
    
    # Load existing documents from storage if not in session state
    if not st.session_state.documents_processed:
        st.session_state.documents_processed = []
    
    # Check if we have documents in storage but not in session state
    if storage_info.get('chunks_in_memory', 0) > 0 and not any(
        doc.get('property_id') == st.session_state.selected_property 
        for doc in st.session_state.documents_processed
    ):
        # Extract document names from metadata
        if rag_chatbot and rag_chatbot.document_metadata:
            # Group by source document
            doc_sources = {}
            for metadata in rag_chatbot.document_metadata:
                source = metadata.get('source_document', 'Unknown')
                doc_type = metadata.get('document_type', 'Unknown')
                if source not in doc_sources:
                    doc_sources[source] = {
                        'name': source,
                        'type': doc_type,
                        'pages': 0,
                        'property_id': st.session_state.selected_property
                    }
                doc_sources[source]['pages'] += 1
            
            # Add to session state
            for doc_data in doc_sources.values():
                if doc_data not in st.session_state.documents_processed:
                    st.session_state.documents_processed.append(doc_data)
            
            # Show success message
            st.success(f"✅ Loaded {len(doc_sources)} previously uploaded documents from storage")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload HOA documents for this property (PDF format)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload Declaration, Bylaws, Rules, or other HOA governing documents"
    )
    
    # Process uploaded files
    if uploaded_files:
        st.subheader("📋 Document Classification")
        st.write("Please classify each document before processing:")
        
        for uploaded_file in uploaded_files:
            # Check if file was already processed
            if uploaded_file.name not in [doc['name'] for doc in st.session_state.documents_processed]:
                # Document type selection
                doc_type = st.selectbox(
                    f"Document type for: {uploaded_file.name}",
                    options=["HOA Bylaws", "Other"],
                    key=f"type_{uploaded_file.name}",
                    help="Select 'HOA Bylaws' for governing documents, 'Other' for additional materials"
                )
                
                # Process button
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
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
                        
                        # Progress tracking variables
                        progress_key = f"progress_{uploaded_file.name}"
                        
                        # Create progress bar and status for this file
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Progress display section that updates from session state
                        progress_container = st.container()
                        
                        # Initialize progress in session state
                        if progress_key not in st.session_state:
                            st.session_state[progress_key] = {
                                'progress': 0,
                                'message': 'Starting...',
                                'timestamp': time.time()
                            }
                        
                        # Update progress bar from session state
                        with progress_container:
                            if progress_key in st.session_state:
                                progress_data = st.session_state[progress_key]
                                progress_bar.progress(progress_data['progress'] / 100)
                                status_text.text(f"📄 {uploaded_file.name}: {progress_data['message']}")
                        
                        def update_progress(progress, message):
                            # Store progress in session state for UI updates
                            st.session_state[progress_key] = {
                                'progress': progress,
                                'message': message,
                                'timestamp': time.time()
                            }
                            
                            # Log progress for debugging
                            print(f"📄 {uploaded_file.name}: {message} ({progress:.1f}%)")
                        
                        # Parse the PDF with progress tracking (using existing temp file)
                        document = parse_pdf(tmp_file_path, uploaded_file.name, update_progress)
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        if document and document.get_all_text():
                            # Add to processed documents
                            doc_data = {
                                'name': uploaded_file.name,
                                'type': doc_type,
                                'text': document.get_all_text(),
                                'pages': len(document.pages),
                                'property_id': st.session_state.selected_property
                            }
                            
                            st.session_state.documents_processed.append(doc_data)
                            
                            # Add to RAG chatbot for this property
                            if rag_chatbot:
                                rag_chatbot.add_documents([doc_data])
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"✅ Processed: {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to extract text from: {uploaded_file.name}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Document summary with removal functionality
    property_docs = [doc for doc in st.session_state.documents_processed 
                    if doc.get('property_id') == st.session_state.selected_property]
    
    if property_docs:
        st.subheader("📊 Document Summary")
        st.write(f"**Total Documents:** {len(property_docs)}")
        st.write(f"**Total Pages:** {sum(doc['pages'] for doc in property_docs)}")
        st.write(f"**Total Chunks:** {storage_info.get('chunks_in_memory', 0)}")
        
        # Document type statistics
        bylaws_count = len([doc for doc in property_docs if doc.get('type') == 'HOA Bylaws'])
        other_count = len([doc for doc in property_docs if doc.get('type') == 'Other'])
        st.write(f"**HOA Bylaws:** {bylaws_count} documents")
        st.write(f"**Other Documents:** {other_count} documents")
        
        st.write("**Documents:**")
        
        # Get document list from RAG chatbot for accurate chunk counts
        rag_documents = rag_chatbot.get_document_list() if rag_chatbot else []
        rag_doc_dict = {doc['name']: doc for doc in rag_documents}
        
        for doc in property_docs:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Get chunk count from RAG storage
                chunk_count = rag_doc_dict.get(doc['name'], {}).get('chunk_count', 0)
                doc_type_icon = "📋" if doc.get('type') == "HOA Bylaws" else "📄"
                st.write(f"{doc_type_icon} {doc['name']} ({doc['pages']} pages, {chunk_count} chunks) - **{doc.get('type', 'Unknown')}**")
            
            with col2:
                if st.button(f"🗑️ Remove", key=f"remove_{doc['name']}", type="secondary"):
                    if rag_chatbot:
                        result = rag_chatbot.remove_document(doc['name'])
                        if result['success']:
                            # Remove from session state
                            st.session_state.documents_processed = [
                                d for d in st.session_state.documents_processed 
                                if not (d.get('property_id') == st.session_state.selected_property and d.get('name') == doc['name'])
                            ]
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error(result['message'])
                    else:
                        st.error("RAG chatbot not available")
    else:
        st.info("No documents uploaded for this property yet. Upload some documents to get started!")


def show_chat_page(user_manager):
    """Display chat page."""
    
    st.header("💬 Chat with HOA Bot")
    
    if not st.session_state.selected_property:
        st.warning("Please select a property first to chat about its documents.")
        return
    
    selected_prop = user_manager.get_property(st.session_state.selected_property)
    if not selected_prop:
        st.error("Selected property not found.")
        return
    
    st.subheader(f"Chatting about: {selected_prop['nickname']}")
    
    # Reflection toggle
    col1, col2 = st.columns([1, 3])
    with col1:
        enable_reflection = st.checkbox(
            "🧠 Enable Reflection", 
            value=True, 
            help="Enable multi-step reflection to improve response quality (may take longer)"
        )
    with col2:
        if enable_reflection:
            st.info("Reflection enabled: Responses will be analyzed and improved for better accuracy and completeness.")
        else:
            st.info("Reflection disabled: Faster responses with standard quality.")
    
    # Check if documents are available (either in session state or RAG storage)
    property_docs = [doc for doc in st.session_state.documents_processed 
                    if doc.get('property_id') == st.session_state.selected_property]
    
    # If no documents in session state, check RAG chatbot storage
    if not property_docs:
        rag_config = get_rag_chatbot_config()
        rag_chatbot = create_rag_chatbot(
            property_id=st.session_state.selected_property,
            vector_store_backend=rag_config['vector_store_backend'],
            vector_store_config=rag_config['vector_store_config']
        )
        storage_info = rag_chatbot.get_storage_info() if rag_chatbot else {}
        
        if storage_info.get('chunks_in_memory', 0) > 0:
            # Documents exist in storage, load them into session state
            if rag_chatbot and rag_chatbot.document_metadata:
                # Group by source document
                doc_sources = {}
                for metadata in rag_chatbot.document_metadata:
                    source = metadata.get('source_document', 'Unknown')
                    doc_type = metadata.get('document_type', 'Unknown')
                    if source not in doc_sources:
                        doc_sources[source] = {
                            'name': source,
                            'type': doc_type,
                            'pages': 0,
                            'property_id': st.session_state.selected_property
                        }
                    doc_sources[source]['pages'] += 1
                
                # Add to session state
                for doc_data in doc_sources.values():
                    if doc_data not in st.session_state.documents_processed:
                        st.session_state.documents_processed.append(doc_data)
                
                # Update property_docs
                property_docs = [doc for doc in st.session_state.documents_processed 
                               if doc.get('property_id') == st.session_state.selected_property]
    
    if not property_docs:
        st.info("No documents uploaded for this property yet. Upload some documents first!")
        return
    
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
        placeholder="e.g., What are the rental restrictions? How are board members elected?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Send", type="primary"):
            if user_input.strip():
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input.strip()
                })
                
                # Get chatbot response with reflection setting
                rag_config = get_rag_chatbot_config()
                rag_chatbot = create_rag_chatbot(
                    property_id=st.session_state.selected_property,
                    enable_reflection=enable_reflection,
                    vector_store_backend=rag_config['vector_store_backend'],
                    vector_store_config=rag_config['vector_store_config']
                )
                if rag_chatbot:
                    spinner_text = "Reflecting and improving response..." if enable_reflection else "Thinking..."
                    with st.spinner(spinner_text):
                        response = rag_chatbot.chat(user_input.strip())
                    
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
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


def show_settings_page(user_manager):
    """Display user settings page."""
    
    st.header("⚙️ Account Settings")
    
    user_info = user_manager.get_user_info(st.session_state.current_user)
    if not user_info:
        st.error("User information not found.")
        return
    
    # Display user info
    st.subheader("Account Information")
    st.write(f"**Name:** {user_info['name']}")
    st.write(f"**Email:** {user_info['email']}")
    st.write(f"**Username:** {user_info['username']}")
    
    # Change password
    st.subheader("Change Password")
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        submit_button = st.form_submit_button("Change Password")
        
        if submit_button:
            if not all([current_password, new_password, confirm_password]):
                st.error("Please fill in all password fields.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            elif len(new_password) < 8:
                st.error("New password must be at least 8 characters long.")
            else:
                success, message = user_manager.change_password(
                    st.session_state.current_user,
                    current_password,
                    new_password
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Vector Store Configuration
    st.subheader("Vector Store Configuration")
    
    # Show current configuration
    rag_config = get_rag_chatbot_config()
    st.info(f"**Current Backend:** {rag_config['vector_store_backend'].upper()}")
    
    if rag_config['vector_store_backend'] == 'pinecone':
        st.success("✅ Pinecone configured - using cloud-based vector storage")
        st.write("**Benefits:** Scalable, persistent, accessible from anywhere")
    else:
        st.info("ℹ️ FAISS configured - using local vector storage")
        st.write("**Benefits:** Fast, no external dependencies, works offline")
    
    # Configuration instructions
    with st.expander("🔧 How to change vector store backend"):
        st.write("""
        **To use Pinecone (cloud storage):**
        1. Sign up at [pinecone.io](https://pinecone.io)
        2. Create an index with dimension 384
        3. Set environment variables:
           - `VECTOR_STORE_BACKEND=pinecone`
           - `PINECONE_API_KEY=your_api_key`
           - `PINECONE_INDEX_NAME=your_index_name` (optional)
        
        **To use FAISS (local storage):**
        1. Set environment variable: `VECTOR_STORE_BACKEND=faiss`
        2. Or leave unset (FAISS is the default)
        """)
    
    st.write("---")
    
    # Storage management
    st.subheader("Storage Management")
    
    rag_config = get_rag_chatbot_config()
    rag_chatbot = create_rag_chatbot(
        property_id=st.session_state.selected_property,
        vector_store_backend=rag_config['vector_store_backend'],
        vector_store_config=rag_config['vector_store_config']
    )
    if rag_chatbot:
        storage_info = rag_chatbot.get_storage_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Storage", f"{storage_info['total_size_mb']} MB")
            st.metric("Chunks in Memory", storage_info['chunks_in_memory'])
        
        with col2:
            st.metric("Vectors in Index", storage_info['vectors_in_index'])
            st.metric("Files on Disk", len(storage_info['files']))
        
        # Show vector store information
        if 'vector_store' in storage_info:
            vs_info = storage_info['vector_store']
            st.write("**Vector Store Info:**")
            st.write(f"• Backend: {vs_info.get('backend', 'Unknown')}")
            st.write(f"• Total Vectors: {vs_info.get('total_vectors', 0)}")
            if vs_info.get('backend') == 'Pinecone':
                st.write(f"• Index Name: {vs_info.get('index_name', 'Unknown')}")
            elif vs_info.get('backend') == 'FAISS':
                st.write(f"• Storage Path: {vs_info.get('storage_path', 'Unknown')}")
        
        # Show file details
        if storage_info['files']:
            st.write("**Storage Files:**")
            for filename, info in storage_info['files'].items():
                st.write(f"• {filename}: {info['size_mb']} MB")
        
        # Document management
        st.write("---")
        st.subheader("Document Management")
        
        # Get list of all documents
        documents = rag_chatbot.get_document_list()
        
        if documents:
            st.write(f"**Total Documents:** {len(documents)}")
            
            # Create a table for document management
            for doc in documents:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📄 {doc['name']}")
                
                with col2:
                    st.write(f"**{doc['chunk_count']} chunks**")
                
                with col3:
                    if st.button(f"🗑️ Remove", key=f"settings_remove_{doc['name']}", type="secondary"):
                        result = rag_chatbot.remove_document(doc['name'])
                        if result['success']:
                            # Remove from session state if it exists there
                            st.session_state.documents_processed = [
                                d for d in st.session_state.documents_processed 
                                if not (d.get('property_id') == st.session_state.selected_property and d.get('name') == doc['name'])
                            ]
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error(result['message'])
        else:
            st.info("No documents in storage")
        
        # Clear data button
        st.write("---")
        if st.button("🗑️ Clear All Data", type="secondary", help="Remove all documents and chunks from memory and disk"):
            result = rag_chatbot.clear_all_data()
            if result['success']:
                st.success(result['message'])
                # Clear session state
                st.session_state.documents_processed = []
                st.rerun()
            else:
                st.error(result['message'])


if __name__ == "__main__":
    main()
