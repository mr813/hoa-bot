"""Main Streamlit application for HOA Auditor."""

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
    from app.structure import analyze_document_structure, compare_documents
    from app.checklist import ComplianceChecker
    from app.findings import FindingsEngine
    from app.reporting import ReportGenerator
    from app.research import create_research_instance, is_research_enabled
    from app.ui_components import (
        render_disclaimer, render_sidebar_config, render_file_uploader,
        render_processing_status, render_document_summary, render_findings_summary,
        render_findings_list, render_report_section, render_research_panel,
        render_help_section
    )
except (ImportError, OSError):
    # Fallback to relative imports if absolute imports fail
    from .parsing import parse_pdf, validate_pdf_file
    from .structure import analyze_document_structure, compare_documents
    from .checklist import ComplianceChecker
    from .findings import FindingsEngine
    from .reporting import ReportGenerator
    from .research import create_research_instance, is_research_enabled
    from .ui_components import (
        render_disclaimer, render_sidebar_config, render_file_uploader,
        render_processing_status, render_document_summary, render_findings_summary,
        render_findings_list, render_report_section, render_research_panel,
        render_help_section
    )


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'structures' not in st.session_state:
        st.session_state.structures = []
    
    if 'findings_engine' not in st.session_state:
        st.session_state.findings_engine = None
    
    if 'research_enabled' not in st.session_state:
        st.session_state.research_enabled = is_research_enabled()
    
    if 'research_instance' not in st.session_state:
        st.session_state.research_instance = create_research_instance()


def process_uploaded_files(uploaded_files) -> List[Any]:
    """Process uploaded PDF files."""
    structures = []
    
    if not uploaded_files:
        return structures
    
    # Validate files
    for uploaded_file in uploaded_files:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Validate PDF
            is_valid, message = validate_pdf_file(tmp_path)
            if not is_valid:
                st.error(f"Invalid file {uploaded_file.name}: {message}")
                continue
            
            # Parse PDF
            with st.spinner(f"Processing {uploaded_file.name}..."):
                document = parse_pdf(tmp_path, uploaded_file.name)
                structure = analyze_document_structure(document)
                structures.append(structure)
            
            st.success(f"✅ Processed {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    return structures


def run_compliance_audit(structures: List[Any]) -> FindingsEngine:
    """Run the compliance audit on processed documents."""
    if not structures:
        st.error("No documents to audit.")
        return None
    
    # Initialize compliance checker
    checker = ComplianceChecker()
    
    # Run compliance checks
    with st.spinner("Running compliance audit..."):
        issues = checker.check_compliance(structures)
    
    # Process issues into findings
    findings_engine = FindingsEngine()
    findings = findings_engine.process_issues(issues)
    
    st.success(f"✅ Audit complete! Found {len(findings)} issues.")
    
    return findings_engine


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="HOA Auditor - Florida Chapter 718 Compliance",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render disclaimer
    render_disclaimer()
    
    # Render sidebar
    render_sidebar_config()
    
    # Main content
    st.title("🏠 HOA Auditor")
    st.markdown("### Florida Chapter 718 Compliance Checker")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload & Process", 
        "📊 Analysis", 
        "🔍 Findings", 
        "📄 Reports", 
        "❓ Help"
    ])
    
    with tab1:
        st.header("📁 Document Upload & Processing")
        
        # File upload
        uploaded_files = render_file_uploader()
        
        # Process files button
        if uploaded_files and st.button("🔄 Process Documents", type="primary"):
            if len(uploaded_files) > 5:
                st.error("Maximum 5 files allowed.")
                return
            
            # Process files
            structures = process_uploaded_files(uploaded_files)
            
            if structures:
                st.session_state.structures = structures
                st.session_state.documents_processed = True
                st.success("✅ Document processing complete!")
                st.rerun()
    
    with tab2:
        st.header("📊 Document Analysis")
        
        if st.session_state.documents_processed and st.session_state.structures:
            # Show document summary
            render_document_summary(st.session_state.structures)
            
            # Run audit button
            if st.button("🔍 Run Compliance Audit", type="primary"):
                findings_engine = run_compliance_audit(st.session_state.structures)
                
                if findings_engine:
                    st.session_state.findings_engine = findings_engine
                    st.success("✅ Audit complete! Check the Findings tab.")
                    st.rerun()
        else:
            st.info("Please upload and process documents first.")
    
    with tab3:
        st.header("🔍 Findings & Analysis")
        
        if st.session_state.findings_engine:
            # Show findings summary
            render_findings_summary(st.session_state.findings_engine)
            
            # Show detailed findings
            render_findings_list(
                st.session_state.findings_engine, 
                st.session_state.research_instance
            )
            
            # Research panel
            if st.session_state.research_enabled:
                render_research_panel(st.session_state.research_instance)
        else:
            st.info("Please run the compliance audit first.")
    
    with tab4:
        st.header("📄 Reports & Export")
        
        if st.session_state.findings_engine:
            # Initialize report generator
            report_generator = ReportGenerator()
            
            # Show report generation options
            render_report_section(st.session_state.findings_engine, report_generator)
        else:
            st.info("Please run the compliance audit first.")
    
    with tab5:
        render_help_section()


if __name__ == "__main__":
    main()
