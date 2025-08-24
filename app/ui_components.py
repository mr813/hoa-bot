"""UI components for the HOA Auditor Streamlit app."""

import streamlit as st
from typing import Dict, List, Any, Optional
from app.utils import get_severity_color, get_severity_icon, format_file_size, format_statute_reference
from app.findings import Finding


def render_disclaimer():
    """Render the legal disclaimer banner."""
    st.warning(
        "⚠️ **DISCLAIMER:** This application is for educational purposes only and does not constitute legal advice. "
        "All findings should be reviewed by qualified legal counsel before taking any action.",
        icon="⚠️"
    )


def render_sidebar_config():
    """Render sidebar configuration options."""
    st.sidebar.title("🏠 HOA Auditor")
    st.sidebar.markdown("Florida Chapter 718 Compliance Checker")
    
    # Research settings
    if st.session_state.get('research_enabled', False):
        st.sidebar.subheader("🔬 Research Settings")
        
        available_models = [
            "sonar", "sonar-pro", "llama-3.1-8b-instruct", 
            "llama-3.1-70b-instruct", "mixtral-8x7b-instruct"
        ]
        
        selected_model = st.sidebar.selectbox(
            "AI Model",
            available_models,
            index=0,
            help="Select the AI model for research assistance"
        )
        
        st.session_state['selected_model'] = selected_model
        
        st.sidebar.info(
            "Research features are rate-limited. Please be patient with API calls.",
            icon="ℹ️"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 1.0.0")


def render_file_uploader():
    """Render the file uploader component."""
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload your HOA documents (PDF format)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload up to 5 PDF files (Declaration, Bylaws, Rules, etc.)"
    )
    
    if uploaded_files:
        st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
        
        # Display file information
        file_info = []
        for file in uploaded_files:
            file_info.append({
                'name': file.name,
                'size': format_file_size(file.size),
                'type': file.type
            })
        
        st.subheader("📋 Uploaded Files")
        for info in file_info:
            st.write(f"• **{info['name']}** ({info['size']})")
    
    return uploaded_files


def render_processing_status(processing_step: str, progress: float = 0.0):
    """Render processing status with progress bar."""
    st.subheader(f"🔄 {processing_step}")
    
    if progress > 0:
        progress_bar = st.progress(progress)
        if progress >= 1.0:
            progress_bar.empty()
            st.success("✅ Processing complete!")


def render_document_summary(structures: List[Any]):
    """Render document analysis summary."""
    st.header("📊 Document Analysis")
    
    if not structures:
        st.info("No documents analyzed yet.")
        return
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents", len(structures))
    
    with col2:
        total_pages = sum(len(s.document.pages) for s in structures)
        st.metric("Total Pages", total_pages)
    
    with col3:
        ocr_used = sum(1 for s in structures if s.document.ocr_used)
        st.metric("OCR Used", ocr_used)
    
    # Document details
    st.subheader("📄 Document Details")
    
    for structure in structures:
        with st.expander(f"📋 {structure.document.file_name}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {structure.doc_type.title()}")
                st.write(f"**Confidence:** {structure.confidence:.1%}")
                st.write(f"**Pages:** {len(structure.document.pages)}")
                st.write(f"**Sections:** {len(structure.sections)}")
            
            with col2:
                st.write(f"**Hierarchy Level:** {structure.hierarchy_level}")
                st.write(f"**OCR Used:** {'Yes' if structure.document.ocr_used else 'No'}")
                st.write(f"**Text Density:** {structure.document.text_density:.0f} chars/page")
            
            # Key values
            if structure.key_values:
                st.write("**Key Values:**")
                for key, value in structure.key_values.items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")


def render_findings_summary(findings_engine):
    """Render findings summary."""
    st.header("🔍 Audit Results")
    
    summary = findings_engine.get_findings_summary()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Findings", summary['total_findings'])
    
    with col2:
        critical_count = summary['severity_counts'].get('likely_conflict', 0)
        st.metric("Critical Issues", critical_count, delta=None)
    
    with col3:
        counsel_count = summary['severity_counts'].get('needs_counsel', 0)
        st.metric("Needs Counsel", counsel_count, delta=None)
    
    with col4:
        ai_count = summary.get('ai_aided_count', 0)
        st.metric("AI Enhanced", ai_count, delta=None)
    
    # Severity breakdown
    st.subheader("📈 Findings by Severity")
    
    severity_data = summary['severity_counts']
    if severity_data:
        for severity, count in severity_data.items():
            if count > 0:
                color = get_severity_color(severity)
                icon = get_severity_icon(severity)
                st.write(f"{icon} **{severity.replace('_', ' ').title()}:** {count}")


def render_findings_list(findings_engine, research_instance=None):
    """Render the findings list with filtering options."""
    st.header("📋 Detailed Findings")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "likely_conflict", "needs_counsel", "likely_compliant", "info"],
            help="Filter findings by severity level"
        )
    
    with col2:
        topic_filter = st.selectbox(
            "Filter by Topic",
            ["All"] + list(set(f.topic for f in findings_engine.findings)),
            help="Filter findings by topic"
        )
    
    # Apply filters
    filtered_findings = findings_engine.findings
    
    if severity_filter != "All":
        filtered_findings = [f for f in filtered_findings if f.severity == severity_filter]
    
    if topic_filter != "All":
        filtered_findings = [f for f in filtered_findings if f.topic == topic_filter]
    
    # Display findings
    if not filtered_findings:
        st.info("No findings match the selected filters.")
        return
    
    for i, finding in enumerate(filtered_findings):
        render_finding_card(finding, i, research_instance)


def render_finding_card(finding: Finding, index: int, research_instance=None):
    """Render a single finding card."""
    color = get_severity_color(finding.severity)
    icon = get_severity_icon(finding.severity)
    
    with st.container():
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding-left: 1rem; margin: 1rem 0;">
            <h4>{icon} {finding.description}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Finding details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Severity:** {finding.severity.replace('_', ' ').title()}")
            st.write(f"**File:** {finding.file_name} ({finding.doc_type})")
            st.write(f"**Evidence:** {finding.evidence_snippet}")
            
            if finding.statute_citations:
                st.write("**Statutes:**")
                for citation in finding.statute_citations:
                    st.write(f"• {format_statute_reference(citation['section'])} - {citation['note']}")
            
            st.write(f"**Suggestion:** {finding.suggestion}")
        
        with col2:
            # Research button
            if research_instance and research_instance.is_enabled():
                if st.button(f"🔬 Research", key=f"research_{index}"):
                    with st.spinner("Researching..."):
                        research_result = research_instance.enrich_finding(
                            finding.description, 
                            finding.statute_refs
                        )
                        
                        if research_result.get('success'):
                            findings_engine = st.session_state.get('findings_engine')
                            if findings_engine:
                                findings_engine.enrich_finding_with_research(index, research_result)
                                st.success("Research completed!")
                                st.rerun()
                        else:
                            st.error(f"Research failed: {research_result.get('error', 'Unknown error')}")
            
            # AI research badge
            if finding.ai_aided:
                st.markdown('<span style="background: #007cba; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">AI-aided research</span>', unsafe_allow_html=True)
        
        # Show AI research if available
        if finding.ai_research and finding.ai_research.get('summary_bullets'):
            st.subheader("🔬 Research Insights")
            for bullet in finding.ai_research['summary_bullets']:
                st.write(f"• {bullet}")
            
            if finding.ai_research.get('sources'):
                st.write("**Sources:**")
                for source in finding.ai_research['sources']:
                    st.write(f"• [{source['title']}]({source['url']})")
        
        st.markdown("---")


def render_report_section(findings_engine, report_generator):
    """Render the report generation section."""
    st.header("📄 Generate Reports")
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Findings Report")
        
        report_format = st.selectbox(
            "Report Format",
            ["Markdown", "HTML", "PDF"],
            help="Select the format for the findings report"
        )
        
        if st.button("📄 Generate Report"):
            with st.spinner("Generating report..."):
                findings_data = findings_engine.get_findings_for_report()
                
                if report_format == "Markdown":
                    report_content = report_generator.generate_findings_report(findings_data, 'markdown')
                    st.text_area("Report Content", report_content, height=400)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Markdown",
                        data=report_content,
                        file_name="hoa_audit_report.md",
                        mime="text/markdown"
                    )
                
                elif report_format == "HTML":
                    report_content = report_generator.generate_findings_report(findings_data, 'html')
                    st.text_area("Report Content", report_content, height=400)
                    
                    st.download_button(
                        label="📥 Download HTML",
                        data=report_content,
                        file_name="hoa_audit_report.html",
                        mime="text/html"
                    )
                
                elif report_format == "PDF":
                    # Generate PDF
                    pdf_path = "temp_audit_report.pdf"
                    if report_generator.generate_pdf_report(findings_data, pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📥 Download PDF",
                                data=f.read(),
                                file_name="hoa_audit_report.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.error("Failed to generate PDF report")
    
    with col2:
        st.subheader("✉️ Board Letter")
        
        association_name = st.text_input(
            "Association Name",
            value="Board of Directors",
            help="Name of the association board"
        )
        
        owner_name = st.text_input(
            "Your Name",
            value="Unit Owner",
            help="Your name for the letter signature"
        )
        
        if st.button("✉️ Generate Letter"):
            with st.spinner("Generating letter..."):
                findings_data = findings_engine.get_findings_for_report()
                letter_content = report_generator.generate_board_letter(
                    findings_data, association_name, owner_name
                )
                
                st.text_area("Letter Content", letter_content, height=400)
                
                st.download_button(
                    label="📥 Download Letter",
                    data=letter_content,
                    file_name="board_letter.md",
                    mime="text/markdown"
                )
    
    # JSON Export
    st.subheader("📊 Export Data")
    
    if st.button("📥 Export JSON"):
        findings_data = findings_engine.get_findings_for_report()
        json_data = findings_data
        
        st.download_button(
            label="📥 Download JSON",
            data=str(json_data),
            file_name="findings_data.json",
            mime="application/json"
        )


def render_research_panel(research_instance):
    """Render the research assistance panel."""
    if not research_instance or not research_instance.is_enabled():
        return
    
    st.header("🔬 Research Assistant")
    
    # Research options
    research_option = st.selectbox(
        "Research Topic",
        [
            "Statute Summary",
            "Hierarchy Explanation",
            "Custom Query"
        ],
        help="Select a research topic"
    )
    
    if research_option == "Statute Summary":
        statute_ref = st.text_input(
            "Statute Reference",
            value="718.112",
            help="Enter Florida statute reference (e.g., 718.112)"
        )
        
        if st.button("🔍 Research Statute"):
            with st.spinner("Researching statute..."):
                result = research_instance.get_statute_summary(statute_ref)
                render_research_result(result)
    
    elif research_option == "Hierarchy Explanation":
        if st.button("🔍 Explain Hierarchy"):
            with st.spinner("Researching hierarchy principle..."):
                result = research_instance.get_hierarchy_explanation()
                render_research_result(result)
    
    elif research_option == "Custom Query":
        query = st.text_area(
            "Custom Research Query",
            help="Enter your research question about Florida condominium law"
        )
        
        if st.button("🔍 Research Query") and query:
            with st.spinner("Researching..."):
                result = research_instance.ask_perplexity(query)
                render_research_result(result)


def render_research_result(result: Dict[str, Any]):
    """Render research results."""
    if result.get('success'):
        st.subheader("🔬 Research Results")
        
        if result.get('summary_bullets'):
            st.write("**Key Points:**")
            for bullet in result['summary_bullets']:
                st.write(f"• {bullet}")
        
        if result.get('sources'):
            st.write("**Sources:**")
            for source in result['sources']:
                st.write(f"• [{source['title']}]({source['url']})")
        
        if result.get('raw_text'):
            with st.expander("Full Response"):
                st.write(result['raw_text'])
    else:
        st.error(f"Research failed: {result.get('error', 'Unknown error')}")


def render_help_section():
    """Render help and information section."""
    st.header("❓ Help & Information")
    
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Getting Started
        
        1. **Upload Documents:** Upload your HOA governing documents (Declaration, Bylaws, Rules, etc.)
        2. **Run Audit:** Click "Run Audit" to analyze documents against Florida Chapter 718
        3. **Review Findings:** Examine the findings and use filters to focus on specific issues
        4. **Generate Reports:** Create reports and board letters for documentation
        5. **Research (Optional):** Use AI research assistance for enhanced analysis
        
        ### Document Types
        
        - **Declaration:** The master document that establishes the condominium
        - **Bylaws:** Rules for association governance and operations
        - **Rules & Regulations:** Specific use and conduct rules
        - **Resolutions:** Board decisions and policies
        
        ### Severity Levels
        
        - **🚨 Likely Conflict:** Potential violation of Florida law
        - **⚠️ Needs Counsel:** Requires legal review and advice
        - **✅ Likely Compliant:** Appears to meet legal requirements
        - **ℹ️ Info:** Informational findings
        """)
    
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        ### System Requirements
        
        - Python 3.10+
        - Tesseract OCR (for PDF text extraction)
        - Poppler (for PDF processing)
        
        ### Optional Features
        
        - **Perplexity API:** Enable AI research assistance
        - **PDF Generation:** Requires WeasyPrint or ReportLab
        
        ### File Limits
        
        - Maximum 5 PDF files
        - Maximum 300 pages total
        - Maximum 50MB per file
        """)
    
    with st.expander("⚖️ Legal Disclaimer"):
        st.markdown("""
        ### Important Legal Information
        
        This application is for **educational purposes only** and does not constitute legal advice.
        
        **Limitations:**
        - Analysis is based on document content and pattern matching
        - May not capture all legal nuances or recent law changes
        - Does not replace professional legal review
        
        **Recommendations:**
        - Always consult qualified legal counsel
        - Review all findings with your attorney
        - Consider local legal requirements
        - Keep documentation updated
        
        **No Liability:**
        The developers and operators of this tool assume no liability for any actions taken based on the analysis provided.
        """)
