"""HOA Auditor - Florida Chapter 718 Compliance Checker."""

__version__ = "1.0.0"
__author__ = "HOA Auditor Team"
__description__ = "A Streamlit app for auditing HOA compliance against Florida Chapter 718"

from app.main import main
from app.parsing import Document, parse_pdf
from app.structure import DocumentStructure, analyze_document_structure
from app.checklist import ComplianceChecker
from app.findings import Finding, FindingsEngine
from app.reporting import ReportGenerator
from app.research import PerplexityResearch, is_research_enabled

__all__ = [
    'main',
    'Document',
    'parse_pdf',
    'DocumentStructure',
    'analyze_document_structure',
    'ComplianceChecker',
    'Finding',
    'FindingsEngine',
    'ReportGenerator',
    'PerplexityResearch',
    'is_research_enabled'
]
