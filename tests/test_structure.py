"""Tests for the structure module."""

import unittest
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.append('..')

from app.structure import DocumentStructure, analyze_document_structure, compare_documents, get_document_summary
from app.parsing import Document


class TestDocumentStructure(unittest.TestCase):
    """Test the DocumentStructure class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.document = Document("test.pdf")
        self.document.add_page(1, "Declaration of condominium content")
        self.document.add_page(2, "More declaration content")
        
        self.structure = DocumentStructure(self.document)
    
    def test_document_structure_initialization(self):
        """Test document structure initialization."""
        self.assertEqual(self.structure.document, self.document)
        self.assertEqual(self.structure.doc_type, "unknown")
        self.assertEqual(self.structure.confidence, 0.0)
        self.assertEqual(len(self.structure.sections), 0)
        self.assertEqual(len(self.structure.key_values), 0)
        self.assertEqual(self.structure.hierarchy_level, 0)
    
    def test_classify_document_declaration(self):
        """Test document classification for declaration."""
        # Add declaration keywords to document
        self.document.add_page(3, "common elements limited common elements percentage of ownership")
        
        doc_type = self.structure.classify_document()
        
        self.assertEqual(doc_type, "declaration")
        self.assertEqual(self.structure.doc_type, "declaration")
        self.assertEqual(self.structure.hierarchy_level, 1)
        self.assertGreater(self.structure.confidence, 0.0)
    
    def test_classify_document_bylaws(self):
        """Test document classification for bylaws."""
        # Clear document and add bylaws content
        self.document.pages = []
        self.document.add_page(1, "bylaws board of directors meetings voting procedures quorum")
        
        doc_type = self.structure.classify_document()
        
        self.assertEqual(doc_type, "bylaws")
        self.assertEqual(self.structure.doc_type, "bylaws")
        self.assertEqual(self.structure.hierarchy_level, 2)
    
    def test_classify_document_rules(self):
        """Test document classification for rules."""
        # Clear document and add rules content
        self.document.pages = []
        self.document.add_page(1, "rules and regulations house rules use restrictions pet policies")
        
        doc_type = self.structure.classify_document()
        
        self.assertEqual(doc_type, "rules")
        self.assertEqual(self.structure.doc_type, "rules")
        self.assertEqual(self.structure.hierarchy_level, 3)
    
    def test_extract_sections(self):
        """Test section extraction."""
        # Add content with sections
        self.document.add_page(4, """
        ARTICLE I - NAME AND PURPOSE
        This is the first article.
        
        SECTION 1.1 - DEFINITIONS
        This is section 1.1.
        """)
        
        sections = self.structure.extract_sections()
        
        self.assertGreater(len(sections), 0)
        self.assertEqual(sections[0]['number'], 'I')
        self.assertEqual(sections[0]['title'], 'NAME AND PURPOSE')
    
    def test_extract_key_values(self):
        """Test key value extraction."""
        # Add content with key values
        self.document.add_page(5, "minimum rental term 6 months transfer fee 2.5%")
        
        key_values = self.structure.extract_key_values()
        
        self.assertIn('rental_minimum', key_values)
        self.assertEqual(key_values['rental_minimum'], 6)
        self.assertIn('transfer_fee_percent', key_values)
        self.assertEqual(key_values['transfer_fee_percent'], 2.5)
    
    def test_get_section_by_title(self):
        """Test finding sections by title."""
        # Add sections first
        self.structure.extract_sections()
        
        # Add content with sections
        self.document.add_page(6, """
        ARTICLE II - RENTAL RULES
        Rental rules content.
        """)
        
        self.structure.extract_sections()
        
        section = self.structure.get_section_by_title(['rental', 'rules'])
        self.assertIsNotNone(section)
        self.assertIn('rental', section['title'].lower())
    
    def test_get_sections_by_level(self):
        """Test getting sections by hierarchy level."""
        # Add sections first
        self.structure.extract_sections()
        
        # Add content with different level sections
        self.document.add_page(7, """
        ARTICLE III - GOVERNANCE
        SECTION A - BOARD DUTIES
        SECTION B - MEETINGS
        """)
        
        self.structure.extract_sections()
        
        level_1_sections = self.structure.get_sections_by_level(1)
        level_2_sections = self.structure.get_sections_by_level(2)
        
        self.assertGreater(len(level_1_sections), 0)
        self.assertGreater(len(level_2_sections), 0)


class TestDocumentAnalysis(unittest.TestCase):
    """Test document analysis functions."""
    
    def test_analyze_document_structure(self):
        """Test document structure analysis."""
        document = Document("test.pdf")
        document.add_page(1, "Declaration of condominium common elements")
        
        structure = analyze_document_structure(document)
        
        self.assertIsInstance(structure, DocumentStructure)
        self.assertEqual(structure.document, document)
        self.assertNotEqual(structure.doc_type, "unknown")
        self.assertGreater(len(structure.sections), 0)
    
    def test_compare_documents(self):
        """Test document comparison."""
        # Create test documents
        doc1 = Document("declaration.pdf")
        doc1.add_page(1, "Declaration of condominium")
        
        doc2 = Document("bylaws.pdf")
        doc2.add_page(1, "Bylaws board of directors")
        
        # Create structures
        structure1 = DocumentStructure(doc1)
        structure1.classify_document()
        
        structure2 = DocumentStructure(doc2)
        structure2.classify_document()
        
        comparison = compare_documents([structure1, structure2])
        
        self.assertIn('hierarchy_conflicts', comparison)
        self.assertIn('contradictions', comparison)
        self.assertIn('missing_documents', comparison)
        self.assertIn('recommendations', comparison)
    
    def test_get_document_summary(self):
        """Test document summary generation."""
        document = Document("test.pdf")
        document.add_page(1, "Test content")
        
        structure = DocumentStructure(document)
        structure.classify_document()
        structure.extract_sections()
        structure.extract_key_values()
        
        summary = get_document_summary(structure)
        
        self.assertIn('file_name', summary)
        self.assertIn('doc_type', summary)
        self.assertIn('confidence', summary)
        self.assertIn('page_count', summary)
        self.assertIn('section_count', summary)
        self.assertIn('key_values', summary)
        self.assertIn('hierarchy_level', summary)
        self.assertIn('ocr_used', summary)
        self.assertIn('text_density', summary)


class TestDocumentComparison(unittest.TestCase):
    """Test document comparison functionality."""
    
    def test_missing_declaration(self):
        """Test detection of missing declaration."""
        # Create only bylaws document
        doc = Document("bylaws.pdf")
        doc.add_page(1, "Bylaws content")
        
        structure = DocumentStructure(doc)
        structure.classify_document()
        
        comparison = compare_documents([structure])
        
        missing_docs = comparison['missing_documents']
        declaration_missing = any(doc['type'] == 'declaration' for doc in missing_docs)
        
        self.assertTrue(declaration_missing)
    
    def test_hierarchy_conflicts(self):
        """Test detection of hierarchy conflicts."""
        # Create declaration with rental minimum
        decl_doc = Document("declaration.pdf")
        decl_doc.add_page(1, "minimum rental term 12 months")
        
        # Create rules with lower rental minimum
        rules_doc = Document("rules.pdf")
        rules_doc.add_page(1, "minimum rental term 6 months")
        
        decl_structure = DocumentStructure(decl_doc)
        decl_structure.classify_document()
        decl_structure.extract_key_values()
        
        rules_structure = DocumentStructure(rules_doc)
        rules_structure.classify_document()
        rules_structure.extract_key_values()
        
        comparison = compare_documents([decl_structure, rules_structure])
        
        # Should detect hierarchy conflicts
        self.assertGreater(len(comparison['hierarchy_conflicts']), 0)


if __name__ == '__main__':
    unittest.main()
