"""Tests for the parsing module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the modules to test
import sys
sys.path.append('..')

from app.parsing import Document, extract_text_with_pymupdf, extract_text_with_ocr, should_use_ocr, extract_sections_from_text, extract_key_values, validate_pdf_file


class TestDocument(unittest.TestCase):
    """Test the Document class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.document = Document("test.pdf")
    
    def test_document_initialization(self):
        """Test document initialization."""
        self.assertEqual(self.document.file_name, "test.pdf")
        self.assertEqual(len(self.document.pages), 0)
        self.assertEqual(self.document.raw_text, "")
        self.assertEqual(self.document.text_density, 0.0)
        self.assertFalse(self.document.ocr_used)
    
    def test_add_page(self):
        """Test adding pages to document."""
        self.document.add_page(1, "Page 1 content")
        self.assertEqual(len(self.document.pages), 1)
        self.assertEqual(self.document.pages[0]['no'], 1)
        self.assertEqual(self.document.pages[0]['text'], "Page 1 content")
        self.assertFalse(self.document.pages[0]['ocr_used'])
    
    def test_get_page_text(self):
        """Test getting text from specific page."""
        self.document.add_page(1, "Page 1 content")
        self.document.add_page(2, "Page 2 content")
        
        self.assertEqual(self.document.get_page_text(1), "Page 1 content")
        self.assertEqual(self.document.get_page_text(2), "Page 2 content")
        self.assertEqual(self.document.get_page_text(3), "")  # Non-existent page
    
    def test_get_all_text(self):
        """Test getting all text from document."""
        self.document.add_page(1, "Page 1 content")
        self.document.add_page(2, "Page 2 content")
        
        expected_text = "Page 1 content Page 2 content"
        self.assertEqual(self.document.get_all_text(), expected_text)
    
    def test_calculate_text_density(self):
        """Test text density calculation."""
        self.document.add_page(1, "Short")
        self.document.add_page(2, "Longer content here")
        
        density = self.document.calculate_text_density()
        expected_density = (5 + 18) / 2  # Average characters per page
        self.assertEqual(density, expected_density)


class TestTextExtraction(unittest.TestCase):
    """Test text extraction functions."""
    
    @patch('app.parsing.fitz')
    def test_extract_text_with_pymupdf(self, mock_fitz):
        """Test PyMuPDF text extraction."""
        # Mock the fitz document
        mock_doc = Mock()
        mock_doc.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creator': 'Test Creator',
            'producer': 'Test Producer'
        }
        mock_doc.__len__ = Mock(return_value=2)
        
        # Mock pages
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Page 2 content"
        
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        
        mock_fitz.open.return_value = mock_doc
        
        pages_text, metadata = extract_text_with_pymupdf("test.pdf")
        
        self.assertEqual(len(pages_text), 2)
        self.assertEqual(pages_text[0], "Page 1 content")
        self.assertEqual(pages_text[1], "Page 2 content")
        self.assertEqual(metadata['title'], 'Test Document')
        self.assertEqual(metadata['page_count'], 2)
    
    def test_should_use_ocr(self):
        """Test OCR decision logic."""
        # Test with low text density
        low_density_texts = ["", "a", "short"]
        self.assertTrue(should_use_ocr(low_density_texts))
        
        # Test with high text density
        high_density_texts = ["This is a much longer text with many more characters to exceed the threshold"]
        self.assertFalse(should_use_ocr(high_density_texts))


class TestSectionExtraction(unittest.TestCase):
    """Test section extraction functions."""
    
    def test_extract_sections_from_text(self):
        """Test section extraction from text."""
        text = """
        ARTICLE I - NAME AND PURPOSE
        This is the first article content.
        
        SECTION 1.1 - DEFINITIONS
        This is section 1.1 content.
        
        SECTION 1.2 - RULES
        This is section 1.2 content.
        """
        
        sections = extract_sections_from_text(text)
        
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]['number'], 'I')
        self.assertEqual(sections[0]['title'], 'NAME AND PURPOSE')
        self.assertEqual(sections[1]['number'], '1.1')
        self.assertEqual(sections[1]['title'], 'DEFINITIONS')
    
    def test_extract_key_values(self):
        """Test key value extraction."""
        text = """
        The minimum rental term shall be 6 months.
        Transfer fee is 2.5% of the sale price.
        Maximum fine is $1,000 for violations.
        """
        
        key_values = extract_key_values(text)
        
        self.assertEqual(key_values.get('rental_minimum'), 6)
        self.assertEqual(key_values.get('transfer_fee_percent'), 2.5)
        self.assertEqual(key_values.get('max_fine_amount'), 1000)


class TestPDFValidation(unittest.TestCase):
    """Test PDF validation functions."""
    
    def test_validate_pdf_file_nonexistent(self):
        """Test validation of non-existent file."""
        is_valid, message = validate_pdf_file("nonexistent.pdf")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", message)
    
    def test_validate_pdf_file_size_limit(self):
        """Test validation of oversized file."""
        # Create a temporary file that's too large
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write more than 50MB
            tmp_file.write(b'0' * (51 * 1024 * 1024))
            tmp_path = tmp_file.name
        
        try:
            is_valid, message = validate_pdf_file(tmp_path)
            self.assertFalse(is_valid)
            self.assertIn("50MB", message)
        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()
