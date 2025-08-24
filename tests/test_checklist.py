"""Tests for the checklist module."""

import unittest
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.append('..')

from app.checklist import ComplianceChecker
from app.structure import DocumentStructure
from app.parsing import Document


class TestComplianceChecker(unittest.TestCase):
    """Test the ComplianceChecker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = ComplianceChecker()
    
    def test_checklist_loading(self):
        """Test that checklist loads properly."""
        self.assertIsNotNone(self.checker.checklist)
        self.assertIn('topics', self.checker.checklist)
        self.assertIn('statute_sections', self.checker.checklist)
        self.assertIn('hierarchy_rules', self.checker.checklist)
    
    def test_check_compliance_empty(self):
        """Test compliance check with empty structures."""
        issues = self.checker.check_compliance([])
        self.assertEqual(len(issues), 0)
    
    def test_check_rental_minimum(self):
        """Test rental minimum compliance check."""
        # Create test document with rental minimum
        document = Document("test.pdf")
        document.add_page(1, "minimum rental term 3 months")
        
        structure = DocumentStructure(document)
        structure.classify_document()
        structure.extract_key_values()
        
        issues = self.checker.check_compliance([structure])
        
        # Should find rental minimum issues
        rental_issues = [issue for issue in issues if issue['topic'] == 'rental_minimum']
        self.assertGreater(len(rental_issues), 0)
    
    def test_check_transfer_fee_cap(self):
        """Test transfer fee cap compliance check."""
        # Create test document with high transfer fee
        document = Document("test.pdf")
        document.add_page(1, "transfer fee 15% of sale price")
        
        structure = DocumentStructure(document)
        structure.classify_document()
        structure.extract_key_values()
        
        issues = self.checker.check_compliance([structure])
        
        # Should find transfer fee issues
        fee_issues = [issue for issue in issues if issue['topic'] == 'transfer_fee_cap']
        self.assertGreater(len(fee_issues), 0)
    
    def test_check_fines_procedure(self):
        """Test fines procedure compliance check."""
        # Create test document missing required elements
        document = Document("test.pdf")
        document.add_page(1, "fines will be assessed for violations")
        # Missing: notice, hearing, committee, appeal
        
        structure = DocumentStructure(document)
        structure.classify_document()
        structure.extract_key_values()
        
        issues = self.checker.check_compliance([structure])
        
        # Should find fines procedure issues
        fines_issues = [issue for issue in issues if issue['topic'] == 'fines_procedure']
        self.assertGreater(len(fines_issues), 0)
    
    def test_check_records_access(self):
        """Test records access compliance check."""
        # Create test document without records provisions
        document = Document("test.pdf")
        document.add_page(1, "general association rules and regulations")
        # Missing: records, inspection, access
        
        structure = DocumentStructure(document)
        structure.classify_document()
        structure.extract_key_values()
        
        issues = self.checker.check_compliance([structure])
        
        # Should find records access issues
        records_issues = [issue for issue in issues if issue['topic'] == 'records_access']
        self.assertGreater(len(records_issues), 0)
    
    def test_get_statute_description(self):
        """Test statute description retrieval."""
        description = self.checker.get_statute_description("718.112")
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)
    
    def test_get_topic_by_id(self):
        """Test topic retrieval by ID."""
        topic = self.checker.get_topic_by_id("rental_minimum")
        self.assertIsNotNone(topic)
        self.assertEqual(topic['id'], "rental_minimum")
        self.assertIn('topic', topic)
        self.assertIn('statute_refs', topic)


if __name__ == '__main__':
    unittest.main()
