"""Document structure detection and classification module."""

import re
from typing import Dict, List, Any, Optional
from app.parsing import Document, extract_sections_from_text, extract_key_values


class DocumentStructure:
    """Class to hold document structure information."""
    
    def __init__(self, document: Document):
        self.document = document
        self.doc_type = "unknown"
        self.confidence = 0.0
        self.sections: List[Dict[str, Any]] = []
        self.key_values: Dict[str, Any] = {}
        self.hierarchy_level = 0
    
    def classify_document(self) -> str:
        """Classify document type based on content analysis."""
        text = self.document.get_all_text().lower()
        
        # Declaration indicators
        declaration_keywords = [
            'declaration of condominium',
            'condominium declaration',
            'master deed',
            'declaration of covenants',
            'restrictions and easements',
            'plat and plan',
            'unit boundaries',
            'common elements',
            'limited common elements',
            'percentage of ownership',
            'voting rights',
            'maintenance responsibility'
        ]
        
        # Bylaws indicators
        bylaws_keywords = [
            'bylaws',
            'articles of incorporation',
            'board of directors',
            'officers',
            'meetings',
            'voting procedures',
            'quorum',
            'elections',
            'duties and powers',
            'amendment procedures'
        ]
        
        # Rules and regulations indicators
        rules_keywords = [
            'rules and regulations',
            'house rules',
            'use restrictions',
            'pet policies',
            'parking rules',
            'noise restrictions',
            'architectural guidelines',
            'maintenance standards',
            'enforcement procedures',
            'penalties and fines'
        ]
        
        # Calculate scores
        declaration_score = sum(1 for keyword in declaration_keywords if keyword in text)
        bylaws_score = sum(1 for keyword in bylaws_keywords if keyword in text)
        rules_score = sum(1 for keyword in rules_keywords if keyword in text)
        
        # Normalize scores by text length
        text_length = len(text)
        if text_length > 0:
            declaration_score /= text_length / 1000  # Normalize per 1000 characters
            bylaws_score /= text_length / 1000
            rules_score /= text_length / 1000
        
        # Determine document type
        max_score = max(declaration_score, bylaws_score, rules_score)
        
        if max_score > 0.1:  # Threshold for classification
            if declaration_score == max_score:
                self.doc_type = "declaration"
                self.confidence = min(declaration_score * 10, 1.0)
                self.hierarchy_level = 1
            elif bylaws_score == max_score:
                self.doc_type = "bylaws"
                self.confidence = min(bylaws_score * 10, 1.0)
                self.hierarchy_level = 2
            elif rules_score == max_score:
                self.doc_type = "rules"
                self.confidence = min(rules_score * 10, 1.0)
                self.hierarchy_level = 3
        else:
            # Try pattern-based classification
            self.doc_type = self._classify_by_patterns(text)
            self.confidence = 0.5
        
        return self.doc_type
    
    def _classify_by_patterns(self, text: str) -> str:
        """Classify document using pattern matching."""
        # Declaration patterns
        declaration_patterns = [
            r'declaration\s+of\s+condominium',
            r'unit\s+\d+',
            r'percentage\s+of\s+ownership',
            r'common\s+elements',
            r'limited\s+common\s+elements'
        ]
        
        # Bylaws patterns
        bylaws_patterns = [
            r'article\s+[ivx]+',
            r'section\s+\d+',
            r'board\s+of\s+directors',
            r'annual\s+meeting',
            r'quorum'
        ]
        
        # Rules patterns
        rules_patterns = [
            r'rule\s+\d+',
            r'regulation\s+\d+',
            r'no\s+(?:pets|smoking|parking)',
            r'quiet\s+hours',
            r'architectural\s+review'
        ]
        
        # Count matches
        declaration_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                                for pattern in declaration_patterns)
        bylaws_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in bylaws_patterns)
        rules_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for pattern in rules_patterns)
        
        # Return best match
        if declaration_matches > bylaws_matches and declaration_matches > rules_matches:
            return "declaration"
        elif bylaws_matches > rules_matches:
            return "bylaws"
        elif rules_matches > 0:
            return "rules"
        else:
            return "other"
    
    def extract_sections(self) -> List[Dict[str, Any]]:
        """Extract sections from document."""
        text = self.document.get_all_text()
        self.sections = extract_sections_from_text(text)
        
        # Add page information to sections
        for section in self.sections:
            section['pages'] = self._find_section_pages(section)
        
        return self.sections
    
    def _find_section_pages(self, section: Dict[str, Any]) -> List[int]:
        """Find which pages a section appears on."""
        pages = []
        section_text = section.get('title', '') + ' ' + section.get('content', '')
        
        for page in self.document.pages:
            if any(keyword.lower() in page['text'].lower() 
                  for keyword in section_text.split()[:5]):  # Check first 5 words
                pages.append(page['no'])
        
        return pages
    
    def extract_key_values(self) -> Dict[str, Any]:
        """Extract key values from document."""
        text = self.document.get_all_text()
        self.key_values = extract_key_values(text)
        return self.key_values
    
    def get_section_by_title(self, title_keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Find section by title keywords."""
        for section in self.sections:
            section_title = section.get('title', '').lower()
            if any(keyword.lower() in section_title for keyword in title_keywords):
                return section
        return None
    
    def get_sections_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Get sections by hierarchy level."""
        return [section for section in self.sections if section.get('level') == level]


def analyze_document_structure(document: Document) -> DocumentStructure:
    """Analyze document structure and return DocumentStructure object."""
    structure = DocumentStructure(document)
    
    # Classify document type
    structure.classify_document()
    
    # Extract sections
    structure.extract_sections()
    
    # Extract key values
    structure.extract_key_values()
    
    return structure


def compare_documents(structures: List[DocumentStructure]) -> Dict[str, Any]:
    """Compare multiple documents for conflicts and hierarchy issues."""
    comparison = {
        'hierarchy_conflicts': [],
        'contradictions': [],
        'missing_documents': [],
        'recommendations': []
    }
    
    # Check for required document types
    doc_types = [s.doc_type for s in structures]
    
    if 'declaration' not in doc_types:
        comparison['missing_documents'].append({
            'type': 'declaration',
            'importance': 'critical',
            'description': 'Declaration of Condominium is required for proper analysis'
        })
    
    if 'bylaws' not in doc_types:
        comparison['missing_documents'].append({
            'type': 'bylaws',
            'importance': 'high',
            'description': 'Bylaws provide important governance information'
        })
    
    # Check for hierarchy conflicts
    declaration_structure = next((s for s in structures if s.doc_type == 'declaration'), None)
    bylaws_structure = next((s for s in structures if s.doc_type == 'bylaws'), None)
    rules_structures = [s for s in structures if s.doc_type == 'rules']
    
    if declaration_structure and bylaws_structure:
        conflicts = _check_hierarchy_conflicts(declaration_structure, bylaws_structure)
        comparison['hierarchy_conflicts'].extend(conflicts)
    
    if declaration_structure and rules_structures:
        for rules_structure in rules_structures:
            conflicts = _check_hierarchy_conflicts(declaration_structure, rules_structure)
            comparison['hierarchy_conflicts'].extend(conflicts)
    
    # Check for contradictions in key values
    contradictions = _check_value_contradictions(structures)
    comparison['contradictions'].extend(contradictions)
    
    return comparison


def _check_hierarchy_conflicts(higher_doc: DocumentStructure, lower_doc: DocumentStructure) -> List[Dict[str, Any]]:
    """Check for conflicts between higher and lower hierarchy documents."""
    conflicts = []
    
    # Check for rules that contradict declaration
    if higher_doc.doc_type == 'declaration' and lower_doc.doc_type in ['bylaws', 'rules']:
        higher_text = higher_doc.document.get_all_text().lower()
        lower_text = lower_doc.document.get_all_text().lower()
        
        # Check for rental restrictions
        if 'rental' in lower_text and 'rental' in higher_text:
            # Look for conflicts in rental terms
            higher_rental_section = higher_doc.get_section_by_title(['rental', 'lease'])
            lower_rental_section = lower_doc.get_section_by_title(['rental', 'lease'])
            
            if higher_rental_section and lower_rental_section:
                # Compare rental terms
                higher_min = higher_doc.key_values.get('rental_minimum')
                lower_min = lower_doc.key_values.get('rental_minimum')
                
                if higher_min and lower_min and lower_min < higher_min:
                    conflicts.append({
                        'type': 'rental_minimum_conflict',
                        'higher_doc': higher_doc.document.file_name,
                        'lower_doc': lower_doc.document.file_name,
                        'higher_value': higher_min,
                        'lower_value': lower_min,
                        'description': f'Rental minimum in {lower_doc.doc_type} ({lower_min}) conflicts with declaration ({higher_min})'
                    })
    
    return conflicts


def _check_value_contradictions(structures: List[DocumentStructure]) -> List[Dict[str, Any]]:
    """Check for contradictions in key values across documents."""
    contradictions = []
    
    # Group by key value type
    rental_minimums = [(s.doc_type, s.key_values.get('rental_minimum')) 
                      for s in structures if s.key_values.get('rental_minimum')]
    transfer_fees = [(s.doc_type, s.key_values.get('transfer_fee_percent')) 
                    for s in structures if s.key_values.get('transfer_fee_percent')]
    fine_amounts = [(s.doc_type, s.key_values.get('max_fine_amount')) 
                   for s in structures if s.key_values.get('max_fine_amount')]
    
    # Check for contradictions
    if len(set(value for _, value in rental_minimums)) > 1:
        contradictions.append({
            'type': 'rental_minimum_contradiction',
            'values': rental_minimums,
            'description': 'Different rental minimum values found across documents'
        })
    
    if len(set(value for _, value in transfer_fees)) > 1:
        contradictions.append({
            'type': 'transfer_fee_contradiction',
            'values': transfer_fees,
            'description': 'Different transfer fee percentages found across documents'
        })
    
    if len(set(value for _, value in fine_amounts)) > 1:
        contradictions.append({
            'type': 'fine_amount_contradiction',
            'values': fine_amounts,
            'description': 'Different maximum fine amounts found across documents'
        })
    
    return contradictions


def get_document_summary(structure: DocumentStructure) -> Dict[str, Any]:
    """Get a summary of document structure."""
    return {
        'file_name': structure.document.file_name,
        'doc_type': structure.doc_type,
        'confidence': structure.confidence,
        'page_count': len(structure.document.pages),
        'section_count': len(structure.sections),
        'key_values': structure.key_values,
        'hierarchy_level': structure.hierarchy_level,
        'ocr_used': structure.document.ocr_used,
        'text_density': structure.document.text_density
    }
