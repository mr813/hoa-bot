"""Florida Chapter 718 compliance checklist module."""

import re
from typing import Dict, List, Any, Optional
from app.utils import load_json_file, get_assets_path
from app.structure import DocumentStructure


class ComplianceChecker:
    """Class to check documents against Florida Chapter 718 requirements."""
    
    def __init__(self):
        self.checklist = self._load_checklist()
        self.statute_sections = self.checklist.get('statute_sections', {})
        self.hierarchy_rules = self.checklist.get('hierarchy_rules', [])
    
    def _load_checklist(self) -> Dict[str, Any]:
        """Load the Florida Chapter 718 checklist."""
        checklist_path = get_assets_path() / "state_pack_fl_718.json"
        return load_json_file(str(checklist_path))
    
    def check_compliance(self, structures: List[DocumentStructure]) -> List[Dict[str, Any]]:
        """Check all documents against Florida Chapter 718 requirements."""
        issues = []
        
        for topic in self.checklist.get('topics', []):
            topic_issues = self._check_topic(topic, structures)
            issues.extend(topic_issues)
        
        return issues
    
    def _check_topic(self, topic: Dict[str, Any], structures: List[DocumentStructure]) -> List[Dict[str, Any]]:
        """Check a specific topic against all documents."""
        issues = []
        topic_id = topic.get('id', '')
        topic_name = topic.get('topic', '')
        statute_refs = topic.get('statute_refs', [])
        logic_type = topic.get('logic_type', 'compliance')
        
        if logic_type == 'extraction':
            issues.extend(self._check_extraction_topic(topic, structures))
        elif logic_type == 'compliance':
            issues.extend(self._check_compliance_topic(topic, structures))
        elif logic_type == 'hierarchy':
            issues.extend(self._check_hierarchy_topic(topic, structures))
        
        return issues
    
    def _check_extraction_topic(self, topic: Dict[str, Any], structures: List[DocumentStructure]) -> List[Dict[str, Any]]:
        """Check extraction-based topics (rental minimum, transfer fees, etc.)."""
        issues = []
        topic_id = topic.get('id', '')
        keywords = topic.get('keywords', [])
        
        for structure in structures:
            text = structure.document.get_all_text().lower()
            
            # Check if topic is mentioned
            if any(keyword in text for keyword in keywords):
                # Extract relevant information
                if topic_id == 'rental_minimum':
                    issues.extend(self._check_rental_minimum(structure))
                elif topic_id == 'transfer_fee_cap':
                    issues.extend(self._check_transfer_fee_cap(structure))
                elif topic_id == 'rental_approval_scope':
                    issues.extend(self._check_rental_approval_scope(structure))
        
        return issues
    
    def _check_compliance_topic(self, topic: Dict[str, Any], structures: List[DocumentStructure]) -> List[Dict[str, Any]]:
        """Check compliance-based topics (fines, meetings, records, etc.)."""
        issues = []
        topic_id = topic.get('id', '')
        keywords = topic.get('keywords', [])
        
        for structure in structures:
            text = structure.document.get_all_text().lower()
            
            if any(keyword in text for keyword in keywords):
                if topic_id == 'fines_procedure':
                    issues.extend(self._check_fines_procedure(structure))
                elif topic_id == 'meetings_notice':
                    issues.extend(self._check_meetings_notice(structure))
                elif topic_id == 'records_access':
                    issues.extend(self._check_records_access(structure))
                elif topic_id == 'collections_safe_harbor':
                    issues.extend(self._check_collections_safe_harbor(structure))
                elif topic_id == 'budgets_reserves_SIRS':
                    issues.extend(self._check_budgets_reserves(structure))
                elif topic_id == 'elections_recall':
                    issues.extend(self._check_elections_recall(structure))
        
        return issues
    
    def _check_hierarchy_topic(self, topic: Dict[str, Any], structures: List[DocumentStructure]) -> List[Dict[str, Any]]:
        """Check hierarchy-based topics (declaration supremacy)."""
        issues = []
        topic_id = topic.get('id', '')
        
        if topic_id == 'declaration_supremacy':
            issues.extend(self._check_declaration_supremacy(structures))
        
        return issues
    
    def _check_rental_minimum(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check rental minimum requirements."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Look for rental minimum patterns
        patterns = [
            r'minimum\s+(?:rental|lease)\s+(?:term|duration|period)\s*(?:of\s*)?(\d+)\s*(?:month|year)',
            r'(\d+)\s*(?:month|year)\s+minimum\s+(?:rental|lease)',
            r'no\s+(?:rental|lease)\s+(?:less\s+than|under)\s+(\d+)\s*(?:month|year)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                minimum = int(match)
                
                # Check against Florida requirements
                if minimum < 1:
                    issues.append({
                        'topic': 'rental_minimum',
                        'severity': 'likely_conflict',
                        'description': f'Rental minimum of {minimum} months may conflict with Florida law',
                        'evidence': f'Found rental minimum of {minimum} months',
                        'statute_refs': ['718.110(13)', '718.112(2)(i)'],
                        'file_name': structure.document.file_name,
                        'doc_type': structure.doc_type,
                        'suggestion': 'Consider minimum 1 month rental term or consult legal counsel'
                    })
        
        return issues
    
    def _check_transfer_fee_cap(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check transfer fee caps."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Look for transfer fee patterns
        patterns = [
            r'transfer\s+fee\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*transfer\s+fee',
            r'sale\s+fee\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                fee_percent = float(match)
                
                # Check against reasonable limits
                if fee_percent > 10.0:
                    issues.append({
                        'topic': 'transfer_fee_cap',
                        'severity': 'needs_counsel',
                        'description': f'Transfer fee of {fee_percent}% may be excessive',
                        'evidence': f'Found transfer fee of {fee_percent}%',
                        'statute_refs': ['718.112(2)(i)'],
                        'file_name': structure.document.file_name,
                        'doc_type': structure.doc_type,
                        'suggestion': 'Review transfer fee reasonableness and consult legal counsel'
                    })
        
        return issues
    
    def _check_rental_approval_scope(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check rental approval scope and requirements."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Check for overly broad approval requirements
        broad_patterns = [
            r'board\s+approval\s+required\s+for\s+all\s+rentals',
            r'no\s+rental\s+without\s+board\s+approval',
            r'rental\s+application\s+required',
        ]
        
        for pattern in broad_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append({
                    'topic': 'rental_approval_scope',
                    'severity': 'needs_counsel',
                    'description': 'Broad rental approval requirements may conflict with Florida law',
                    'evidence': 'Found broad rental approval requirements',
                    'statute_refs': ['718.110(13)', '718.112(2)(i)'],
                    'file_name': structure.document.file_name,
                    'doc_type': structure.doc_type,
                    'suggestion': 'Review rental approval scope and consult legal counsel'
                })
        
        return issues
    
    def _check_fines_procedure(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check fines and suspension procedures."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Check for required elements
        required_elements = [
            'notice',
            'hearing',
            'committee',
            'appeal',
            'due process'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in text:
                missing_elements.append(element)
        
        if missing_elements:
            issues.append({
                'topic': 'fines_procedure',
                'severity': 'likely_conflict',
                'description': f'Missing required elements in fines procedure: {", ".join(missing_elements)}',
                'evidence': f'Fines procedure missing: {", ".join(missing_elements)}',
                'statute_refs': ['718.303(3)', '718.303(4)'],
                'file_name': structure.document.file_name,
                'doc_type': structure.doc_type,
                'suggestion': 'Ensure fines procedure includes notice, hearing, and appeal rights'
            })
        
        return issues
    
    def _check_meetings_notice(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check meeting notice requirements."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Check for required notice elements
        notice_elements = [
            'notice',
            'agenda',
            'quorum',
            'voting'
        ]
        
        missing_elements = []
        for element in notice_elements:
            if element not in text:
                missing_elements.append(element)
        
        if missing_elements:
            issues.append({
                'topic': 'meetings_notice',
                'severity': 'needs_counsel',
                'description': f'Meeting procedures may be missing required elements: {", ".join(missing_elements)}',
                'evidence': f'Meeting procedures missing: {", ".join(missing_elements)}',
                'statute_refs': ['718.112(2)(c)', '718.112(2)(d)'],
                'file_name': structure.document.file_name,
                'doc_type': structure.doc_type,
                'suggestion': 'Review meeting notice and procedure requirements'
            })
        
        return issues
    
    def _check_records_access(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check records access and inspection rights."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Check for records access provisions
        if 'records' not in text and 'inspection' not in text:
            issues.append({
                'topic': 'records_access',
                'severity': 'likely_conflict',
                'description': 'No records access or inspection provisions found',
                'evidence': 'Missing records access provisions',
                'statute_refs': ['718.111(12)', '718.111(13)'],
                'file_name': structure.document.file_name,
                'doc_type': structure.doc_type,
                'suggestion': 'Add records access and inspection provisions'
            })
        
        return issues
    
    def _check_collections_safe_harbor(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check collections and safe harbor provisions."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Check for safe harbor provisions
        if 'safe harbor' not in text and 'payment plan' not in text:
            issues.append({
                'topic': 'collections_safe_harbor',
                'severity': 'needs_counsel',
                'description': 'No safe harbor or payment plan provisions found',
                'evidence': 'Missing safe harbor provisions',
                'statute_refs': ['718.116(6)', '718.116(11)'],
                'file_name': structure.document.file_name,
                'doc_type': structure.doc_type,
                'suggestion': 'Consider adding safe harbor and payment plan provisions'
            })
        
        return issues
    
    def _check_budgets_reserves(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check budget and reserve requirements."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Check for SIRS (Structural Integrity Reserve Study) references
        if 'sirs' not in text and 'structural integrity' not in text:
            issues.append({
                'topic': 'budgets_reserves_SIRS',
                'severity': 'info',
                'description': 'No SIRS (Structural Integrity Reserve Study) provisions found',
                'evidence': 'Missing SIRS provisions',
                'statute_refs': ['718.112(2)(f)', '718.112(2)(g)'],
                'file_name': structure.document.file_name,
                'doc_type': structure.doc_type,
                'suggestion': 'Consider adding SIRS requirements (statute changed; consult counsel)'
            })
        
        return issues
    
    def _check_elections_recall(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """Check election and recall procedures."""
        issues = []
        text = structure.document.get_all_text().lower()
        
        # Check for election procedures
        election_elements = [
            'election',
            'voting',
            'ballot',
            'candidate'
        ]
        
        missing_elements = []
        for element in election_elements:
            if element not in text:
                missing_elements.append(element)
        
        if missing_elements:
            issues.append({
                'topic': 'elections_recall',
                'severity': 'needs_counsel',
                'description': f'Election procedures may be missing elements: {", ".join(missing_elements)}',
                'evidence': f'Election procedures missing: {", ".join(missing_elements)}',
                'statute_refs': ['718.112(2)(d)', '718.112(2)(j)'],
                'file_name': structure.document.file_name,
                'doc_type': structure.doc_type,
                'suggestion': 'Review election and recall procedures'
            })
        
        return issues
    
    def _check_declaration_supremacy(self, structures: List[DocumentStructure]) -> List[Dict[str, Any]]:
        """Check declaration supremacy over other documents."""
        issues = []
        
        # Find declaration
        declaration = next((s for s in structures if s.doc_type == 'declaration'), None)
        if not declaration:
            return issues
        
        # Check other documents for potential conflicts
        for structure in structures:
            if structure.doc_type in ['bylaws', 'rules']:
                # Look for potential conflicts
                conflicts = self._find_potential_conflicts(declaration, structure)
                issues.extend(conflicts)
        
        return issues
    
    def _find_potential_conflicts(self, declaration: DocumentStructure, other: DocumentStructure) -> List[Dict[str, Any]]:
        """Find potential conflicts between declaration and other documents."""
        conflicts = []
        
        # Check for rental restrictions
        if 'rental' in other.document.get_all_text().lower():
            decl_rental = declaration.get_section_by_title(['rental', 'lease'])
            other_rental = other.get_section_by_title(['rental', 'lease'])
            
            if decl_rental and other_rental:
                # Compare rental terms
                decl_min = declaration.key_values.get('rental_minimum')
                other_min = other.key_values.get('rental_minimum')
                
                if decl_min and other_min and other_min < decl_min:
                    conflicts.append({
                        'topic': 'declaration_supremacy',
                        'severity': 'likely_conflict',
                        'description': f'Rental minimum in {other.doc_type} conflicts with declaration',
                        'evidence': f'{other.doc_type} allows {other_min} months, declaration requires {decl_min} months',
                        'statute_refs': ['718.104(4)', '718.110(4)'],
                        'file_name': other.document.file_name,
                        'doc_type': other.doc_type,
                        'suggestion': f'Amend {other.doc_type} to match declaration minimum of {decl_min} months'
                    })
        
        return conflicts
    
    def get_statute_description(self, statute_ref: str) -> str:
        """Get description of a statute section."""
        return self.statute_sections.get(statute_ref, statute_ref)
    
    def get_topic_by_id(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get topic by ID."""
        for topic in self.checklist.get('topics', []):
            if topic.get('id') == topic_id:
                return topic
        return None
