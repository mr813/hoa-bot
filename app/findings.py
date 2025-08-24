"""Findings engine for normalizing and categorizing compliance issues."""

from typing import Dict, List, Any, Optional
from app.utils import truncate_text, format_statute_reference


class Finding:
    """Class to represent a compliance finding."""
    
    def __init__(self, issue: Dict[str, Any]):
        self.topic = issue.get('topic', '')
        self.severity = issue.get('severity', 'info')
        self.description = issue.get('description', '')
        self.evidence = issue.get('evidence', '')
        self.statute_refs = issue.get('statute_refs', [])
        self.file_name = issue.get('file_name', '')
        self.doc_type = issue.get('doc_type', '')
        self.suggestion = issue.get('suggestion', '')
        self.user_doc_citations = []
        self.statute_citations = []
        self.ai_aided = False
        self.ai_research = None
        
        # Process evidence
        self.evidence_snippet = truncate_text(self.evidence, 300)
        
        # Process citations
        self._process_citations()
    
    def _process_citations(self):
        """Process and format citations."""
        # User document citations
        if self.file_name:
            self.user_doc_citations.append({
                'file': self.file_name,
                'section_title': self._extract_section_title(),
                'page': self._extract_page_reference()
            })
        
        # Statute citations
        for statute_ref in self.statute_refs:
            self.statute_citations.append({
                'section': statute_ref,
                'note': self._get_statute_note(statute_ref)
            })
    
    def _extract_section_title(self) -> str:
        """Extract section title from evidence."""
        # Simple extraction - could be enhanced
        words = self.evidence.split()[:5]  # First 5 words
        return ' '.join(words) if words else ''
    
    def _extract_page_reference(self) -> Optional[int]:
        """Extract page reference from evidence."""
        # Look for page numbers in evidence
        import re
        page_match = re.search(r'page\s+(\d+)', self.evidence, re.IGNORECASE)
        if page_match:
            return int(page_match.group(1))
        return None
    
    def _get_statute_note(self, statute_ref: str) -> str:
        """Get note for statute reference."""
        statute_notes = {
            '718.104': 'Declaration of condominium',
            '718.110': 'Amendment of declaration',
            '718.111': 'The association',
            '718.112': 'Bylaws',
            '718.116': 'Assessments and collections',
            '718.303': 'Obligations of owners and fines'
        }
        return statute_notes.get(statute_ref, '')
    
    def add_ai_research(self, research_data: Dict[str, Any]):
        """Add AI research data to finding."""
        self.ai_aided = True
        self.ai_research = research_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary."""
        return {
            'topic': self.topic,
            'severity': self.severity,
            'description': self.description,
            'evidence': self.evidence,
            'evidence_snippet': self.evidence_snippet,
            'statute_refs': self.statute_refs,
            'file_name': self.file_name,
            'doc_type': self.doc_type,
            'suggestion': self.suggestion,
            'user_doc_citations': self.user_doc_citations,
            'statute_citations': self.statute_citations,
            'ai_aided': self.ai_aided,
            'ai_research': self.ai_research
        }


class FindingsEngine:
    """Engine for processing and managing findings."""
    
    def __init__(self):
        self.findings: List[Finding] = []
        self.severity_weights = {
            'likely_conflict': 3,
            'needs_counsel': 2,
            'likely_compliant': 1,
            'info': 0
        }
    
    def process_issues(self, issues: List[Dict[str, Any]]) -> List[Finding]:
        """Process raw issues into normalized findings."""
        self.findings = []
        
        for issue in issues:
            finding = Finding(issue)
            self.findings.append(finding)
        
        # Sort by severity
        self.findings.sort(key=lambda f: self.severity_weights.get(f.severity, 0), reverse=True)
        
        return self.findings
    
    def get_findings_by_severity(self, severity: str) -> List[Finding]:
        """Get findings filtered by severity."""
        return [f for f in self.findings if f.severity == severity]
    
    def get_findings_by_topic(self, topic: str) -> List[Finding]:
        """Get findings filtered by topic."""
        return [f for f in self.findings if f.topic == topic]
    
    def get_top_findings(self, limit: int = 10) -> List[Finding]:
        """Get top findings by severity."""
        return self.findings[:limit]
    
    def get_findings_summary(self) -> Dict[str, Any]:
        """Get summary statistics of findings."""
        severity_counts = {}
        topic_counts = {}
        
        for finding in self.findings:
            # Count by severity
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
            # Count by topic
            topic_counts[finding.topic] = topic_counts.get(finding.topic, 0) + 1
        
        return {
            'total_findings': len(self.findings),
            'severity_counts': severity_counts,
            'topic_counts': topic_counts,
            'ai_aided_count': len([f for f in self.findings if f.ai_aided])
        }
    
    def enrich_finding_with_research(self, finding_index: int, research_data: Dict[str, Any]) -> bool:
        """Enrich a finding with AI research data."""
        if 0 <= finding_index < len(self.findings):
            self.findings[finding_index].add_ai_research(research_data)
            return True
        return False
    
    def get_finding_by_index(self, index: int) -> Optional[Finding]:
        """Get finding by index."""
        if 0 <= index < len(self.findings):
            return self.findings[index]
        return None
    
    def export_findings(self) -> List[Dict[str, Any]]:
        """Export all findings as dictionaries."""
        return [finding.to_dict() for finding in self.findings]
    
    def get_findings_for_report(self) -> Dict[str, Any]:
        """Get findings organized for report generation."""
        return {
            'summary': self.get_findings_summary(),
            'findings': self.export_findings(),
            'top_findings': [f.to_dict() for f in self.get_top_findings(5)],
            'by_severity': {
                severity: [f.to_dict() for f in self.get_findings_by_severity(severity)]
                for severity in ['likely_conflict', 'needs_counsel', 'likely_compliant', 'info']
            },
            'by_topic': {
                topic: [f.to_dict() for f in self.get_findings_by_topic(topic)]
                for topic in set(f.topic for f in self.findings)
            }
        }


def normalize_issue_severity(issue: Dict[str, Any]) -> str:
    """Normalize issue severity based on content and context."""
    severity = issue.get('severity', 'info')
    
    # Override based on topic importance
    critical_topics = ['declaration_supremacy', 'fines_procedure', 'records_access']
    if issue.get('topic') in critical_topics and severity == 'needs_counsel':
        severity = 'likely_conflict'
    
    # Override based on evidence strength
    evidence = issue.get('evidence', '').lower()
    if 'missing' in evidence and 'required' in evidence:
        severity = 'likely_conflict'
    
    return severity


def categorize_findings(findings: List[Finding]) -> Dict[str, List[Finding]]:
    """Categorize findings by type and priority."""
    categories = {
        'critical': [],
        'high_priority': [],
        'medium_priority': [],
        'low_priority': [],
        'informational': []
    }
    
    for finding in findings:
        if finding.severity == 'likely_conflict':
            categories['critical'].append(finding)
        elif finding.severity == 'needs_counsel':
            categories['high_priority'].append(finding)
        elif finding.severity == 'likely_compliant':
            categories['medium_priority'].append(finding)
        else:
            categories['informational'].append(finding)
    
    return categories


def generate_finding_recommendations(findings: List[Finding]) -> List[str]:
    """Generate actionable recommendations from findings."""
    recommendations = []
    
    # Group by topic
    topic_findings = {}
    for finding in findings:
        if finding.topic not in topic_findings:
            topic_findings[finding.topic] = []
        topic_findings[finding.topic].append(finding)
    
    # Generate recommendations
    for topic, topic_findings_list in topic_findings.items():
        if topic == 'rental_minimum':
            recommendations.append("Review and standardize rental minimum requirements across all documents")
        elif topic == 'fines_procedure':
            recommendations.append("Ensure fines and suspension procedures include notice, hearing, and appeal rights")
        elif topic == 'declaration_supremacy':
            recommendations.append("Review all rules and bylaws for conflicts with declaration provisions")
        elif topic == 'records_access':
            recommendations.append("Add comprehensive records access and inspection provisions")
        elif topic == 'meetings_notice':
            recommendations.append("Review meeting notice and procedure requirements for compliance")
    
    return recommendations


def validate_finding(finding: Finding) -> bool:
    """Validate that a finding has all required fields."""
    required_fields = ['topic', 'severity', 'description', 'evidence']
    
    for field in required_fields:
        if not getattr(finding, field, None):
            return False
    
    return True
