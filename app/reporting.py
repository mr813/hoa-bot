"""Reporting module for generating findings reports and board letters."""

import json
import markdown
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from app.utils import get_templates_path, save_json_file, format_statute_reference


class ReportGenerator:
    """Class to generate various types of reports."""
    
    def __init__(self):
        self.templates_path = get_templates_path()
        self.jinja_env = None
        
        if JINJA2_AVAILABLE:
            self.jinja_env = Environment(loader=FileSystemLoader(str(self.templates_path)))
    
    def generate_findings_report(self, findings_data: Dict[str, Any], output_format: str = 'markdown') -> str:
        """Generate findings report in specified format."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_report(findings_data)
        
        # Load template
        template = self.jinja_env.get_template('report.md.j2')
        
        # Prepare template data
        template_data = {
            'generated_date': datetime.now().strftime('%B %d, %Y'),
            'summary': findings_data.get('summary', {}),
            'findings': findings_data.get('findings', []),
            'top_findings': findings_data.get('top_findings', []),
            'by_severity': findings_data.get('by_severity', {}),
            'by_topic': findings_data.get('by_topic', {}),
            'format_statute_ref': format_statute_reference
        }
        
        # Render template
        markdown_content = template.render(**template_data)
        
        if output_format == 'html':
            return self._convert_markdown_to_html(markdown_content)
        elif output_format == 'pdf':
            return self._convert_markdown_to_pdf(markdown_content)
        else:
            return markdown_content
    
    def generate_board_letter(self, findings_data: Dict[str, Any], 
                            association_name: str = "Board of Directors",
                            owner_name: str = "Unit Owner") -> str:
        """Generate draft board letter."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_letter(findings_data, association_name, owner_name)
        
        # Load template
        template = self.jinja_env.get_template('board_letter.md.j2')
        
        # Get top issues for letter
        top_issues = findings_data.get('top_findings', [])[:3]
        
        # Prepare template data
        template_data = {
            'generated_date': datetime.now().strftime('%B %d, %Y'),
            'association_name': association_name,
            'owner_name': owner_name,
            'top_issues': top_issues,
            'total_findings': findings_data.get('summary', {}).get('total_findings', 0),
            'critical_count': len(findings_data.get('by_severity', {}).get('likely_conflict', [])),
            'format_statute_ref': format_statute_reference
        }
        
        # Render template
        return template.render(**template_data)
    
    def export_findings_json(self, findings_data: Dict[str, Any], file_path: str) -> bool:
        """Export findings as JSON file."""
        return save_json_file(findings_data, file_path)
    
    def generate_pdf_report(self, findings_data: Dict[str, Any], file_path: str) -> bool:
        """Generate PDF report."""
        try:
            markdown_content = self.generate_findings_report(findings_data, 'markdown')
            
            if WEASYPRINT_AVAILABLE:
                return self._generate_pdf_with_weasyprint(markdown_content, file_path)
            elif REPORTLAB_AVAILABLE:
                return self._generate_pdf_with_reportlab(findings_data, file_path)
            else:
                return False
                
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False
    
    def _generate_simple_report(self, findings_data: Dict[str, Any]) -> str:
        """Generate simple report without Jinja2."""
        report = []
        report.append("# HOA Compliance Audit Report")
        report.append(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
        report.append("")
        
        # Summary
        summary = findings_data.get('summary', {})
        report.append("## Executive Summary")
        report.append(f"- Total Findings: {summary.get('total_findings', 0)}")
        
        severity_counts = summary.get('severity_counts', {})
        for severity, count in severity_counts.items():
            report.append(f"- {severity.replace('_', ' ').title()}: {count}")
        
        report.append("")
        
        # Top Findings
        report.append("## Top Findings")
        top_findings = findings_data.get('top_findings', [])
        for i, finding in enumerate(top_findings[:5], 1):
            report.append(f"### {i}. {finding.get('description', '')}")
            report.append(f"**Severity:** {finding.get('severity', '')}")
            report.append(f"**Evidence:** {finding.get('evidence_snippet', '')}")
            report.append(f"**Suggestion:** {finding.get('suggestion', '')}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_simple_letter(self, findings_data: Dict[str, Any], 
                               association_name: str, owner_name: str) -> str:
        """Generate simple board letter without Jinja2."""
        letter = []
        letter.append(f"Dear {association_name},")
        letter.append("")
        letter.append(f"I am writing to bring to your attention several compliance issues identified during my review of our condominium association's governing documents against Florida Chapter 718 requirements.")
        letter.append("")
        
        # Add top issues
        top_issues = findings_data.get('top_findings', [])[:3]
        for issue in top_issues:
            letter.append(f"- {issue.get('description', '')}")
        
        letter.append("")
        letter.append("I believe these issues warrant your attention and may require consultation with legal counsel.")
        letter.append("")
        letter.append("I would welcome the opportunity to discuss these findings and work collaboratively to ensure our association's compliance with Florida law.")
        letter.append("")
        letter.append(f"Sincerely,")
        letter.append(f"{owner_name}")
        letter.append(f"{datetime.now().strftime('%B %d, %Y')}")
        
        return "\n".join(letter)
    
    def _convert_markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML."""
        html_content = markdown.markdown(markdown_content)
        
        # Add basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>HOA Compliance Audit Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .finding {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007cba; background: #f9f9f9; }}
                .severity-critical {{ border-left-color: #dc3545; }}
                .severity-high {{ border-left-color: #fd7e14; }}
                .severity-medium {{ border-left-color: #ffc107; }}
                .severity-low {{ border-left-color: #28a745; }}
                .ai-badge {{ background: #007cba; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        return styled_html
    
    def _convert_markdown_to_pdf(self, markdown_content: str) -> str:
        """Convert markdown to PDF (returns HTML for PDF generation)."""
        return self._convert_markdown_to_html(markdown_content)
    
    def _generate_pdf_with_weasyprint(self, html_content: str, file_path: str) -> bool:
        """Generate PDF using WeasyPrint."""
        try:
            HTML(string=html_content).write_pdf(file_path)
            return True
        except Exception as e:
            print(f"WeasyPrint PDF generation failed: {e}")
            return False
    
    def _generate_pdf_with_reportlab(self, findings_data: Dict[str, Any], file_path: str) -> bool:
        """Generate PDF using ReportLab."""
        try:
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("HOA Compliance Audit Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Date
            date_text = f"Generated: {datetime.now().strftime('%B %d, %Y')}"
            date_para = Paragraph(date_text, styles['Normal'])
            story.append(date_para)
            story.append(Spacer(1, 12))
            
            # Summary
            summary = findings_data.get('summary', {})
            summary_title = Paragraph("Executive Summary", styles['Heading2'])
            story.append(summary_title)
            story.append(Spacer(1, 12))
            
            summary_text = f"Total Findings: {summary.get('total_findings', 0)}"
            summary_para = Paragraph(summary_text, styles['Normal'])
            story.append(summary_para)
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"ReportLab PDF generation failed: {e}")
            return False
    
    def create_report_package(self, findings_data: Dict[str, Any], 
                            output_dir: str, 
                            association_name: str = "Board of Directors",
                            owner_name: str = "Unit Owner") -> Dict[str, str]:
        """Create a complete report package with multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        package_files = {}
        
        # Generate JSON export
        json_path = output_path / "findings.json"
        if self.export_findings_json(findings_data, str(json_path)):
            package_files['json'] = str(json_path)
        
        # Generate Markdown report
        markdown_content = self.generate_findings_report(findings_data, 'markdown')
        md_path = output_path / "audit_report.md"
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            package_files['markdown'] = str(md_path)
        except Exception as e:
            print(f"Error writing markdown file: {e}")
        
        # Generate HTML report
        html_content = self.generate_findings_report(findings_data, 'html')
        html_path = output_path / "audit_report.html"
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            package_files['html'] = str(html_path)
        except Exception as e:
            print(f"Error writing HTML file: {e}")
        
        # Generate PDF report
        pdf_path = output_path / "audit_report.pdf"
        if self.generate_pdf_report(findings_data, str(pdf_path)):
            package_files['pdf'] = str(pdf_path)
        
        # Generate board letter
        letter_content = self.generate_board_letter(findings_data, association_name, owner_name)
        letter_path = output_path / "board_letter.md"
        try:
            with open(letter_path, 'w', encoding='utf-8') as f:
                f.write(letter_content)
            package_files['letter'] = str(letter_path)
        except Exception as e:
            print(f"Error writing board letter: {e}")
        
        return package_files
