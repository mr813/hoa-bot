"""Utility functions for HOA Auditor."""

import re
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and return JSON file contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
    
    return text


def extract_page_numbers(text: str) -> List[int]:
    """Extract page numbers from text."""
    page_patterns = [
        r'page\s+(\d+)',
        r'p\.\s*(\d+)',
        r'(\d+)\s*of\s*\d+',  # "1 of 50"
    ]
    
    pages = set()
    for pattern in page_patterns:
        matches = re.findall(pattern, text.lower())
        pages.update(int(match) for match in matches)
    
    return sorted(list(pages))


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to specified length, preserving word boundaries."""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can find a good break point
        return truncated[:last_space] + "..."
    
    return truncated + "..."


def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return Path(filename).suffix.lower()


def is_pdf_file(filename: str) -> bool:
    """Check if file is a PDF."""
    return get_file_extension(filename) == '.pdf'


def format_statute_reference(statute_ref: str) -> str:
    """Format statute reference for display."""
    # Convert "718.112(2)(c)" to "§ 718.112(2)(c)"
    if statute_ref.startswith('718.'):
        return f"§ {statute_ref}"
    return statute_ref


def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        'likely_conflict': '#ff4b4b',  # Red
        'needs_counsel': '#ffa500',    # Orange
        'likely_compliant': '#00ff00', # Green
        'info': '#0066cc'              # Blue
    }
    return colors.get(severity, '#666666')


def get_severity_icon(severity: str) -> str:
    """Get icon for severity level."""
    icons = {
        'likely_conflict': '🚨',
        'needs_counsel': '⚠️',
        'likely_compliant': '✅',
        'info': 'ℹ️'
    }
    return icons.get(severity, '📄')


def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 100:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:100-len(ext)] + ext
    return sanitized


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def get_app_version() -> str:
    """Get application version."""
    return "1.0.0"


def get_assets_path() -> Path:
    """Get path to assets directory."""
    return Path(__file__).parent.parent / "assets"


def get_templates_path() -> Path:
    """Get path to templates directory."""
    return Path(__file__).parent.parent / "templates"
