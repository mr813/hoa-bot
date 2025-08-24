"""PDF parsing and text extraction module."""

import fitz  # PyMuPDF
import io
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Check for OCR dependencies with better error handling
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR dependencies not available - pdf2image or pytesseract not installed")

# Check for system dependencies
try:
    import subprocess
    result = subprocess.run(['which', 'pdftoppm'], capture_output=True, text=True)
    POPPLER_AVAILABLE = result.returncode == 0
except:
    POPPLER_AVAILABLE = False

try:
    import subprocess
    result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
    TESSERACT_AVAILABLE = result.returncode == 0
except:
    TESSERACT_AVAILABLE = False

from app.utils import clean_text, truncate_text


class Document:
    """Document class to hold parsed PDF data."""
    
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.pages: List[Dict[str, Any]] = []
        self.raw_text = ""
        self.meta: Dict[str, Any] = {}
        self.text_density = 0.0
        self.ocr_used = False
    
    def add_page(self, page_num: int, text: str, ocr_used: bool = False):
        """Add a page to the document."""
        self.pages.append({
            'no': page_num,
            'text': clean_text(text),
            'ocr_used': ocr_used
        })
    
    def get_page_text(self, page_num: int) -> str:
        """Get text from specific page."""
        for page in self.pages:
            if page['no'] == page_num:
                return page['text']
        return ""
    
    def get_all_text(self) -> str:
        """Get all text from document."""
        if not self.raw_text:
            self.raw_text = " ".join(page['text'] for page in self.pages)
        return self.raw_text
    
    def calculate_text_density(self) -> float:
        """Calculate text density (characters per page)."""
        if not self.pages:
            return 0.0
        
        total_chars = sum(len(page['text']) for page in self.pages)
        self.text_density = total_chars / len(self.pages)
        return self.text_density


def extract_text_with_pymupdf(pdf_path: str) -> Tuple[List[str], Dict[str, Any]]:
    """Extract text from PDF using PyMuPDF."""
    pages_text = []
    metadata = {}
    
    try:
        doc = fitz.open(pdf_path)
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'page_count': len(doc)
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages_text.append(text)
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {e}")
        return [], {}
    
    return pages_text, metadata


def extract_text_with_ocr(pdf_path: str, progress_callback=None) -> List[str]:
    """Extract text from PDF using OCR (fallback method)."""
    if not OCR_AVAILABLE:
        print("OCR not available - pdf2image or pytesseract not installed")
        return []
    
    if not POPPLER_AVAILABLE:
        print("Poppler not available - cannot convert PDF to images")
        return []
    
    if not TESSERACT_AVAILABLE:
        print("Tesseract not available - cannot perform OCR")
        return []
    
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        pages_text = []
        total_pages = len(images)
        
        for i, image in enumerate(images):
            # Update progress if callback provided
            if progress_callback:
                progress = (i / total_pages) * 100
                progress_callback(progress, f"Processing page {i+1} of {total_pages}")
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            pages_text.append(text)
        
        # Final progress update
        if progress_callback:
            progress_callback(100, "OCR processing completed")
        
        return pages_text
        
    except Exception as e:
        print(f"Error extracting text with OCR: {e}")
        return []


def should_use_ocr(pages_text: List[str]) -> bool:
    """Determine if OCR should be used based on text density."""
    if not pages_text:
        return True
    
    # Calculate average text density
    total_chars = sum(len(text) for text in pages_text)
    avg_density = total_chars / len(pages_text)
    
    # If average density is less than 100 characters per page, use OCR
    return avg_density < 100


def parse_pdf(file_path: str, file_name: str, progress_callback=None) -> Document:
    """Parse PDF file and return Document object."""
    document = Document(file_name)
    
    # Try PyMuPDF first
    pages_text, metadata = extract_text_with_pymupdf(file_path)
    document.meta = metadata
    
    # Check if we need OCR and if it's available
    if should_use_ocr(pages_text) and OCR_AVAILABLE and POPPLER_AVAILABLE and TESSERACT_AVAILABLE:
        print(f"Low text density detected for {file_name}, using OCR...")
        ocr_pages_text = extract_text_with_ocr(file_path, progress_callback)
        
        if ocr_pages_text:
            pages_text = ocr_pages_text
            document.ocr_used = True
            print(f"OCR completed for {file_name}")
        else:
            print(f"OCR failed for {file_name}, using PyMuPDF text")
    elif should_use_ocr(pages_text) and not (OCR_AVAILABLE and POPPLER_AVAILABLE and TESSERACT_AVAILABLE):
        print(f"Low text density detected for {file_name}, but OCR not available. Using PyMuPDF text.")
    
    # Add pages to document
    for page_num, text in enumerate(pages_text, 1):
        document.add_page(page_num, text, document.ocr_used)
    
    # Calculate text density
    document.calculate_text_density()
    
    return document


def extract_sections_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract section headings and content from text."""
    sections = []
    
    # Common section patterns
    section_patterns = [
        r'(?:ARTICLE|Article)\s+([IVX]+|[A-Z]|\d+)\.?\s*[-–—]?\s*(.+)',
        r'(?:SECTION|Section)\s+(\d+[A-Z]?\.?\d*)\.?\s*[-–—]?\s*(.+)',
        r'(\d+\.)\s*(.+)',
        r'([A-Z][A-Z\s]+):\s*(.+)',
    ]
    
    lines = text.split('\n')
    current_section = None
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check for section headers
        for pattern in section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                
                # Close previous section
                if current_section:
                    current_section['end_line'] = line_num - 1
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'number': section_num,
                    'title': section_title,
                    'start_line': line_num,
                    'content': []
                }
                break
        
        # Add line to current section
        if current_section:
            current_section['content'].append(line)
    
    # Add final section
    if current_section:
        current_section['end_line'] = len(lines) - 1
        sections.append(current_section)
    
    return sections


def determine_section_level(section_num: str) -> int:
    """Determine the hierarchical level of a section."""
    # Roman numerals (I, II, III) = level 1
    if re.match(r'^[IVX]+$', section_num, re.IGNORECASE):
        return 1
    
    # Letters (A, B, C) = level 2
    if re.match(r'^[A-Z]$', section_num):
        return 2
    
    # Numbers with dots (1., 2.) = level 1
    if re.match(r'^\d+\.$', section_num):
        return 1
    
    # Numbers with letters (1A, 2B) = level 2
    if re.match(r'^\d+[A-Z]$', section_num):
        return 2
    
    # Default to level 1
    return 1


def extract_key_values(text: str) -> Dict[str, Any]:
    """Extract key values from document text."""
    key_values = {}
    
    # Rental minimum patterns
    rental_patterns = [
        r'minimum\s+(?:rental|lease)\s+(?:term|duration|period)\s*(?:of\s*)?(\d+)\s*(?:month|year)',
        r'(\d+)\s*(?:month|year)\s+minimum\s+(?:rental|lease)',
        r'no\s+(?:rental|lease)\s+(?:less\s+than|under)\s+(\d+)\s*(?:month|year)',
    ]
    
    for pattern in rental_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            key_values['rental_minimum'] = int(match.group(1))
            break
    
    # Transfer fee patterns
    fee_patterns = [
        r'transfer\s+fee\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*transfer\s+fee',
        r'sale\s+fee\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
    ]
    
    for pattern in fee_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            key_values['transfer_fee_percent'] = float(match.group(1))
            break
    
    # Fine amount patterns
    fine_patterns = [
        r'fine\s*(?:of\s*)?\$?(\d+(?:,\d+)?)',
        r'\$?(\d+(?:,\d+)?)\s*fine',
        r'maximum\s+fine\s*(?:of\s*)?\$?(\d+(?:,\d+)?)',
    ]
    
    for pattern in fine_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            key_values['max_fine_amount'] = int(amount_str)
            break
    
    return key_values


def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """Validate that file is a readable PDF and return status with message."""
    try:
        # Check if file exists
        if not Path(file_path).exists():
            return False, "File does not exist"
        
        # Check file size
        file_size = Path(file_path).stat().st_size
        if file_size == 0:
            return False, "File is empty"
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, "File size exceeds 100MB limit"
        
        # Try to open with PyMuPDF
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        
        if page_count == 0:
            return False, "PDF has no pages"
        
        if page_count > 1000:  # Reasonable page limit
            return False, f"PDF has too many pages ({page_count})"
        
        return True, f"Valid PDF with {page_count} pages"
        
    except fitz.FileDataError as e:
        return False, f"PDF file is corrupted or password protected: {str(e)}"
    except Exception as e:
        # Check if it's a file type error by examining the error message
        error_msg = str(e).lower()
        if "not a pdf" in error_msg or "invalid" in error_msg or "cannot open" in error_msg:
            return False, f"File is not a valid PDF: {str(e)}"
        else:
            return False, f"PDF validation failed: {str(e)}"


def get_pdf_info(file_path: str) -> Dict[str, Any]:
    """Get basic information about a PDF file."""
    try:
        doc = fitz.open(file_path)
        info = {
            'page_count': len(doc),
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'file_size': Path(file_path).stat().st_size
        }
        doc.close()
        return info
    except Exception as e:
        print(f"Error getting PDF info: {e}")
        return {}
