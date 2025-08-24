"""PDF parsing and text extraction module."""

import fitz  # PyMuPDF
import io
import re
import logging
import time
import traceback
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory_usage(stage: str):
    """Log current memory usage for debugging."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"💾 Memory usage at {stage}: {memory_mb:.1f} MB")
    except Exception as e:
        logger.warning(f"⚠️ Could not log memory usage: {e}")

# Check for OCR dependencies with better error handling
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    logger.info("✅ OCR dependencies loaded successfully")
except ImportError as e:
    OCR_AVAILABLE = False
    logger.warning(f"⚠️ OCR dependencies not available: {e}")
    print("Warning: OCR dependencies not available - pdf2image or pytesseract not installed")

# Check for system dependencies
try:
    import subprocess
    result = subprocess.run(['which', 'pdftoppm'], capture_output=True, text=True)
    POPPLER_AVAILABLE = result.returncode == 0
    logger.info(f"📄 Poppler available: {POPPLER_AVAILABLE}")
except Exception as e:
    POPPLER_AVAILABLE = False
    logger.warning(f"⚠️ Poppler check failed: {e}")

try:
    import subprocess
    result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
    TESSERACT_AVAILABLE = result.returncode == 0
    logger.info(f"🔍 Tesseract available: {TESSERACT_AVAILABLE}")
except Exception as e:
    TESSERACT_AVAILABLE = False
    logger.warning(f"⚠️ Tesseract check failed: {e}")

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
    logger.info(f"🔄 Starting PyMuPDF text extraction for: {pdf_path}")
    start_time = time.time()
    
    pages_text = []
    metadata = {}
    
    try:
        logger.info(f"📖 Opening PDF file: {pdf_path}")
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        logger.info(f"📄 PDF has {page_count} pages")
        
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'page_count': page_count
        }
        logger.info(f"📋 PDF metadata: {metadata}")
        
        for page_num in range(page_count):
            try:
                logger.debug(f"📄 Processing page {page_num + 1}/{page_count}")
                page = doc.load_page(page_num)
                text = page.get_text()
                pages_text.append(text)
                logger.debug(f"✅ Page {page_num + 1} processed, text length: {len(text)}")
            except Exception as page_error:
                logger.error(f"❌ Error processing page {page_num + 1}: {page_error}")
                pages_text.append("")  # Add empty text for failed page
        
        doc.close()
        logger.info(f"✅ PyMuPDF extraction completed in {time.time() - start_time:.2f}s")
        logger.info(f"📊 Total text extracted: {sum(len(text) for text in pages_text)} characters")
        
    except Exception as e:
        logger.error(f"❌ Error extracting text with PyMuPDF: {e}")
        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        print(f"Error extracting text with PyMuPDF: {e}")
        return [], {}
    
    return pages_text, metadata


def extract_text_with_ocr(pdf_path: str, progress_callback=None) -> List[str]:
    """Extract text from PDF using OCR (fallback method) with memory-efficient processing."""
    logger.info(f"🔄 Starting OCR text extraction for: {pdf_path}")
    log_memory_usage("OCR start")
    start_time = time.time()
    
    if not OCR_AVAILABLE:
        logger.error("❌ OCR not available - pdf2image or pytesseract not installed")
        print("OCR not available - pdf2image or pytesseract not installed")
        return []
    
    if not POPPLER_AVAILABLE:
        logger.error("❌ Poppler not available - cannot convert PDF to images")
        print("Poppler not available - cannot convert PDF to images")
        return []
    
    if not TESSERACT_AVAILABLE:
        logger.error("❌ Tesseract not available - cannot perform OCR")
        print("Tesseract not available - cannot perform OCR")
        return []
    
    try:
        # First, get the total number of pages without loading all images
        logger.info(f"📄 Getting page count...")
        try:
            page_count = len(convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1))
            logger.info(f"📄 Total pages detected: {page_count}")
        except Exception as e:
            logger.error(f"❌ Failed to get page count: {e}")
            # Try with lower DPI if page count detection fails
            try:
                page_count = len(convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1))
                logger.info(f"📄 Total pages detected (low DPI): {page_count}")
            except Exception as e2:
                logger.error(f"❌ Failed to get page count even with low DPI: {e2}")
                raise Exception(f"Could not process PDF: {e2}")
        
        # Adjust DPI based on page count to prevent memory issues
        if page_count > 100:
            dpi = 200  # Lower DPI for very large documents
            logger.info(f"📊 Large document detected ({page_count} pages), using DPI=200")
        elif page_count > 50:
            dpi = 250  # Medium DPI for medium documents
            logger.info(f"📊 Medium document detected ({page_count} pages), using DPI=250")
        else:
            dpi = 300  # Full DPI for small documents
            logger.info(f"📊 Small document detected ({page_count} pages), using DPI=300")
        
        # Adjust batch size based on page count
        if page_count > 200:
            batch_size = 3  # Smaller batches for very large documents
        elif page_count > 100:
            batch_size = 4  # Medium batches for large documents
        else:
            batch_size = 5  # Normal batches for smaller documents
        
        pages_text = []
        
        # Process pages in smaller batches to avoid memory issues
        total_batches = (page_count + batch_size - 1) // batch_size
        
        logger.info(f"🔄 Processing {page_count} pages in {total_batches} batches of {batch_size}")
        
        for batch_num in range(total_batches):
            start_page = batch_num * batch_size + 1
            end_page = min((batch_num + 1) * batch_size, page_count)
            
            logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches}: pages {start_page}-{end_page}")
            
            try:
                # Convert only this batch of pages to images
                images = convert_from_path(
                    pdf_path, 
                    dpi=dpi, 
                    first_page=start_page, 
                    last_page=end_page
                )
                
                # Process each page in the batch
                for i, image in enumerate(images):
                    page_num = start_page + i
                    try:
                        logger.debug(f"🔍 Processing OCR for page {page_num}/{page_count}")
                        page_start_time = time.time()
                        
                        # Update progress if callback provided
                        if progress_callback:
                            progress = (page_num / page_count) * 100
                            progress_callback(progress, f"Processing page {page_num} of {page_count}")
                        
                        # Extract text using OCR
                        text = pytesseract.image_to_string(image)
                        pages_text.append(text)
                        
                        page_time = time.time() - page_start_time
                        logger.debug(f"✅ Page {page_num} OCR completed in {page_time:.2f}s, text length: {len(text)}")
                        
                        # Force garbage collection to free memory
                        import gc
                        gc.collect()
                        
                    except Exception as page_error:
                        logger.error(f"❌ Error processing OCR for page {page_num}: {page_error}")
                        logger.error(f"📋 Page error traceback: {traceback.format_exc()}")
                        pages_text.append("")  # Add empty text for failed page
                
                # Clear images from memory after processing batch
                del images
                import gc
                gc.collect()
                
                log_memory_usage(f"after batch {batch_num + 1}")
                logger.info(f"✅ Batch {batch_num + 1} completed, processed {end_page - start_page + 1} pages")
                
            except Exception as batch_error:
                logger.error(f"❌ Error processing batch {batch_num + 1}: {batch_error}")
                logger.error(f"📋 Batch error traceback: {traceback.format_exc()}")
                # Add empty text for failed pages in this batch
                for _ in range(end_page - start_page + 1):
                    pages_text.append("")
        
        # Final progress update
        if progress_callback:
            progress_callback(100, "OCR processing completed")
        
        total_time = time.time() - start_time
        log_memory_usage("OCR completion")
        logger.info(f"✅ OCR extraction completed in {total_time:.2f}s")
        logger.info(f"📊 Total text extracted: {sum(len(text) for text in pages_text)} characters")
        
        return pages_text
        
    except Exception as e:
        logger.error(f"❌ Error extracting text with OCR: {e}")
        logger.error(f"📋 Full OCR error traceback: {traceback.format_exc()}")
        print(f"Error extracting text with OCR: {e}")
        return []


def should_use_ocr(pages_text: List[str]) -> bool:
    """Determine if OCR should be used based on text density and system resources."""
    if not pages_text:
        logger.info("📄 No text extracted, will use OCR")
        return True
    
    # Check system memory before deciding on OCR
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_available_gb = memory.available / (1024**3)
        logger.info(f"💾 Available memory: {memory_available_gb:.1f} GB")
        
        # Skip OCR if memory is low (less than 1GB available)
        if memory_available_gb < 1.0:
            logger.warning(f"⚠️ Low memory detected ({memory_available_gb:.1f} GB), skipping OCR")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Could not check memory: {e}")
    
    # Calculate average text density
    total_chars = sum(len(text) for text in pages_text)
    avg_density = total_chars / len(pages_text)
    
    logger.info(f"📊 Text density analysis: {total_chars} total chars, {len(pages_text)} pages, {avg_density:.1f} chars/page")
    
    # If average density is less than 100 characters per page, use OCR
    should_ocr = avg_density < 100
    logger.info(f"🔍 OCR decision: {'Use OCR' if should_ocr else 'Use PyMuPDF text'} (threshold: 100 chars/page)")
    
    return should_ocr


def parse_pdf(file_path: str, file_name: str, progress_callback=None) -> Document:
    """Parse PDF file and return Document object."""
    logger.info(f"🚀 Starting PDF parsing for: {file_name}")
    logger.info(f"📁 File path: {file_path}")
    start_time = time.time()
    
    document = Document(file_name)
    
    try:
        # Try PyMuPDF first
        logger.info(f"📖 Attempting PyMuPDF extraction...")
        pages_text, metadata = extract_text_with_pymupdf(file_path)
        document.meta = metadata
        
        # Check if we need OCR and if it's available
        if should_use_ocr(pages_text) and OCR_AVAILABLE and POPPLER_AVAILABLE and TESSERACT_AVAILABLE:
            logger.info(f"🔍 Low text density detected for {file_name}, switching to OCR...")
            print(f"Low text density detected for {file_name}, using OCR...")
            
            # Add timeout for OCR processing to prevent hanging (thread-safe approach)
            import threading
            import time
            
            ocr_result = []
            ocr_error = []
            ocr_completed = threading.Event()
            
            def ocr_worker():
                try:
                    result = extract_text_with_ocr(file_path, progress_callback)
                    ocr_result.append(result)
                except Exception as e:
                    ocr_error.append(e)
                finally:
                    ocr_completed.set()
            
            # Start OCR in a separate thread
            ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
            ocr_thread.start()
            
            # Wait for completion with timeout (10 minutes)
            if ocr_completed.wait(timeout=600):  # 10 minutes timeout
                if ocr_error:
                    logger.error(f"❌ OCR processing failed for {file_name}: {ocr_error[0]}")
                    print(f"OCR processing failed for {file_name}, using PyMuPDF text")
                elif ocr_result and ocr_result[0]:
                    pages_text = ocr_result[0]
                    document.ocr_used = True
                    logger.info(f"✅ OCR completed successfully for {file_name}")
                    print(f"OCR completed for {file_name}")
                else:
                    logger.warning(f"⚠️ OCR failed for {file_name}, falling back to PyMuPDF text")
                    print(f"OCR failed for {file_name}, using PyMuPDF text")
            else:
                logger.error(f"❌ OCR processing timed out for {file_name}")
                print(f"OCR processing timed out for {file_name}, using PyMuPDF text")
                
        elif should_use_ocr(pages_text) and not (OCR_AVAILABLE and POPPLER_AVAILABLE and TESSERACT_AVAILABLE):
            logger.warning(f"⚠️ Low text density detected for {file_name}, but OCR not available. Using PyMuPDF text.")
            print(f"Low text density detected for {file_name}, but OCR not available. Using PyMuPDF text.")
        
        # Add pages to document
        logger.info(f"📄 Adding {len(pages_text)} pages to document...")
        for page_num, text in enumerate(pages_text, 1):
            document.add_page(page_num, text, document.ocr_used)
        
        # Calculate text density
        document.calculate_text_density()
        logger.info(f"📊 Final text density: {document.text_density:.1f} chars/page")
        
        total_time = time.time() - start_time
        logger.info(f"✅ PDF parsing completed in {total_time:.2f}s")
        logger.info(f"📋 Document summary: {len(pages_text)} pages, {sum(len(text) for text in pages_text)} total chars")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ Error in parse_pdf for {file_name}: {e}")
        logger.error(f"📋 Full parse_pdf traceback: {traceback.format_exc()}")
        raise


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
    logger.info(f"🔍 Validating PDF file: {file_path}")
    
    try:
        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"❌ File does not exist: {file_path}")
            return False, "File does not exist"
        
        # Check file size
        file_size = Path(file_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"📊 File size: {file_size_mb:.2f} MB")
        
        if file_size == 0:
            logger.error("❌ File is empty")
            return False, "File is empty"
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            logger.error(f"❌ File size exceeds 100MB limit: {file_size_mb:.2f} MB")
            return False, "File size exceeds 100MB limit"
        
        # Try to open with PyMuPDF
        logger.info("📖 Attempting to open PDF with PyMuPDF...")
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        
        logger.info(f"📄 PDF page count: {page_count}")
        
        if page_count == 0:
            logger.error("❌ PDF has no pages")
            return False, "PDF has no pages"
        
        if page_count > 1000:  # Reasonable page limit
            logger.error(f"❌ PDF has too many pages: {page_count}")
            return False, f"PDF has too many pages ({page_count})"
        
        logger.info(f"✅ PDF validation successful: {page_count} pages, {file_size_mb:.2f} MB")
        return True, f"Valid PDF with {page_count} pages"
        
    except fitz.FileDataError as e:
        logger.error(f"❌ PDF file is corrupted or password protected: {str(e)}")
        return False, f"PDF file is corrupted or password protected: {str(e)}"
    except Exception as e:
        logger.error(f"❌ PDF validation error: {str(e)}")
        logger.error(f"📋 Validation error traceback: {traceback.format_exc()}")
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
