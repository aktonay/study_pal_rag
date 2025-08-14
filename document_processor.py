import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import PyPDF2
import fitz  # PyMuPDF
import pdfplumber
from docx import Document
from typing import List, Dict
import tiktoken
import hashlib
import json
from datetime import datetime
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Initialize OCR reader if available
        self.ocr_reader = None
        self.ocr_available = OCR_AVAILABLE  # Use instance variable
        if self.ocr_available:
            try:
                # Initialize with English and Bengali language support
                self.ocr_reader = easyocr.Reader(['en', 'bn'], gpu=False)
                print("✓ EasyOCR initialized with English and Bengali support")
            except Exception as e:
                print(f"Warning: Could not initialize EasyOCR: {e}")
                self.ocr_available = False  # Use instance variable
        
        # Create persistent storage directory
        self.storage_dir = "document_storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Document registry to track processed documents
        self.registry_file = os.path.join(self.storage_dir, "document_registry.json")
        self.document_registry = self.load_document_registry()
    
    def load_document_registry(self) -> Dict:
        """Load document registry from file"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading document registry: {e}")
        return {}
    
    def save_document_registry(self):
        """Save document registry to file"""
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving document registry: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content to detect duplicates"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_document_processed(self, file_path: str, filename: str) -> bool:
        """Check if document is already processed"""
        try:
            file_hash = self.get_file_hash(file_path)
            
            # Check by hash first (most reliable)
            for doc_id, doc_info in self.document_registry.items():
                if doc_info.get('file_hash') == file_hash:
                    print(f"Document {filename} already processed (same content)")
                    return True
            
            # Check by filename (less reliable but useful)
            for doc_id, doc_info in self.document_registry.items():
                if doc_info.get('original_filename') == filename:
                    # Same filename, check if content is similar
                    if doc_info.get('file_size') == os.path.getsize(file_path):
                        print(f"Document {filename} likely already processed (same name and size)")
                        return True
            
            return False
        except Exception as e:
            print(f"Error checking if document is processed: {e}")
            return False
    
    def extract_text_with_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (most robust method)"""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n" + page_text + "\n"
            doc.close()
            return text
        except Exception as e:
            print(f"PyMuPDF extraction failed for {file_path}: {e}")
            return ""
    
    def extract_text_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber (good for tables and complex layouts)"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n" + page_text + "\n"
            return text
        except Exception as e:
            print(f"pdfplumber extraction failed for {file_path}: {e}")
            return ""
    
    def extract_text_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n" + page_text + "\n"
            return text
        except Exception as e:
            print(f"PyPDF2 extraction failed for {file_path}: {e}")
            return ""
    
    def extract_text_with_easyocr_from_image(self, image_bytes: bytes, page_num: int = 1) -> str:
        """Extract text from image bytes using EasyOCR"""
        if not self.ocr_reader:
            return ""
        
        try:
            # EasyOCR can work directly with image bytes
            result = self.ocr_reader.readtext(image_bytes, detail=0, paragraph=True)
            if result:
                text = ' '.join(result)
                return f"\n--- Page {page_num} ---\n" + text + "\n"
            return ""
        except Exception as e:
            print(f"EasyOCR failed for page {page_num}: {e}")
            return ""
    
    def extract_text_with_easyocr_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using EasyOCR on each page as image"""
        if not self.ocr_available or not self.ocr_reader:
            print("❌ EasyOCR not available. Install with: pip install easyocr")
            return ""
        
        text = ""
        try:
            print("🔍 Attempting EasyOCR extraction...")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                print(f"🔍 Processing page {page_num + 1} with EasyOCR...")
                
                # Convert page to image
                page = doc.load_page(page_num)
                
                # Get image with higher DPI for better OCR accuracy
                matrix = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=matrix)
                
                # Convert to bytes
                img_bytes = pix.tobytes("png")
                
                # Extract text using EasyOCR
                page_text = self.extract_text_with_easyocr_from_image(img_bytes, page_num + 1)
                
                if page_text.strip():
                    text += page_text
            
            doc.close()
            return text
            
        except Exception as e:
            print(f"EasyOCR extraction failed for {file_path}: {e}")
            return ""
    
    def extract_text_from_image_file(self, image_path: str) -> str:
        """Extract text from standalone image file using EasyOCR"""
        if not self.ocr_available or not self.ocr_reader:
            print("❌ EasyOCR not available. Install with: pip install easyocr")
            return ""
        
        try:
            print(f"🔍 Extracting text from image: {os.path.basename(image_path)}")
            result = self.ocr_reader.readtext(image_path, detail=0, paragraph=True)
            
            if result:
                text = ' '.join(result)
                print(f"✓ Successfully extracted text using EasyOCR ({len(text)} characters)")
                return text
            else:
                print("❌ No text found in image")
                return ""
                
        except Exception as e:
            print(f"EasyOCR failed for image {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using multiple methods with EasyOCR fallback"""
        print(f"Attempting to extract text from: {os.path.basename(file_path)}")
        
        # Method 1: PyMuPDF (most reliable for text-based PDFs)
        text = self.extract_text_with_pymupdf(file_path)
        if text.strip():
            print(f"✓ Successfully extracted text using PyMuPDF ({len(text)} characters)")
            return text
        
        # Method 2: pdfplumber (good for complex layouts)
        print("PyMuPDF failed, trying pdfplumber...")
        text = self.extract_text_with_pdfplumber(file_path)
        if text.strip():
            print(f"✓ Successfully extracted text using pdfplumber ({len(text)} characters)")
            return text
        
        # Method 3: PyPDF2 (basic fallback)
        print("pdfplumber failed, trying PyPDF2...")
        text = self.extract_text_with_pypdf2(file_path)
        if text.strip():
            print(f"✓ Successfully extracted text using PyPDF2 ({len(text)} characters)")
            return text
        
        # Method 4: EasyOCR (for scanned PDFs)
        print("All text extraction methods failed, trying EasyOCR...")
        text = self.extract_text_with_easyocr_from_pdf(file_path)
        if text.strip():
            print(f"✓ Successfully extracted text using EasyOCR ({len(text)} characters)")
            return text
        
        # If all methods fail
        print(f"❌ All extraction methods (including EasyOCR) failed for {os.path.basename(file_path)}")
        
        # Additional debugging information
        try:
            doc = fitz.open(file_path)
            print(f"PDF Info: {len(doc)} pages, encrypted: {doc.is_encrypted}")
            
            # Check if pages contain text or just images
            for i in range(min(3, len(doc))):  # Check first 3 pages
                page = doc.load_page(i)
                text_dict = page.get_text("dict")
                has_text = any(block.get("type") == 0 for block in text_dict.get("blocks", []))
                has_images = any(block.get("type") == 1 for block in text_dict.get("blocks", []))
                print(f"Page {i+1}: Contains text blocks: {has_text}, Contains images: {has_images}")
            
            doc.close()
        except Exception as e:
            print(f"Could not analyze PDF structure: {e}")
        
        return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for para_num, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            print(f"✓ Successfully extracted text from DOCX ({len(text)} characters)")
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for both English and Bengali"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Bengali characters
        # Keep alphanumeric, Bengali unicode range, punctuation, and spaces
        text = re.sub(r'[^\w\s\u0980-\u09FF।,;:.!?()[\]{}\'\""-]', '', text)
        
        # Remove excessive newlines but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        return text.strip()
    
    def detect_language_content(self, text: str) -> str:
        """Detect primary language of document content"""
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        english_chars = sum(1 for char in text if char.isalpha() and not ('\u0980' <= char <= '\u09FF'))
        total_chars = bengali_chars + english_chars
        
        if total_chars == 0:
            return "unknown"
        
        bengali_ratio = bengali_chars / total_chars
        
        if bengali_ratio > 0.4:
            return "bengali"
        elif bengali_ratio < 0.1:
            return "english"
        else:
            return "mixed"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except:
            return len(text.split())  # Fallback for non-English text
    
    def chunk_text_smart(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Smart chunking that respects language boundaries"""
        if not text.strip():
            return []
        
        # Detect primary language
        primary_language = self.detect_language_content(text)
        
        # Split by different delimiters based on language
        if primary_language == "bengali" or primary_language == "mixed":
            # Bengali text uses । as sentence delimiter
            sentences = re.split(r'[।.!?]+', text)
        else:
            # English text
            sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self.count_tokens(sentence)
            
            # If sentence is too long, split it further
            if sentence_tokens > self.chunk_size:
                # Split long sentence by commas or other delimiters
                sub_sentences = re.split(r'[,;:]', sentence)
                for sub_sentence in sub_sentences:
                    sub_sentence = sub_sentence.strip()
                    if sub_sentence:
                        self._add_sentence_to_chunk(sub_sentence, chunks, current_chunk, current_tokens, metadata)
            else:
                # Regular sentence processing
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_data = {
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        'language': self.detect_language_content(current_chunk),
                        'metadata': metadata or {}
                    }
                    chunks.append(chunk_data)
                    
                    # Start new chunk with overlap
                    overlap_text = self.get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'language': self.detect_language_content(current_chunk),
                'metadata': metadata or {}
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _add_sentence_to_chunk(self, sentence: str, chunks: List[Dict], current_chunk: str, current_tokens: int, metadata: Dict):
        """Helper method to add sentence to chunk (used for long sentences)"""
        sentence_tokens = self.count_tokens(sentence)
        
        if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
            # Save current chunk
            chunk_data = {
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'language': self.detect_language_content(current_chunk),
                'metadata': metadata or {}
            }
            chunks.append(chunk_data)
            
            # Start new chunk
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens
    
    def get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= self.chunk_overlap:
                return text
            
            overlap_tokens = tokens[-self.chunk_overlap:]
            return self.encoding.decode(overlap_tokens)
        except:
            # Fallback for non-English text
            words = text.split()
            if len(words) <= self.chunk_overlap // 4:  # Approximate token to word ratio
                return text
            return " ".join(words[-(self.chunk_overlap // 4):])
    
    def process_document(self, file_path: str, force_reprocess: bool = False) -> List[Dict]:
        """Main method to process a document with duplicate detection"""
        filename = os.path.basename(file_path)
        
        # Check if already processed (unless forced)
        if not force_reprocess and self.is_document_processed(file_path, filename):
            print(f"Skipping {filename} - already processed")
            return []
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            raw_text = self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            raw_text = self.extract_text_from_docx(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            # Support for standalone image files
            raw_text = self.extract_text_from_image_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if not raw_text.strip():
            print(f"❌ No text extracted from {filename}")
            return []
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Generate document ID and hash
        doc_id = hashlib.md5(f"{filename}_{datetime.now().isoformat()}".encode()).hexdigest()
        file_hash = self.get_file_hash(file_path)
        
        # Create metadata
        metadata = {
            'doc_id': doc_id,
            'filename': filename,
            'original_filename': filename,
            'file_path': file_path,
            'file_type': file_extension,
            'file_hash': file_hash,
            'file_size': os.path.getsize(file_path),
            'total_chars': len(cleaned_text),
            'processed_at': datetime.now().isoformat(),
            'primary_language': self.detect_language_content(cleaned_text)
        }
        
        # Chunk the text
        chunks = self.chunk_text_smart(cleaned_text, metadata)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk['metadata']['chunk_id'] = i
            chunk['metadata']['total_chunks'] = len(chunks)
        
        # Register the document
        self.document_registry[doc_id] = {
            'original_filename': filename,
            'file_hash': file_hash,
            'file_size': metadata['file_size'],
            'processed_at': metadata['processed_at'],
            'total_chunks': len(chunks),
            'primary_language': metadata['primary_language']
        }
        
        # Save registry
        self.save_document_registry()
        
        print(f"✅ Processed {filename}: {len(chunks)} chunks (Language: {metadata['primary_language']})")
        return chunks
    
    def get_processed_documents_info(self) -> List[Dict]:
        """Get information about all processed documents"""
        documents_info = []
        for doc_id, doc_info in self.document_registry.items():
            documents_info.append({
                'doc_id': doc_id,
                'filename': doc_info['original_filename'],
                'processed_at': doc_info['processed_at'],
                'total_chunks': doc_info['total_chunks'],
                'primary_language': doc_info['primary_language']
            })
        return sorted(documents_info, key=lambda x: x['processed_at'], reverse=True)
    
    def clear_document_registry(self):
        """Clear all processed documents registry"""
        self.document_registry = {}
        self.save_document_registry()
        print("Document registry cleared")