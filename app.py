"""
🇮🇳 Indian Legal Document Simplifier - Render Production Version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIXED VERSION with proper chunking verification and logging

Version: 2.1.0-render
"""

import os
import re
import gc
import json
import hashlib
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Production configuration optimized for Render free tier"""
    
    # API Keys
    HF_TOKEN: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    
    # Models
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    QA_MODEL: str = "deepset/minilm-uncased-squad2"
    
    # Memory Management
    MAX_DOCUMENTS: int = 25
    MAX_CHUNKS_PER_DOC: int = 50
    MAX_TOTAL_CHUNKS: int = 300
    EMBEDDING_CACHE_SIZE: int = 150
    EMBEDDING_DIM: int = 384
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHUNKING CONFIGURATION - Key settings!
    # ═══════════════════════════════════════════════════════════════════════════
    CHUNK_SIZE_WORDS: int = 150          # Target words per chunk
    CHUNK_OVERLAP_WORDS: int = 30        # Overlap between chunks
    MIN_CHUNK_LENGTH_CHARS: int = 100    # Minimum characters per chunk
    MAX_CHUNK_LENGTH_CHARS: int = 2000   # Maximum characters per chunk
    
    # Retrieval
    TOP_K_RETRIEVAL: int = 4
    SIMILARITY_THRESHOLD: float = 0.25
    
    # Timeouts
    API_TIMEOUT: int = 12
    MAX_RETRIES: int = 2
    RETRY_DELAY: float = 1.5
    
    # File Handling
    MAX_FILE_SIZE: int = 5 * 1024 * 1024
    UPLOAD_FOLDER: str = "/tmp/legal_uploads"
    ALLOWED_EXTENSIONS: set = field(default_factory=lambda: {'pdf', 'docx', 'txt'})
    
    # Indian Legal Domain
    INDIAN_COURTS: List[str] = field(default_factory=lambda: [
        "Supreme Court", "High Court", "District Court", "Sessions Court",
        "NCLT", "NCLAT", "ITAT", "SAT", "NGT", "CAT", "Consumer Forum",
        "Labour Court", "Family Court", "Tribunal", "Magistrate"
    ])
    
    INDIAN_ACTS: List[str] = field(default_factory=lambda: [
        "Indian Contract Act", "Indian Penal Code", "IPC", "CrPC", "CPC",
        "Constitution", "Companies Act", "GST Act", "Income Tax Act",
        "SEBI Act", "RBI Act", "Arbitration Act", "Consumer Protection Act",
        "RTI Act", "IT Act", "POCSO", "NDPS Act", "Motor Vehicles Act",
        "Negotiable Instruments Act", "Transfer of Property Act",
        "Indian Evidence Act", "Limitation Act", "Specific Relief Act"
    ])
    
    LEGAL_TERMS: List[str] = field(default_factory=lambda: [
        "FIR", "chargesheet", "bail", "anticipatory bail", "writ",
        "PIL", "suo motu", "caveat", "injunction", "stay order",
        "decree", "judgment", "appeal", "revision", "SLP",
        "affidavit", "vakalatnama", "pleading", "petition",
        "plaintiff", "defendant", "appellant", "respondent",
        "complainant", "accused", "witness", "evidence",
        "jurisdiction", "limitation", "cause of action"
    ])
    
    OBLIGATION_KEYWORDS: Dict[str, List[str]] = field(default_factory=lambda: {
        "mandatory": ["shall", "must", "is required to", "has to", "obligated to"],
        "permissive": ["may", "can", "is permitted to", "is entitled to", "has the right"],
        "prohibitive": ["shall not", "must not", "cannot", "prohibited", "forbidden"],
        "conditional": ["if", "unless", "provided that", "subject to", "in case of", "where"]
    })

config = Config()

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING - Enhanced for debugging
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more visibility
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("LegalRAG")

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def memory_cleanup(func):
    """Decorator to force garbage collection after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()
    return wrapper

def safe_request(url: str, headers: Dict, payload: Dict, timeout: int = 12) -> Optional[Dict]:
    """Make HTTP request with retry logic"""
    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
                continue
            else:
                logger.warning(f"API error {response.status_code}: {response.text[:100]}")
                return None
                
        except requests.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY)
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CHUNKING - FIXED AND ENHANCED
# ═══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    """
    Robust text chunking for legal documents
    
    Strategies:
    1. Try to split by legal sections (ARTICLE, CLAUSE, etc.)
    2. Fall back to sentence-based chunking
    3. Maintain overlap for context continuity
    """
    
    # Legal section markers for intelligent splitting
    SECTION_MARKERS = [
        r'\n\s*(?:ARTICLE|Article)\s+[IVXLCDM\d]+',
        r'\n\s*(?:CLAUSE|Clause)\s+[IVXLCDM\d]+',
        r'\n\s*(?:SECTION|Section)\s+[IVXLCDM\d]+',
        r'\n\s*(?:CHAPTER|Chapter)\s+[IVXLCDM\d]+',
        r'\n\s*(?:PART|Part)\s+[IVXLCDM\d]+',
        r'\n\s*(?:SCHEDULE|Schedule)\s+[IVXLCDM\d]+',
        r'\n\s*\d+\.\s+[A-Z][A-Z\s]+:',  # Numbered clauses like "1. DEFINITIONS:"
        r'\n\s*(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF)',
        r'\n\s*(?:DEFINITIONS?|INTERPRETATION|TERM AND TERMINATION|PAYMENT|LIABILITY|INDEMNITY|CONFIDENTIALITY|DISPUTE RESOLUTION)[\s:]+',
    ]
    
    @classmethod
    def chunk_document(cls, text: str) -> List[Dict]:
        """
        Main chunking method - returns list of chunk dictionaries
        
        Args:
            text: Full document text
            
        Returns:
            List of {'text': str, 'word_count': int, 'char_count': int, 'chunk_id': str, 'method': str}
        """
        if not text or len(text.strip()) < 50:
            logger.warning("Text too short for chunking")
            return []
        
        # Clean the text first
        text = cls._clean_text(text)
        logger.info(f"📄 Starting chunking: {len(text)} chars, ~{len(text.split())} words")
        
        # Try section-based chunking first
        chunks = cls._chunk_by_sections(text)
        
        if chunks:
            logger.info(f"✅ Section-based chunking: {len(chunks)} chunks")
            return chunks
        
        # Fall back to sentence-based chunking
        chunks = cls._chunk_by_sentences(text)
        logger.info(f"✅ Sentence-based chunking: {len(chunks)} chunks")
        
        return chunks
    
    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers
        text = re.sub(r'Page\s*\d+\s*(?:of\s*\d+)?', '', text, flags=re.IGNORECASE)
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Restore paragraph breaks for section detection
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)
        
        return text.strip()
    
    @classmethod
    def _chunk_by_sections(cls, text: str) -> List[Dict]:
        """Split by legal section markers"""
        # Combine all section patterns
        combined_pattern = '|'.join(f'(?={p})' for p in cls.SECTION_MARKERS)
        
        try:
            sections = re.split(combined_pattern, text, flags=re.IGNORECASE)
        except:
            return []
        
        # Filter empty sections
        sections = [s.strip() for s in sections if s and len(s.strip()) >= config.MIN_CHUNK_LENGTH_CHARS]
        
        if len(sections) <= 1:
            # No good section splits found
            return []
        
        chunks = []
        for i, section in enumerate(sections):
            # If section is too long, further split by sentences
            if len(section) > config.MAX_CHUNK_LENGTH_CHARS:
                sub_chunks = cls._chunk_by_sentences(section)
                for j, sub in enumerate(sub_chunks):
                    sub['chunk_id'] = f"sec{i}_sub{j}"
                    sub['method'] = 'section+sentence'
                chunks.extend(sub_chunks)
            else:
                word_count = len(section.split())
                chunks.append({
                    'text': section,
                    'word_count': word_count,
                    'char_count': len(section),
                    'chunk_id': f"sec{i}",
                    'method': 'section'
                })
        
        # Limit chunks per document
        return chunks[:config.MAX_CHUNKS_PER_DOC]
    
    @classmethod
    def _chunk_by_sentences(cls, text: str) -> List[Dict]:
        """Split by sentences with overlap"""
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        if not sentences:
            # If no sentence splits, just chunk by character count
            return cls._chunk_by_chars(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence exceeds the limit
            if current_word_count + sentence_words > config.CHUNK_SIZE_WORDS and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                
                if len(chunk_text) >= config.MIN_CHUNK_LENGTH_CHARS:
                    chunks.append({
                        'text': chunk_text,
                        'word_count': current_word_count,
                        'char_count': len(chunk_text),
                        'chunk_id': f"chunk{chunk_index}",
                        'method': 'sentence'
                    })
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_words = 0
                
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words > config.CHUNK_OVERLAP_WORDS:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_words += s_words
                
                current_chunk = overlap_sentences
                current_word_count = overlap_words
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Don't forget the last chunk!
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= config.MIN_CHUNK_LENGTH_CHARS:
                chunks.append({
                    'text': chunk_text,
                    'word_count': current_word_count,
                    'char_count': len(chunk_text),
                    'chunk_id': f"chunk{chunk_index}",
                    'method': 'sentence'
                })
        
        return chunks[:config.MAX_CHUNKS_PER_DOC]
    
    @classmethod
    def _chunk_by_chars(cls, text: str) -> List[Dict]:
        """Fallback: chunk by character count"""
        chunks = []
        chunk_size = config.MAX_CHUNK_LENGTH_CHARS
        overlap = 200  # Character overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence end
                for i in range(min(200, end - start)):
                    if text[end - i] in '.!?':
                        end = end - i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= config.MIN_CHUNK_LENGTH_CHARS:
                chunks.append({
                    'text': chunk_text,
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text),
                    'chunk_id': f"char{chunk_index}",
                    'method': 'character'
                })
                chunk_index += 1
            
            start = end - overlap
            
            if start >= len(text) - overlap:
                break
        
        return chunks[:config.MAX_CHUNKS_PER_DOC]

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PROCESSOR (Uses TextChunker)
# ═══════════════════════════════════════════════════════════════════════════════

class TextProcessor:
    """Text processing utilities for legal documents"""
    
    SECTION_PATTERN = re.compile(
        r'(?:Section|Sec\.|S\.)\s*(\d+[A-Za-z]?(?:\s*[\(\[][^)\]]+[\)\]])?)',
        re.IGNORECASE
    )
    CASE_CITATION_PATTERN = re.compile(
        r'(\d{4}\s*[\(\[]\d+[\)\]]\s*SCC\s*\d+|AIR\s*\d{4}\s*\w+\s*\d+)',
        re.IGNORECASE
    )
    DATE_PATTERN = re.compile(
        r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        re.IGNORECASE
    )
    MONETARY_PATTERN = re.compile(
        r'(?:Rs\.?|₹|INR)\s*([\d,]+(?:\.\d{2})?)\s*(?:crore|lakh|thousand|lakhs|crores)?',
        re.IGNORECASE
    )
    
    @classmethod
    def extract_sections(cls, text: str) -> List[str]:
        matches = cls.SECTION_PATTERN.findall(text)
        return list(set(matches))[:10]
    
    @classmethod
    def extract_citations(cls, text: str) -> List[str]:
        matches = cls.CASE_CITATION_PATTERN.findall(text)
        return list(set(matches))[:10]
    
    @classmethod
    def extract_dates(cls, text: str) -> List[str]:
        matches = cls.DATE_PATTERN.findall(text)
        return list(set(matches))[:10]
    
    @classmethod
    def extract_monetary(cls, text: str) -> List[str]:
        matches = cls.MONETARY_PATTERN.findall(text)
        return [f"₹{m}" for m in set(matches)][:10]

# ═══════════════════════════════════════════════════════════════════════════════
# INDIAN LEGAL NER
# ═══════════════════════════════════════════════════════════════════════════════

class IndianLegalNER:
    """Named Entity Recognition for Indian Legal Documents"""
    
    def __init__(self):
        self._court_pattern = self._build_pattern(config.INDIAN_COURTS)
        self._act_pattern = self._build_pattern(config.INDIAN_ACTS)
        self._term_pattern = self._build_pattern(config.LEGAL_TERMS)
    
    def _build_pattern(self, terms: List[str]) -> re.Pattern:
        escaped = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
        return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
    
    def extract_all(self, text: str) -> Dict[str, List[str]]:
        entities = {}
        
        courts = list(set(self._court_pattern.findall(text)))
        if courts:
            entities['courts'] = courts[:5]
        
        acts = list(set(self._act_pattern.findall(text)))
        if acts:
            entities['acts'] = acts[:5]
        
        terms = list(set(self._term_pattern.findall(text)))
        if terms:
            entities['legal_terms'] = terms[:10]
        
        sections = TextProcessor.extract_sections(text)
        if sections:
            entities['sections'] = sections[:5]
        
        citations = TextProcessor.extract_citations(text)
        if citations:
            entities['citations'] = citations[:5]
        
        dates = TextProcessor.extract_dates(text)
        if dates:
            entities['dates'] = dates[:5]
        
        amounts = TextProcessor.extract_monetary(text)
        if amounts:
            entities['amounts'] = amounts[:5]
        
        return entities
    
    def extract_obligations(self, text: str) -> Dict[str, List[str]]:
        obligations = {}
        sentences = re.split(r'[.!?]', text)
        
        for obl_type, keywords in config.OBLIGATION_KEYWORDS.items():
            pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
            matching = []
            
            for sentence in sentences:
                if pattern.search(sentence):
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 20:
                        matching.append(clean_sentence[:200])
            
            if matching:
                obligations[obl_type] = matching[:3]
        
        return obligations
    
    def get_key_concepts(self, text: str) -> List[str]:
        concepts = set()
        text_lower = text.lower()
        
        for court in config.INDIAN_COURTS:
            if court.lower() in text_lower:
                concepts.add(court)
        
        for act in config.INDIAN_ACTS:
            if act.lower() in text_lower:
                concepts.add(act)
        
        for term in config.LEGAL_TERMS:
            if term.lower() in text_lower:
                concepts.add(term)
        
        return list(concepts)[:15]

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """Process legal documents with proper chunking"""
    
    @staticmethod
    @memory_cleanup
    def extract_text(file_path: str, filename: str) -> str:
        """Extract text from document files"""
        ext = filename.lower().rsplit('.', 1)[-1]
        
        logger.info(f"📂 Extracting text from {filename} (type: {ext})")
        
        try:
            if ext == 'txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                logger.info(f"   TXT: {len(text)} characters extracted")
                return text
            
            elif ext == 'pdf':
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text_parts = []
                
                max_pages = min(len(reader.pages), 50)
                logger.info(f"   PDF: Processing {max_pages} pages")
                
                for i in range(max_pages):
                    try:
                        page_text = reader.pages[i].extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"   Error on page {i}: {e}")
                
                text = '\n'.join(text_parts)
                del reader
                logger.info(f"   PDF: {len(text)} characters extracted")
                return text
            
            elif ext == 'docx':
                from docx import Document
                doc = Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs[:500] if p.text.strip()]
                text = '\n'.join(paragraphs)
                del doc
                logger.info(f"   DOCX: {len(text)} characters extracted")
                return text
            
            else:
                raise ValueError(f"Unsupported format: {ext}")
                
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
            raise
    
    @staticmethod
    def process(file_path: str, filename: str) -> Dict:
        """Process document and return chunks"""
        
        logger.info(f"🔄 Processing document: {filename}")
        
        # Step 1: Extract text
        raw_text = DocumentProcessor.extract_text(file_path, filename)
        
        if not raw_text or len(raw_text) < 100:
            raise ValueError("Document is empty or too short (< 100 chars)")
        
        logger.info(f"📝 Raw text: {len(raw_text)} chars, ~{len(raw_text.split())} words")
        
        # Step 2: Chunk the text
        chunks = TextChunker.chunk_document(raw_text)
        
        if not chunks:
            raise ValueError("Could not create any chunks from document")
        
        # Log chunk details
        logger.info(f"✂️ Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:5]):  # Show first 5
            logger.debug(f"   Chunk {i}: {chunk['word_count']} words, {chunk['char_count']} chars, method={chunk['method']}")
        if len(chunks) > 5:
            logger.debug(f"   ... and {len(chunks) - 5} more chunks")
        
        return {
            'filename': filename,
            'chunks': chunks,
            'total_chunks': len(chunks),
            'character_count': len(raw_text),
            'word_count': len(raw_text.split())
        }

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingService:
    """Embedding service with caching"""
    
    def __init__(self):
        self.url = f"https://api-inference.huggingface.co/models/{config.EMBED_MODEL}"
        self.headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._lock = threading.Lock()
        logger.info(f"📊 Embedding service initialized: {config.EMBED_MODEL}")
    
    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text[:300].encode()).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: np.ndarray):
        with self._lock:
            if len(self._cache) >= config.EMBEDDING_CACHE_SIZE:
                remove_count = config.EMBEDDING_CACHE_SIZE // 5
                for old_key in self._cache_order[:remove_count]:
                    self._cache.pop(old_key, None)
                self._cache_order = self._cache_order[remove_count:]
            
            self._cache[key] = embedding
            self._cache_order.append(key)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if not text or len(text.strip()) < 10:
            return np.zeros(config.EMBEDDING_DIM)
        
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Truncate for API
        text = text[:500].strip()
        
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        
        result = safe_request(self.url, self.headers, payload, config.API_TIMEOUT)
        
        if result is not None:
            try:
                arr = np.array(result, dtype=np.float32)
                
                if len(arr.shape) == 3:
                    embedding = arr[0].mean(axis=0)
                elif len(arr.shape) == 2:
                    embedding = arr.mean(axis=0)
                elif len(arr.shape) == 1:
                    embedding = arr
                else:
                    embedding = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                self._add_to_cache(cache_key, embedding)
                return embedding
                
            except Exception as e:
                logger.error(f"Embedding parse error: {e}")
        
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    
    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE WITH HYBRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class HybridVectorStore:
    """In-memory vector store with hybrid search"""
    
    def __init__(self, max_chunks: int = 300):
        self.max_chunks = max_chunks
        self.documents: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        self._lock = threading.Lock()
        logger.info(f"💾 Vector store initialized (max: {max_chunks} chunks)")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords for sparse matching"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'were', 'said', 'each', 'which', 'their',
                     'will', 'from', 'this', 'that', 'with', 'they', 'would',
                     'there', 'what', 'about', 'when', 'make', 'like', 'just'}
        
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return list(set(keywords))
    
    def add(self, text: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """Add document to store"""
        with self._lock:
            if len(self.documents) >= self.max_chunks:
                remove_count = self.max_chunks // 5
                logger.info(f"🗑️ Store full, removing {remove_count} oldest chunks")
                
                self.documents = self.documents[remove_count:]
                self.embeddings = self.embeddings[remove_count:]
                
                # Rebuild keyword index
                self.keyword_index.clear()
                for idx, doc in enumerate(self.documents):
                    keywords = self._extract_keywords(doc['text'])
                    for kw in keywords:
                        self.keyword_index[kw].append(idx)
            
            doc_idx = len(self.documents)
            self.documents.append({
                'text': text,
                'metadata': metadata,
                'added_at': datetime.now().isoformat()
            })
            self.embeddings.append(embedding)
            
            # Update keyword index
            keywords = self._extract_keywords(text)
            for kw in keywords:
                self.keyword_index[kw].append(doc_idx)
            
            return True
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, 
                      k: int = 4, alpha: float = 0.7) -> List[Dict]:
        """Hybrid search: dense + sparse"""
        if not self.documents:
            logger.warning("⚠️ Search called but store is empty")
            return []
        
        with self._lock:
            n_docs = len(self.documents)
            logger.debug(f"🔍 Searching {n_docs} documents")
            
            # Dense scores
            embeddings_matrix = np.vstack(self.embeddings)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            dense_scores = np.dot(embeddings_matrix, query_norm)
            
            # Sparse scores
            query_keywords = self._extract_keywords(query)
            logger.debug(f"   Query keywords: {query_keywords[:10]}")
            
            sparse_scores = np.zeros(n_docs)
            for kw in query_keywords:
                if kw in self.keyword_index:
                    for idx in self.keyword_index[kw]:
                        if idx < n_docs:
                            sparse_scores[idx] += 1
            
            if sparse_scores.max() > 0:
                sparse_scores = sparse_scores / sparse_scores.max()
            
            # Combine
            combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
            
            # Top k
            top_indices = np.argsort(combined_scores)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                score = float(combined_scores[idx])
                if score >= config.SIMILARITY_THRESHOLD:
                    results.append({
                        'text': self.documents[idx]['text'],
                        'metadata': self.documents[idx]['metadata'],
                        'score': score,
                        'dense_score': float(dense_scores[idx]),
                        'sparse_score': float(sparse_scores[idx])
                    })
            
            logger.debug(f"   Found {len(results)} relevant results")
            return results
    
    def count(self) -> int:
        return len(self.documents)
    
    def get_filenames(self) -> List[str]:
        filenames = set()
        for doc in self.documents:
            if 'metadata' in doc and 'filename' in doc['metadata']:
                filenames.add(doc['metadata']['filename'])
        return list(filenames)
    
    def clear(self):
        with self._lock:
            self.documents.clear()
            self.embeddings.clear()
            self.keyword_index.clear()
        logger.info("🗑️ Vector store cleared")

# ═══════════════════════════════════════════════════════════════════════════════
# QA SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class QAService:
    """Question Answering Service"""
    
    def __init__(self):
        self.url = f"https://api-inference.huggingface.co/models/{config.QA_MODEL}"
        self.headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        
        self.synonyms = {
            'terminate': ['end', 'cancel', 'conclude'],
            'liability': ['responsibility', 'obligation'],
            'breach': ['violation', 'non-compliance', 'default'],
            'penalty': ['fine', 'punishment', 'sanction'],
            'contract': ['agreement', 'deed'],
        }
        logger.info(f"🤖 QA service initialized: {config.QA_MODEL}")
    
    def expand_query(self, query: str) -> str:
        expanded = query
        query_lower = query.lower()
        
        for term, syns in self.synonyms.items():
            if term in query_lower:
                expanded += ' ' + ' '.join(syns[:2])
        
        return expanded
    
    def answer(self, question: str, context: str) -> Dict:
        context = context[:3000]
        
        payload = {
            "inputs": {
                "question": question,
                "context": context
            },
            "options": {"wait_for_model": True}
        }
        
        result = safe_request(self.url, self.headers, payload, config.API_TIMEOUT)
        
        if result:
            return {
                'answer': result.get('answer', 'Could not determine answer'),
                'confidence': round(result.get('score', 0), 4),
                'start': result.get('start', 0),
                'end': result.get('end', 0)
            }
        
        return {
            'answer': 'Unable to process question. Please try again.',
            'confidence': 0,
            'start': 0,
            'end': 0
        }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RAG SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class IndianLegalRAG:
    """Main RAG System"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info("🚀 Initializing Indian Legal RAG System...")
        
        self.store = HybridVectorStore(max_chunks=config.MAX_TOTAL_CHUNKS)
        self.embedder = EmbeddingService()
        self.qa = QAService()
        self.ner = IndianLegalNER()
        
        self._initialized = True
        logger.info("✅ Indian Legal RAG System ready!")
    
    @memory_cleanup
    def ingest_document(self, file_path: str, filename: str) -> Dict:
        """Ingest a document with proper chunking"""
        try:
            logger.info(f"📥 Ingesting: {filename}")
            
            # Process document - THIS IS WHERE CHUNKING HAPPENS
            doc_data = DocumentProcessor.process(file_path, filename)
            
            chunks_added = 0
            chunks_failed = 0
            
            logger.info(f"📦 Processing {len(doc_data['chunks'])} chunks...")
            
            for i, chunk in enumerate(doc_data['chunks']):
                if self.store.count() >= config.MAX_TOTAL_CHUNKS:
                    logger.warning(f"⚠️ Max chunks reached, stopping at {i}")
                    break
                
                # Get embedding
                embedding = self.embedder.get_embedding(chunk['text'])
                
                if embedding is not None and np.any(embedding):
                    self.store.add(
                        text=chunk['text'],
                        embedding=embedding,
                        metadata={
                            'filename': filename,
                            'chunk_id': chunk['chunk_id'],
                            'word_count': chunk['word_count'],
                            'method': chunk['method']
                        }
                    )
                    chunks_added += 1
                else:
                    chunks_failed += 1
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"   Progress: {i + 1}/{len(doc_data['chunks'])} chunks")
            
            logger.info(f"✅ Ingestion complete: {chunks_added} added, {chunks_failed} failed")
            
            return {
                'success': True,
                'filename': filename,
                'chunks_added': chunks_added,
                'chunks_failed': chunks_failed,
                'total_chunks': doc_data['total_chunks'],
                'document_words': doc_data['word_count'],
                'document_chars': doc_data['character_count']
            }
            
        except Exception as e:
            logger.error(f"❌ Ingest error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    @memory_cleanup
    def query(self, question: str) -> Dict:
        """Answer a question"""
        try:
            start_time = time.time()
            
            logger.info(f"❓ Query: {question[:50]}...")
            
            if self.store.count() == 0:
                return {
                    'success': False,
                    'answer': 'No documents uploaded yet. Please upload a legal document first.',
                    'confidence': 0,
                    'sources': [],
                    'entities': {},
                    'concepts': []
                }
            
            # Expand and embed query
            expanded_query = self.qa.expand_query(question)
            query_embedding = self.embedder.get_embedding(expanded_query)
            
            # Search
            results = self.store.hybrid_search(
                query=expanded_query,
                query_embedding=query_embedding,
                k=config.TOP_K_RETRIEVAL
            )
            
            if not results:
                return {
                    'success': False,
                    'answer': 'No relevant information found for your question.',
                    'confidence': 0,
                    'sources': [],
                    'entities': {},
                    'concepts': []
                }
            
            # Combine context
            context = "\n\n".join([r['text'] for r in results])
            logger.debug(f"   Context length: {len(context)} chars from {len(results)} chunks")
            
            # Get answer
            qa_result = self.qa.answer(question, context)
            
            # Extract entities
            entities = self.ner.extract_all(context)
            concepts = self.ner.get_key_concepts(context)
            obligations = self.ner.extract_obligations(context)
            
            # Prepare sources
            sources = [{
                'filename': r['metadata'].get('filename', 'Unknown'),
                'chunk_id': r['metadata'].get('chunk_id', 'N/A'),
                'relevance_score': round(r['score'], 4),
                'preview': r['text'][:200] + '...' if len(r['text']) > 200 else r['text']
            } for r in results]
            
            processing_time = round(time.time() - start_time, 2)
            
            logger.info(f"✅ Answer generated in {processing_time}s (confidence: {qa_result['confidence']})")
            
            return {
                'success': True,
                'answer': qa_result['answer'],
                'confidence': qa_result['confidence'],
                'sources': sources,
                'entities': entities,
                'concepts': concepts,
                'obligations': obligations,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'answer': f'Error: {str(e)}',
                'confidence': 0,
                'sources': [],
                'entities': {},
                'concepts': []
            }
    
    def get_stats(self) -> Dict:
        return {
            'total_chunks': self.store.count(),
            'max_chunks': config.MAX_TOTAL_CHUNKS,
            'documents': self.store.get_filenames(),
            'document_count': len(self.store.get_filenames()),
            'cache_size': len(self.embedder._cache),
            'status': 'operational' if self._initialized else 'initializing'
        }
    
    def clear_all(self):
        self.store.clear()
        self.embedder.clear_cache()
        gc.collect()
        logger.info("🗑️ All data cleared")

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = IndianLegalRAG()
    return _rag_instance

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.1.0-render'
    })

@app.route('/api/stats')
def stats():
    try:
        return jsonify(get_rag().get_stats())
    except Exception as e:
        return jsonify({'error': str(e), 'total_chunks': 0, 'status': 'error'})

@app.route('/api/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            return jsonify({
                'success': False, 
                'error': f'Invalid file type. Allowed: {", ".join(config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            result = get_rag().ingest_document(filepath, filename)
        finally:
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'success': False, 'error': 'Question is required'}), 400
        
        if len(question) > 500:
            return jsonify({'success': False, 'error': 'Question too long (max 500 chars)'}), 400
        
        result = get_rag().query(question)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Ask error: {e}")
        gc.collect()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    try:
        get_rag().clear_all()
        return jsonify({'success': True, 'message': 'All data cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Debug endpoint to check chunks
@app.route('/api/debug/chunks')
def debug_chunks():
    """Debug endpoint to see stored chunks"""
    try:
        rag = get_rag()
        chunks_info = []
        
        for i, doc in enumerate(rag.store.documents[:10]):  # First 10
            chunks_info.append({
                'index': i,
                'text_preview': doc['text'][:100] + '...',
                'metadata': doc['metadata'],
                'added_at': doc['added_at']
            })
        
        return jsonify({
            'total_chunks': rag.store.count(),
            'sample_chunks': chunks_info,
            'filenames': rag.store.get_filenames()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large (max 5MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    gc.collect()
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    
    logger.info(f"🇮🇳 Starting Indian Legal RAG v2.1.0 on port {port}")
    logger.info(f"   Debug mode: {debug}")
    logger.info(f"   Chunk size: {config.CHUNK_SIZE_WORDS} words")
    logger.info(f"   Overlap: {config.CHUNK_OVERLAP_WORDS} words")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
