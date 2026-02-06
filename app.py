"""
🇮🇳 Indian Legal Document Simplifier - Render Production Version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Optimized for Render Free Tier:
- Memory: <450MB (limit: 512MB)
- Cold Start: <20s
- Request Timeout: <25s (limit: 30s)

Core Features Preserved:
✅ Hybrid Search (Dense + BM25-like scoring)
✅ Indian Legal NER (Courts, Acts, Sections)
✅ Legal Concept Extraction
✅ Question Answering with Confidence
✅ Document Chunking with Overlap
✅ Source Attribution

Author: LegalMind AI
Version: 2.0.0-render
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

from flask import Flask, render_template, request, jsonify, Response
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
    
    # Models - Using lightweight, fast models
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, fast
    QA_MODEL: str = "deepset/minilm-uncased-squad2"  # Fast QA model
    
    # Memory Management (Critical for 512MB limit)
    MAX_DOCUMENTS: int = 25
    MAX_CHUNKS_PER_DOC: int = 40
    MAX_TOTAL_CHUNKS: int = 300
    EMBEDDING_CACHE_SIZE: int = 150
    EMBEDDING_DIM: int = 384
    
    # Chunking (Optimized for legal documents)
    CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 75
    MIN_CHUNK_LENGTH: int = 50
    
    # Retrieval
    TOP_K_RETRIEVAL: int = 4
    SIMILARITY_THRESHOLD: float = 0.3
    
    # Timeouts (Render has 30s limit)
    API_TIMEOUT: int = 12
    MAX_RETRIES: int = 2
    RETRY_DELAY: float = 1.5
    
    # File Handling
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
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
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
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

def timed_execution(func):
    """Decorator to log execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
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
                # Model loading
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
# TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class TextProcessor:
    """Efficient text processing for legal documents"""
    
    # Indian legal abbreviations
    ABBREVIATIONS = {
        "IPC": "Indian Penal Code",
        "CrPC": "Code of Criminal Procedure", 
        "CPC": "Code of Civil Procedure",
        "FIR": "First Information Report",
        "PIL": "Public Interest Litigation",
        "SLP": "Special Leave Petition",
        "u/s": "under section",
        "r/w": "read with",
        "w.e.f.": "with effect from",
        "Hon'ble": "Honorable",
        "Ld.": "Learned",
        "vs": "versus",
        "v.": "versus"
    }
    
    # Regex patterns (compiled for performance)
    SECTION_PATTERN = re.compile(
        r'(?:Section|Sec\.|S\.)\s*(\d+[A-Za-z]?(?:\s*[\(\[][^)\]]+[\)\]])?)',
        re.IGNORECASE
    )
    CASE_CITATION_PATTERN = re.compile(
        r'(\d{4}\s*[\(\[]\d+[\)\]]\s*SCC\s*\d+|AIR\s*\d{4}\s*\w+\s*\d+|\[\d{4}\]\s*\d+\s*\w+\s*\d+)',
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
    def clean_text(cls, text: str) -> str:
        """Clean and normalize legal text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers/headers
        text = re.sub(r'Page\s*\d+\s*(?:of\s*\d+)?', '', text, flags=re.IGNORECASE)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove control characters but keep legal punctuation
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    @classmethod
    def smart_chunk(cls, text: str, chunk_size: int = 350, overlap: int = 75) -> List[Dict]:
        """
        Semantic chunking optimized for legal documents
        Respects section boundaries and maintains context
        """
        if not text or len(text) < config.MIN_CHUNK_LENGTH:
            return []
        
        chunks = []
        
        # First, try to split by major legal sections
        section_markers = [
            r'(?=\n\s*(?:ARTICLE|CLAUSE|SECTION|CHAPTER|PART|SCHEDULE)\s+[IVXLCDM\d]+)',
            r'(?=\n\s*\d+\.\s+[A-Z])',  # Numbered clauses
            r'(?=\n\s*(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF))',
            r'(?=\n\s*(?:DEFINITIONS?|INTERPRETATION|TERM|PAYMENT|LIABILITY|INDEMNITY|TERMINATION|DISPUTE)[\s:])',
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(section_markers)
        sections = re.split(combined_pattern, text, flags=re.IGNORECASE)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is small enough, keep as one chunk
            words = section.split()
            if len(words) <= chunk_size:
                if len(section) >= config.MIN_CHUNK_LENGTH:
                    chunks.append({
                        'text': section,
                        'word_count': len(words),
                        'hash': hashlib.md5(section.encode()).hexdigest()[:8]
                    })
                continue
            
            # Split large sections by sentences
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', section)
            
            current_chunk = []
            current_word_count = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                if current_word_count + sentence_words > chunk_size:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text) >= config.MIN_CHUNK_LENGTH:
                            chunks.append({
                                'text': chunk_text,
                                'word_count': current_word_count,
                                'hash': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                            })
                        
                        # Keep overlap
                        overlap_words = 0
                        overlap_sentences = []
                        for s in reversed(current_chunk):
                            s_words = len(s.split())
                            if overlap_words + s_words > overlap:
                                break
                            overlap_sentences.insert(0, s)
                            overlap_words += s_words
                        
                        current_chunk = overlap_sentences
                        current_word_count = overlap_words
                
                current_chunk.append(sentence)
                current_word_count += sentence_words
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= config.MIN_CHUNK_LENGTH:
                    chunks.append({
                        'text': chunk_text,
                        'word_count': current_word_count,
                        'hash': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                    })
        
        # Limit chunks per document
        return chunks[:config.MAX_CHUNKS_PER_DOC]
    
    @classmethod
    def extract_sections(cls, text: str) -> List[str]:
        """Extract section references from text"""
        matches = cls.SECTION_PATTERN.findall(text)
        return list(set(matches))[:10]
    
    @classmethod
    def extract_citations(cls, text: str) -> List[str]:
        """Extract case citations"""
        matches = cls.CASE_CITATION_PATTERN.findall(text)
        return list(set(matches))[:10]
    
    @classmethod
    def extract_dates(cls, text: str) -> List[str]:
        """Extract dates from text"""
        matches = cls.DATE_PATTERN.findall(text)
        return list(set(matches))[:10]
    
    @classmethod
    def extract_monetary(cls, text: str) -> List[str]:
        """Extract monetary amounts"""
        matches = cls.MONETARY_PATTERN.findall(text)
        return [f"₹{m}" for m in set(matches)][:10]

# ═══════════════════════════════════════════════════════════════════════════════
# INDIAN LEGAL NER
# ═══════════════════════════════════════════════════════════════════════════════

class IndianLegalNER:
    """Named Entity Recognition for Indian Legal Documents"""
    
    def __init__(self):
        # Pre-compile patterns for performance
        self._court_pattern = self._build_pattern(config.INDIAN_COURTS)
        self._act_pattern = self._build_pattern(config.INDIAN_ACTS)
        self._term_pattern = self._build_pattern(config.LEGAL_TERMS)
    
    def _build_pattern(self, terms: List[str]) -> re.Pattern:
        """Build regex pattern from terms"""
        escaped = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
        return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
    
    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """Extract all entity types from text"""
        entities = {}
        
        # Courts
        courts = list(set(self._court_pattern.findall(text)))
        if courts:
            entities['courts'] = courts[:5]
        
        # Acts
        acts = list(set(self._act_pattern.findall(text)))
        if acts:
            entities['acts'] = acts[:5]
        
        # Legal terms
        terms = list(set(self._term_pattern.findall(text)))
        if terms:
            entities['legal_terms'] = terms[:10]
        
        # Sections
        sections = TextProcessor.extract_sections(text)
        if sections:
            entities['sections'] = sections[:5]
        
        # Citations
        citations = TextProcessor.extract_citations(text)
        if citations:
            entities['citations'] = citations[:5]
        
        # Dates
        dates = TextProcessor.extract_dates(text)
        if dates:
            entities['dates'] = dates[:5]
        
        # Monetary
        amounts = TextProcessor.extract_monetary(text)
        if amounts:
            entities['amounts'] = amounts[:5]
        
        return entities
    
    def extract_obligations(self, text: str) -> Dict[str, List[str]]:
        """Extract legal obligations by type"""
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
        """Get key legal concepts from text"""
        concepts = set()
        text_lower = text.lower()
        
        # Check for courts
        for court in config.INDIAN_COURTS:
            if court.lower() in text_lower:
                concepts.add(court)
        
        # Check for acts
        for act in config.INDIAN_ACTS:
            if act.lower() in text_lower:
                concepts.add(act)
        
        # Check for terms
        for term in config.LEGAL_TERMS:
            if term.lower() in text_lower:
                concepts.add(term)
        
        return list(concepts)[:15]

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """Process legal documents with memory efficiency"""
    
    @staticmethod
    @memory_cleanup
    def extract_text(file_path: str, filename: str) -> str:
        """Extract text from document files"""
        ext = filename.lower().rsplit('.', 1)[-1]
        
        try:
            if ext == 'txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif ext == 'pdf':
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text_parts = []
                
                # Limit pages to prevent memory issues
                max_pages = min(len(reader.pages), 50)
                
                for i in range(max_pages):
                    try:
                        page_text = reader.pages[i].extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {i}: {e}")
                        continue
                
                del reader
                return '\n'.join(text_parts)
            
            elif ext == 'docx':
                from docx import Document
                doc = Document(file_path)
                
                # Limit paragraphs
                paragraphs = [p.text for p in doc.paragraphs[:500] if p.text.strip()]
                
                del doc
                return '\n'.join(paragraphs)
            
            else:
                raise ValueError(f"Unsupported format: {ext}")
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    @staticmethod
    def process(file_path: str, filename: str) -> Dict:
        """Process document and return chunks"""
        # Extract text
        raw_text = DocumentProcessor.extract_text(file_path, filename)
        
        if not raw_text or len(raw_text) < 100:
            raise ValueError("Document is empty or too short")
        
        # Clean text
        clean_text = TextProcessor.clean_text(raw_text)
        
        # Create chunks
        chunks = TextProcessor.smart_chunk(clean_text)
        
        if not chunks:
            raise ValueError("Could not create chunks from document")
        
        return {
            'filename': filename,
            'chunks': chunks,
            'total_chunks': len(chunks),
            'character_count': len(clean_text)
        }

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingService:
    """Embedding service with caching and memory management"""
    
    def __init__(self):
        self.url = f"https://api-inference.huggingface.co/models/{config.EMBED_MODEL}"
        self.headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._lock = threading.Lock()
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(text[:300].encode()).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: np.ndarray):
        """Add to cache with size limit"""
        with self._lock:
            if len(self._cache) >= config.EMBEDDING_CACHE_SIZE:
                # Remove oldest entries
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
        
        # Check cache
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
                
                # Handle different response shapes
                if len(arr.shape) == 3:
                    embedding = arr[0].mean(axis=0)
                elif len(arr.shape) == 2:
                    embedding = arr.mean(axis=0)
                elif len(arr.shape) == 1:
                    embedding = arr
                else:
                    embedding = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
                
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # Cache it
                self._add_to_cache(cache_key, embedding)
                
                return embedding
                
            except Exception as e:
                logger.error(f"Embedding parse error: {e}")
        
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    
    def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts"""
        return [self.get_embedding(t) for t in texts]
    
    def clear_cache(self):
        """Clear embedding cache"""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE (In-Memory with Hybrid Search)
# ═══════════════════════════════════════════════════════════════════════════════

class HybridVectorStore:
    """
    In-memory vector store with hybrid search
    Combines dense (cosine) and sparse (keyword) retrieval
    """
    
    def __init__(self, max_chunks: int = 300):
        self.max_chunks = max_chunks
        self.documents: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords for sparse matching"""
        # Convert to lowercase and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'were', 'said', 'each', 'which', 'their',
                     'will', 'from', 'this', 'that', 'with', 'they', 'would'}
        
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return list(set(keywords))
    
    def add(self, text: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """Add document to store"""
        with self._lock:
            # Check capacity
            if len(self.documents) >= self.max_chunks:
                # Remove oldest 20%
                remove_count = self.max_chunks // 5
                self.documents = self.documents[remove_count:]
                self.embeddings = self.embeddings[remove_count:]
                
                # Rebuild keyword index
                self.keyword_index.clear()
                for idx, doc in enumerate(self.documents):
                    keywords = self._extract_keywords(doc['text'])
                    for kw in keywords:
                        self.keyword_index[kw].append(idx)
            
            # Add new document
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
        """
        Hybrid search combining dense and sparse retrieval
        alpha: weight for dense similarity (1-alpha for sparse)
        """
        if not self.documents:
            return []
        
        with self._lock:
            n_docs = len(self.documents)
            
            # Dense similarity scores
            embeddings_matrix = np.vstack(self.embeddings)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            dense_scores = np.dot(embeddings_matrix, query_norm)
            
            # Sparse (keyword) scores
            query_keywords = self._extract_keywords(query)
            sparse_scores = np.zeros(n_docs)
            
            for kw in query_keywords:
                if kw in self.keyword_index:
                    for idx in self.keyword_index[kw]:
                        if idx < n_docs:
                            sparse_scores[idx] += 1
            
            # Normalize sparse scores
            if sparse_scores.max() > 0:
                sparse_scores = sparse_scores / sparse_scores.max()
            
            # Combine scores
            combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
            
            # Get top k
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
            
            return results
    
    def count(self) -> int:
        """Get document count"""
        return len(self.documents)
    
    def get_filenames(self) -> List[str]:
        """Get list of indexed filenames"""
        filenames = set()
        for doc in self.documents:
            if 'metadata' in doc and 'filename' in doc['metadata']:
                filenames.add(doc['metadata']['filename'])
        return list(filenames)
    
    def clear(self):
        """Clear all data"""
        with self._lock:
            self.documents.clear()
            self.embeddings.clear()
            self.keyword_index.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# QA SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class QAService:
    """Question Answering Service"""
    
    def __init__(self):
        self.url = f"https://api-inference.huggingface.co/models/{config.QA_MODEL}"
        self.headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        
        # Query expansion for legal domain
        self.synonyms = {
            'terminate': ['end', 'cancel', 'conclude', 'discontinue'],
            'liability': ['responsibility', 'obligation', 'accountability'],
            'breach': ['violation', 'non-compliance', 'default'],
            'penalty': ['fine', 'punishment', 'sanction'],
            'rights': ['entitlements', 'privileges', 'powers'],
            'sue': ['file case', 'legal action', 'prosecute'],
            'contract': ['agreement', 'deed', 'covenant']
        }
    
    def expand_query(self, query: str) -> str:
        """Expand query with legal synonyms"""
        expanded = query
        query_lower = query.lower()
        
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                expanded += ' ' + ' '.join(synonyms[:2])
        
        return expanded
    
    def answer(self, question: str, context: str) -> Dict:
        """Get answer from QA model"""
        # Limit context size
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
    """Main RAG System for Indian Legal Documents"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for memory efficiency"""
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
    @timed_execution
    def ingest_document(self, file_path: str, filename: str) -> Dict:
        """Ingest a legal document"""
        try:
            # Process document
            doc_data = DocumentProcessor.process(file_path, filename)
            
            chunks_added = 0
            
            for chunk in doc_data['chunks']:
                # Check total capacity
                if self.store.count() >= config.MAX_TOTAL_CHUNKS:
                    logger.warning("Maximum chunk limit reached")
                    break
                
                # Get embedding
                embedding = self.embedder.get_embedding(chunk['text'])
                
                # Add to store
                if embedding is not None and np.any(embedding):
                    self.store.add(
                        text=chunk['text'],
                        embedding=embedding,
                        metadata={
                            'filename': filename,
                            'chunk_hash': chunk['hash'],
                            'word_count': chunk['word_count']
                        }
                    )
                    chunks_added += 1
            
            logger.info(f"📄 Ingested {filename}: {chunks_added} chunks")
            
            return {
                'success': True,
                'filename': filename,
                'chunks_added': chunks_added,
                'total_chunks': doc_data['total_chunks']
            }
            
        except Exception as e:
            logger.error(f"Ingest error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @memory_cleanup
    @timed_execution
    def query(self, question: str) -> Dict:
        """Answer a legal question"""
        try:
            start_time = time.time()
            
            # Check if we have documents
            if self.store.count() == 0:
                return {
                    'success': False,
                    'answer': 'No documents have been uploaded yet. Please upload a legal document first.',
                    'confidence': 0,
                    'sources': [],
                    'entities': {},
                    'concepts': []
                }
            
            # Expand query
            expanded_query = self.qa.expand_query(question)
            
            # Get query embedding
            query_embedding = self.embedder.get_embedding(expanded_query)
            
            # Hybrid search
            results = self.store.hybrid_search(
                query=expanded_query,
                query_embedding=query_embedding,
                k=config.TOP_K_RETRIEVAL
            )
            
            if not results:
                return {
                    'success': False,
                    'answer': 'Could not find relevant information for your question.',
                    'confidence': 0,
                    'sources': [],
                    'entities': {},
                    'concepts': []
                }
            
            # Combine context
            context = "\n\n".join([r['text'] for r in results])
            
            # Get answer
            qa_result = self.qa.answer(question, context)
            
            # Extract entities and concepts
            entities = self.ner.extract_all(context)
            concepts = self.ner.get_key_concepts(context)
            obligations = self.ner.extract_obligations(context)
            
            # Prepare sources
            sources = [{
                'filename': r['metadata'].get('filename', 'Unknown'),
                'relevance_score': round(r['score'], 4),
                'preview': r['text'][:200] + '...' if len(r['text']) > 200 else r['text']
            } for r in results]
            
            processing_time = round(time.time() - start_time, 2)
            
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
            logger.error(f"Query error: {e}")
            return {
                'success': False,
                'answer': f'Error processing your question: {str(e)}',
                'confidence': 0,
                'sources': [],
                'entities': {},
                'concepts': []
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_chunks': self.store.count(),
            'max_chunks': config.MAX_TOTAL_CHUNKS,
            'documents': self.store.get_filenames(),
            'document_count': len(self.store.get_filenames()),
            'cache_size': len(self.embedder._cache),
            'status': 'operational' if self._initialized else 'initializing'
        }
    
    def clear_all(self):
        """Clear all data and caches"""
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

# Ensure upload folder exists
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Lazy initialization for faster cold start
_rag_instance = None

def get_rag():
    """Get or create RAG instance (lazy loading)"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = IndianLegalRAG()
    return _rag_instance

# ───────────────────────────────────────────────────────────────────────────────
# ROUTES
# ───────────────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint - responds quickly for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0-render'
    })

@app.route('/api/stats')
def stats():
    """Get system statistics"""
    try:
        return jsonify(get_rag().get_stats())
    except Exception as e:
        return jsonify({
            'error': str(e),
            'total_chunks': 0,
            'status': 'error'
        })

@app.route('/api/upload', methods=['POST'])
def upload():
    """Upload and process a document"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Validate extension
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            return jsonify({
                'success': False, 
                'error': f'Invalid file type. Allowed: {", ".join(config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Process document
            result = get_rag().ingest_document(filepath, filename)
        finally:
            # Always cleanup
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        gc.collect()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    """Answer a legal question"""
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
    """Clear all data"""
    try:
        get_rag().clear_all()
        return jsonify({'success': True, 'message': 'All data cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ───────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ───────────────────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False, 
        'error': f'File too large. Maximum size is {config.MAX_FILE_SIZE // (1024*1024)}MB'
    }), 413

@app.errorhandler(500)
def server_error(e):
    gc.collect()
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    gc.collect()
    return jsonify({'success': False, 'error': 'An error occurred'}), 500

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"🇮🇳 Starting Indian Legal RAG on port {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
