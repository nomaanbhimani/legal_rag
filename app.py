"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LEGALMIND INDIA v2.1.0                                   ║
║              AI-Powered Indian Legal Document Assistant                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Features:                                                                    ║
║  • RAG Architecture with InLegalBERT embeddings                              ║
║  • PDF, DOCX, TXT file upload support                                        ║
║  • Semantic chunking for legal documents                                     ║
║  • Vector similarity search                                                  ║
║  • Extractive QA with RoBERTa                                               ║
║  • Legal text simplification                                                 ║
║  • Indian legal entity extraction                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Optimized for: Render Free Tier (512MB RAM, 30s timeout)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import hashlib
import time
import gc
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename




# PDF Support
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ PyPDF2 not installed - PDF support disabled")

try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False

# DOCX Support
try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️ python-docx not installed - DOCX support disabled")

class Config:
    """
    Application Configuration
    Optimized for Render Free Tier (512MB RAM)
    """
    
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_API_BASE = "https://api-inference.huggingface.co/models"
    API_TIMEOUT = 25  
    
    EMBEDDING_MODEL = "law-ai/InLegalBERT"
    QA_MODEL = "deepset/roberta-base-squad2"
    
    CHUNK_SIZE = 400           
    CHUNK_OVERLAP = 80         
    MIN_CHUNK_SIZE = 50        
    
    TOP_K_RESULTS = 3          
    MAX_CONTEXT_LENGTH = 2000  
    
    EMBEDDING_CACHE_SIZE = 100
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_TEXT_LENGTH = 100000          # 100k characters
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'rtf'}
    UPLOAD_FOLDER = tempfile.gettempdir()
    
    MAX_DOCUMENT_SIZE = 50000  # 50k characters per document
    MAX_QUESTION_LENGTH = 500




@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    doc_id: str
    chunk_idx: int
    filename: str = ""
    section_title: Optional[str] = None
    doc_type: str = "general"
    page_number: Optional[int] = None


@dataclass
class SearchResult:
    """Search result with similarity score"""
    chunk: Chunk
    score: float


@dataclass
class Answer:
    """Complete answer response"""
    answer: str
    confidence: float
    simplified: Optional[str] = None
    concepts: List[Dict] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    processing_time_ms: int = 0


LEGAL_TERMS = {
    # === Latin/Legal Terms ===
    "suo moto": "on its own motion (without anyone filing a case)",
    "suo motu": "on its own motion (without anyone filing a case)",
    "prima facie": "at first glance; based on initial evidence",
    "res judicata": "a matter already decided by court that cannot be reopened",
    "locus standi": "right or capacity to bring a case to court",
    "inter alia": "among other things",
    "ab initio": "from the beginning; from the start",
    "de novo": "starting fresh, as if for the first time",
    "ultra vires": "beyond one's legal power or authority",
    "intra vires": "within one's legal power or authority",
    "bona fide": "in good faith; genuine",
    "mala fide": "in bad faith; with ill intent",
    "ipso facto": "by the fact itself; automatically",
    "ex parte": "from one side only; without the other party",
    "ad hoc": "for this specific purpose",
    "status quo": "the existing state of affairs",
    "sine qua non": "an essential condition; something absolutely necessary",
    "per incuriam": "through lack of care (decision made in ignorance of law)",
    "obiter dictum": "a remark made in passing; not binding as precedent",
    "ratio decidendi": "the reason for the decision; binding part of judgment",
    "stare decisis": "to stand by decided matters; follow precedents",
    "amicus curiae": "friend of the court; neutral advisor",
    "sub judice": "under judicial consideration; pending in court",
    "caveat emptor": "let the buyer beware",
    "quid pro quo": "something for something; mutual exchange",
    "habeas corpus": "produce the body; order to bring detained person to court",
    "mandamus": "we command; order to perform official duty",
    "certiorari": "to be informed; order to review lower court's decision",
    "quo warranto": "by what authority; challenge to someone's right to office",
    "prohibition": "order stopping lower court from exceeding its powers",
    
    # === Indian Criminal Law Terms ===
    "cognizable offence": "serious crime where police can arrest without warrant",
    "non-cognizable offence": "less serious crime requiring warrant for arrest",
    "bailable offence": "crime where accused has the right to be released on bail",
    "non-bailable offence": "serious crime where bail is at court's discretion",
    "anticipatory bail": "bail granted before arrest in anticipation of arrest",
    "regular bail": "bail granted to person already in judicial custody",
    "interim bail": "temporary bail granted until main application is decided",
    "default bail": "bail granted due to failure to file chargesheet in time",
    "FIR": "First Information Report - first written document about a crime",
    "chargesheet": "formal document filed by police listing charges after investigation",
    "cognizance": "court's formal acknowledgment that it will hear a case",
    "remand": "sending accused back to custody (judicial or police)",
    "judicial custody": "custody in jail under court's supervision",
    "police custody": "custody with police for investigation purposes",
    "parole": "temporary release from prison on conditions",
    "probation": "supervised release in community instead of prison",
    "compounding": "settling a criminal case between parties with court permission",
    "abetment": "helping, encouraging or instigating someone to commit a crime",
    "mens rea": "guilty mind; criminal intent required for crime",
    "actus reus": "guilty act; physical action constituting crime",
    "culpable homicide": "causing death with intention or knowledge",
    "murder": "culpable homicide with specific intentions under Section 300 IPC",
    
    # === Indian Civil Law Terms ===
    "plaintiff": "person who files a civil case (sues someone)",
    "defendant": "person against whom civil case is filed",
    "decree": "formal expression of court's decision in civil case",
    "judgment": "statement of reasons for the decision",
    "order": "court's decision on procedural matters",
    "injunction": "court order to do or refrain from doing something",
    "temporary injunction": "interim order until final decision",
    "permanent injunction": "final order after case is decided",
    "specific performance": "order to fulfill contract exactly as agreed",
    "damages": "monetary compensation for loss or injury",
    "mesne profits": "profits from property during wrongful possession",
    "restitution": "restoring property or rights to rightful owner",
    
    # === Constitutional Terms ===
    "fundamental rights": "basic rights guaranteed by Constitution (Part III)",
    "directive principles": "guidelines for government policy (Part IV)",
    "fundamental duties": "duties of citizens under Constitution (Part IVA)",
    "judicial review": "court's power to examine validity of laws and actions",
    "constitutional remedy": "right to approach court for rights violation",
    "writ jurisdiction": "power of High Courts and Supreme Court to issue writs",
    "public interest litigation": "case filed for public benefit, not personal",
    "PIL": "Public Interest Litigation",
    
    # === Contract Law Terms ===
    "consideration": "something of value exchanged in a contract",
    "indemnity": "promise to compensate for loss or damage",
    "guarantee": "promise to pay if the principal debtor defaults",
    "force majeure": "unforeseeable circumstances preventing contract performance",
    "liquidated damages": "pre-determined compensation for breach",
    "unliquidated damages": "damages to be determined by court",
    "specific performance": "court order to fulfill contract exactly",
    "novation": "replacing old contract or party with new one",
    "rescission": "cancellation of contract from the beginning",
    "privity of contract": "only parties to contract can sue on it",
    "breach of contract": "failure to fulfill contractual obligations",
    
    # === Property Law Terms ===
    "easement": "right to use another's property for specific purpose",
    "encumbrance": "claim or liability attached to property",
    "lien": "right to retain property until debt is paid",
    "mortgage": "transfer of interest in property as security for loan",
    "conveyance": "transfer of property from one person to another",
    "title": "legal ownership of property",
    "adverse possession": "acquiring ownership through long continuous possession",
    "partition": "division of joint property among co-owners",
    
    # === Evidence Law Terms ===
    "admissible evidence": "evidence that court can consider",
    "hearsay": "second-hand information not directly witnessed",
    "circumstantial evidence": "indirect evidence inferring facts",
    "documentary evidence": "evidence in form of documents",
    "oral evidence": "testimony given by witnesses",
    "burden of proof": "duty to prove facts in dispute",
    "beyond reasonable doubt": "standard of proof in criminal cases",
    "preponderance of probability": "standard of proof in civil cases",
    
    # === Archaic Legal Terms (common in documents) ===
    "herein": "in this document",
    "hereinafter": "from this point onwards in this document",
    "hereto": "to this document",
    "hereby": "by means of this document",
    "thereof": "of that; of the thing mentioned",
    "therein": "in that; in the thing mentioned",
    "wherein": "in which",
    "whereof": "of which",
    "aforesaid": "mentioned earlier in this document",
    "forthwith": "immediately; without delay",
    "notwithstanding": "despite; in spite of",
    "pursuant to": "in accordance with; following",
    "inter se": "among themselves",
    "mutatis mutandis": "with necessary changes being made",
    "pari passu": "on equal footing; proportionally",
}

# ─── Citation Patterns for Indian Courts ─────────────────────────────────────────
CITATION_PATTERNS = [
    # AIR Citations: AIR 2020 SC 1234
    (r'AIR\s+(\d{4})\s+(SC|Del|Bom|Cal|Mad|Kar|All|Pat|Ori|Ker|AP|Guj|MP|Raj|HP|Pun|J&K|Chh|Jhar|Utt|Tri|Meg|Mani|Nag|Gau|Sik)\s+(\d+)', 'AIR'),
    # SCC Citations: (2020) 5 SCC 123
    (r'\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)', 'SCC'),
    # SCC Online: 2020 SCC OnLine SC 123
    (r'(\d{4})\s+SCC\s+OnLine\s+(SC|Del|Bom|Cal|Mad|Kar|All)\s+(\d+)', 'SCC_Online'),
    # SCR Citations: (2020) 5 SCR 123
    (r'\((\d{4})\)\s+(\d+)\s+SCR\s+(\d+)', 'SCR'),
    # Criminal Law Journal
    (r'Cr\.?L\.?J\.?\s+(\d{4})\s+(\d+)', 'CrLJ'),
    # MANU Citations
    (r'MANU/[A-Z]{2}/\d+/\d{4}', 'MANU'),
    # Scale Citations
    (r'\((\d{4})\)\s+(\d+)\s+Scale\s+(\d+)', 'Scale'),
    # JT Citations
    (r'JT\s+(\d{4})\s*\(\d+\)\s+SC\s+(\d+)', 'JT'),
]

# ─── Section/Article Patterns ────────────────────────────────────────────────────
SECTION_PATTERNS = [
    # Section X of Act
    r'[Ss]ection\s+(\d+[A-Z]?(?:\s*\([^)]+\))?)\s+(?:of\s+)?(?:the\s+)?([A-Za-z\s]+(?:Act|Code)(?:,?\s+\d{4})?)',
    # Section X IPC/CrPC/CPC
    r'[Ss]ection\s+(\d+[A-Z]?)\s+(IPC|CrPC|CPC|IT\s+Act|NDPS|POCSO|NI\s+Act|MVA)',
    # Sec. X
    r'[Ss]ec\.?\s+(\d+[A-Z]?)\s+(IPC|CrPC|CPC)',
    # Sections X, Y and Z
    r'[Ss]ections?\s+(\d+[A-Z]?(?:\s*,\s*\d+[A-Z]?)*(?:\s+(?:and|&|to)\s+\d+[A-Z]?)?)',
    # Article X of Constitution
    r'[Aa]rticle\s+(\d+[A-Z]?)(?:\s*\([^)]+\))?(?:\s+of\s+(?:the\s+)?Constitution)?',
    # Order X Rule Y
    r'[Oo]rder\s+([IVXLCDM]+|\d+)\s+[Rr]ule\s+(\d+)',
    # Rule X of Rules
    r'[Rr]ule\s+(\d+)\s+(?:of\s+)?(?:the\s+)?([A-Za-z\s]+[Rr]ules)',
]


def extract_citations(text: str) -> List[Dict]:
    """
    Extract Indian legal citations from text.
    
    Supports: AIR, SCC, SCR, CrLJ, MANU, Scale, JT citations
    
    Args:
        text: Input text to search for citations
        
    Returns:
        List of citation dictionaries with citation, source, and year
    """
    citations = []
    seen = set()
    
    for pattern, source in CITATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            citation_text = match.group(0).strip()
            citation_lower = citation_text.lower()
            
            if citation_lower not in seen:
                seen.add(citation_lower)
                
                # Extract year from groups
                year = None
                for group in match.groups():
                    if group and isinstance(group, str) and group.isdigit() and len(group) == 4:
                        year = group
                        break
                
                citations.append({
                    "citation": citation_text,
                    "source": source,
                    "year": year
                })
    
    return citations[:10]  # Limit for performance


def extract_legal_concepts(text: str) -> List[Dict]:
    """
    Extract legal concepts and statutory references from text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of concept dictionaries with term, definition, and category
    """
    concepts = []
    text_lower = text.lower()
    seen = set()
    
    # Check for legal terms (longest first to avoid partial matches)
    sorted_terms = sorted(LEGAL_TERMS.items(), key=lambda x: -len(x[0]))
    
    for term, definition in sorted_terms:
        term_lower = term.lower()
        if term_lower in text_lower and term_lower not in seen:
            seen.add(term_lower)
            concepts.append({
                "term": term,
                "definition": definition,
                "category": "legal_term"
            })
    
    # Extract section/article references
    for pattern in SECTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            ref = match.group(0).strip()
            ref_lower = ref.lower()
            
            if ref_lower not in seen and len(ref) > 5:
                seen.add(ref_lower)
                concepts.append({
                    "term": ref,
                    "definition": "Statutory reference",
                    "category": "statute"
                })
    
    return concepts[:15]  # Limit for performance


def simplify_legal_text(text: str) -> str:
    """
    Simplify legal jargon to plain, understandable language.
    
    Args:
        text: Legal text with jargon
        
    Returns:
        Simplified text with explanations
    """
    if not text:
        return text
    
    simplified = text
    
    # Replace legal terms with explanations (longest first)
    sorted_terms = sorted(LEGAL_TERMS.items(), key=lambda x: -len(x[0]))
    
    for term, definition in sorted_terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, simplified, re.IGNORECASE):
            # Add explanation in parentheses (only first occurrence)
            replacement = f"{term} ({definition})"
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE, count=1)
    
    # Simplify archaic constructions
    archaic_replacements = [
        (r'\bshall\b', 'will'),
        (r'\bhereby\b', 'by this'),
        (r'\bwhereof\b', 'of which'),
        (r'\bpursuant to\b', 'following'),
        (r'\bin lieu of\b', 'instead of'),
        (r'\bwith respect to\b', 'regarding'),
        (r'\bthe said\b', 'the mentioned'),
        (r'\baforesaid\b', 'mentioned earlier'),
        (r'\bin accordance with\b', 'following'),
        (r'\bsubject to\b', 'depending on'),
        (r'\binter alia\b', 'among other things'),
        (r'\bprima facie\b', 'at first glance'),
        (r'\bforthwith\b', 'immediately'),
        (r'\bnotwithstanding\b', 'despite'),
    ]
    
    for pattern, replacement in archaic_replacements:
        simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
    
    return simplified


def identify_document_type(text: str) -> str:
    """
    Identify the type of Indian legal document.
    
    Supported types: judgment, contract, statute, petition, notice, 
                     affidavit, chargesheet, fir, general
    
    Args:
        text: Document text to analyze
        
    Returns:
        Document type string
    """
    text_lower = text.lower()[:5000]  # Check first 5000 chars for speed
    
    patterns = {
        "judgment": [
            r'\bjudgment\b', r'\bjudgement\b', r'\border\b', r'\bdecree\b',
            r'\bhon[\'\.]?ble\b', r'\bhonourable\b', r'\bthe\s+court\s+held\b',
            r'\bappeal\s+(is\s+)?(allowed|dismissed)\b', r'\bwrit\s+petition\b',
            r'\bversus\b', r'\bv\.\s', r'\bappellant\b', r'\brespondent\b',
            r'\blearned\s+counsel\b', r'\bthe\s+court\s+observed\b'
        ],
        "contract": [
            r'\bagreement\b', r'\bcontract\b', r'\bparties\s+hereto\b',
            r'\bwhereas\b', r'\bnow\s+therefore\b', r'\bconsideration\b',
            r'\bindemnif', r'\btermination\b', r'\bforce\s+majeure\b',
            r'\bconfidentiality\b', r'\bgoverning\s+law\b'
        ],
        "statute": [
            r'\bact,?\s+\d{4}\b', r'\bbe\s+it\s+enacted\b', r'\bregulation\b',
            r'\bordinance\b', r'\bchapter\s+[ivx\d]+\b', r'\bpreliminary\b',
            r'\bdefinitions?\b.*\bmeans\b', r'\bshort\s+title\b'
        ],
        "petition": [
            r'\bpetition\b', r'\bpetitioner\b', r'\bwrit\b', r'\bprayer\b',
            r'\brelief\s+sought\b', r'\bpublic\s+interest\s+litigation\b',
            r'\bpil\b', r'\bgrounds?\s*:\b', r'\bmost\s+respectfully\b'
        ],
        "notice": [
            r'\bnotice\b', r'\bsummons\b', r'\bhereby\s+(give|serve)\b',
            r'\bshow\s+cause\b', r'\breply\s+within\b', r'\blegal\s+notice\b'
        ],
        "affidavit": [
            r'\baffidavit\b', r'\bi\s+.*solemnly\s+affirm\b', r'\bdeponent\b',
            r'\bverification\b', r'\bsworn\s+before\b', r'\bnotary\b'
        ],
        "chargesheet": [
            r'\bchargesheet\b', r'\bcharge[\-\s]?sheet\b', r'\bfinal\s+report\b',
            r'\binvestigation\b.*\baccused\b', r'\bfir\s+no\b'
        ],
        "fir": [
            r'\bfirst\s+information\s+report\b', r'\bf\.?i\.?r\.?\b',
            r'\bcomplainant\b', r'\boccurrence\b', r'\bpolice\s+station\b'
        ]
    }
    
    scores = {}
    for doc_type, type_patterns in patterns.items():
        score = sum(len(re.findall(p, text_lower)) for p in type_patterns)
        scores[doc_type] = score
    
    max_score = max(scores.values()) if scores else 0
    if max_score < 2:  # Require at least 2 matches for confidence
        return "general"
    
    return max(scores, key=scores.get)


class DocumentProcessor:
    """
    Handles extraction of text from various document formats.
    
    Supported formats:
    - PDF (using PyPDF2 and pdfplumber)
    - DOCX/DOC (using python-docx)
    - TXT/RTF (plain text)
    """
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get lowercase file extension"""
        if '.' not in filename:
            return ''
        return filename.rsplit('.', 1)[1].lower()
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> Dict[str, Any]:
        """
        Extract text from PDF file.
        
        Uses pdfplumber for complex layouts, falls back to PyPDF2.
        
        Args:
            file_content: PDF file bytes
            
        Returns:
            Dictionary with success status, text, and metadata
        """
        if not PDF_SUPPORT:
            return {
                "success": False,
                "error": "PDF support not available. Install PyPDF2.",
                "text": ""
            }
        
        try:
            text_parts = []
            page_texts = []
            
            # Try pdfplumber first (better for complex layouts, tables)
            if PDFPLUMBER_SUPPORT:
                try:
                    with pdfplumber.open(BytesIO(file_content)) as pdf:
                        for i, page in enumerate(pdf.pages):
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                text_parts.append(page_text)
                                page_texts.append({"page": i + 1, "text": page_text})
                except Exception as e:
                    print(f"pdfplumber failed: {e}, falling back to PyPDF2")
                    text_parts = []
                    page_texts = []
            
            # Fallback to PyPDF2
            if not text_parts:
                reader = PdfReader(BytesIO(file_content))
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
                        page_texts.append({"page": i + 1, "text": page_text})
            
            full_text = "\n\n".join(text_parts)
            
            return {
                "success": True,
                "text": full_text,
                "pages": len(page_texts),
                "page_texts": page_texts
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"PDF extraction failed: {str(e)}",
                "text": ""
            }
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> Dict[str, Any]:
        """
        Extract text from DOCX file.
        
        Extracts text from paragraphs and tables.
        
        Args:
            file_content: DOCX file bytes
            
        Returns:
            Dictionary with success status and text
        """
        if not DOCX_SUPPORT:
            return {
                "success": False,
                "error": "DOCX support not available. Install python-docx.",
                "text": ""
            }
        
        try:
            doc = DocxDocument(BytesIO(file_content))
            
            paragraphs = []
            
            # Extract from paragraphs
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            
            # Extract from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        paragraphs.append(row_text)
            
            full_text = "\n\n".join(paragraphs)
            
            return {
                "success": True,
                "text": full_text,
                "paragraphs": len(paragraphs)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"DOCX extraction failed: {str(e)}",
                "text": ""
            }
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes) -> Dict[str, Any]:
        """
        Extract text from TXT/RTF file.
        
        Tries multiple encodings to handle different file sources.
        
        Args:
            file_content: Text file bytes
            
        Returns:
            Dictionary with success status and text
        """
        try:
            # Try common encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii', 'iso-8859-1']
            text = None
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if text is None:
                # Last resort: decode with errors replaced
                text = file_content.decode('utf-8', errors='replace')
            
            return {
                "success": True,
                "text": text.strip()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Text extraction failed: {str(e)}",
                "text": ""
            }
    
    @classmethod
    def extract_text(cls, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text from file based on extension.
        
        Args:
            file_content: File bytes
            filename: Original filename with extension
            
        Returns:
            Dictionary with extracted text and metadata
        """
        ext = cls.get_file_extension(filename)
        
        if ext == 'pdf':
            return cls.extract_text_from_pdf(file_content)
        elif ext in ['docx', 'doc']:
            return cls.extract_text_from_docx(file_content)
        elif ext in ['txt', 'rtf', 'text']:
            return cls.extract_text_from_txt(file_content)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {ext}",
                "text": ""
            }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text for better processing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common OCR issues
        text = re.sub(r'(\w)\s{2,}(\w)', r'\1 \2', text)
        
        # Remove page numbers and common headers/footers
        text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()




class LegalChunker:
    """
    Structure-aware semantic chunking for legal documents.
    
    Preserves legal document structure by detecting:
    - Section/Article boundaries
    - Clause markers
    - Paragraph breaks
    - Legal document headers
    
    This improves retrieval accuracy compared to fixed-size chunking.
    """
    
    # Boundary patterns for Indian legal documents
    BOUNDARY_PATTERNS = [
        # Section markers
        r'^(?:Section|Sec\.?)\s+\d+[A-Z]?\b',
        r'^(?:Article|Art\.?)\s+\d+\b',
        r'^(?:Clause|Cl\.?)\s+\d+\b',
        # Chapter/Part markers
        r'^(?:Chapter|Ch\.?)\s+[IVX\d]+\b',
        r'^(?:Part|Pt\.?)\s+[IVX\d]+\b',
        r'^(?:Schedule|Sch\.?)\s+[IVX\d]+\b',
        # Rule markers
        r'^(?:Rule|R\.?)\s+\d+\b',
        # Numbered items
        r'^\d+\.\s+[A-Z]',
        r'^\d+\)\s+[A-Z]',
        r'^\([a-z]\)\s+',
        r'^\([ivx]+\)\s+',
        # Legal document sections
        r'^(?:WHEREAS|NOW THEREFORE|PROVIDED THAT|BE IT ENACTED)',
        r'^(?:JUDGMENT|ORDER|DECREE|PETITION|AFFIDAVIT)',
        r'^(?:IN THE MATTER OF|BETWEEN|AND|VERSUS)',
        r'^(?:PRAYER|GROUNDS|RELIEF|VERIFICATION)',
        r'^(?:PREAMBLE|DEFINITIONS|INTERPRETATION)',
    ]
    
    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 80,
        min_size: int = 50
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks for context continuity
            min_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
        
        # Compile boundary regex for performance
        self.boundary_regex = re.compile(
            '|'.join(self.BOUNDARY_PATTERNS),
            re.MULTILINE | re.IGNORECASE
        )
    
    def _find_boundaries(self, text: str) -> List[int]:
        """Find all potential chunk boundaries in text"""
        boundaries = [0]
        
        # Add regex-matched section boundaries
        for match in self.boundary_regex.finditer(text):
            boundaries.append(match.start())
        
        # Add paragraph boundaries (double newlines)
        for match in re.finditer(r'\n\n+', text):
            boundaries.append(match.start())
        
        # Add sentence boundaries for long sections
        for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])', text):
            boundaries.append(match.start())
        
        # Sort and deduplicate
        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))
        
        return boundaries
    
    def _extract_section_title(self, text: str) -> Optional[str]:
        """Extract section title from chunk beginning"""
        text_stripped = text.strip()
        
        # Try to match boundary pattern
        match = self.boundary_regex.match(text_stripped)
        if match:
            # Get the full first line as title
            end_idx = text_stripped.find('\n', match.end())
            if end_idx == -1:
                end_idx = min(len(text_stripped), match.end() + 100)
            return text_stripped[match.start():end_idx].strip()[:100]
        
        # Fall back to first line if it looks like a title
        first_line = text_stripped.split('\n')[0].strip()
        if 10 < len(first_line) < 150 and first_line[0].isupper():
            return first_line[:100]
        
        return None
    
    def chunk(
        self,
        text: str,
        doc_id: str = "doc",
        filename: str = ""
    ) -> List[Chunk]:
        """
        Chunk document while preserving legal structure.
        
        Args:
            text: Document text to chunk
            doc_id: Unique document identifier
            filename: Original filename
            
        Returns:
            List of Chunk objects
        """
        if not text or len(text.strip()) < self.min_size:
            return []
        
        # Clean and normalize text
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        # Find chunk boundaries
        boundaries = self._find_boundaries(text)
        
        chunks = []
        current_text = ""
        chunk_idx = 0
        
        for i in range(len(boundaries) - 1):
            segment = text[boundaries[i]:boundaries[i + 1]].strip()
            
            if not segment:
                continue
            
            # Check if adding segment exceeds chunk size
            if len(current_text) + len(segment) > self.chunk_size and current_text:
                # Save current chunk
                section_title = self._extract_section_title(current_text)
                chunks.append(Chunk(
                    text=current_text.strip(),
                    doc_id=doc_id,
                    chunk_idx=chunk_idx,
                    filename=filename,
                    section_title=section_title
                ))
                chunk_idx += 1
                
                # Keep overlap for context continuity
                if len(current_text) > self.overlap:
                    overlap_text = current_text[-self.overlap:]
                else:
                    overlap_text = ""
                current_text = overlap_text + " " + segment
            else:
                # Add segment to current chunk
                current_text = (current_text + " " + segment).strip() if current_text else segment
        
        # Add final chunk if it meets minimum size
        if current_text and len(current_text.strip()) >= self.min_size:
            section_title = self._extract_section_title(current_text)
            chunks.append(Chunk(
                text=current_text.strip(),
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                filename=filename,
                section_title=section_title
            ))
        
        return chunks



class EmbeddingEngine:
    """
    API-based embedding engine using InLegalBERT.
    
    Uses Hugging Face Inference API to avoid loading model in memory.
    Includes caching to reduce API calls and improve response time.
    
    InLegalBERT is trained on 5M+ Indian legal documents and achieves
    state-of-the-art performance on Indian legal NLP tasks.
    """
    
    def __init__(self):
        """Initialize the embedding engine"""
        self.api_url = f"{Config.HF_API_BASE}/{Config.EMBEDDING_MODEL}"
        self.headers = {
            "Authorization": f"Bearer {Config.HF_TOKEN}"
        } if Config.HF_TOKEN else {}
        
        # LRU cache for embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text (first 500 chars)"""
        return hashlib.md5(text[:500].encode()).hexdigest()
    
    def _manage_cache(self, key: str, value: np.ndarray):
        """LRU cache management to prevent memory bloat"""
        if key in self._cache:
            return
        
        # Remove oldest entry if cache is full
        if len(self._cache) >= Config.EMBEDDING_CACHE_SIZE:
            old_key = self._cache_order.pop(0)
            if old_key in self._cache:
                del self._cache[old_key]
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.
        
        Uses cache if available, otherwise calls HuggingFace API.
        
        Args:
            text: Text to embed
            
        Returns:
            768-dimensional numpy array
        """
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Truncate text to avoid API limits
            truncated = text[:1000]
            
            # Call HuggingFace API
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": truncated,
                    "options": {"wait_for_model": True, "use_cache": True}
                },
                timeout=Config.API_TIMEOUT
            )
            
            # Handle model loading (503)
            if response.status_code == 503:
                time.sleep(3)
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": truncated, "options": {"wait_for_model": True}},
                    timeout=Config.API_TIMEOUT
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats from HF API
            arr = np.array(result, dtype=np.float32)
            
            if len(arr.shape) == 3:
                # [Batch, Sequence, Hidden] -> Mean pooling
                embedding = arr[0].mean(axis=0)
            elif len(arr.shape) == 2:
                # [Sequence, Hidden] -> Mean pooling
                embedding = arr.mean(axis=0)
            else:
                embedding = arr
            
            # Cache the result
            self._manage_cache(cache_key, embedding)
            return embedding
            
        except requests.exceptions.Timeout:
            print(f"Embedding API timeout")
            return np.zeros(768, dtype=np.float32)
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
        self._cache_order.clear()




class LightVectorStore:
    """
    Lightweight in-memory vector store using numpy.
    
    Optimized for small-medium datasets (< 1000 chunks).
    Uses efficient numpy operations for similarity search.
    No external database dependencies.
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        """
        Initialize vector store.
        
        Args:
            embedding_engine: Engine to use for generating embeddings
        """
        self.embedding_engine = embedding_engine
        self.chunks: List[Chunk] = []
        self.embeddings: List[np.ndarray] = []
        self.embedding_matrix: Optional[np.ndarray] = None
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
        """
        for chunk in chunks:
            embedding = self.embedding_engine.get_embedding(chunk.text)
            self.chunks.append(chunk)
            self.embeddings.append(embedding)
        
        # Rebuild embedding matrix for efficient search
        if self.embeddings:
            self.embedding_matrix = np.vstack(self.embeddings)
        
        gc.collect()  # Free memory
    
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Search for chunks similar to query.
        
        Uses cosine similarity with vectorized numpy operations.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        if not self.chunks or self.embedding_matrix is None:
            return []
        
        # Get query embedding
        query_embedding = self.embedding_engine.get_embedding(query)
        
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        
        # Compute all similarities at once (vectorized)
        doc_norms = np.linalg.norm(self.embedding_matrix, axis=1)
        doc_norms[doc_norms == 0] = 1  # Avoid division by zero
        
        similarities = np.dot(self.embedding_matrix, query_embedding) / (doc_norms * query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(similarities[idx])
            ))
        
        return results
    
    def count(self) -> int:
        """Get number of chunks in store"""
        return len(self.chunks)
    
    def get_documents(self) -> List[Dict]:
        """Get list of unique documents with chunk counts"""
        docs = {}
        for chunk in self.chunks:
            if chunk.doc_id not in docs:
                docs[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "doc_type": chunk.doc_type,
                    "chunks": 0
                }
            docs[chunk.doc_id]["chunks"] += 1
        return list(docs.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its chunks from the store.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was found and deleted
        """
        initial_count = len(self.chunks)
        
        # Find indices to keep
        indices_to_keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        
        if len(indices_to_keep) == initial_count:
            return False
        
        # Rebuild lists with remaining items
        self.chunks = [self.chunks[i] for i in indices_to_keep]
        self.embeddings = [self.embeddings[i] for i in indices_to_keep]
        
        # Rebuild embedding matrix
        if self.embeddings:
            self.embedding_matrix = np.vstack(self.embeddings)
        else:
            self.embedding_matrix = None
        
        gc.collect()
        return True
    
    def clear(self):
        """Clear all data from the store"""
        self.chunks = []
        self.embeddings = []
        self.embedding_matrix = None
        gc.collect()




class QAEngine:
    """
    Extractive Question Answering using Hugging Face API.
    
    Uses RoBERTa-Squad2, a state-of-the-art extractive QA model.
    Extracts answer spans from provided context.
    """
    
    def __init__(self):
        """Initialize the QA engine"""
        self.api_url = f"{Config.HF_API_BASE}/{Config.QA_MODEL}"
        self.headers = {
            "Authorization": f"Bearer {Config.HF_TOKEN}"
        } if Config.HF_TOKEN else {}
    
    def answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Get answer for question given context.
        
        Args:
            question: Question to answer
            context: Context containing the answer
            
        Returns:
            Dictionary with answer and confidence score
        """
        try:
            # Truncate context to avoid timeouts
            context = context[:Config.MAX_CONTEXT_LENGTH]
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": {
                        "question": question,
                        "context": context
                    },
                    "options": {"wait_for_model": True}
                },
                timeout=Config.API_TIMEOUT
            )
            
            # Handle model loading
            if response.status_code == 503:
                time.sleep(3)
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "inputs": {"question": question, "context": context},
                        "options": {"wait_for_model": True}
                    },
                    timeout=Config.API_TIMEOUT
                )
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "answer": result.get("answer", "No answer found in the documents."),
                "confidence": round(result.get("score", 0.0), 4)
            }
            
        except requests.exceptions.Timeout:
            return {
                "answer": "Request timed out. Please try again with a simpler question.",
                "confidence": 0.0
            }
        except Exception as e:
            print(f"QA error: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0
            }




class LegalRAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) Pipeline for Indian Legal Documents.
    
    Pipeline Flow:
    1. Document Upload → Text Extraction → Chunking → Embedding → Storage
    2. Query → Enhancement → Retrieval → Context Aggregation → QA → Simplification
    
    Components:
    - DocumentProcessor: Extract text from PDF/DOCX/TXT
    - LegalChunker: Structure-aware document chunking
    - EmbeddingEngine: InLegalBERT embeddings via API
    - LightVectorStore: In-memory vector similarity search
    - QAEngine: Extractive QA with RoBERTa
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = LightVectorStore(self.embedding_engine)
        self.qa_engine = QAEngine()
        self.chunker = LegalChunker(
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP,
            min_size=Config.MIN_CHUNK_SIZE
        )
        self.doc_processor = DocumentProcessor()
        
        print("✅ LegalRAG Pipeline initialized successfully")
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance query with domain-specific terms for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query with additional relevant terms
        """
        expansions = {
            "bail": "bail custody arrest release bond surety anticipatory",
            "murder": "murder death kill IPC 302 homicide culpable",
            "divorce": "divorce marriage separation maintenance alimony custody",
            "property": "property land ownership transfer deed title conveyance",
            "contract": "contract agreement breach damages consideration parties",
            "fir": "FIR complaint police report cognizable first information",
            "arrest": "arrest custody detention bail remand warrant",
            "cheque": "cheque dishonour bounce NI Act 138 negotiable",
            "theft": "theft stolen IPC 378 robbery larceny",
            "fraud": "fraud cheating IPC 420 deception misrepresentation",
            "writ": "writ petition habeas mandamus certiorari prohibition",
            "appeal": "appeal revision review appellate higher court",
            "evidence": "evidence witness testimony proof documentary",
            "compensation": "compensation damages relief remedy award",
            "fundamental": "fundamental rights constitution article basic",
            "section": "section provision clause subsection act",
        }
        
        query_lower = query.lower()
        for keyword, expansion in expansions.items():
            if keyword in query_lower:
                return f"{query} {expansion}"
        
        return query
    
    def add_document_from_file(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        Add document from file upload.
        
        Handles text extraction, cleaning, chunking, and indexing.
        
        Args:
            file_content: File bytes
            filename: Original filename
            
        Returns:
            Result dictionary with success status and metadata
        """
        start_time = time.time()
        
        # Validate file size
        if len(file_content) > Config.MAX_FILE_SIZE:
            return {
                "success": False,
                "message": f"File too large. Maximum size is {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            }
        
        # Validate file type
        if not self.doc_processor.allowed_file(filename):
            return {
                "success": False,
                "message": f"File type not supported. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            }
        
        # Extract text
        extraction_result = self.doc_processor.extract_text(file_content, filename)
        
        if not extraction_result["success"]:
            return {
                "success": False,
                "message": f"Text extraction failed: {extraction_result.get('error', 'Unknown error')}"
            }
        
        text = extraction_result["text"]
        text = self.doc_processor.clean_text(text)
        
        if not text or len(text.strip()) < 50:
            return {
                "success": False,
                "message": "Could not extract meaningful text from file"
            }
        
        # Add the extracted text as document
        return self._add_document_internal(text, filename, extraction_result)
    
    def add_document(
        self,
        content: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Add document from text content.
        
        Args:
            content: Document text
            filename: Document name
            
        Returns:
            Result dictionary
        """
        return self._add_document_internal(content, filename)
    
    def _add_document_internal(
        self,
        content: str,
        filename: str,
        file_info: Dict = None
    ) -> Dict[str, Any]:
        """
        Internal method to add document to the system.
        
        Args:
            content: Document text
            filename: Document name
            file_info: Optional extraction metadata
            
        Returns:
            Result dictionary
        """
        start_time = time.time()
        
        # Validate content
        if not content or not content.strip():
            return {"success": False, "message": "Empty document content"}
        
        # Truncate if too long
        if len(content) > Config.MAX_TEXT_LENGTH:
            content = content[:Config.MAX_TEXT_LENGTH]
        
        # Identify document type
        doc_type = identify_document_type(content)
        
        # Generate unique document ID
        doc_id = hashlib.md5(
            f"{filename}_{content[:100]}_{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Chunk the document
        chunks = self.chunker.chunk(content, doc_id, filename)
        
        if not chunks:
            return {
                "success": False,
                "message": "Could not create chunks from document content"
            }
        
        # Set document type for all chunks
        for chunk in chunks:
            chunk.doc_type = doc_type
        
        # Add chunks to vector store
        self.vector_store.add_chunks(chunks)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "doc_type": doc_type,
            "chunks_created": len(chunks),
            "processing_time_ms": processing_time
        }
        
        if file_info:
            result["pages"] = file_info.get("pages", 0)
        
        return result
    
    def query(self, question: str, simplify: bool = True) -> Answer:
        """
        Process a question and return an answer.
        
        Pipeline:
        1. Check if documents exist
        2. Enhance query with legal terms
        3. Retrieve relevant chunks
        4. Aggregate context
        5. Run extractive QA
        6. Extract legal concepts and citations
        7. Simplify answer (optional)
        
        Args:
            question: User question
            simplify: Whether to simplify the answer
            
        Returns:
            Answer object with all results
        """
        start_time = time.time()
        
        # Check for documents
        if self.vector_store.count() == 0:
            return Answer(
                answer="No documents uploaded yet. Please upload legal documents first to ask questions.",
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Validate question
        if not question or not question.strip():
            return Answer(
                answer="Please provide a valid question.",
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Enhance query for better retrieval
        enhanced_query = self._enhance_query(question)
        
        # Retrieve relevant chunks
        results = self.vector_store.search(enhanced_query, top_k=Config.TOP_K_RESULTS)
        
        if not results:
            return Answer(
                answer="No relevant information found for your question in the uploaded documents.",
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Aggregate context from retrieved chunks
        context_parts = []
        sources = []
        
        for result in results:
            context_parts.append(result.chunk.text)
            sources.append({
                "doc_id": result.chunk.doc_id,
                "filename": result.chunk.filename,
                "section": result.chunk.section_title or "Section",
                "doc_type": result.chunk.doc_type,
                "score": round(result.score, 3),
                "preview": result.chunk.text[:200] + "..." if len(result.chunk.text) > 200 else result.chunk.text
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Get answer from QA model
        qa_result = self.qa_engine.answer(question, context)
        
        # Extract legal concepts and citations
        concepts = extract_legal_concepts(context)
        citations = extract_citations(context)
        
        # Simplify answer if requested
        simplified = None
        if simplify and qa_result["answer"] and qa_result["confidence"] > 0:
            simplified = simplify_legal_text(qa_result["answer"])
            # Only include if different from original
            if simplified == qa_result["answer"]:
                simplified = None
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return Answer(
            answer=qa_result["answer"],
            confidence=qa_result["confidence"],
            simplified=simplified,
            concepts=concepts,
            citations=citations,
            sources=sources,
            processing_time_ms=processing_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_chunks": self.vector_store.count(),
            "documents": self.vector_store.get_documents(),
            "embedding_model": Config.EMBEDDING_MODEL,
            "qa_model": Config.QA_MODEL,
            "supported_formats": list(Config.ALLOWED_EXTENSIONS),
            "pdf_support": PDF_SUPPORT,
            "docx_support": DOCX_SUPPORT
        }
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document by ID"""
        success = self.vector_store.delete_document(doc_id)
        return {
            "success": success,
            "message": "Document deleted successfully" if success else "Document not found"
        }
    
    def clear(self):
        """Clear all documents and caches"""
        self.vector_store.clear()
        self.embedding_engine.clear_cache()
        gc.collect()



# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE

# Global pipeline instance (singleton)
_pipeline: Optional[LegalRAGPipeline] = None


def get_pipeline() -> LegalRAGPipeline:
    """Get or create the pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = LegalRAGPipeline()
    return _pipeline




@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "models": {
            "embedding": Config.EMBEDDING_MODEL,
            "qa": Config.QA_MODEL
        },
        "capabilities": {
            "pdf": PDF_SUPPORT,
            "docx": DOCX_SUPPORT,
            "formats": list(Config.ALLOWED_EXTENSIONS)
        },
        "stats": {
            "chunks": stats["total_chunks"],
            "documents": len(stats["documents"])
        }
    })


@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Upload document endpoint.
    
    Accepts either:
    - File upload (multipart/form-data with 'file' field)
    - JSON with 'content' and 'filename' fields
    """
    try:
        pipeline = get_pipeline()
        
        # Check for file upload
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"success": False, "message": "No file selected"})
            
            filename = secure_filename(file.filename)
            file_content = file.read()
            
            result = pipeline.add_document_from_file(file_content, filename)
            return jsonify(result)
        
        # Check for JSON text upload
        data = request.json
        if data:
            content = data.get('content', '').strip()
            filename = data.get('filename', 'document.txt').strip()
            
            if not content:
                return jsonify({"success": False, "message": "No content provided"})
            
            result = pipeline.add_document(content, filename)
            return jsonify(result)
        
        return jsonify({"success": False, "message": "No file or content provided"})
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Ask question endpoint.
    
    Expects JSON with:
    - question: The question to ask
    - simplify: Whether to simplify the answer (optional, default True)
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"answer": "No data provided", "confidence": 0})
        
        question = data.get('question', '').strip()
        simplify = data.get('simplify', True)
        
        if not question:
            return jsonify({"answer": "Please provide a question", "confidence": 0})
        
        if len(question) > Config.MAX_QUESTION_LENGTH:
            return jsonify({
                "answer": f"Question too long. Maximum {Config.MAX_QUESTION_LENGTH} characters.",
                "confidence": 0
            })
        
        pipeline = get_pipeline()
        answer = pipeline.query(question, simplify=simplify)
        
        return jsonify({
            "answer": answer.answer,
            "confidence": answer.confidence,
            "simplified": answer.simplified,
            "concepts": answer.concepts or [],
            "citations": answer.citations or [],
            "sources": answer.sources or [],
            "processing_time_ms": answer.processing_time_ms
        })
        
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"answer": f"Error: {str(e)}", "confidence": 0})


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        pipeline = get_pipeline()
        return jsonify(pipeline.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents"""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()
        return jsonify({"documents": stats["documents"]})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a specific document"""
    try:
        pipeline = get_pipeline()
        result = pipeline.delete_document(doc_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear all documents"""
    try:
        pipeline = get_pipeline()
        pipeline.clear()
        return jsonify({"success": True, "message": "All documents cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ─── Error Handlers ──────────────────────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "message": f"File too large. Maximum size is {Config.MAX_FILE_SIZE // (1024*1024)}MB"
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
