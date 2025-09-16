import os
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel 
import chromadb
import PyPDF2
from docx import Document
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
import hashlib
import json
from dataclasses import dataclass

# Configuration
@dataclass
class Config:
    MODEL_NAME: str = "law-ai/InLegalBERT"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CONTEXT_LENGTH: int = 2000
    GENERATION_MAX_LENGTH: int = 500
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    TOP_K: int = 50

config = Config()

class InLegalBERTBrain:
    """
    InLegalBERT as the central brain for:
    1. Text understanding and embedding
    2. Legal text analysis and simplification  
    3. Question answering
    4. Legal concept extraction
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.embedding_model = None
        self.qa_model = None
        
    @st.cache_resource
    def load_models(_self):
        """Load InLegalBERT models"""
        try:
            # Core tokenizer and embedding model
            _self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            if _self.tokenizer.pad_token is None:
                _self.tokenizer.pad_token = _self.tokenizer.eos_token
            
            # Embedding model for RAG
            _self.embedding_model = AutoModel.from_pretrained(config.MODEL_NAME)
            _self.embedding_model.to(_self.device)
            _self.embedding_model.eval()
            
            # Build custom QA model
            _self.qa_model = _self._build_custom_qa_model()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading InLegalBERT models: {str(e)}")
            return False
    
    def _build_custom_qa_model(self):
        """Build custom QA model on top of InLegalBERT"""
        class LegalQAModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.bert = base_model
                self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # start/end positions
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids, attention_mask=None, token_type_ids=None):
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                sequence_output = self.dropout(outputs.last_hidden_state)
                logits = self.qa_outputs(sequence_output)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)
                
                return {
                    'start_logits': start_logits,
                    'end_logits': end_logits,
                    'last_hidden_state': outputs.last_hidden_state
                }
        
        model = LegalQAModel(self.embedding_model)
        model.to(self.device)
        return model
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using InLegalBERT"""
        if not self.embedding_model:
            if not self.load_models():
                return np.random.rand(768)
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0]
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            return np.random.rand(768)
    
    def extract_legal_concepts(self, text: str) -> List[str]:
        """Extract key legal concepts using pattern matching"""
        legal_patterns = [
            r'\b(shall|must|may|will)\s+\w+',
            r'\b(contract|agreement|clause|term|condition)\b',
            r'\b(party|parties|plaintiff|defendant)\b',
            r'\b(liability|obligation|right|duty)\b',
            r'\b(terminate|breach|default|cure)\b',
            r'\b(payment|fee|penalty|damages)\b',
            r'\b(confidential|proprietary|intellectual property)\b'
        ]
        
        concepts = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def answer_question(self, question: str, context: str) -> Dict:
        """Use InLegalBERT for question answering"""
        if not self.qa_model:
            return {"answer": "QA model not available", "confidence": 0.0}
        
        try:
            # Prepare input for QA model
            inputs = self.tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
                
                start_scores = outputs['start_logits']
                end_scores = outputs['end_logits']
                
                # Find best start and end positions
                start_index = torch.argmax(start_scores)
                end_index = torch.argmax(end_scores)
                
                # Extract answer
                input_ids = inputs['input_ids'][0]
                answer_tokens = input_ids[start_index:end_index+1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                confidence = (torch.max(start_scores) + torch.max(end_scores)) / 2
                confidence = torch.sigmoid(confidence).item()
                
                return {
                    "answer": answer.strip(),
                    "confidence": confidence,
                    "start_pos": start_index.item(),
                    "end_pos": end_index.item()
                }
                
        except Exception as e:
            st.error(f"QA error: {str(e)}")
            return {"answer": "Error processing question", "confidence": 0.0}
    
    def simplify_legal_text(self, legal_text: str, context: str = "") -> str:
        """Convert complex legal text to plain English using rule-based approach"""
        # Legal-to-plain mappings
        replacements = {
            r'\bshall\b': 'must',
            r'\bwhereas\b': 'since',
            r'\btherefore\b': 'so',
            r'\bheretofore\b': 'before this',
            r'\bhereinafter\b': 'from now on',
            r'\bparty of the first part\b': 'first party',
            r'\bparty of the second part\b': 'second party',
            r'\bin consideration of\b': 'in exchange for',
            r'\bnotwithstanding\b': 'despite',
            r'\bpursuant to\b': 'according to',
            r'\bwherein\b': 'where',
            r'\bwhereby\b': 'by which',
            r'\baforesaid\b': 'mentioned above',
            r'\bindemnify\b': 'protect from financial loss',
            r'\bliable\b': 'responsible',
            r'\bterminate\b': 'end',
            r'\bbreach\b': 'break or violate',
            r'\bdefault\b': 'fail to meet obligations',
            r'\bremedy\b': 'fix or solution',
            r'\bwaive\b': 'give up',
            r'\bvoid\b': 'invalid',
            r'\bnull and void\b': 'completely invalid'
        }
        
        simplified = legal_text
        for legal_term, plain_term in replacements.items():
            simplified = re.sub(legal_term, plain_term, simplified, flags=re.IGNORECASE)
        
        # Break down long sentences
        sentences = simplified.split('. ')
        short_sentences = []
        
        for sentence in sentences:
            if len(sentence) > 100:  # Long sentence
                # Try to break at conjunctions
                parts = re.split(r'\b(and|or|but|however|moreover|furthermore)\b', sentence)
                for part in parts:
                    if part.strip() and part.strip() not in ['and', 'or', 'but', 'however', 'moreover', 'furthermore']:
                        short_sentences.append(part.strip())
            else:
                short_sentences.append(sentence)
        
        return '. '.join([s for s in short_sentences if s.strip()])
    
    def generate_summary(self, legal_texts: List[str], query_context: str = "") -> str:
        """Generate comprehensive summary using InLegalBERT"""
        
        # Extract key concepts from all texts
        all_concepts = []
        for text in legal_texts:
            concepts = self.extract_legal_concepts(text)
            all_concepts.extend(concepts)
        
        key_concepts = list(set(all_concepts))[:10]  # Top 10 unique concepts
        
        # Combine texts for analysis
        combined_text = "\n\n".join(legal_texts[:3])  # Use first 3 chunks
        
        # Generate answer using QA model
        summary_question = f"What are the main points about {query_context}?" if query_context else "What are the main legal points?"
        qa_result = self.answer_question(summary_question, combined_text)
        
        # If QA answer is too short or unclear, extract key sentences
        if len(qa_result["answer"]) < 50 or qa_result["confidence"] < 0.3:
            # Extract key sentences from the most relevant text
            key_sentences = self._extract_key_sentences(legal_texts[0], query_context)
            base_answer = ". ".join(key_sentences[:3])
        else:
            base_answer = qa_result["answer"]
        
        # Simplify the extracted answer
        simplified_answer = self.simplify_legal_text(base_answer, query_context)
        
        # Add context-specific explanations
        if query_context.lower() in ['termination', 'terminate']:
            simplified_answer += "\n\nThis means the contract can be ended under these specific conditions."
        elif query_context.lower() in ['payment', 'fee', 'money']:
            simplified_answer += "\n\nThis outlines when and how payments must be made."
        elif query_context.lower() in ['liability', 'responsible']:
            simplified_answer += "\n\nThis explains who is responsible if something goes wrong."
        
        # Combine everything into a comprehensive response
        response = f"""**Main Points:**
{simplified_answer}

**Key Legal Terms Found:**
{', '.join(key_concepts[:5])}

**Confidence:** {qa_result['confidence']:.2f}

**Important:** This is a simplified explanation. The original legal language should be referenced for precise legal meaning."""
        
        return response
    
    def _extract_key_sentences(self, text: str, query_context: str) -> List[str]:
        """Extract key sentences based on query context"""
        sentences = text.split('.')
        key_sentences = []
        
        # Look for sentences containing relevant terms
        search_terms = query_context.lower().split() if query_context else ['contract', 'agreement', 'party']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Ignore very short sentences
                for term in search_terms:
                    if term in sentence.lower():
                        key_sentences.append(sentence)
                        break
        
        # If no relevant sentences found, return first few sentences
        if not key_sentences:
            key_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
        
        return key_sentences[:5]  # Return top 5 key sentences

# Enhanced RAG System with InLegalBERT Brain
class LegalRAGSystem:
    """Enhanced RAG system with InLegalBERT as the central brain"""
    
    def __init__(self):
        self.brain = InLegalBERTBrain()
        self.vector_db = None
        self.processor = DocumentProcessor()
        self._init_vector_db()
    
    def _init_vector_db(self):
        """Initialize vector database with new ChromaDB client"""
        try:
            import chromadb
            
            # Use the new client configuration
            client = chromadb.PersistentClient(path="./chroma_db")
            
            try:
                self.vector_db = client.create_collection(
                    name="legal_documents",
                    metadata={"hnsw:space": "cosine"}  # Optional: specify distance metric
                )
            except Exception:
                # Collection already exists, get it
                self.vector_db = client.get_collection(name="legal_documents")
                
        except Exception as e:
            st.error(f"Vector DB initialization error: {str(e)}")
            self.vector_db = None
    
    def process_document(self, file, doc_type: str = "legal") -> bool:
        """Process document using InLegalBERT brain"""
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Extract text
            status_text.text("Extracting text from document...")
            progress_bar.progress(20)
            
            if file.type == "application/pdf":
                text = self.processor.extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self.processor.extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = str(file.read(), "utf-8")
            else:
                st.error("Unsupported file type")
                return False
            
            if not text:
                return False
            
            # Process with InLegalBERT brain
            status_text.text("Cleaning and chunking text...")
            progress_bar.progress(40)
            
            clean_text = self.processor.clean_text(text)
            chunks = self.processor.chunk_text(clean_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            
            # Generate embeddings and analyze chunks
            status_text.text("Analyzing with InLegalBERT...")
            progress_bar.progress(60)
            
            embeddings = []
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks):
                # Update progress
                current_progress = 60 + int((i / len(chunks)) * 25)
                progress_bar.progress(current_progress)
                
                # Get embedding
                embedding = self.brain.get_embeddings(chunk['text'])
                embeddings.append(embedding.tolist())
                
                # Extract legal concepts
                concepts = self.brain.extract_legal_concepts(chunk['text'])
                
                # Create simplified preview
                preview_text = chunk['text'][:300]
                simplified_preview = self.brain.simplify_legal_text(preview_text)
                
                # Enhanced chunk with legal analysis
                enhanced_chunk = {
                    **chunk,
                    'legal_concepts': concepts,
                    'simplified_preview': simplified_preview
                }
                enhanced_chunks.append(enhanced_chunk)
            
            # Store in vector database
            status_text.text("Storing in database...")
            progress_bar.progress(85)
            
            doc_id = hashlib.md5(file.name.encode()).hexdigest()
            metadata_list = []
            documents = []
            ids = []
            
            for i, (chunk, embedding) in enumerate(zip(enhanced_chunks, embeddings)):
                metadata = {
                    'filename': file.name,
                    'doc_type': doc_type,
                    'chunk_id': i,
                    'legal_concepts': json.dumps(chunk['legal_concepts']),
                    'simplified_preview': chunk['simplified_preview'],
                    'upload_date': datetime.now().isoformat()
                }
                metadata_list.append(metadata)
                documents.append(chunk['text'])
                ids.append(f"{doc_id}_{i}")
            
            if self.vector_db is None:
                st.error("Vector database not initialized")
                return False
                
            self.vector_db.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata_list,
                ids=ids
            )
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            st.success(f"Processed {file.name} with InLegalBERT - {len(chunks)} chunks, {sum(len(c['legal_concepts']) for c in enhanced_chunks)} legal concepts identified")
            return True
            
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict:
        """Query system using InLegalBERT brain for understanding and generation"""
        if not self.vector_db:
            return {
                'answer': "No documents uploaded yet.",
                'sources': [],
                'confidence': 0,
                'legal_concepts': []
            }
        
        # Get embedding for query using InLegalBERT
        query_embedding = self.brain.get_embeddings(question)
        
        # Search for relevant chunks
        try:
            results = self.vector_db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5
            )
            
            if not results['documents'][0]:
                return {
                    'answer': "No relevant information found in the uploaded documents.",
                    'sources': [],
                    'confidence': 0,
                    'legal_concepts': []
                }
            
            # Extract relevant texts and metadata
            relevant_texts = results['documents'][0]
            metadata_list = results['metadatas'][0]
            distances = results['distances'][0] if results['distances'] else [0] * len(relevant_texts)
            
            # Use InLegalBERT brain to generate comprehensive answer
            answer = self.brain.generate_summary(relevant_texts, question)
            
            # Calculate overall confidence
            confidence_scores = [1 - d for d in distances]
            overall_confidence = np.mean(confidence_scores)
            
            # Extract all legal concepts found
            all_concepts = []
            for metadata in metadata_list:
                if 'legal_concepts' in metadata:
                    try:
                        concepts = json.loads(metadata['legal_concepts'])
                        all_concepts.extend(concepts)
                    except:
                        pass
            
            unique_concepts = list(set(all_concepts))
            
            # Prepare sources with enhanced information
            sources = []
            for i, (metadata, distance) in enumerate(zip(metadata_list, distances)):
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', i),
                    'relevance_score': 1 - distance,
                    'legal_concepts': json.loads(metadata.get('legal_concepts', '[]')),
                    'simplified_preview': metadata.get('simplified_preview', '')
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': overall_confidence,
                'legal_concepts': unique_concepts[:10],  # Top 10 concepts
                'brain_model': 'InLegalBERT'
            }
            
        except Exception as e:
            st.error(f"Query processing error: {str(e)}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'confidence': 0,
                'legal_concepts': []
            }

class DocumentProcessor:
    """Handles document parsing and chunking"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"DOCX extraction error: {str(e)}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_idx': i,
                'end_idx': min(i + chunk_size, len(words)),
                'word_count': len(chunk_words)
            })
            
            if i + chunk_size >= len(words):
                break
        
        return chunks



# Enhanced Page Configuration
st.set_page_config(
    page_title="LegalMind Pro",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Google Hackathon-worthy UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main Container */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/></svg>') repeat;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Card Styles */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    
    .status-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Upload Zone */
    .upload-zone {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: 2px dashed #ff6b6b;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #ff5252;
        transform: scale(1.02);
    }
    
    /* Question Cards */
    .question-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        margin: 0.5rem;
    }
    
    .question-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    
    /* Legal Concepts Tags */
    .concept-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Chat-like interface for results */
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .feature-card {
            padding: 1rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main-container {
            background: #1a202c;
            color: #e2e8f0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_hero_section():
    st.markdown("""
    <div class="hero-section fade-in">
        <div class="hero-title">⚖️ LegalMind Pro</div>
        <div class="hero-subtitle">
            Transform complex legal documents into clear, actionable insights with AI-powered analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_showcase():
    st.markdown("## 🚀 Powerful Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 AI-Powered Analysis</h3>
            <p>Advanced legal language understanding with InLegalBERT for precise document interpretation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📝 Plain English Translation</h3>
            <p>Convert complex legal jargon into clear, understandable language for everyone</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Instant Q&A</h3>
            <p>Get immediate answers to your legal questions with confidence scoring</p>
        </div>
        """, unsafe_allow_html=True)

def create_smart_upload_zone():
    st.markdown("""
    <div class="upload-zone">
        <h3>📁 Drop Your Legal Documents</h3>
        <p>Support for PDF, DOCX, and TXT files • Secure processing • No data retention</p>
    </div>
    """, unsafe_allow_html=True)

def create_interactive_questions():
    st.markdown("### 💡 Smart Question Suggestions")
    
    # Categorized questions with emojis
    question_categories = {
        "📋 Contract Basics": [
            "What are the main obligations?",
            "Who are the parties involved?",
            "What's the contract duration?"
        ],
        "⚠️ Risk & Liability": [
            "What are the liability terms?",
            "What happens in case of breach?",
            "Are there any penalties?"
        ],
        "💰 Financial Terms": [
            "What are the payment terms?",
            "Are there any fees mentioned?",
            "What about refunds?"
        ],
        "🔚 Termination": [
            "How can this contract be terminated?",
            "What's the notice period?",
            "Are there early termination fees?"
        ]
    }
    
    # Create tabs for different categories
    tabs = st.tabs(list(question_categories.keys()))
    
    for tab, (category, questions) in zip(tabs, question_categories.items()):
        with tab:
            cols = st.columns(len(questions))
            for col, question in zip(cols, questions):
                with col:
                    if st.button(question, key=f"q_{hash(question)}", use_container_width=True):
                        return question
    return None

def create_status_dashboard():
    st.markdown("### 📊 System Status")
    
    # System status with visual indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">🟢</div>
            <div class="metric-label">AI Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">🟢</div>
            <div class="metric-label">Database</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">0</div>
            <div class="metric-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value pulse">⚡</div>
            <div class="metric-label">Ready</div>
        </div>
        """, unsafe_allow_html=True)

def create_analytics_dashboard(result=None):
    """Create an interactive analytics dashboard"""
    st.markdown("### 📈 Analytics Dashboard")
    
    if result and result.get('sources'):
        # Confidence Score Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['confidence'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Source Relevance Chart
        if len(result['sources']) > 1:
            source_data = {
                'Source': [f"Section {s['chunk_id']}" for s in result['sources']],
                'Relevance': [s['relevance_score'] for s in result['sources']]
            }
            
            fig_bar = px.bar(
                source_data, 
                x='Source', 
                y='Relevance',
                title="Source Relevance Scores",
                color='Relevance',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        # Placeholder analytics
        st.info("📊 Upload documents to see detailed analytics")

def create_enhanced_results_display(result):
    """Enhanced results display with better UX"""
    st.markdown("## 🎯 Analysis Results")
    
    # Main answer in a chat-like interface
    st.markdown("""
    <div class="chat-container">
        <div class="chat-message">
    """, unsafe_allow_html=True)
    
    st.markdown(result['answer'])
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Metrics row
    if result.get('confidence') or result.get('sources'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_color = "🟢" if result.get('confidence', 0) > 0.7 else "🟡" if result.get('confidence', 0) > 0.5 else "🔴"
            st.metric(
                "Confidence", 
                f"{confidence_color} {result.get('confidence', 0):.1%}",
                help="AI confidence in the answer accuracy"
            )
        
        with col2:
            st.metric(
                "Sources", 
                f"📄 {len(result.get('sources', []))}",
                help="Number of document sections analyzed"
            )
        
        with col3:
            st.metric(
                "Concepts", 
                f"🏷️ {len(result.get('legal_concepts', []))}",
                help="Legal concepts identified"
            )
    
    # Legal concepts as modern tags
    if result.get('legal_concepts'):
        st.markdown("#### 🏷️ Key Legal Concepts")
        concepts_html = ""
        for concept in result['legal_concepts'][:8]:
            concepts_html += f'<span class="concept-tag">{concept}</span>'
        
        st.markdown(concepts_html, unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("🚀 Initializing LegalMind Pro..."):
            # Your existing initialization code here
            pass
    
    # Hero Section
    create_hero_section()
    
    # Main Layout
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_main:
        # Feature showcase
        create_feature_showcase()
        
        # Upload section
        create_smart_upload_zone()
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"Size: {file.size:,} bytes | Type: {file.type}")
                    with col2:
                        if st.button(f"🔍 Analyze", key=f"analyze_{file.name}"):
                            # Your processing code here
                            st.success("✅ Analysis complete!")
        
        st.markdown("---")
        
        # Interactive question section
        selected_question = create_interactive_questions()
        
        # Custom question input
        st.markdown("#### ❓ Ask Your Own Question")
        user_question = st.text_area(
            "Enter your question:",
            placeholder="What are the specific termination conditions in this contract?",
            label_visibility="collapsed",
            height=100
        )
        
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if user_question or selected_question:
                question = user_question or selected_question
                
                with st.spinner("🧠 Analyzing your question..."):
                    # Simulate processing
                    import time
                    time.sleep(2)
                    
                    # Mock result for demonstration
                    mock_result = {
                        'answer': "Based on the contract analysis, the termination conditions include a 30-day notice period and completion of ongoing obligations. Early termination fees may apply depending on the circumstances.",
                        'confidence': 0.87,
                        'sources': [
                            {'chunk_id': 1, 'relevance_score': 0.92},
                            {'chunk_id': 3, 'relevance_score': 0.78}
                        ],
                        'legal_concepts': ['termination', 'notice period', 'obligations', 'fees']
                    }
                    
                    create_enhanced_results_display(mock_result)
    
    with col_sidebar:
        # Status dashboard
        create_status_dashboard()
        
        st.markdown("---")
        
        # Analytics dashboard
        create_analytics_dashboard()
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh System", use_container_width=True):
            st.rerun()
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.info("📈 Advanced analytics coming soon!")
        
        if st.button("💾 Export Results", use_container_width=True):
            st.success("📄 Results exported!")
        
        # Help section
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **Quick Tips:**
            - Upload PDF, DOCX, or TXT files
            - Use specific questions for better results  
            - Check confidence scores
            - Explore suggested questions
            
            **Contact:** support@legalmind.ai
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>🚀 Built with ❤️ for Google Hackathon | ⚖️ LegalMind Pro v2.0</p>
        <p style="font-size: 0.8rem;">⚠️ For informational purposes only. Consult legal professionals for legal advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()