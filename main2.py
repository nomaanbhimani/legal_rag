import os
import streamlit as st
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    AutoModelForQuestionAnswering, pipeline, BertForSequenceClassification
)
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

def main():
    st.set_page_config(
        page_title="InLegalBERT Document Analyzer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 InLegalBERT Legal Document Analyzer")
    st.markdown("**Powered by InLegalBERT**: Fine-tuned legal AI for understanding, analysis, and plain English conversion")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("Loading InLegalBERT brain..."):
            st.session_state.rag_system = LegalRAGSystem()
            # Pre-load the brain
            st.session_state.rag_system.brain.load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("🧠 System Status")
        
        # Model status indicators
        brain = st.session_state.rag_system.brain
        
        status_indicators = [
            ("Embedding Model", brain.embedding_model is not None),
            ("Q&A Model", brain.qa_model is not None),
            ("Vector Database", st.session_state.rag_system.vector_db is not None)
        ]
        
        for component, is_active in status_indicators:
            if is_active:
                st.success(f"✅ {component}")
            else:
                st.error(f"❌ {component}")
        
        st.divider()
        
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload legal documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload legal documents for analysis"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if st.button(f"🔍 Analyze {file.name}", key=f"process_{file.name}"):
                    with st.spinner(f"Analyzing {file.name}..."):
                        success = st.session_state.rag_system.process_document(file)
        
        st.divider()
        
        # Quick stats
        if hasattr(st.session_state.rag_system, 'vector_db') and st.session_state.rag_system.vector_db:
            try:
                count = st.session_state.rag_system.vector_db.count()
                st.metric("📊 Document Chunks", count)
            except:
                st.metric("📊 Document Chunks", "Error")
        else:
            st.metric("📊 Document Chunks", "0")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Sample questions in a more compact format
        st.subheader("🎯 Quick Questions")
        
        quick_questions = [
            "What are the key obligations?",
            "Explain termination clauses",
            "What are the liability terms?",
            "Summarize payment terms",
            "What about confidentiality?"
        ]
        
        # Display quick questions in columns
        q_cols = st.columns(len(quick_questions))
        for i, question in enumerate(quick_questions):
            with q_cols[i]:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.current_question = question
        
        st.divider()
        
        # Custom question input
        user_question = st.text_area(
            "Ask your specific question:",
            height=80,
            placeholder="What happens if I want to terminate this contract early?"
        )
        
        if st.button("🔍 Analyze", type="primary"):
            if user_question.strip():
                st.session_state.current_question = user_question
        
        # Process and display results
        if hasattr(st.session_state, 'current_question'):
            with st.spinner("Analyzing..."):
                result = st.session_state.rag_system.query(st.session_state.current_question)
                
                st.subheader("📋 Analysis Results")
                st.markdown(result['answer'])
                
                # Metrics in columns
                if result['sources']:
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("🎯 Confidence", f"{result['confidence']:.2f}")
                    with metric_col2:
                        st.metric("📄 Sources", len(result['sources']))
                
                # Legal concepts as tags
                if result['legal_concepts']:
                    st.subheader("📌 Key Terms")
                    concept_container = st.container()
                    with concept_container:
                        cols = st.columns(4)
                        for i, concept in enumerate(result['legal_concepts'][:8]):
                            with cols[i % 4]:
                                st.info(concept)
                
                # Source information in expandable sections
                if result['sources']:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"📄 {source['filename']} - Section {source['chunk_id']} (Score: {source['relevance_score']:.2f})"):
                            if source.get('simplified_preview'):
                                st.write(source['simplified_preview'])
                            if source.get('legal_concepts'):
                                st.caption(f"Terms: {', '.join(source['legal_concepts'][:3])}")
    
    with col2:
        st.header("📊 Analytics")
        
        # Document statistics
        if st.session_state.rag_system.vector_db:
            try:
                count = st.session_state.rag_system.vector_db.count()
                
                if count > 0:
                    # Get sample concepts
                    sample_results = st.session_state.rag_system.vector_db.query(
                        query_embeddings=[st.session_state.rag_system.brain.get_embeddings("contract").tolist()],
                        n_results=min(3, count)
                    )
                    
                    if sample_results and sample_results['metadatas']:
                        all_concepts = []
                        for metadata in sample_results['metadatas'][0]:
                            if 'legal_concepts' in metadata:
                                try:
                                    concepts = json.loads(metadata['legal_concepts'])
                                    all_concepts.extend(concepts)
                                except:
                                    pass
                        
                        unique_concepts = list(set(all_concepts))[:6]
                        if unique_concepts:
                            st.subheader("🔍 Found Terms")
                            for concept in unique_concepts:
                                st.caption(f"• {concept}")
                
            except Exception as e:
                st.error("Unable to load analytics")
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh System"):
            st.rerun()
        
        if st.button("📊 Show Statistics"):
            if st.session_state.rag_system.vector_db:
                try:
                    count = st.session_state.rag_system.vector_db.count()
                    st.info(f"Total chunks: {count}")
                except:
                    st.error("Cannot load statistics")
        
        st.divider()
        
        # Configuration info
        with st.expander("⚙️ Configuration"):
            st.caption(f"Model: {config.MODEL_NAME}")
            st.caption(f"Chunk Size: {config.CHUNK_SIZE}")
            st.caption(f"Temperature: {config.TEMPERATURE}")
    
    # Footer
    st.markdown("---")
    
    footer_col1, footer_col2 = st.columns(2)
    
    with footer_col1:
        st.caption("**System Status**")
        brain = st.session_state.rag_system.brain
        
        status_items = [
            ("InLegalBERT", "🟢" if brain.embedding_model else "🔴"),
            ("Database", "🟢" if st.session_state.rag_system.vector_db else "🔴"),
            ("Q&A", "🟢" if brain.qa_model else "🔴")
        ]
        
        status_text = " | ".join([f"{name} {status}" for name, status in status_items])
        st.caption(status_text)
    
    with footer_col2:
        st.warning("⚠️ **Disclaimer:** Educational purposes only. Not legal advice. Consult legal professionals for legal matters.")
    
    # Debug mode (hidden by default)
    if st.sidebar.checkbox("🔧 Debug Mode"):
        with st.expander("Debug Information"):
            brain = st.session_state.rag_system.brain
            
            debug_data = {
                "embedding_model": brain.embedding_model is not None,
                "qa_model": brain.qa_model is not None,
                "device": str(brain.device),
                "model": config.MODEL_NAME
            }
            
            st.json(debug_data)
            
            if hasattr(st.session_state, 'current_question'):
                st.caption(f"Last question: {st.session_state.current_question}")
                concepts = brain.extract_legal_concepts(st.session_state.current_question)
                if concepts:
                    st.caption(f"Query concepts: {', '.join(concepts)}")

if __name__ == "__main__":
    main()