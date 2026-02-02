import os
import streamlit as st
import chromadb
import PyPDF2
from docx import Document
from typing import List, Dict, Tuple, Optional
import numpy as np
import requests
import re
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

# Configuration
@dataclass
class Config:
    EMBED_MODEL: str = "nlpaueb/legal-bert-small-uncased"
    QA_MODEL: str = "deepset/roberta-base-squad2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    HF_TOKEN: str = st.secrets.get("HF_TOKEN", "")

config = Config()

class LegalMindBrain:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        self.embed_url = f"https://api-inference.huggingface.co/models/{config.EMBED_MODEL}"
        self.qa_url = f"https://api-inference.huggingface.co/models/{config.QA_MODEL}"

    def _query_api(self, url: str, payload: dict):
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json()
        except Exception as e:
            st.error(f"API Connection Error: {e}")
            return None

    def load_models(self):
        # On Cloud, we just check if the token exists
        return bool(config.HF_TOKEN)

    def get_embeddings(self, text: str) -> np.ndarray:
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        output = self._query_api(self.embed_url, payload)
        try:
            embeddings = np.array(output)
            if len(embeddings.shape) > 1:
                return embeddings[0][0] if len(embeddings.shape) == 3 else embeddings[0]
            return embeddings
        except:
            return np.random.rand(768)

    def extract_legal_concepts(self, text: str) -> List[str]:
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
            concepts.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        return list(set(concepts))

    def answer_question(self, question: str, context: str) -> Dict:
        payload = {"inputs": {"question": question, "context": context}, "options": {"wait_for_model": True}}
        result = self._query_api(self.qa_url, payload)
        if isinstance(result, dict) and "answer" in result:
            return {"answer": result["answer"], "confidence": result.get("score", 0.0)}
        return {"answer": "No specific answer found in the text.", "confidence": 0.0}

    def simplify_legal_text(self, legal_text: str) -> str:
        replacements = {r'\bshall\b': 'must', r'\bwhereas\b': 'since', r'\bpursuant to\b': 'according to', r'\bindemnify\b': 'protect'}
        simplified = legal_text
        for p, r in replacements.items():
            simplified = re.sub(p, r, simplified, flags=re.IGNORECASE)
        return simplified

    def generate_summary(self, legal_texts: List[str], query_context: str = "") -> str:
        combined_text = " ".join(legal_texts[:3])
        qa_result = self.answer_question(query_context or "Summary", combined_text)
        simplified = self.simplify_legal_text(qa_result["answer"])
        
        return f"**Main Points:**\n{simplified}\n\n**Confidence:** {qa_result['confidence']:.2f}"

# --- Document Processor & RAG System (Core Logic Maintained) ---

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([p.extract_text() for p in reader.pages])

    @staticmethod
    def extract_text_from_docx(file):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    @staticmethod
    def chunk_text(text: str):
        words = text.split()
        return [{'text': ' '.join(words[i:i+config.CHUNK_SIZE])} for i in range(0, len(words), config.CHUNK_SIZE-config.CHUNK_OVERLAP)]

class LegalRAGSystem:
    def __init__(self):
        self.brain = LegalMindBrain()
        self.processor = DocumentProcessor()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.vector_db = self.client.get_or_create_collection(name="legal_documents")

    def process_document(self, file):
        progress = st.progress(0)
        status = st.empty()
        status.text("Extracting text...")
        
        if file.type == "application/pdf": text = self.processor.extract_text_from_pdf(file)
        elif "word" in file.type: text = self.processor.extract_text_from_docx(file)
        else: text = str(file.read(), "utf-8")
        
        chunks = self.processor.chunk_text(text)
        for i, chunk in enumerate(chunks):
            status.text(f"Processing chunk {i+1}/{len(chunks)}...")
            emb = self.brain.get_embeddings(chunk['text'])
            concepts = self.brain.extract_legal_concepts(chunk['text'])
            self.vector_db.add(
                ids=[f"{hashlib.md5(file.name.encode()).hexdigest()}_{i}"],
                embeddings=[emb.tolist()],
                documents=[chunk['text']],
                metadatas=[{"concepts": json.dumps(concepts), "filename": file.name}]
            )
            progress.progress((i + 1) / len(chunks))
        status.text("Complete!")
        return True

    def query(self, question: str):
        q_emb = self.brain.get_embeddings(question)
        results = self.vector_db.query(query_embeddings=[q_emb.tolist()], n_results=4)
        if not results['documents'][0]: return {"answer": "No data found.", "sources": [], "confidence": 0, "legal_concepts": []}
        
        answer = self.brain.generate_summary(results['documents'][0], question)
        all_concepts = []
        for m in results['metadatas'][0]:
            all_concepts.extend(json.loads(m.get('concepts', '[]')))
            
        return {
            "answer": answer,
            "confidence": results['distances'][0][0] if results['distances'] else 0.5,
            "legal_concepts": list(set(all_concepts)),
            "sources": [{"filename": m['filename'], "text": d} for m, d in zip(results['metadatas'][0], results['documents'][0])]
        }

# --- ORIGINAL UI LOGIC ---

def main():
    st.set_page_config(page_title="LegalMind Analyzer", page_icon="🧠", layout="wide")
    st.title("🧠 LegalMind Legal Document Analyzer")

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = LegalRAGSystem()

    with st.sidebar:
        st.header("🧠 System Status")
        brain = st.session_state.rag_system.brain
        st.success("✅ API Connected") if brain.load_models() else st.error("❌ Token Missing")
        st.success("✅ Database Active")
        
        st.divider()
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader("Upload", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                if st.button(f"🔍 Analyze {file.name}"):
                    st.session_state.rag_system.process_document(file)
        
        if st.session_state.rag_system.vector_db:
            st.metric("📊 Document Chunks", st.session_state.rag_system.vector_db.count())

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Questions")
        quick_questions = ["What are the key obligations?", "Explain termination clauses", "Summarize payment terms"]
        q_cols = st.columns(len(quick_questions))
        for i, question in enumerate(quick_questions):
            if q_cols[i].button(question): st.session_state.current_question = question

        user_question = st.text_area("Custom Question:", height=80, placeholder="Ask something...")
        if st.button("🔍 Analyze", type="primary") and user_question:
            st.session_state.current_question = user_question

        if hasattr(st.session_state, 'current_question'):
            with st.spinner("Analyzing..."):
                result = st.session_state.rag_system.query(st.session_state.current_question)
                st.subheader("📋 Analysis Results")
                st.markdown(result['answer'])
                
                m1, m2 = st.columns(2)
                m1.metric("🎯 Relevance", f"{result['confidence']:.2f}")
                m2.metric("📄 Sources Found", len(result['sources']))

                if result['legal_concepts']:
                    st.subheader("📌 Key Terms")
                    cols = st.columns(4)
                    for i, concept in enumerate(result['legal_concepts'][:8]):
                        cols[i % 4].info(concept)

    with col2:
        st.header("📊 Analytics & Sources")
        if hasattr(st.session_state, 'current_question') and result['sources']:
            for i, source in enumerate(result['sources']):
                with st.expander(f"Source {i+1}: {source['filename']}"):
                    st.caption(source['text'][:300] + "...")

    st.warning("⚠️ Prototype only. Consult a legal professional.")

if __name__ == "__main__":
    main()
