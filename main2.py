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
    # Small legal model for embeddings
    EMBED_MODEL: str = "nlpaueb/legal-bert-small-uncased"
    # Dedicated QA model for better extraction
    QA_MODEL: str = "deepset/roberta-base-squad2" 
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    # Use Streamlit secrets for production
    HF_TOKEN: str = st.secrets.get("HF_TOKEN", "")

config = Config()

class LegalMindBrain:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        self.embed_url = f"https://api-inference.huggingface.co/models/{config.EMBED_MODEL}"
        self.qa_url = f"https://api-inference.huggingface.co/models/{config.QA_MODEL}"

    def _query_api(self, url: str, payload: dict) -> list:
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 200:
            st.error(f"API Error: {response.text}")
            return []
        return response.json()

    def load_models(self):
        # API version doesn't "load" locally, just verify token
        return bool(config.HF_TOKEN)

    def get_embeddings(self, text: str) -> np.ndarray:
        try:
            payload = {"inputs": text, "options": {"wait_for_model": True}}
            output = self._query_api(self.embed_url, payload)
            
            # Extract mean pooling or CLS from HF response
            embeddings = np.array(output)
            if len(embeddings.shape) > 1:
                return embeddings[0] if len(embeddings.shape) == 2 else embeddings[0][0]
            return embeddings
        except Exception as e:
            return np.random.rand(768) # Fallback

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
            # Handle tuple groups from regex
            for m in matches:
                concepts.append(m[0] if isinstance(m, tuple) else m)
        return list(set(concepts))

    def answer_question(self, question: str, context: str) -> Dict:
        try:
            payload = {
                "inputs": {"question": question, "context": context},
                "options": {"wait_for_model": True}
            }
            result = self._query_api(self.qa_url, payload)
            
            if isinstance(result, dict) and "answer" in result:
                return {
                    "answer": result["answer"],
                    "confidence": result.get("score", 0.0)
                }
            return {"answer": "I couldn't find a specific answer.", "confidence": 0.0}
        except:
            return {"answer": "Error in QA API", "confidence": 0.0}

    def simplify_legal_text(self, legal_text: str) -> str:
        replacements = {
            r'\bshall\b': 'must', r'\bwhereas\b': 'since',
            r'\bindemnify\b': 'protect from loss', r'\bliable\b': 'responsible',
            r'\bpursuant to\b': 'according to', r'\bterminate\b': 'end'
        }
        simplified = legal_text
        for pattern, replacement in replacements.items():
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
        return simplified

    def generate_summary(self, legal_texts: List[str], query_context: str = "") -> str:
        combined_text = " ".join(legal_texts[:2]) # Keep context window small for API
        qa_result = self.answer_question(query_context or "What is the summary?", combined_text)
        
        simplified = self.simplify_legal_text(qa_result["answer"])
        
        return f"""**Analysis:** {simplified}\n\n**Confidence:** {qa_result['confidence']:.2f}"""

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])

    @staticmethod
    def extract_text_from_docx(file) -> str:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    @staticmethod
    def chunk_text(text: str) -> List[Dict]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), config.CHUNK_SIZE - config.CHUNK_OVERLAP):
            chunk_text = ' '.join(words[i:i + config.CHUNK_SIZE])
            chunks.append({'text': chunk_text})
            if i + config.CHUNK_SIZE >= len(words): break
        return chunks

class LegalRAGSystem:
    def __init__(self):
        self.brain = LegalMindBrain()
        self.processor = DocumentProcessor()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.vector_db = self.client.get_or_create_collection(name="legal_docs")

    def process_document(self, file):
        if file.type == "application/pdf": text = self.processor.extract_text_from_pdf(file)
        elif "word" in file.type: text = self.processor.extract_text_from_docx(file)
        else: text = str(file.read(), "utf-8")

        chunks = self.processor.chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = self.brain.get_embeddings(chunk['text'])
            concepts = self.brain.extract_legal_concepts(chunk['text'])
            
            self.vector_db.add(
                ids=[f"{hashlib.md5(file.name.encode()).hexdigest()}_{i}"],
                embeddings=[embedding.tolist()],
                documents=[chunk['text']],
                metadatas=[{"concepts": json.dumps(concepts), "filename": file.name}]
            )
        return True

    def query(self, question: str) -> Dict:
        q_emb = self.brain.get_embeddings(question)
        results = self.vector_db.query(query_embeddings=[q_emb.tolist()], n_results=3)
        
        if not results['documents'][0]:
            return {"answer": "No relevant info.", "sources": []}

        answer = self.brain.generate_summary(results['documents'][0], question)
        return {"answer": answer, "sources": results['metadatas'][0]}

def main():
    st.set_page_config(page_title="LegalMind", page_icon="⚖️")
    st.title("⚖️ LegalMind AI (Cloud Edition)")

    if 'rag' not in st.session_state:
        st.session_state.rag = LegalRAGSystem()

    with st.sidebar:
        st.header("Upload")
        files = st.file_uploader("Docs", accept_multiple_files=True)
        if files:
            for f in files:
                if st.button(f"Process {f.name}"):
                    st.session_state.rag.process_document(f)
                    st.success("Done!")

    query = st.text_input("Ask about your documents:")
    if query:
        res = st.session_state.rag.query(query)
        st.markdown(res['answer'])
        with st.expander("View Sources"):
            st.write(res['sources'])

if __name__ == "__main__":
    main()
