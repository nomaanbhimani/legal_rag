import os, hashlib, json, re
from flask import Flask, render_template, request, jsonify
import chromadb
import numpy as np
import requests
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

# Configuration matches your OG Config class
HF_TOKEN = os.getenv("HF_TOKEN")
EMBED_MODEL = "nlpaueb/legal-bert-small-uncased"
QA_MODEL = "deepset/roberta-base-squad2" 

class LegalMindBrain:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        self.embed_url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL}"
        self.qa_url = f"https://api-inference.huggingface.co/models/{QA_MODEL}"

    def get_embeddings(self, text: str):
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        res = requests.post(self.embed_url, headers=self.headers, json=payload).json()
        try:
            # Matches your OG logic: extracting the CLS token equivalent
            arr = np.array(res)
            if len(arr.shape) == 3: return arr[0][0] # [Batch, Seq, Dim] -> [Dim]
            return arr[0] if len(arr.shape) == 2 else arr
        except: return np.zeros(768)

    def extract_legal_concepts(self, text: str):
        # Your exact Regex logic from the OG code
        patterns = [r'\b(shall|must|may|will)\s+\w+', r'\b(contract|agreement|clause|term|condition)\b',
                    r'\b(party|parties|plaintiff|defendant)\b', r'\b(liability|obligation|right|duty)\b']
        concepts = []
        for p in patterns:
            matches = re.findall(p, text, re.IGNORECASE)
            concepts.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        return list(set(concepts))

    def answer_question(self, question: str, context: str):
        payload = {"inputs": {"question": question, "context": context}, "options": {"wait_for_model": True}}
        res = requests.post(self.qa_url, headers=self.headers, json=payload).json()
        return {"answer": res.get("answer", "No answer found"), "confidence": res.get("score", 0.0)}

# Global Instance (Replaces st.session_state)
class LegalRAGSystem:
    def __init__(self):
        self.brain = LegalMindBrain()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.db = self.client.get_or_create_collection(name="legal_docs")

    def query(self, question: str):
        q_emb = self.brain.get_embeddings(question)
        results = self.db.query(query_embeddings=[q_emb.tolist()], n_results=3)
        if not results['documents'][0]: return {"answer": "No context found.", "concepts": []}
        
        context = " ".join(results['documents'][0])
        qa_res = self.brain.answer_question(question, context)
        concepts = self.brain.extract_legal_concepts(context)
        
        return {
            "answer": qa_res["answer"],
            "confidence": round(qa_res["confidence"], 2),
            "concepts": concepts[:5],
            "sources": results['metadatas'][0]
        }

rag = LegalRAGSystem()

@app.route('/')
def home(): return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    return jsonify(rag.query(data.get('question')))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
    
