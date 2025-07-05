# scripts/train_model.py

import os
import pandas as pd
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Constants
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
POLICY_FOLDER = "./data/policies/"
OUTPUT_DIR = "./models/faiss_indexes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Excel Files ===
obligations = pd.read_excel("./data/obligation_mapping.xlsx")
kpci = pd.read_excel("./data/kpci.xlsx")
nfri = pd.read_excel("./data/nfri.xlsx")

# === Extract Obligations Text ===
obligations["combined_text"] = obligations["Obligation Title"].fillna("") + " " + obligations["Obligation Summary"].fillna("")

# === Extract KPCI Text ===
kpci["combined_text"] = kpci["Control Title"].fillna("") + " " + kpci["Control Description"].fillna("")

# === Extract NFRI Text ===
nfri["combined_text"] = nfri["Issue Title"].fillna("") + " " + nfri["Issue Rationale"].fillna("") + " " + nfri["Resolution Summary"].fillna("")

# === Load Policy PDFs ===
def read_policy_pdfs(folder_path):
    policy_ids, policy_texts = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            try:
                pdf = PdfReader(file_path)
                text = " ".join([page.extract_text() or "" for page in pdf.pages])
                policy_id = file.replace(".pdf", "")
                policy_ids.append(policy_id)
                policy_texts.append(text)
            except Exception as e:
                print(f"❌ Failed to read {file}: {e}")
    return policy_ids, policy_texts

policy_ids, policy_texts = read_policy_pdfs(POLICY_FOLDER)

# === Embedding Helper ===
def embed_text(texts):
    return MODEL.encode(texts, show_progress_bar=True)

# === Embed & Save Function ===
def build_faiss_index(embeddings, ids, name):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, os.path.join(OUTPUT_DIR, f"{name}.index"))
    pd.Series(ids).to_csv(os.path.join(OUTPUT_DIR, f"{name}_ids.csv"), index=False)

# === Generate Embeddings ===
print("🔄 Embedding obligations...")
obligation_embeddings = embed_text(obligations["combined_text"])

print("🔄 Embedding policies...")
policy_embeddings = embed_text(policy_texts)

print("🔄 Embedding NFRIs...")
nfri_embeddings = embed_text(nfri["combined_text"])

print("🔄 Embedding KPCIs...")
kpci_embeddings = embed_text(kpci["combined_text"])

# === Save Vector Indexes ===
build_faiss_index(policy_embeddings, policy_ids, "policy")
build_faiss_index(nfri_embeddings, nfri["Issue ID"].tolist(), "nfri")
build_faiss_index(kpci_embeddings, kpci["Control ID"].tolist(), "kpci")

print("✅ Model training & FAISS indexes saved.")
