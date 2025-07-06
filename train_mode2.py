# scripts/train_model.py

import os
import pandas as pd
import numpy as np
import faiss
import spacy
from PyPDF2 import PdfReader

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

def embed(texts):
    return np.array([nlp(text).vector for text in texts]).astype('float32')

# Paths
POLICY_FOLDER = "./data/policies/"
OUTPUT_DIR = "./models/faiss_indexes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Excel files
obligations = pd.read_excel("./data/obligation_mapping.xlsx")
kpci = pd.read_excel("./data/kpci.xlsx")
nfri = pd.read_excel("./data/nfri.xlsx")

# Combine text fields
obligations["combined_text"] = obligations["Obligation Title"].fillna("") + " " + obligations["Obligation Summary"].fillna("")
kpci["combined_text"] = kpci["Control Title"].fillna("") + " " + kpci["Control Description"].fillna("")
nfri["combined_text"] = (
    nfri["Issue Title"].fillna("") + " " +
    nfri["Issue Rationale"].fillna("") + " " +
    nfri["Resolution Summary"].fillna("")
)

# Load and read policy PDFs
def read_policy_pdfs(folder_path):
    ids, texts = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            try:
                path = os.path.join(folder_path, file)
                pdf = PdfReader(path)
                text = " ".join([page.extract_text() or "" for page in pdf.pages])
                ids.append(file.replace(".pdf", ""))
                texts.append(text)
            except Exception as e:
                print(f"Failed to read {file}: {e}")
    return ids, texts

policy_ids, policy_texts = read_policy_pdfs(POLICY_FOLDER)

# Embed
print("Embedding obligations...")
obligation_embeddings = embed(obligations["combined_text"])
print("Embedding policies...")
policy_embeddings = embed(policy_texts)
print("Embedding NFRIs...")
nfri_embeddings = embed(nfri["combined_text"])
print("Embedding KPCIs...")
kpci_embeddings = embed(kpci["combined_text"])

# Save FAISS indexes
def save_index(embeddings, ids, name):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, f"{name}.index"))
    pd.Series(ids).to_csv(os.path.join(OUTPUT_DIR, f"{name}_ids.csv"), index=False)

save_index(policy_embeddings, policy_ids, "policy")
save_index(nfri_embeddings, nfri["Issue ID"].tolist(), "nfri")
save_index(kpci_embeddings, kpci["Control ID"].tolist(), "kpci")

print("✅ FAISS indexes saved successfully.")
