# scripts/predict.py

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_DIR = "./models/faiss_indexes/"

def load_index(index_name):
    index = faiss.read_index(f"{INDEX_DIR}{index_name}.index")
    ids = pd.read_csv(f"{INDEX_DIR}{index_name}_ids.csv", header=None)[0].tolist()
    return index, ids

def predict_top_k(text, index, ids, k=3):
    embedding = MODEL.encode([text])
    D, I = index.search(embedding, k)
    return [ids[i] for i in I[0]]

# === Load Indexes ===
policy_index, policy_ids = load_index("policy")
nfri_index, nfri_ids = load_index("nfri")
kpci_index, kpci_ids = load_index("kpci")

# === Input Obligation ===
new_obligation_title = "Encrypt financial customer records"
new_obligation_summary = "Ensure that all financial records are stored securely with proper encryption mechanisms and backups."

combined_input = new_obligation_title + " " + new_obligation_summary

# === Predict ===
predicted_policies = predict_top_k(combined_input, policy_index, policy_ids)
predicted_nfris = predict_top_k(combined_input, nfri_index, nfri_ids)
predicted_kpcis = predict_top_k(combined_input, kpci_index, kpci_ids)

# === Output ===
print("🔮 Predicted Policy IDs:", predicted_policies)
print("🔮 Predicted NFRI IDs:", predicted_nfris)
print("🔮 Predicted KPCI IDs:", predicted_kpcis)
