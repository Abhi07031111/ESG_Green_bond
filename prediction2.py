# scripts/predict.py

import faiss
import pandas as pd
import spacy
import numpy as np

INDEX_DIR = "./models/faiss_indexes/"
nlp = spacy.load("en_core_web_md")

def embed(text):
    return nlp(text).vector.astype('float32')

def load_index(name):
    index = faiss.read_index(f"{INDEX_DIR}{name}.index")
    ids = pd.read_csv(f"{INDEX_DIR}{name}_ids.csv", header=None)[0].tolist()
    return index, ids

def search(text, index, ids, top_k=3):
    vector = np.array([embed(text)]).astype('float32')
    _, indices = index.search(vector, top_k)
    return [ids[i] for i in indices[0]]

# Load FAISS indexes
policy_index, policy_ids = load_index("policy")
nfri_index, nfri_ids = load_index("nfri")
kpci_index, kpci_ids = load_index("kpci")

# Input obligation
title = "Encrypt financial customer records"
summary = "Ensure that all financial records are stored securely with proper encryption mechanisms and backups."
combined = title + " " + summary

# Predict
predicted_policies = search(combined, policy_index, policy_ids)
predicted_nfris = search(combined, nfri_index, nfri_ids)
predicted_kpcis = search(combined, kpci_index, kpci_ids)

# Output
print("🔐 Predicted Policy IDs:", predicted_policies)
print("⚠️  Predicted NFRI IDs:", predicted_nfris)
print("🛡️  Predicted KPCI IDs:", predicted_kpcis)
