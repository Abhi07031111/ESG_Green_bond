# scripts/predict.py

import faiss
import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = "./models/faiss_indexes/"
nlp = spacy.load("en_core_web_md")

# === Embed with SpaCy
def embed_spacy(text):
    return nlp(text).vector.astype('float32')

# === Load FAISS index + ID list
def load_index(name):
    index = faiss.read_index(f"{INDEX_DIR}{name}.index")
    ids = pd.read_csv(f"{INDEX_DIR}{name}_ids.csv", header=None)[0].tolist()
    return index, ids

# === Hybrid Search Function
def hybrid_search(text, index, ids, corpus_texts, top_k=3):
    # Vector-based (SpaCy)
    vector = np.array([embed_spacy(text)])
    _, faiss_indices = index.search(vector, top_k)

    # TF-IDF based
    tfidf = TfidfVectorizer().fit(corpus_texts)
    tfidf_vecs = tfidf.transform(corpus_texts)
    query_tfidf = tfidf.transform([text])
    cosine_scores = cosine_similarity(query_tfidf, tfidf_vecs).flatten()
    tfidf_top_k = cosine_scores.argsort()[-top_k:][::-1]

    # Combine results (weighted average)
    scores = {}
    for i in faiss_indices[0]:
        scores[ids[i]] = scores.get(ids[i], 0) + 1.0  # FAISS weight
    for i in tfidf_top_k:
        scores[ids[i]] = scores.get(ids[i], 0) + 0.7  # TF-IDF weight

    # Return top final matches
    return sorted(scores, key=scores.get, reverse=True)[:top_k]

# === Load Indexes and Corpora
policy_index, policy_ids = load_index("policy")
nfri_index, nfri_ids = load_index("nfri")
kpci_index, kpci_ids = load_index("kpci")

policy_corpus = pd.read_csv(f"{INDEX_DIR}policy_ids.csv", header=None)[0].tolist()
nfri_corpus = pd.read_csv(f"{INDEX_DIR}nfri_ids.csv", header=None)[0].tolist()
kpci_corpus = pd.read_csv(f"{INDEX_DIR}kpci_ids.csv", header=None)[0].tolist()

# === Input Obligation
title = "Encrypt financial customer records"
summary = "Ensure that all financial records are stored securely with proper encryption mechanisms and backups."
combined = title + " " + summary

# === Predict
predicted_policies = hybrid_search(combined, policy_index, policy_ids, policy_corpus)
predicted_nfris = hybrid_search(combined, nfri_index, nfri_ids, nfri_corpus)
predicted_kpcis = hybrid_search(combined, kpci_index, kpci_ids, kpci_corpus)

# === Output
print("🔐 Predicted Policy IDs:", predicted_policies)
print("⚠️  Predicted NFRI IDs:", predicted_nfris)
print("🛡️  Predicted KPCI IDs:", predicted_kpcis)
