"""
rag_engine.py
=============
RAG engine for the Literary Analysis System - Marginalia.
Imported by Streamlit in Phase 7.

Usage:
    from rag_engine import RAGEngine
    rag = RAGEngine(drive_path, openai_api_key)
    response = rag.ask("What themes appear in my notes?")
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Encapsulates the full RAG pipeline for use in Streamlit.
    Loads all data on initialization and exposes ask() as the main interface.
    """

    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, drive_path: str, openai_api_key: str):
        self.drive_path = drive_path
        self.client     = OpenAI(api_key=openai_api_key)
        self._load_data()
        self._load_bert()
        logger.info("RAGEngine initialized.")

    def _load_pickle(self, filename: str) -> object:
        """Carga un archivo pickle desde drive_path. Lanza FileNotFoundError si no existe."""
        filepath = os.path.join(self.drive_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filename} not found at {filepath}.")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _load_data(self):
        """Carga embeddings, DataFrames y perfil lector."""
        self.bert_embeddings    = normalize(self._load_pickle("bert_embeddings.pkl"), norm="l2")
        self.catalog_embeddings = self._load_pickle("catalog_embeddings.pkl")
        self.reader_profile     = self._load_pickle("reader_profile.pkl")
        self.df_notas = pd.read_csv(os.path.join(self.drive_path, "user_notes_clustered.csv"))
        self.df_catalog = pd.read_csv(os.path.join(self.drive_path, "books_clean.csv"))
        self.titles_read = set(self.df_notas["title"].str.lower().str.strip().tolist())
        logger.info("Data loaded. Notes: %d | Catalog: %d", len(self.df_notas), len(self.df_catalog))

    def _load_bert(self):
        """Carga el modelo BERT para vectorizar queries."""
        self.tokenizer  = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.bert_model.eval()
        logger.info("BERT model loaded: %s", self.MODEL_NAME)

    def _embed_query(self, text: str) -> np.ndarray:
        """Convierte una query en un embedding BERT normalizado de 768 dimensiones."""
        inputs = self.tokenizer(text, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        token_emb = outputs.last_hidden_state          # (1, n_tokens, 768)
        mask      = inputs["attention_mask"]
        mask_exp  = mask.unsqueeze(-1).expand(token_emb.size()).float()
        emb = (token_emb * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return normalize(emb.squeeze().numpy().reshape(1, -1), norm="l2")[0]

    def retrieve(self, query: str, top_k_notes: int = 3, top_k_books: int = 5) -> dict:
        """Recupera las notas y libros del catálogo más relevantes para una query."""
        query_vec    = self._embed_query(query)
        note_sims    = cosine_similarity(query_vec.reshape(1, -1), self.bert_embeddings)[0]
        catalog_sims = cosine_similarity(query_vec.reshape(1, -1), self.catalog_embeddings)[0]

        top_notes = []
        for idx in note_sims.argsort()[::-1][:top_k_notes]:
            row = self.df_notas.iloc[idx]
            top_notes.append({
                "title":      row["title"],
                "author":     row["author"],
                "note_text":  row["note_text"],
                "rating":     row["rating"],
                "tags":       row.get("tags", ""),
                "similarity": round(float(note_sims[idx]), 4)
            })

        top_books = []
        for idx in [
            i for i in catalog_sims.argsort()[::-1]
            if str(self.df_catalog.iloc[i]["title"]).lower().strip() not in self.titles_read
        ][:top_k_books]:
            row = self.df_catalog.iloc[idx]
            top_books.append({
                "title":       row["title"],
                "authors":     str(row["author_names"]).strip("[]'\""),
                "description": str(row["description"])[:300],
                "similarity":  round(float(catalog_sims[idx]), 4)
            })

        return {"notes": top_notes, "books": top_books}

    def ask(self, query: str, top_k_notes: int = 3, top_k_books: int = 5) -> str:
        """Interfaz principal: pregunta en lenguaje natural → respuesta GPT-4o personalizada."""
        context  = self.retrieve(query, top_k_notes, top_k_books)

        system_prompt = (
            "You are a personal literary assistant with deep knowledge "
            "of the user's reading history and preferences.\n\n"
            "Your role:\n"
            "- Answer questions about the user's notes and reading patterns\n"
            "- Recommend books from the catalog that match the user's profile\n"
            "- Help rediscover forgotten ideas from past readings\n\n"
            "Rules:\n"
            "- Base answers ONLY on the context provided\n"
            "- Be warm, specific, reference the user's actual notes\n"
            "- When recommending, always explain WHY it matches the profile\n"
            "- Keep responses concise but insightful\n"
            f"- IMPORTANT: Always respond in English, regardless of the language of the question."
        )

        profile_summary = "USER'S READING PROFILE:\n"
        for key, cl in self.reader_profile.items():
            label = cl.get("label", f"Cluster {cl['cluster_id']}")
            profile_summary += (
                f"- {label}: "
                f"Books: {', '.join(cl['books'])} | "
                f"Themes: {', '.join(cl['top_words'][:4])} | "
                f"Avg rating: {cl['avg_rating']}/5\n"
            )

        notes_ctx = "\nRELEVANT PERSONAL NOTES:\n"
        for n in context["notes"]:
            notes_ctx += (
                f"Book: '{n['title']}' by {n['author']} "
                f"(Rating: {n['rating']}/5)\n"
                f"Note: {n['note_text']}\n\n"
            )

        books_ctx = "\nRELEVANT CATALOG BOOKS:\n"
        for b in context["books"]:
            books_ctx += (
                f"Title: '{b['title']}' by {b['author_names']}\n"
                f"Description: {b['description']}\n\n"
            )

        user_prompt = (
            profile_summary + notes_ctx + books_ctx +
            f"\nUSER QUESTION: {query}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content
