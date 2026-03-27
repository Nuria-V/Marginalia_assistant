# Marginalia — Your Personal Literary Assistant

> A personal literary analysis system that learns from your reading notes,
> builds your unique reader profile, and recommends books through conversation.

<!-- Add app screenshot here after deployment -->

---

## What is Marginalia?

Marginalia are the handwritten notes readers leave in the margins of books —
personal traces of thought, reaction, and meaning. This project takes that
idea into the digital age.

You write notes about books you have read. Marginalia analyzes those notes,
discovers patterns in your reading preferences, builds a narrative-emotional
reader profile, and uses it to recommend books you have never read but are
likely to love. You can also ask the system questions about your own reading
history in natural language.

---

## Live Demo

**[Try Marginalia on Streamlit](#)** *(link after deployment)*

---

## Pipeline Overview

Marginalia integrates five technologies in a sequential pipeline:

```
Personal notes
     │
     ▼
[TF-IDF]  ──────────────────── Keyword extraction & cluster labeling
     │
     ▼
[BERT embeddings]  ──────────── Semantic vectorization of notes (768 dims)
     │
     ▼
[K-Means clustering]  ────────── Reader profile: 5 narrative clusters
     │
     ▼
[Recommendation engine]  ─────── Cosine similarity vs 78,811-book catalog
     │
     ▼
[RAG + GPT-4o]  ──────────────── Conversational interface over your notes
     │
     ▼
[Streamlit frontend]  ────────── Interactive web app
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| NLP — keyword analysis | `scikit-learn` TF-IDF | Vectorize notes, extract cluster keywords |
| NLP — semantic embeddings | `sentence-transformers` BERT (`all-mpnet-base-v2`, 768 dims) | Deep semantic representation of notes and catalog |
| ML — unsupervised learning | `scikit-learn` K-Means (K=5) | Cluster notes into reader profile groups |
| Recommendation engine | Cosine similarity | Match reader profile against 78,811 books |
| Generative AI — RAG | OpenAI GPT-4o + BERT retriever | Conversational interface over personal notes |
| Frontend | Streamlit | Interactive web application |
| Data | Google Books API + Goodreads database | Book catalog (title, author, description, pages, year, ISBN) |
| Deployment | Streamlit Community Cloud + GitHub | Production hosting and CI/CD |
| Artifact storage | Hugging Face Datasets (private) | Store `.pkl` embeddings and processed CSVs |

---

## Reader Profile

The system analyzes reading notes and groups them into 5 narrative clusters
using K-Means on BERT embeddings. Each cluster becomes a dimension of the
reader profile:

| Cluster | Label | Sample books |
|---|---|---|
| 0 | Inner Worlds | Ask the Dust, Crime and Punishment, Factotum, The Stranger, Notes from Underground |
| 1 | Speculative Minds | Dune, Sapiens, Ubik, Flowers for Algernon, The Man in the High Castle |
| 2 | Political & Dystopian | 1984, Brave New World, Frankenstein, Animal Farm, The Catcher in the Rye |
| 3 | The Seeker's Path | The Name of the Wind, The Little Prince, Siddhartha, Pride and Prejudice, Perfume |
| 4 | The Dark & The Strange | Let the Right One In, The Metamorphosis, The Raven, The Fall of the House of Usher |

Cluster quality was evaluated using silhouette score across K=2 to K=6.
K=5 was selected as the optimal balance between mathematical fit and
narrative coherence.

---

## Features

**Reader Profile sidebar**
Persistent panel showing your 5 reading clusters with narrative labels,
theme keywords, and average ratings — always visible regardless of the active tab.

**Recommendations tab**
Top-6 book recommendations per cluster, pre-calculated via cosine similarity
between cluster centroids and 78,811 BERT-encoded catalog descriptions.
Each card shows title, author, description preview, and a similarity score
visualized as a progress bar.

**Ask your library tab**
Conversational RAG interface powered by GPT-4o. Ask anything about your
reading history in English. The system retrieves the most semantically
relevant notes and catalog books, builds a contextualized prompt, and
generates a personalized response. Includes suggested questions and full
conversation history within the session.

**Add a note tab**
Form to add new reading notes via `NotesManager`. Validated fields:
title, author, reflection text (min. 20 chars), rating (1–5), tags,
and date read. New notes feed into future pipeline updates.

---

## Project Structure

```
marginalia/
├── app.py                  # Streamlit frontend — Phase 7
├── rag_engine.py           # RAG engine (BERT retriever + GPT-4o) — Phase 6
├── notes_manager.py        # Note lifecycle management — Phase 2
├── requirements.txt        # Python dependencies
└── README.md
```

Artifact files (`.pkl`, `.csv`) are stored in a private Hugging Face Dataset
and downloaded to `/tmp` at app startup via `hf_hub_download`.

---

## Development Phases

The project was built across 8 sequential phases in Google Colab:

| Phase | Description | Output |
|---|---|---|
| 1 | Dataset setup — Google Books API + Goodreads, cleaning, deduplication | `books_clean.csv` — 78,811 books |
| 2 | Notes collection — `NotesManager`, validation, CSV schema | `user_notes.csv` — 50 personal reading notes |
| 3 | NLP processing — incremental BERT + TF-IDF pipeline | `bert_embeddings.pkl`, `tfidf_matrix.pkl` |
| 4 | Clustering — K-Means, silhouette analysis, reader profile | `reader_profile.pkl`, `user_notes_clustered.csv` |
| 5 | Recommendation engine — catalog embeddings, cosine similarity | `catalog_embeddings.pkl`, `recommendations.pkl` |
| 6 | RAG — BERT retriever, GPT-4o generator, `RAGEngine` class | `rag_engine.py` |
| 7 | Streamlit frontend — tabs, sidebar, chat interface | `app.py` |
| 8 | Deployment — GitHub + Streamlit Community Cloud | Live app |

---

## Deployment Architecture

```
GitHub repo (marginalia)
     │
     │  connected via Streamlit GitHub integration
     ▼
Streamlit Community Cloud (public)
     │
     │  hf_hub_download() at startup
     ▼
HF Dataset: nm-valles-00/marginalia-data (private)
     └── bert_embeddings.pkl
     └── catalog_embeddings.pkl
     └── reader_profile.pkl
     └── recommendations.pkl
     └── cluster_model.pkl
     └── tfidf_matrix.pkl
     └── tfidf_vectorizer.pkl
     └── user_notes_clustered.csv
     └── books_clean.csv
```

Secrets configured in Streamlit Community Cloud:
- `HF_TOKEN` — read access to private dataset
- `HF_DATASET_REPO` — dataset repository id
- `OPENAI_API_KEY` — GPT-4o access

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/nm-valles-00/marginalia.git
cd marginalia

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="hf_..."
export HF_DATASET_REPO="nm-valles-00/marginalia-data"
export OPENAI_API_KEY="sk-..."

# Run
streamlit run app.py
```

On first run, artifacts are downloaded from HF Dataset to `/tmp/literary_data`.
Subsequent runs use the local cache.

---

## Updating the Pipeline

When new reading notes are added, re-run Phases 3–5 in Colab to regenerate
embeddings and recommendations, then re-upload the updated `.pkl` files to
the HF Dataset. The app picks up the new artifacts on next startup.

```
Add notes (Phase 2)
    → Re-run Phase 3 (BERT + TF-IDF)
    → Re-run Phase 4 (K-Means clustering)
    → Re-run Phase 5 (recommendation engine)
    → Re-upload .pkl files to HF Dataset
    → Restart Streamlit app
```

Catalog embeddings (`catalog_embeddings.pkl`) never need to be regenerated
unless the book catalog itself changes.

---

## Dataset

The book catalog was built by combining the **Google Books API** and a
**Goodreads database**. After cleaning, year filtering, description length
filtering (100–2500 chars), and fuzzy deduplication (via `rapidfuzz`),
the final dataset contains **78,811 books** with fields:
`title`, `description`, `author_names`, `detail_categories`, `num_pages`,
`publication_year`, `isbn`, `language_code`.

The user notes dataset contains **50 personal reading notes** covering 50 books
across all 5 reader profile clusters.

---

## Author

**Nuria** — Data Science & ML Bootcamp Final Project

---

## License

MIT
