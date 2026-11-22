# Zero-Hallucination YouTube AI Learning Suite

A bilingual (English & Roman Urdu) YouTube video learning assistant powered by Hierarchical Chapter Indexing, Hybrid Retrieval (Dense Embeddings + BM25), Local SQuAD 2.0 Extractive Verification, and Gemini synthesis.

## Key Features
- **Hierarchical Chapter Indexing:** Segments video transcripts into chronological macro chapters with LLM-generated topic headings and 500-character overlapping chunks.
- **Hybrid Retrieval & Guardrails:** Combines `BAAI/bge-small-en-v1.5` cosine similarity with `BM25Okapi` keyword scoring via Reciprocal Rank Fusion (RRF) and a normalized cosine distance threshold (`1.12`) to block out-of-topic queries.
- **Local SQuAD 2.0 Fast Path:** Integrates a fine-tuned DistilBERT QA model ([`Dilawer-khan/distilbert-squad2-qa`](https://huggingface.co/Dilawer-khan/distilbert-squad2-qa)) that verifies exact answer spans and calculates no-answer probabilities locally before invoking an LLM. High-confidence factual answers (`>= 85%`) bypass the LLM API entirely.
- **Bilingual Q&A (English & Roman Urdu):** Automatically detects Roman Urdu queries, translates them for retrieval, and generates grounded responses with clickable timestamp jump links and highlighted transcript proof.
- **Interactive Quiz Generator:** Creates 5 or 10 video-grounded multiple-choice questions (MCQs) with instant grading and explanations.
- **Structured Study Notes:** Compiles chronological, chapter-by-chapter study notes in English or Roman Urdu with one-click Markdown (`.md`) download.
- **Multi-Video Isolation:** Stores each video in an isolated ChromaDB collection (`vid_<video_id>`) for instant switching without cross-video contamination.

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Akhoon-CodeNinja/youtube-ai-rag-assistant.git](https://github.com/Akhoon-CodeNinja/youtube-ai-rag-assistant.git)
cd youtube-ai-rag-assistant
```

### 2. Create a Virtual Environment & Install Dependencies
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
*Note: On the first launch, the app will automatically download and cache `BAAI/bge-small-en-v1.5` (~130 MB) and the fine-tuned SQuAD 2.0 verifier `Dilawer-khan/distilbert-squad2-qa` (~265 MB) from Hugging Face Hub. Subsequent startups take ~2 seconds.*

---

## How to Use
1. Open the app in your browser at `http://localhost:8501`.
2. Enter your **Gemini API Key** in the left sidebar (obtain a free key from [Google AI Studio](https://aistudio.google.com/)).
3. Paste any **YouTube Video URL** and click **Process Video**.
4. Once indexed, use the three tabs:
   - **Q&A Assistant:** Ask factual or conceptual questions in English or Roman Urdu.
   - **Quiz Generator:** Generate and submit interactive MCQs based strictly on the video.
   - **Study Notes:** Generate and download structured `.md` revision notes.

---

## Model Training & Evaluation Notebook
- The complete fine-tuning and SQuAD 2.0 evaluation pipeline (`EM`, `F1`, and `no_answer_probability` calibration) for `Dilawer-khan/distilbert-squad2-qa` is included in [`notebooks/distilbert_squad2_training_and_eval.ipynb`](notebooks/distilbert_squad2_training_and_eval.ipynb).
