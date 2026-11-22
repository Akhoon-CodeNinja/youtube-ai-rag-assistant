# Zero-Hallucination YouTube AI Learning Suite

A multi-video YouTube RAG assistant featuring Hierarchical Chapter Indexing, Hybrid Search (Dense BGE Embeddings + BM25), Local SQuAD 2.0 DistilBERT Extractive Verification, Bilingual Support (English & Roman Urdu), an Interactive MCQ Quiz Generator, and Study Notes Export.

## Architecture
1. **Hierarchical Indexing:** Segments video transcripts into macro chapters with LLM-generated headings and 500-character overlapping chunks.
2. **Hybrid Retrieval + Guardrails:** Combines `BAAI/bge-small-en-v1.5` cosine similarity with `BM25Okapi` keyword scoring via Reciprocal Rank Fusion (RRF) and normalized distance thresholding (`1.12`).
3. **Local SQuAD 2.0 Fast Path:** Uses a fine-tuned `distilbert-base-uncased` QA model to verify answer spans and compute no-answer probabilities locally before invoking an LLM.
4. **Learning Tools:** Generates video-grounded MCQ quizzes and downloadable Markdown study notes in English or Roman Urdu.

## Setup & Installation
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Gemini API key:
   ```bash
   set GEMINI_API_KEY=your_api_key_here
   ```
3. *(Optional)* Place your fine-tuned SQuAD 2.0 DistilBERT model folder (`final_distilbert_model`) in the project root to enable local extractive verification.
4. Run the application:
   ```bash
   streamlit run app.py
   ```
