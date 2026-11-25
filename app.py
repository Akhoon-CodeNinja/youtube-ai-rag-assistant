import os
import json
import time
import urllib.request
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

app = FastAPI(title="AI Video Learning Suite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
DB_DIR = "./chroma_db"
METADATA_FILE = "videos_metadata.json"
SQUAD_MODEL_PATH = "Dilawer-khan/distilbert-squad2-qa"
DISTANCE_THRESHOLD = 1.12
SQUAD_FAST_PATH_CONFIDENCE = 85.0

print("Starting AI Retrieval Engine (BGE-small)...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading Local Fine-Tuned SQuAD 2.0 DistilBERT Verifier...")
qa_tokenizer = AutoTokenizer.from_pretrained(SQUAD_MODEL_PATH)
qa_model = AutoModelForQuestionAnswering.from_pretrained(SQUAD_MODEL_PATH).to("cpu").eval()
print("Backend Ready!")


class IndexRequest(BaseModel):
    url: str
    api_key: str = ""


class AskRequest(BaseModel):
    video_id: str
    question: str
    api_key: str = ""


class GenerateRequest(BaseModel):
    video_id: str
    language: str = "English"
    num_questions: int = 5
    api_key: str = ""


def load_video_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_video_metadata(video_id, title):
    data = load_video_metadata()
    data[video_id] = title
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_youtube_title(video_id):
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            info = json.loads(response.read().decode("utf-8"))
            return info.get("title", f"Video ({video_id})")
    except Exception:
        return f"Video ({video_id})"


def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None


def get_db(video_id):
    collection_name = f"vid_{video_id.replace('-', '_')}"
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        persist_directory=DB_DIR,
    )


def is_roman_urdu(text):
    roman_urdu_markers = {
        "kya", "kia", "kyun", "kese", "kaise", "hai", "hain", "tha", "thi",
        "mein", "ko", "ka", "ki", "ke", "se", "aur", "batao", "batayein",
        "konsa", "kitne", "kitni", "iska", "uska", "hum", "aap", "nahi", "nhi",
        "wala", "wali", "wale", "ho", "raha", "rahi", "kar", "kr", "smjha"
    }
    words = {w.lower().strip("?.!,") for w in text.split()}
    return len(words.intersection(roman_urdu_markers)) >= 1


def fast_translate_to_english(roman_text, api_key):
    if not api_key:
        return roman_text
    client = genai.Client(api_key=api_key, http_options={"timeout": 20000})
    prompt = (
        "Translate the following Roman Urdu question into clear, natural English. "
        "Return ONLY the English translation and nothing else.\n\n"
        f"Roman Urdu: {roman_text}"
    )
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            if res and res.text:
                return res.text.strip()
        except Exception:
            continue
    return roman_text


def generate_chapter_headings(sections, api_key):
    if not api_key:
        return {str(i): f"Section {i + 1} ({int(s['start'])}s - {int(s['end'])}s)" for i, s in enumerate(sections)}

    client = genai.Client(api_key=api_key, http_options={"timeout": 45000})
    section_summaries = []
    for idx, sec in enumerate(sections):
        preview = sec["text"][:1500]
        section_summaries.append(f"Section {idx} ({int(sec['start'])}s - {int(sec['end'])}s):\n{preview}")

    prompt = (
        "Read the following chronological sections of a YouTube video transcript. "
        "For each section, write ONE concise, descriptive topic heading (10-18 words) summarizing the main concepts, "
        "examples, and terms discussed in that section. "
        "Return ONLY a valid JSON object mapping section index strings ('0', '1', ...) to their heading strings.\n\n"
        + "\n\n---\n\n".join(section_summaries)
    )

    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            if response and response.text:
                return json.loads(response.text)
        except Exception:
            continue

    return {str(i): f"Section {i + 1} ({int(s['start'])}s - {int(s['end'])}s)" for i, s in enumerate(sections)}


def build_hierarchical_chunks(transcript, api_key):
    full_text = ""
    char_to_time = []

    for item in transcript:
        text = (item.text if hasattr(item, "text") else item["text"]).replace("\n", " ").strip()
        start = item.start if hasattr(item, "start") else item["start"]
        if not text:
            continue
        start_idx = len(full_text)
        full_text += text + " "
        char_to_time.append((start_idx, start))

    total_len = len(full_text)
    if total_len == 0:
        return []

    num_sections = min(6, max(1, total_len // 2500))
    section_size = total_len // num_sections
    sections = []

    for i in range(num_sections):
        s_start = i * section_size
        s_end = total_len if i == num_sections - 1 else (i + 1) * section_size
        sec_text = full_text[s_start:s_end].strip()

        t_start, t_end = 0, 0
        for c_idx, ts in char_to_time:
            if c_idx <= s_start:
                t_start = ts
            if c_idx <= s_end:
                t_end = ts

        sections.append({"start": t_start, "end": t_end, "text": sec_text, "char_start": s_start, "char_end": s_end})

    headings_map = generate_chapter_headings(sections, api_key)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[". ", "? ", "! ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    documents = []
    search_cursor = 0
    for chunk in chunks:
        pos = full_text.find(chunk, search_cursor)
        if pos == -1:
            pos = full_text.find(chunk)
        else:
            search_cursor = pos + max(1, len(chunk) - 100)

        chunk_time = 0
        for char_idx, timestamp in char_to_time:
            if char_idx <= pos:
                chunk_time = timestamp
            else:
                break

        chapter_heading = "General Overview"
        for idx, sec in enumerate(sections):
            if sec["char_start"] <= pos <= sec["char_end"]:
                chapter_heading = headings_map.get(str(idx), f"Section {idx + 1}")
                break

        contextualized_content = f"[Chapter Topic: {chapter_heading}]\n{chunk.strip()}"
        documents.append(
            Document(
                page_content=contextualized_content,
                metadata={"start_time": chunk_time, "chapter": chapter_heading},
            )
        )

    return documents


def hierarchical_hybrid_retrieve(db, query, k=4):
    all_records = db.get()
    all_texts = all_records.get("documents", [])
    all_metas = all_records.get("metadatas", [])

    if not all_texts:
        return [], 2.0

    dense_results = db.similarity_search_with_score(query, k=min(12, len(all_texts)))
    best_dense_score = dense_results[0][1] if dense_results else 2.0

    if best_dense_score > DISTANCE_THRESHOLD:
        return [], best_dense_score

    tokenized_corpus = [doc.lower().split() for doc in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:12]

    rrf_scores = {}
    doc_lookup = {}

    for rank, (doc, _) in enumerate(dense_results):
        key = doc.page_content
        doc_lookup[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))

    for rank, idx in enumerate(bm25_ranked_indices):
        if bm25_scores[idx] > 0:
            key = all_texts[idx]
            if key not in doc_lookup:
                doc_lookup[key] = Document(page_content=key, metadata=all_metas[idx])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))

    ranked_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

    text_to_idx = {text: idx for idx, text in enumerate(all_texts)}
    selected_indices = set()
    for key in ranked_keys:
        idx = text_to_idx.get(key)
        if idx is not None:
            for neighbor in (idx - 1, idx, idx + 1):
                if 0 <= neighbor < len(all_texts):
                    selected_indices.add(neighbor)

    sorted_indices = sorted(selected_indices)
    final_docs = []
    if sorted_indices:
        group = [sorted_indices[0]]
        for idx in sorted_indices[1:]:
            if idx == group[-1] + 1:
                group.append(idx)
            else:
                merged_text = "\n".join(all_texts[i] for i in group)
                final_docs.append(Document(page_content=merged_text, metadata=all_metas[group[0]]))
                group = [idx]
        merged_text = "\n".join(all_texts[i] for i in group)
        final_docs.append(Document(page_content=merged_text, metadata=all_metas[group[0]]))

    return final_docs, best_dense_score


def verify_with_squad_distilbert(question, context_chunks):
    t0 = time.perf_counter()
    best_overall_span = ""
    best_quote_sentence = ""
    best_timestamp = 0
    lowest_no_ans_prob = 1.0

    for doc in context_chunks:
        lines = doc.page_content.split("\n")
        clean_context = " ".join(l for l in lines if not l.startswith("[Chapter Topic:"))

        inputs = qa_tokenizer(
            question,
            clean_context,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
        )
        sequence_ids = inputs.sequence_ids()
        offsets = inputs["offset_mapping"]

        model_inputs = {
            k: torch.tensor([v])
            for k, v in inputs.items()
            if k in ["input_ids", "attention_mask"]
        }

        with torch.no_grad():
            outputs = qa_model(**model_inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        cls_score = start_logits[0].item() + end_logits[0].item()

        best_score = float("-inf")
        best_answer = ""
        best_char_range = (0, 0)

        for start in range(len(sequence_ids)):
            if sequence_ids[start] != 1:
                continue
            for end in range(start, min(start + 30, len(sequence_ids))):
                if sequence_ids[end] != 1:
                    continue
                score = start_logits[start].item() + end_logits[end].item()
                if score > best_score:
                    best_score = score
                    char_start = offsets[start][0]
                    char_end = offsets[end][1]
                    best_answer = clean_context[char_start:char_end].strip()
                    best_char_range = (char_start, char_end)

        difference = cls_score - best_score
        no_answer_prob = float(1 / (1 + np.exp(-np.clip(difference, -50, 50))))

        if no_answer_prob < lowest_no_ans_prob and best_answer:
            lowest_no_ans_prob = no_answer_prob
            best_overall_span = best_answer
            best_timestamp = int(doc.metadata.get("start_time", 0))

            c_start, c_end = best_char_range
            sent_start = max(0, clean_context.rfind(".", 0, c_start) + 1)
            sent_end = clean_context.find(".", c_end)
            if sent_end == -1:
                sent_end = len(clean_context)
            raw_sentence = clean_context[sent_start:sent_end].strip()
            best_quote_sentence = raw_sentence

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "extracted_span": best_overall_span,
        "highlighted_quote": best_quote_sentence,
        "timestamp": best_timestamp,
        "no_answer_probability": round(lowest_no_ans_prob, 4),
        "answerable_confidence": round((1.0 - lowest_no_ans_prob) * 100, 1),
        "latency_ms": latency_ms,
    }


def get_full_clean_transcript(db):
    records = db.get()
    docs = records.get("documents", [])
    metas = records.get("metadatas", [])
    if not docs:
        return ""

    paired = sorted(zip(docs, metas), key=lambda x: x[1].get("start_time", 0))
    formatted_sections = []
    seen_chapters = set()

    for text, meta in paired:
        chapter = meta.get("chapter", "Overview")
        t_sec = int(meta.get("start_time", 0))
        clean_lines = [l for l in text.split("\n") if not l.startswith("[Chapter Topic:")]
        clean_body = " ".join(clean_lines).strip()
        if chapter not in seen_chapters:
            seen_chapters.add(chapter)
            formatted_sections.append(f"\n### {chapter} (Starts at {t_sec}s)\n{clean_body}")
        else:
            formatted_sections.append(clean_body)

    return " ".join(formatted_sections)[:18000]


@app.get("/api/health")
def health():
    return {"status": "online", "saved_videos": load_video_metadata()}


@app.get("/api/videos")
def list_videos():
    return {"videos": load_video_metadata()}


@app.post("/api/process")
def process_video(req: IndexRequest):
    api_key = req.api_key or DEFAULT_GEMINI_KEY
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    db = get_db(video_id)
    existing = db.get()
    saved_meta = load_video_metadata()

    if existing["ids"]:
        title = saved_meta.get(video_id) or fetch_youtube_title(video_id)
        save_video_metadata(video_id, title)
        return {
            "video_id": video_id,
            "title": title,
            "chunks": len(existing["ids"]),
            "status": "cached",
        }

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"]).fetch()
        except Exception:
            first_available = next(iter(transcript_list))
            transcript = first_available.fetch()

        documents = build_hierarchical_chunks(transcript, api_key)
        if not documents:
            raise HTTPException(status_code=400, detail="Transcript was empty.")
        db.add_documents(documents)

        title = fetch_youtube_title(video_id)
        save_video_metadata(video_id, title)
        return {
            "video_id": video_id,
            "title": title,
            "chunks": len(documents),
            "status": "processed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript error: {str(e)}")


@app.post("/api/ask")
def ask_video(req: AskRequest):
    api_key = req.api_key or DEFAULT_GEMINI_KEY
    db = get_db(req.video_id)
    if not db.get()["ids"]:
        raise HTTPException(status_code=404, detail="Video not indexed in ChromaDB.")

    roman_mode = is_roman_urdu(req.question)
    english_query = fast_translate_to_english(req.question, api_key) if roman_mode else req.question

    valid_chunks, best_score = hierarchical_hybrid_retrieve(db, english_query, k=4)
    if not valid_chunks:
        refusal = (
            "Is video mein is topic ka zikar nahi hai."
            if roman_mode
            else "This topic is not mentioned in the video."
        )
        return {
            "answer": refusal,
            "verified": False,
            "confidence": 0.0,
            "latency_ms": 0.0,
            "quote": None,
            "timestamp": 0,
            "mode": "Guardrail Blocked",
        }

    squad_result = verify_with_squad_distilbert(english_query, valid_chunks)
    combined_context = "\n---\n".join(
        f"[Time: {int(d.metadata.get('start_time', 0))}s] {d.page_content}"
        for d in valid_chunks
    )
    start_seconds = (
        squad_result["timestamp"]
        if squad_result["timestamp"] > 0
        else int(valid_chunks[0].metadata.get("start_time", 0))
    )

    use_local_fast_path = (
        not roman_mode
        and squad_result["answerable_confidence"] >= SQUAD_FAST_PATH_CONFIDENCE
        and len(squad_result["extracted_span"]) > 0
    )

    if use_local_fast_path or not api_key:
        if squad_result["no_answer_probability"] >= 0.65:
            return {
                "answer": "This topic is not mentioned in the video." if not roman_mode else "Is video mein is topic ka zikar nahi hai.",
                "verified": False,
                "confidence": squad_result["answerable_confidence"],
                "latency_ms": squad_result["latency_ms"],
                "quote": None,
                "timestamp": start_seconds,
                "mode": "Local SQuAD 2.0",
            }
        return {
            "answer": squad_result["extracted_span"],
            "verified": True,
            "confidence": squad_result["answerable_confidence"],
            "latency_ms": squad_result["latency_ms"],
            "quote": squad_result["highlighted_quote"],
            "timestamp": start_seconds,
            "mode": "Local Fast-Path (0 API Tokens)",
        }

    client = genai.Client(api_key=api_key, http_options={"timeout": 45000})
    target_lang_rule = (
        "Respond in clear, natural Roman Urdu (Urdu written in English alphabets). "
        "If the answer is not in the excerpts, reply ONLY with: 'Is video mein is topic ka zikar nahi hai.'"
        if roman_mode
        else "Respond in clear, concise English. "
        "If the answer is not in the excerpts, reply ONLY with: 'This topic is not mentioned in the video.'"
    )
    system_instruction = (
        "You are a strict fact-extraction assistant. Answer the user's question using ONLY "
        "the provided YouTube transcript excerpts. Do NOT use outside knowledge or invent facts. "
        + target_lang_rule
    )
    squad_hint = ""
    if squad_result["no_answer_probability"] < 0.65 and squad_result["extracted_span"]:
        squad_hint = f"\n\nVerified Extractive Span: {squad_result['extracted_span']}"

    user_message = f"Transcript Excerpts:\n{combined_context}{squad_hint}\n\nQuestion: {english_query}"

    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
            if response and response.text:
                ai_reply = response.text.strip()
                is_refusal = (
                    "not mentioned in the video" in ai_reply.lower()
                    or "zikar nahi hai" in ai_reply.lower()
                )
                return {
                    "answer": ai_reply,
                    "verified": not is_refusal,
                    "confidence": squad_result["answerable_confidence"],
                    "latency_ms": squad_result["latency_ms"],
                    "quote": None if is_refusal else squad_result["highlighted_quote"],
                    "timestamp": start_seconds,
                    "mode": "Hybrid RAG + Gemini",
                }
        except Exception:
            continue

    raise HTTPException(status_code=503, detail="Gemini API unavailable. Check API key or try again.")


@app.post("/api/notes")
def generate_notes(req: GenerateRequest):
    api_key = req.api_key or DEFAULT_GEMINI_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key required for Study Notes.")

    db = get_db(req.video_id)
    transcript_text = get_full_clean_transcript(db)
    if not transcript_text:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    title = load_video_metadata().get(req.video_id, req.video_id)
    lang_directive = (
        "Write the study notes in clear, structured Roman Urdu (Urdu written in English alphabets), keeping technical terms in English."
        if req.language == "Roman Urdu"
        else "Write the study notes in clear, structured English."
    )
    prompt = (
        f"Create comprehensive, well-structured Study Notes for the YouTube video titled '{title}' "
        f"using ONLY the provided transcript. Do NOT add outside information. {lang_directive}\n\n"
        "Format cleanly in Markdown with Executive Overview, Topic-by-Topic Breakdown with timestamps, "
        "Key Definitions/Numbers, and Quick Revision Takeaways.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    client = genai.Client(api_key=api_key, http_options={"timeout": 45000})
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            if res and res.text:
                return {"notes": res.text.strip()}
        except Exception:
            continue

    raise HTTPException(status_code=503, detail="Could not generate study notes.")


@app.post("/api/quiz")
def generate_quiz(req: GenerateRequest):
    api_key = req.api_key or DEFAULT_GEMINI_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key required for Quiz.")

    db = get_db(req.video_id)
    transcript_text = get_full_clean_transcript(db)
    if not transcript_text:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    lang_directive = (
        "Write all questions, options, and explanations in clear Roman Urdu (Urdu written in English alphabets)."
        if req.language == "Roman Urdu"
        else "Write all questions, options, and explanations in clear English."
    )
    prompt = (
        f"Based ONLY on the following YouTube video transcript, create {req.num_questions} multiple-choice questions (MCQs). "
        f"{lang_directive}\n\n"
        "Return ONLY a valid JSON array:\n"
        '[{"question": "...", "options": ["A", "B", "C", "D"], "answer": "Exact option text", "explanation": "..."}]\n\n'
        f"Transcript:\n{transcript_text}"
    )

    client = genai.Client(api_key=api_key, http_options={"timeout": 45000})
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            if res and res.text:
                return {"quiz": json.loads(res.text)}
        except Exception:
            continue

    raise HTTPException(status_code=503, detail="Could not generate quiz.")