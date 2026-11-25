import os
import json
import time
import itertools
import urllib.request
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from groq import Groq
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

KEYS_FILE = "gemini_keys.json"
DB_DIR = "./chroma_db"
METADATA_FILE = "videos_metadata.json"
SQUAD_MODEL_PATH = "./final_distilbert_model" if os.path.exists("./final_distilbert_model") else "Dilawer-khan/distilbert-squad2-qa"
DISTANCE_THRESHOLD = 0.92


def load_keys():
    gemini_keys = []
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                gemini_keys = [k.strip() for k in data.get("gemini_api_keys", []) if k.strip()]
                file_groq = data.get("groq_api_key", "").strip()
                if file_groq and "paste_your_groq_key" not in file_groq:
                    groq_key = file_groq
        except Exception as e:
            print(f"Error reading {KEYS_FILE}: {e}")
    if not gemini_keys:
        env_g = os.getenv("GEMINI_API_KEY", "").strip()
        if env_g:
            gemini_keys = [env_g]
    return gemini_keys, groq_key


GEMINI_KEYS, GROQ_FALLBACK_KEY = load_keys()
KEY_CYCLE = itertools.cycle(GEMINI_KEYS) if GEMINI_KEYS else None
print(f"Loaded {len(GEMINI_KEYS)} Gemini key(s) | Groq Key: {'Ready' if GROQ_FALLBACK_KEY else 'Missing'}")

print("Starting AI Retrieval Engine (BGE-small)...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print(f"Loading Fine-Tuned SQuAD 2.0 DistilBERT from: {SQUAD_MODEL_PATH}")
qa_tokenizer = AutoTokenizer.from_pretrained(SQUAD_MODEL_PATH)
qa_model = AutoModelForQuestionAnswering.from_pretrained(SQUAD_MODEL_PATH).to("cpu").eval()
print("Backend Ready!")



def call_llm_fast(prompt, system_instruction=None, temperature=0.0,
                  json_mode=False, timeout_sec=20):

    last_error = None

    # 1. Gemini Primary
    config_args = {"temperature": temperature}

    if system_instruction:
        config_args["system_instruction"] = system_instruction

    if json_mode:
        config_args["response_mime_type"] = "application/json"

    # SDK timeout milliseconds mein hota hai
    timeout_ms = max(int(timeout_sec * 1000), 15000)

    models = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash"
    ]

    for model_name in models:
        for current_key in GEMINI_KEYS:
            client = None
            try:
                client = genai.Client(
                    api_key=current_key,
                    http_options={"timeout": timeout_ms}
                )

                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_args)
                )

                if res and res.text:
                    print(f"LLM Provider: Gemini ({model_name})")
                    return res.text.strip(), "Gemini LLM"

            except Exception as e:
                last_error = e
                print(f"Gemini {model_name} failed: {type(e).__name__}: {e}")

            finally:
                if client:
                    client.close()

    # 2. Groq Fallback
    if GROQ_FALLBACK_KEY:
        try:
            groq_client = Groq(
                api_key=GROQ_FALLBACK_KEY,
                timeout=20.0
            )

            messages = []

            if system_instruction:
                messages.append({
                    "role": "system",
                    "content": system_instruction
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            kwargs = {
                "model": "openai/gpt-oss-120b",
                "messages": messages,
                "temperature": temperature
            }

            if json_mode:
                kwargs["response_format"] = {
                    "type": "json_object"
                }

            res = groq_client.chat.completions.create(**kwargs)

            if res and res.choices:
                content = res.choices[0].message.content

                if content:
                    print("LLM Provider: Groq (Fallback)")
                    return content.strip(), "Groq LLM"

        except Exception as e:
            last_error = e
            print(f"Groq fallback failed: {type(e).__name__}: {e}")

    raise RuntimeError(f"All LLM providers failed: {last_error}")



class IndexRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    video_id: str
    question: str


class GenerateRequest(BaseModel):
    video_id: str
    language: str = "English"
    num_questions: int = 5


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
        "wala", "wali", "wale", "ho", "raha", "rahi", "kar", "kr", "smjha", "farq", "bunyadi"
    }
    words = {w.lower().strip("?.!,") for w in text.split()}
    return len(words.intersection(roman_urdu_markers)) >= 1


def fast_translate_to_english(roman_text):
    prompt = (
        "Translate this Roman Urdu question into clear English. "
        "Return ONLY the English question, nothing else:\n" + roman_text
    )
    try:
        translated, _ = call_llm_fast(prompt, temperature=0.0, timeout_sec=2)
        return translated
    except Exception:
        return roman_text


def generate_chapter_headings(sections):
    section_summaries = []
    for idx, sec in enumerate(sections):
        preview = sec["text"][:1500]
        section_summaries.append(f"Section {idx} ({int(sec['start'])}s - {int(sec['end'])}s):\n{preview}")

    prompt = (
        "Write ONE short topic heading (8-14 words) for each video transcript section below. "
        "Return ONLY a JSON object mapping index strings ('0', '1', ...) to headings.\n\n"
        + "\n\n---\n\n".join(section_summaries)
    )
    try:
        raw, _ = call_llm_fast(prompt, temperature=0.0, json_mode=True, timeout_sec=4)
        return json.loads(raw)
    except Exception:
        return {str(i): f"Section {i + 1} ({int(s['start'])}s - {int(s['end'])}s)" for i, s in enumerate(sections)}


def build_hierarchical_chunks(transcript):
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

    headings_map = generate_chapter_headings(sections)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=120,
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
            search_cursor = pos + max(1, len(chunk) - 120)

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


def hierarchical_hybrid_retrieve(db, query, k=6):
    all_records = db.get()
    all_texts = all_records.get("documents", [])
    all_metas = all_records.get("metadatas", [])

    if not all_texts:
        return [], 2.0

    dense_results = db.similarity_search_with_score(query, k=min(15, len(all_texts)))
    best_dense_score = dense_results[0][1] if dense_results else 2.0

    if best_dense_score > DISTANCE_THRESHOLD:
        return [], best_dense_score

    tokenized_corpus = [doc.lower().split() for doc in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]

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
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.2 / (60 + rank + 1))

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
    best_timestamp = 0
    best_span_score = float("-inf")
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

        for start in range(len(sequence_ids)):
            if sequence_ids[start] != 1:
                continue
            for end in range(start, min(start + 18, len(sequence_ids))):
                if sequence_ids[end] != 1:
                    continue
                score = start_logits[start].item() + end_logits[end].item()
                if score > best_score:
                    best_score = score
                    char_start = offsets[start][0]
                    char_end = offsets[end][1]
                    best_answer = clean_context[char_start:char_end].strip()

        difference = cls_score - best_score
        no_answer_prob = float(1 / (1 + np.exp(-np.clip(difference, -50, 50))))

        if best_score > best_span_score and best_answer:
            best_span_score = best_score
            lowest_no_ans_prob = no_answer_prob
            best_overall_span = best_answer
            best_timestamp = int(doc.metadata.get("start_time", 0))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    confidence = round(max(1.0, min(99.5, (1.0 - lowest_no_ans_prob) * 100)), 1)
    return {
        "extracted_span": best_overall_span,
        "timestamp": best_timestamp,
        "span_logit_score": round(best_span_score, 2),
        "no_answer_probability": round(lowest_no_ans_prob, 4),
        "answerable_confidence": confidence,
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
    return {
        "status": "online",
        "squad_model": SQUAD_MODEL_PATH,
        "keys_loaded": len(GEMINI_KEYS),
        "saved_videos": load_video_metadata(),
    }


@app.get("/api/videos")
def list_videos():
    return {"videos": load_video_metadata()}


@app.post("/api/process")
def process_video(req: IndexRequest):
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

        documents = build_hierarchical_chunks(transcript)
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
    t_req = time.perf_counter()
    db = get_db(req.video_id)
    if not db.get()["ids"]:
        raise HTTPException(status_code=404, detail="Video not indexed in ChromaDB.")

    roman_mode = is_roman_urdu(req.question)
    english_query = fast_translate_to_english(req.question) if roman_mode else req.question

    valid_chunks, _ = hierarchical_hybrid_retrieve(db, english_query, k=6)
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
            "latency_ms": round((time.perf_counter() - t_req) * 1000, 1),
            "quote": None,
            "timestamp": 0,
            "mode": "Guardrail Blocked",
        }

    squad_result = verify_with_squad_distilbert(english_query, valid_chunks)
    start_seconds = (
        squad_result["timestamp"]
        if squad_result["timestamp"] > 0
        else int(valid_chunks[0].metadata.get("start_time", 0))
    )

    combined_context = "\n---\n".join(
        f"[Time: {int(d.metadata.get('start_time', 0))}s] {d.page_content}"
        for d in valid_chunks
    )

    target_lang_rule = (
        "Answer in concise Roman Urdu (Urdu written in English alphabets). "
        "Keep it strictly to 1-2 short sentences with NO filler or extra commentary. "
        "If the answer is NOT in the excerpts, reply ONLY with: 'Is video mein is topic ka zikar nahi hai.'"
        if roman_mode
        else "Answer in clear, direct English. "
        "Keep it strictly to 1-2 short sentences (under 40 words). Do NOT add extra details. "
        "If the answer is NOT in the excerpts, reply ONLY with: 'This topic is not mentioned in the video.'"
    )
    system_instruction = (
        "You are a concise, strict fact-extraction assistant. Use ONLY the provided transcript excerpts. "
        "Never guess or invent facts. " + target_lang_rule
    )
    user_message = f"Transcript Excerpts:\n{combined_context}\n\nQuestion: {english_query}"

    try:
        ai_reply, provider_tag = call_llm_fast(
            prompt=user_message,
            system_instruction=system_instruction,
            temperature=0.0,
        )
        is_refusal = (
            "not mentioned in the video" in ai_reply.lower()
            or "zikar nahi hai" in ai_reply.lower()
        )
        return {
            "answer": ai_reply,
            "verified": not is_refusal,
            "confidence": squad_result["answerable_confidence"],
            "latency_ms": round((time.perf_counter() - t_req) * 1000, 1),
            "quote": None,
            "timestamp": start_seconds,
            "mode": provider_tag,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/notes")
def generate_notes(req: GenerateRequest):
    db = get_db(req.video_id)
    transcript_text = get_full_clean_transcript(db)
    if not transcript_text:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    title = load_video_metadata().get(req.video_id, req.video_id)
    lang_directive = (
        "Write concise study notes in Roman Urdu (Urdu written in English alphabets), keeping technical terms in English."
        if req.language == "Roman Urdu"
        else "Write concise, crisp study notes in English."
    )
    prompt = (
        f"Create crisp, scannable Study Notes for '{title}' using ONLY the transcript below. {lang_directive}\n\n"
        "Include: Executive Summary (3 bullets), Topic Breakdown table with timestamps, Key Numbers, and 4 Revision Takeaways. "
        "Keep every bullet brief and direct. Use clean Unicode symbols instead of raw LaTeX backslashes, and never use HTML <br> tags.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    try:
        notes_text, _ = call_llm_fast(prompt=prompt, temperature=0.1, timeout_sec=4)
        return {"notes": notes_text}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/quiz")
def generate_quiz(req: GenerateRequest):
    db = get_db(req.video_id)
    transcript_text = get_full_clean_transcript(db)
    if not transcript_text:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    lang_directive = (
        "Write all questions, options, and 1-sentence explanations in Roman Urdu (Urdu written in English alphabets)."
        if req.language == "Roman Urdu"
        else "Write all questions, options, and 1-sentence explanations in clear English."
    )
    prompt = (
        f"Based ONLY on this transcript, create {req.num_questions} MCQs. {lang_directive}\n\n"
        'Return ONLY valid JSON: {"quiz": [{"question": "...", "options": ["A", "B", "C", "D"], "answer": "Exact option text", "explanation": "..."}]}\n\n'
        f"Transcript:\n{transcript_text}"
    )

    try:
        raw, _ = call_llm_fast(prompt=prompt, temperature=0.2, json_mode=True, timeout_sec=4)
        parsed = json.loads(raw)
        quiz_list = parsed.get("quiz", parsed if isinstance(parsed, list) else [])
        return {"quiz": quiz_list}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))