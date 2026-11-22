import streamlit as st
import os
import json
import urllib.request
import numpy as np
import torch
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Video Learning Suite", page_icon="🎓", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_DIR = "./chroma_db"
METADATA_FILE = "videos_metadata.json"
SQUAD_MODEL_PATH = "Dilawer-khan/distilbert-squad2-qa"
DISTANCE_THRESHOLD = 1.12
SQUAD_FAST_PATH_CONFIDENCE = 85.0  # Threshold (%) for zero-API-call local answers

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "study_notes" not in st.session_state:
    st.session_state.study_notes = None

# --- CACHED MODELS ---
@st.cache_resource(show_spinner="Starting AI Retrieval Engine...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="Loading Local Fine-Tuned SQuAD 2.0 DistilBERT Verifier...")
def load_squad_verifier():
    pass
    qa_tokenizer = AutoTokenizer.from_pretrained(SQUAD_MODEL_PATH)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(SQUAD_MODEL_PATH)
    qa_model.to("cpu")
    qa_model.eval()
    return qa_tokenizer, qa_model

embeddings_model = load_embedding_model()
qa_tokenizer, qa_model = load_squad_verifier()

# --- METADATA & YOUTUBE HELPERS ---
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

# --- BILINGUAL HELPERS ---
def is_roman_urdu(text):
    roman_urdu_markers = {
        "kya", "kia", "kyun", "kese", "kaise", "hai", "hain", "tha", "thi",
        "mein", "ko", "ka", "ki", "ke", "se", "aur", "batao", "batayein",
        "konsa", "kitne", "kitni", "iska", "uska", "hum", "aap", "nahi", "nhi",
        "wala", "wali", "wale", "ho", "raha", "rahi", "kar", "kr", "smjha"
    }
    words = {w.lower().strip("?.!,") for w in text.split()}
    return len(words.intersection(roman_urdu_markers)) >= 1

def fast_translate_to_english(roman_text):
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 20000})
    prompt = (
        "Translate the following Roman Urdu question into clear, natural English. "
        "Return ONLY the English translation and nothing else.\n\n"
        f"Roman Urdu: {roman_text}"
    )
    for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            if res and res.text:
                return res.text.strip()
        except Exception:
            continue
    return roman_text

# --- HIERARCHICAL INDEXING ---
def generate_chapter_headings(sections):
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 45000})
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

    try:
        for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                if response and response.text:
                    return json.loads(response.text)
            except Exception:
                continue
    except Exception:
        pass

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

        t_start = 0
        t_end = 0
        for c_idx, ts in char_to_time:
            if c_idx <= s_start:
                t_start = ts
            if c_idx <= s_end:
                t_end = ts

        sections.append({"start": t_start, "end": t_end, "text": sec_text, "char_start": s_start, "char_end": s_end})

    headings_map = generate_chapter_headings(sections)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[". ", "? ", "! ", " ", ""]
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
                metadata={"start_time": chunk_time, "chapter": chapter_heading}
            )
        )

    return documents

# --- HYBRID RETRIEVAL + SQUAD 2.0 LOCAL FAST-PATH VERIFIER ---
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
    if qa_tokenizer is None or qa_model is None:
        return None

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
            return_offsets_mapping=True
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

            # Extract surrounding sentence for UI proof highlight
            c_start, c_end = best_char_range
            sent_start = max(0, clean_context.rfind(".", 0, c_start) + 1)
            sent_end = clean_context.find(".", c_end)
            if sent_end == -1:
                sent_end = len(clean_context)
            raw_sentence = clean_context[sent_start:sent_end].strip()
            best_quote_sentence = raw_sentence.replace(best_answer, f"**{best_answer}**", 1)

    return {
        "extracted_span": best_overall_span,
        "highlighted_quote": best_quote_sentence,
        "timestamp": best_timestamp,
        "no_answer_probability": round(lowest_no_ans_prob, 4),
        "answerable_confidence": round((1.0 - lowest_no_ans_prob) * 100, 1)
    }

# --- QUIZ & STUDY NOTES GENERATORS ---
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

    full_compiled = " ".join(formatted_sections)
    return full_compiled[:18000]  # Safe window for fast generation

def generate_video_quiz(db, num_questions=5, language="English"):
    transcript_text = get_full_clean_transcript(db)
    if not transcript_text:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 45000})
    lang_directive = (
        "Write all questions, options, and explanations in clear Roman Urdu (Urdu written in English alphabets)."
        if language == "Roman Urdu" else
        "Write all questions, options, and explanations in clear English."
    )

    prompt = (
        f"You are an academic exam creator. Based ONLY on the following YouTube video transcript, "
        f"create {num_questions} multiple-choice questions (MCQs) that test key facts and concepts explicitly stated in the video. "
        f"Do NOT include outside facts. {lang_directive}\n\n"
        "Return ONLY a valid JSON array where each element has this exact schema:\n"
        '[{"question": "...", "options": ["Option A", "Option B", "Option C", "Option D"], "answer": "Exact matching string from options", "explanation": "Brief explanation referencing the video"}]\n\n'
        f"Transcript:\n{transcript_text}"
    )

    for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            if res and res.text:
                return json.loads(res.text)
        except Exception:
            continue
    return None

def generate_video_study_notes(db, video_title, language="English"):
    transcript_text = get_full_clean_transcript(db)
    if not transcript_text:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 45000})
    lang_directive = (
        "Write the study notes in clear, structured Roman Urdu (Urdu written in English alphabets), keeping technical terms in English."
        if language == "Roman Urdu" else
        "Write the study notes in clear, structured English."
    )

    prompt = (
        f"Create comprehensive, well-structured Study Notes for the YouTube video titled '{video_title}' "
        f"using ONLY the provided transcript. Do NOT add outside information. {lang_directive}\n\n"
        "Format the notes cleanly in Markdown with:\n"
        "1. **Executive Overview** (2-3 sentences)\n"
        "2. **Topic-by-Topic Breakdown** (organized chronologically with timestamps)\n"
        "3. **Key Definitions, Formulas & Numbers** (bulleted list)\n"
        "4. **Quick Revision Takeaways** (3-5 bullet points)\n\n"
        f"Transcript:\n{transcript_text}"
    )

    for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            if res and res.text:
                return res.text.strip()
        except Exception:
            continue
    return None

# --- VIDEO INDEXING ---
def process_video(video_url):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid link.")
            return False

        collection_name = f"vid_{video_id.replace('-', '_')}"
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_model,
            persist_directory=DB_DIR
        )

        existing_data = db.get()
        if existing_data["ids"]:
            saved_meta = load_video_metadata()
            if video_id not in saved_meta:
                save_video_metadata(video_id, fetch_youtube_title(video_id))
            st.session_state.video_id = video_id
            st.session_state.collection_name = collection_name
            st.session_state.messages = []
            st.session_state.quiz_data = None
            st.session_state.quiz_submitted = False
            st.session_state.study_notes = None
            return "cached"

        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"]).fetch()
        except Exception:
            first_available = next(iter(transcript_list))
            transcript = first_available.fetch()

        documents = build_hierarchical_chunks(transcript)
        db.add_documents(documents)

        save_video_metadata(video_id, fetch_youtube_title(video_id))
        st.session_state.video_id = video_id
        st.session_state.collection_name = collection_name
        st.session_state.messages = []
        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.session_state.study_notes = None
        return "processed"
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return False

# --- UI LAYOUT ---
st.title("🎓 AI Video Learning Suite (Zero-Hallucination RAG + SQuAD 2.0)")

with st.sidebar:
    st.header("🔑 API Key")
    user_api_key = st.text_input("Enter Gemini API Key:", value=GEMINI_API_KEY, type="password")
    if user_api_key:
        GEMINI_API_KEY = user_api_key
    st.divider()
    st.header("⚙️ Video Setup")
    video_url = st.text_input("🔗 YouTube Video URL:")
    if st.button("Process Video", use_container_width=True):
        if video_url:
            with st.spinner("Generating Chapter Headings & Indexing Video..."):
                status = process_video(video_url)
                if status == "cached":
                    st.success("⚡ Loaded from isolated database (0.1s)!")
                elif status == "processed":
                    st.success("✅ Video indexed with Chapter Headings!")
        else:
            st.warning("Please enter a YouTube URL.")

    st.divider()
    st.header("📚 Saved Video Databases")
    saved_videos = load_video_metadata()
    if saved_videos:
        video_options = {f"{title} [{vid}]": vid for vid, title in saved_videos.items()}
        selected_label = st.selectbox(
            "Switch to a processed video:",
            options=list(video_options.keys()),
            key="saved_video_selector"
        )
        if st.button("Load Selected Video", use_container_width=True):
            chosen_id = video_options[selected_label]
            if st.session_state.video_id != chosen_id:
                st.session_state.video_id = chosen_id
                st.session_state.collection_name = f"vid_{chosen_id.replace('-', '_')}"
                st.session_state.messages = []
                st.session_state.quiz_data = None
                st.session_state.quiz_submitted = False
                st.session_state.study_notes = None
                st.rerun()
    else:
        st.caption("No processed videos saved yet.")

if st.session_state.video_id:
    current_title = load_video_metadata().get(st.session_state.video_id, st.session_state.video_id)
    db = Chroma(
        collection_name=st.session_state.collection_name,
        embedding_function=embeddings_model,
        persist_directory=DB_DIR
    )

    col1, col2 = st.columns([1, 1.15])

    with col1:
        st.subheader("📺 Video Player")
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        st.caption(f"**Active Video:** {current_title}")

    with col2:
        tab_chat, tab_quiz, tab_notes = st.tabs(["💬 Q&A Assistant", "📝 Quiz Generator", "📒 Study Notes"])

        # === TAB 1: ZERO-HALLUCINATION CHAT + LOCAL FAST PATH ===
        with tab_chat:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("quote"):
                        st.info(f"📌 **Transcript Proof:** \"{message['quote']}\"")
                    if message.get("timestamp_url"):
                        st.markdown(f"⏱ **[Click to Watch from {message['time_sec']}s]({message['timestamp_url']})**")

            if prompt := st.chat_input("Ask in English or Roman Urdu..."):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Searching Chapters & Running Local SQuAD 2.0 Verification..."):
                        roman_mode = is_roman_urdu(prompt)
                        english_query = prompt

                        if roman_mode:
                            english_query = fast_translate_to_english(prompt)
                            st.caption(f"🌐 **Translated Query (EN):** {english_query}")

                        valid_chunks, best_score = hierarchical_hybrid_retrieve(db, english_query, k=4)

                        if not valid_chunks:
                            refusal = (
                                "Is video mein is topic ka zikar nahi hai."
                                if roman_mode else
                                "This topic is not mentioned in the video."
                            )
                            st.markdown(refusal)
                            with st.expander("🔍 Guardrail Triggered (Blocked Before Model)"):
                                st.write(f"**Best Normalized Distance:** {round(best_score, 4)} (Threshold: {DISTANCE_THRESHOLD})")
                            st.session_state.messages.append({"role": "assistant", "content": refusal})
                        else:
                            squad_result = verify_with_squad_distilbert(english_query, valid_chunks)
                            combined_context = "\n---\n".join([
                                f"[Time: {int(d.metadata.get('start_time', 0))}s] {d.page_content}"
                                for d in valid_chunks
                            ])

                            start_seconds = (
                                squad_result["timestamp"]
                                if (squad_result and squad_result["timestamp"] > 0)
                                else int(valid_chunks[0].metadata.get("start_time", 0))
                            )
                            jump_url = f"https://www.youtube.com/watch?v={st.session_state.video_id}&t={start_seconds}s"

                            with st.expander("🔍 Retrieval + Local DistilBERT SQuAD 2.0 Verification"):
                                st.write(f"**Best Normalized Distance:** {round(best_score, 4)} (Threshold: {DISTANCE_THRESHOLD})")
                                if squad_result:
                                    st.write(
                                        f"**SQuAD 2.0 Extracted Span:** `{squad_result['extracted_span']}` | "
                                        f"**Confidence:** {squad_result['answerable_confidence']}% | "
                                        f"**No-Answer Prob:** {squad_result['no_answer_probability']}"
                                    )
                                st.text(combined_context)

                            # ZERO-API-CALL LOCAL FAST PATH (English factual queries with >= 85% DistilBERT confidence)
                            use_local_fast_path = (
                                not roman_mode
                                and squad_result is not None
                                and squad_result["answerable_confidence"] >= SQUAD_FAST_PATH_CONFIDENCE
                                and len(squad_result["extracted_span"]) > 0
                            )

                            if use_local_fast_path:
                                ai_reply = f"⚡ **Verified Answer (Local SQuAD 2.0 — 0 API Tokens):** {squad_result['extracted_span']}"
                                quote_text = squad_result["highlighted_quote"]
                                st.markdown(ai_reply)
                                if quote_text:
                                    st.info(f"📌 **Transcript Proof:** \"{quote_text}\"")
                                st.markdown(f"⏱ **[Click to Watch from {start_seconds}s]({jump_url})**")

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": ai_reply,
                                    "quote": quote_text,
                                    "timestamp_url": jump_url,
                                    "time_sec": start_seconds
                                })
                            else:
                                # CONCEPTUAL SYNTHESIS & BILINGUAL GENERATION VIA GEMINI
                                client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 45000})
                                target_lang_rule = (
                                    "Respond in clear, natural Roman Urdu (Urdu written in English alphabets). "
                                    "If the answer is not in the excerpts, reply ONLY with: 'Is video mein is topic ka zikar nahi hai.'"
                                    if roman_mode else
                                    "Respond in clear, concise English. "
                                    "If the answer is not in the excerpts, reply ONLY with: 'This topic is not mentioned in the video.'"
                                )

                                system_instruction = (
                                    "You are a strict fact-extraction assistant. Answer the user's question using ONLY "
                                    "the provided YouTube transcript excerpts. Do NOT use outside knowledge or invent facts. "
                                    "Synthesize relevant details explicitly stated in the excerpts to answer clearly. "
                                    + target_lang_rule
                                )

                                squad_hint = ""
                                if squad_result and squad_result["no_answer_probability"] < 0.65 and squad_result["extracted_span"]:
                                    squad_hint = f"\n\nVerified Extractive Span (Local DistilBERT SQuAD 2.0): {squad_result['extracted_span']}"

                                user_message = f"Transcript Excerpts:\n{combined_context}{squad_hint}\n\nQuestion: {english_query}"

                                try:
                                    response = None
                                    last_err = None
                                    for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
                                        try:
                                            response = client.models.generate_content(
                                                model=model_name,
                                                contents=user_message,
                                                config=types.GenerateContentConfig(
                                                    system_instruction=system_instruction,
                                                    temperature=0.0,
                                                )
                                            )
                                            if response and response.text:
                                                break
                                        except Exception as inner_e:
                                            last_err = inner_e
                                            continue

                                    if response is None:
                                        raise last_err

                                    ai_reply = response.text.strip()
                                    is_refusal = (
                                        "not mentioned in the video" in ai_reply.lower()
                                        or "zikar nahi hai" in ai_reply.lower()
                                    )
                                    quote_text = (
                                        squad_result["highlighted_quote"]
                                        if (not is_refusal and squad_result and squad_result["no_answer_probability"] < 0.65)
                                        else None
                                    )

                                    st.markdown(ai_reply)
                                    if quote_text:
                                        st.info(f"📌 **Transcript Proof:** \"{quote_text}\"")
                                    if not is_refusal:
                                        st.markdown(f"⏱ **[Click to Watch from {start_seconds}s]({jump_url})**")

                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": ai_reply,
                                        "quote": quote_text,
                                        "timestamp_url": None if is_refusal else jump_url,
                                        "time_sec": start_seconds
                                    })

                                except Exception as e:
                                    error_msg = str(e)
                                    if "503" in error_msg or "504" in error_msg or "UNAVAILABLE" in error_msg or "DEADLINE_EXCEEDED" in error_msg:
                                        st.warning("⚠️ Google AI server is temporarily busy. Please try again in a moment.")
                                    else:
                                        st.error(f"❌ Error: {error_msg}")

        # === TAB 2: INTERACTIVE QUIZ GENERATOR ===
        with tab_quiz:
            st.subheader("📝 Interactive Video Quiz")
            q_col1, q_col2, q_col3 = st.columns([1, 1, 1])
            with q_col1:
                num_q = st.selectbox("Number of Questions:", [5, 10], index=0)
            with q_col2:
                quiz_lang = st.selectbox("Quiz Language:", ["English", "Roman Urdu"], index=0)
            with q_col3:
                st.write("")
                st.write("")
                gen_quiz_btn = st.button("🎯 Generate Quiz", use_container_width=True)

            if gen_quiz_btn:
                with st.spinner("Creating video-grounded MCQs..."):
                    quiz_items = generate_video_quiz(db, num_questions=num_q, language=quiz_lang)
                    if quiz_items:
                        st.session_state.quiz_data = quiz_items
                        st.session_state.quiz_submitted = False
                    else:
                        st.error("Could not generate quiz. Please try again.")

            if st.session_state.quiz_data:
                with st.form("video_quiz_form"):
                    user_answers = {}
                    for idx, item in enumerate(st.session_state.quiz_data):
                        st.markdown(f"**Q{idx + 1}. {item['question']}**")
                        user_answers[idx] = st.radio(
                            f"Select answer for Q{idx + 1}:",
                            options=item["options"],
                            key=f"quiz_q_{st.session_state.video_id}_{idx}",
                            label_visibility="collapsed"
                        )
                        st.write("")

                    submitted = st.form_submit_button("✅ Submit Quiz & Check Score")
                    if submitted:
                        st.session_state.quiz_submitted = True

                if st.session_state.quiz_submitted:
                    score = 0
                    total = len(st.session_state.quiz_data)
                    st.divider()
                    for idx, item in enumerate(st.session_state.quiz_data):
                        chosen = user_answers.get(idx)
                        correct = item["answer"]
                        if chosen == correct:
                            score += 1
                            st.success(f"**Q{idx + 1}: Correct!** ({correct}) — {item['explanation']}")
                        else:
                            st.error(f"**Q{idx + 1}: Incorrect.** Your answer: *{chosen}* | Correct: **{correct}** — {item['explanation']}")

                    st.metric(label="🏆 Final Quiz Score", value=f"{score} / {total} ({int((score/total)*100)}%)")

        # === TAB 3: STRUCTURED STUDY NOTES GENERATOR ===
        with tab_notes:
            st.subheader("📒 Structured Study Notes")
            n_col1, n_col2 = st.columns([1, 1])
            with n_col1:
                notes_lang = st.selectbox("Notes Language:", ["English", "Roman Urdu"], index=0)
            with n_col2:
                st.write("")
                st.write("")
                gen_notes_btn = st.button("✨ Generate Study Notes", use_container_width=True)

            if gen_notes_btn:
                with st.spinner("Compiling structured study notes from video chapters..."):
                    notes_md = generate_video_study_notes(db, current_title, language=notes_lang)
                    if notes_md:
                        st.session_state.study_notes = notes_md
                    else:
                        st.error("Could not generate notes. Please try again.")

            if st.session_state.study_notes:
                st.download_button(
                    label="📥 Download Notes (.md)",
                    data=st.session_state.study_notes,
                    file_name=f"{st.session_state.video_id}_study_notes.md",
                    mime="text/markdown"
                )
                st.divider()
                st.markdown(st.session_state.study_notes)
else:
    st.info("👈 Paste any YouTube URL in the sidebar and click 'Process Video' to unlock Q&A, Quiz, and Study Notes.")
