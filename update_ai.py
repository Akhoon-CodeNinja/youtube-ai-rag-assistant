from pathlib import Path
from datetime import datetime
import py_compile
import shutil
import sys

TARGET = Path("api_server.py")

if not TARGET.exists():
    print("ERROR: api_server.py not found.")
    print("Run this script from D:\\AI_lab.")
    sys.exit(1)

source = TARGET.read_text(encoding="utf-8")

old_timestamp_block = '''    squad_result = verify_with_squad_distilbert(english_query, valid_chunks)
    start_seconds = (
        squad_result["timestamp"]
        if squad_result["timestamp"] > 0
        else int(valid_chunks[0].metadata.get("start_time", 0))
    )
'''

new_timestamp_block = '''    squad_result = verify_with_squad_distilbert(english_query, valid_chunks)
'''

old_answer_block = '''        ai_reply, provider_tag = call_llm_fast(
            prompt=user_message,
            system_instruction=system_instruction,
            temperature=0.0,
        )
        is_refusal = (
'''

new_answer_block = '''        ai_reply, provider_tag = call_llm_fast(
            prompt=user_message,
            system_instruction=system_instruction,
            temperature=0.0,
        )

        # Choose the citation timestamp from the retrieved chunk that best
        # supports BOTH the question and Gemini's actual answer.
        start_seconds = select_evidence_timestamp(
            english_query,
            ai_reply,
            valid_chunks,
            fallback_timestamp=squad_result["timestamp"],
        )
        print("Citation Timestamp Selected:", start_seconds)

        is_refusal = (
'''

helper_anchor = '''def get_full_clean_transcript(db):
'''

helper_code = '''def select_evidence_timestamp(question, answer, context_chunks, fallback_timestamp=0):
    """
    Select a video timestamp from the retrieved chunk whose text best matches
    both the user's question and the generated answer. DistilBERT's span
    timestamp is retained only as a fallback when no lexical evidence matches.
    """
    import re

    stop_words = {
        "what", "which", "who", "when", "where", "why", "how", "many",
        "much", "was", "were", "is", "are", "the", "a", "an", "of",
        "in", "on", "at", "to", "for", "and", "or", "with", "does",
        "do", "did", "this", "that", "it", "they", "he", "she", "we",
        "you", "i", "used", "use", "using", "about", "from", "into",
        "every", "each", "per", "if", "its", "their", "there", "how"
    }

    def terms(text):
        normalized = re.sub(r"(?<=\\d),(?=\\d)", "", str(text).lower())
        return {
            token for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) > 1 and token not in stop_words
        }

    question_terms = terms(question)
    answer_terms = terms(answer)

    # Answer terms receive greater weight because the timestamp should support
    # the factual content actually returned to the user.
    weights = {}
    for token in question_terms:
        weights[token] = 2
    for token in answer_terms:
        weights[token] = max(weights.get(token, 0), 3)

    best_timestamp = int(fallback_timestamp or 0)
    best_score = 0
    best_timestamp_for_tie = None

    for doc in context_chunks:
        body_lines = [
            line for line in doc.page_content.split("\\n")
            if not line.startswith("[Chapter Topic:")
        ]
        body = " ".join(body_lines)
        doc_terms = terms(body)
        score = sum(weight for token, weight in weights.items() if token in doc_terms)
        timestamp = int(doc.metadata.get("start_time", 0) or 0)

        # On equal evidence scores, prefer the earlier matching occurrence.
        if score > best_score or (
            score == best_score
            and score > 0
            and (best_timestamp_for_tie is None or timestamp < best_timestamp_for_tie)
        ):
            best_score = score
            best_timestamp = timestamp
            best_timestamp_for_tie = timestamp

    print(
        "Citation Evidence Match:",
        f"{best_score} weighted term points",
        f"| Question terms: {len(question_terms)}",
        f"| Answer terms: {len(answer_terms)}",
    )
    return best_timestamp


'''

# Validate that this is the expected source version before changing anything.
if source.count(old_timestamp_block) != 1:
    print("ERROR: Expected timestamp block was not found exactly once.")
    print("No changes made. Your local api_server.py may differ from the uploaded version.")
    sys.exit(1)

if source.count(old_answer_block) != 1:
    print("ERROR: Expected Gemini answer block was not found exactly once.")
    print("No changes made.")
    sys.exit(1)

if source.count(helper_anchor) != 1:
    print("ERROR: Helper insertion point was not found exactly once.")
    print("No changes made.")
    sys.exit(1)

if "def select_evidence_timestamp(" in source:
    print("ERROR: select_evidence_timestamp already exists. No changes made.")
    sys.exit(1)

updated = source.replace(old_timestamp_block, new_timestamp_block, 1)
updated = updated.replace(old_answer_block, new_answer_block, 1)
updated = updated.replace(helper_anchor, helper_code + helper_anchor, 1)

# Compile before touching the original file.
check_path = TARGET.with_name("_api_server_timestamp_check.py")
check_path.write_text(updated, encoding="utf-8")
try:
    py_compile.compile(str(check_path), doraise=True)
finally:
    check_path.unlink(missing_ok=True)

backup = TARGET.with_name(
    f"api_server_backup_timestamp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
)
shutil.copy2(TARGET, backup)
TARGET.write_text(updated, encoding="utf-8")

print("SUCCESS: Evidence-based citation timestamp patch applied.")
print(f"Backup: {backup.name}")
print("Updated: api_server.py")
print("Answer generation, retrieval ranking, thresholds, models, and API keys were not changed.")
print("Citation timestamps now come from the retrieved chunk best matching the question and generated answer.")
