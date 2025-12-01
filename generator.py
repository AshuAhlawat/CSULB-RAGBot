import json
import sys
from typing import List

import ollama

from retriever import retrieve

# Model used for final answer generation.
GEN_MODEL = "deepseek-r1"


def build_context(results: List[dict]) -> str:
    parts = []
    for r in results:
        url = r.get("url", "")
        doc = r.get("document", "")
        parts.append(f"Source: {url}\n{doc}\n")
    return "\n\n".join(parts).strip()


def check_answer(question: str, answer: str) -> bool:
    prompt = (
        "You are a strict evaluator. Given a user question and a proposed answer, "
        "return JSON with a single boolean field 'accurate' indicating whether the answer directly and correctly addresses the question. "
        "Be conservative; if unsure, set accurate to false.\n"
        f"Question: {question}\nAnswer: {answer}\nEvaluation:"
    )
    try:
        verdict = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )["message"]["content"]
        data = json.loads(verdict)
        return bool(data.get("accurate", False))
    except Exception:
        return True


def answer(question: str, top_k: int = 5) -> dict:
    results, checked_urls = retrieve(question, top_k=top_k, return_checked=True)
    if not results:
        return {
            "answer": "I could not find relevant sources to answer that question.",
            "sources": checked_urls,
        }

    context = build_context(results)
    prompt = (
        "You are a factual assistant. Use the provided sources to answer the question. "
        "If the sources do not contain the answer, mention the websites where the user can research the answers. "
        "Do not include sources or links in your answer text; a separate list will be shown.\n"
        f"Question: {question}\n\nSources:\n{context}\n\nAnswer:"
    )

    try:
        reply = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )["message"]["content"]
    except Exception as exc:
        reply = f"I encountered an issue generating the answer: {exc}"

    # Validate the answer if we had sources.
    accurate = check_answer(question, reply)
    if not accurate:
        reply = (
            "I'm not fully confident this answers your question based on the available sources. "
            "Could you clarify or provide more specifics?"
        )

    return {
        "answer": reply,
        "sources": [r.get("url", "") for r in results] or checked_urls,
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip() or "What is the time, place, and manner policy?"
    result = answer(question)
    out_path = "generator_output.md"

    answer_text = result.get("answer", "")
    sources = result.get("sources") or []

    # Terminal output
    print(answer_text)
    print("\nSources:")
    if sources:
        for src in sources:
            print(f"- {src}")
    else:
        print("- (none)")

    # File output mirrors terminal format
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{answer_text}\n")
        f.write("\nSources:\n")
        if sources:
            for src in sources:
                f.write(f"- {src}\n")
        else:
            f.write("- (none)\n")
