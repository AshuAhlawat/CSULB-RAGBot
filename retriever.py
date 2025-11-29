import json
import ollama
import chromadb
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

COLL_NAME = "csulb"
LLM_MODEL = "deepseek-r1"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 5

client = chromadb.PersistentClient(path="chroma_db")
coll = client.get_collection(COLL_NAME)


def clean_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text.strip()


def fetch_latest(url: str, timeout: int = 12):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        if "html" not in ctype:
            return None, "non_html"
        return clean_text(resp.text), "live"
    except Exception as exc:
        return None, f"error:{exc.__class__.__name__}"


def parse_ts(value) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def judge_relevance(query: str, document: str) -> dict:
    snippet = " ".join(document.split()[:max(1000, len(document.split()))])
    prompt = (
        "You are a relevance filter. Given a user question and a document snippet, decide if the snippet likely contains information that answers or helps answer the question. "
        "Respond with JSON containing only a boolean field 'relevant'."
        f"Question: {query}\nSnippet: {snippet}\nAnswer:"
    )
    try:
        reply = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )["message"]["content"]
    except Exception as exc:
        return {"relevant": True}

    try:
        data = json.loads(reply)
        return {"relevant": bool(data.get("relevant", False))}
    except Exception:
        lower = reply.lower()
        return {"relevant": "true" in lower or "yes" in lower}


def strip_extraneous(doc: str) -> str:
    """Ask the LLM to drop navigation/boilerplate and return main content only."""
    prompt = (
        "You will clean a page text. Remove navigation, menus, breadcrumbs, footers, contact blocks, "
        "social links, boilerplate headers/footers, and anything not part of the main body content. "
        "Keep only the main informational content relevant to the page topic. Return plain text only, no JSON, no explanation.\n\n"
        f"Page text:\n{doc}"
    )
    try:
        cleaned = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )["message"]["content"]
        return cleaned.strip() or doc
    except Exception:
        return doc

def retrieve(query: str, top_k=TOP_K):
    qvec = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]

    res = coll.query(query_embeddings=[qvec], n_results=top_k)
    hits = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    results = []
    
    for i, doc, m, dist in tqdm(hits):
        print(f"[process] {i}")
        metadata = m if isinstance(m, dict) else {"meta": str(m)}
        orig_url = metadata.get("url") or metadata.get("source_url") or metadata.get("source") or i
        cached_doc = doc

        last_ts = parse_ts(metadata.get("time"))
        now_utc = datetime.now(timezone.utc)
        needs_refresh = bool(last_ts and (now_utc - last_ts > timedelta(days=1)))

        judge_cached = judge_relevance(query, cached_doc)
        is_relevant = judge_cached.get("relevant", False)
        print(f"  cached relevant={is_relevant} refresh_due={needs_refresh}")

        live_doc = None
        fetch_note = "skip_irrelevant" if not is_relevant else "skipped_recent"

        # Skip work for clearly irrelevant docs.
        if not is_relevant:
            continue

        if needs_refresh and last_ts:
            print("  fetching live copy...")
            live_doc, fetch_note = fetch_latest(orig_url)
            print(f"  fetch_note={fetch_note}")

        if live_doc:
            final_doc = live_doc if live_doc.startswith("URL:") else f"URL: {orig_url}\n\n{live_doc.strip()}"
            print("  cleaning content via LLM...")
            cleaned_doc = strip_extraneous(final_doc)
        else:
            final_doc = cached_doc
            cleaned_doc = strip_extraneous(final_doc)

        if not cleaned_doc.startswith("URL:"):
            cleaned_doc = f"URL: {orig_url}\n\n{cleaned_doc.strip()}"

        print(f"  length raw={len(final_doc)} cleaned={len(cleaned_doc)}")

        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=cleaned_doc)["embedding"]

        metadata = {
            **metadata,
            "url": orig_url,
            "time": now_utc.isoformat(),
        }

        coll.upsert(
            ids=[i],
            embeddings=[embedding],
            documents=[cleaned_doc],
            metadatas=[metadata],
        )

        results.append({
            "url": i,
            "original_document": cached_doc,
            "document": cleaned_doc,
            "score": dist,
        })
    return results

if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "What is the time, place, and manner policy?"
    results = retrieve(q)
    out_path = "retriever_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_path}")
