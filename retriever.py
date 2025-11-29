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


def extract_url_from_doc(doc: str) -> str | None:
    first = doc.splitlines()[0].strip() if doc else ""
    if first.lower().startswith("url:"):
        return first.split("URL:", 1)[1].strip()
    return None


def ensure_scheme(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}"


def parse_ts(value) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def check_relevance(query: str, document: str) -> bool:
    prompt = (
        "You are a strict relevance filter. Given a user question and the full document text, return JSON with a single boolean field 'relevant'. "
        "Only set relevant to true if the document clearly contains information that answers atleast a part of the question. If unsure, set false.\n"
        f"Question: {query}\nDocument:\n{document}\nAnswer:"
    )
    try:
        reply = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )["message"]["content"]
    except Exception:
        return False

    try:
        data = json.loads(reply)
        return bool(data.get("relevant", False))
    except Exception:
        lower = reply.lower()
        return "true" in lower or "yes" in lower


def clean_main(doc: str) -> str:
    prompt = (
        "You will clean a page text. Remove navigation, menus, breadcrumbs, footers, contact blocks, "
        "social links, boilerplate headers/footers, and anything not part of the main body content. "
        "Keep only the main informational content relevant to the page topic. Do NOT add any new content; only remove. "
        "Return plain text only, no JSON, no explanation.\n\n"
        f"Page text:\n{doc}"
    )
    try:
        cleaned = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = cleaned["message"]["content"] if isinstance(cleaned, dict) else str(cleaned)
        content = content.strip()
        # Guard against hallucinated expansions
        if len(content) > len(doc) * 1.05:
            return doc
        return content or doc
    except Exception:
        return doc

def retrieve(query: str, top_k=TOP_K, return_checked: bool = False):
    qvec = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]

    res = coll.query(query_embeddings=[qvec], n_results=top_k)
    hits = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    results = []
    checked_urls = []
    
    for i, doc, m, dist in tqdm(hits):
        print(f"[process] {i}")
        metadata = m if isinstance(m, dict) else {"meta": str(m)}
        orig_url = metadata.get("url") or metadata.get("source_url") or metadata.get("source")
        if not orig_url:
            orig_url = extract_url_from_doc(doc) or i
        orig_url = ensure_scheme(orig_url)
        checked_urls.append(orig_url)
        cached_doc = doc if doc.startswith("URL:") else f"URL: {orig_url}\n\n{doc.strip()}"

        last_ts = parse_ts(metadata.get("time"))
        now_utc = datetime.now(timezone.utc)
        needs_refresh = bool(last_ts and (now_utc - last_ts > timedelta(days=1)))

        print(f"  refresh_due={needs_refresh}")

        live_doc = None
        if needs_refresh and last_ts:
            print("  fetching live copy...")
            live_doc, fetch_note = fetch_latest(orig_url)
            print(f"  fetch_note={fetch_note}")

        if needs_refresh and last_ts:
            final_doc = live_doc if live_doc else cached_doc
            if final_doc and not final_doc.startswith("URL:"):
                final_doc = f"URL: {orig_url}\n\n{final_doc.strip()}"
            cleaned_doc = clean_main(final_doc)
            if not cleaned_doc.startswith("URL:"):
                cleaned_doc = f"URL: {orig_url}\n\n{cleaned_doc.strip()}"
            print(f"  length raw={len(final_doc)} cleaned={len(cleaned_doc)}")

            is_relevant = check_relevance(query, cleaned_doc)
            print(f"  relevance (cleaned)={is_relevant}")

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

            if is_relevant:
                results.append({
                    "url": orig_url,
                    "original_document": cached_doc,
                    "document": cleaned_doc,
                    "score": dist,
                })
        else:
            # Not old: relevance check on cached doc only
            is_relevant = check_relevance(query, cached_doc)
            print(f"  relevance (cached)={is_relevant}")
            if is_relevant:
                results.append({
                    "url": orig_url,
                    "original_document": cached_doc,
                    "document": cached_doc,
                    "score": dist,
                })
    if return_checked:
        return results, checked_urls
    return results

if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "What is the time, place, and manner policy?"
    results = retrieve(q)
    out_path = "retriever_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_path}")
