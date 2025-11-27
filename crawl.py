import os
import queue
import re
import time
import shutil
from datetime import datetime
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

BASE_URLS = [
    "https://www.csulb.edu",
    "https://catalog.csulb.edu",
    "http://catalog.csulb.edu",
]

OUTPUT_DIR = "./data"
LOG_FILE = "crawl_log.csv"

MAX_PAGES_PER_DOMAIN = 5000
REQUEST_DELAY = 0.5
USER_AGENT = "CSULB-RAG-Crawler/0.1 (+rohildalal@gmail.com)"
# How to handle URLs from previous runs: "overwrite", "skip", or "reset" (clears data and log before crawling).
DUPLICATE_MODE = "reset"


def normalize_domain(netloc):
    host = netloc.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def url_in_domain(url, domain):
    parsed = urllib.parse.urlparse(url)
    host = normalize_domain(parsed.netloc)
    return host == domain or host.endswith("." + domain)


def clean_text(html):
    """Return stripped visible text; swallow parser errors and non-text."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text.strip()


def url_to_filename(url):
    encoded = urllib.parse.quote(url, safe="")
    if len(encoded) > 180:
        digest = re.sub(r"[^a-fA-F0-9]", "", encoded)[:16] or "hash"
        encoded = f"{encoded[:120]}-{digest}"
    return f"{encoded}.txt"


def save_text(text, url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filename = url_to_filename(url)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n\n{text}\n")
    return path


def log_csv(row):
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    exists = os.path.exists(LOG_FILE)
    cols = ["url", "status", "timestamp", "text_length", "file_path", "note"]
    pd.DataFrame([row], columns=cols).to_csv(
        LOG_FILE, mode="a", header=not exists, index=False
    )


def load_seen(domain):
    if not os.path.exists(LOG_FILE):
        return set(), 0
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception:
        return set(), 0
    df = df[df["status"] == 200]

    def _domain_from_url(url):
        try:
            return normalize_domain(urllib.parse.urlparse(str(url)).netloc)
        except Exception:
            return ""

    df = df[df["url"].apply(_domain_from_url) == domain]
    urls = set(df["url"].tolist())
    return urls, len(urls)


def fetch_page(url, user_agent, timeout=15):
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    return resp.status_code, resp.text, resp.headers.get("Content-Type", "")


def crawl(base_url):
    domain = normalize_domain(urllib.parse.urlparse(base_url).netloc)
    rp = urllib.robotparser.RobotFileParser()
    try:
        robots_url = urllib.parse.urljoin(base_url, "/robots.txt")
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        rp = None

    seen_ok, already_ok = load_seen(domain)
    seen = set()
    if already_ok >= MAX_PAGES_PER_DOMAIN:
        print(f"[skip] {domain} already has {already_ok} pages (>= max {MAX_PAGES_PER_DOMAIN})")
        return

    start_urls = []
    if rp:
        try:
            sitemaps = rp.site_maps() or []
        except Exception:
            sitemaps = []
        for sm in sitemaps:
            try:
                resp = requests.get(sm, timeout=20)
                tree = ET.fromstring(resp.content)
                for loc in tree.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                    url = (loc.text or "").strip()
                    if url and url_in_domain(url, domain):
                        start_urls.append(url)
            except Exception:
                continue
    if not start_urls:
        start_urls.append(base_url)

    q = queue.Queue()
    queued = set()
    for u in start_urls:
        q.put(u)
        queued.add(u)
    saved = already_ok
    request_idx = 0

    while not q.empty() and saved < MAX_PAGES_PER_DOMAIN:
        url = q.get()
        if url in seen or not url_in_domain(url, domain):
            continue
        seen.add(url)
        already_crawled = url in seen_ok
        skip_save = DUPLICATE_MODE == "skip" and already_crawled
        timestamp = datetime.now().date().isoformat()
        request_idx += 1
        print(f"[{request_idx}/{q.qsize()}] {url}")
        if rp and not rp.can_fetch("*", url):
            log_csv(
                {
                    "url": url,
                    "status": 0,
                    "timestamp": timestamp,
                    "text_length": 0,
                    "file_path": "",
                    "note": "blocked_by_robots",
                }
            )
            continue

        try:
            status, html, content_type = fetch_page(url, USER_AGENT)
        except Exception as exc:
            if not skip_save:
                log_csv(
                    {
                        "url": url,
                        "status": 0,
                        "timestamp": timestamp,
                        "text_length": 0,
                        "file_path": "",
                        "note": str(exc),
                    }
                )
            continue

        if status != 200:
            if not skip_save:
                log_csv(
                    {
                        "url": url,
                        "status": status,
                        "timestamp": timestamp,
                        "text_length": 0,
                        "file_path": "",
                        "note": "",
                    }
                )
            continue

        content_type = (content_type or "").lower()
        if "html" not in content_type:
            if not skip_save:
                log_csv(
                    {
                        "url": url,
                        "status": status,
                        "timestamp": timestamp,
                        "text_length": 0,
                        "file_path": "",
                        "note": f"non_html:{content_type}",
                    }
                )
            continue

        text = clean_text(html)
        should_save = text and not skip_save
        if should_save:
            path = save_text(text, url, OUTPUT_DIR)
            print(f"Saved : {len(text)}\n")
            seen_ok.add(url)
            log_csv(
                {
                    "url": url,
                    "status": status,
                    "timestamp": timestamp,
                    "text_length": len(text),
                    "file_path": path,
                    "note": "",
                }
            )
            if not already_crawled:
                saved += 1

        soup_links = extract_links(html, url)
        for link in soup_links:
            if DUPLICATE_MODE == "skip" and link in seen_ok:
                continue
            if link not in seen and link not in queued and url_in_domain(link, domain):
                q.put(link)
                queued.add(link)

        if REQUEST_DELAY:
            time.sleep(REQUEST_DELAY)


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(base_url, a["href"])
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme in {"http", "https"}:
            links.add(href.split("#", 1)[0])
    return links


def main():
    if DUPLICATE_MODE == "reset":
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        try:
            os.remove(LOG_FILE)
        except FileNotFoundError:
            pass

    for base_url in BASE_URLS:
        print(f"[start] {base_url}")
        crawl(base_url)


if __name__ == "__main__":
    main()
