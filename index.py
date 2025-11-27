import os
import pandas as pd
from datetime import datetime
import ollama
import chromadb
from tqdm import tqdm

COLL_NAME = "csulb"
EMBED_MODEL = "nomic-embed-text" # mxbai-embed-large

crawl_df = pd.read_csv("./crawl_log.csv")
crawl_df.drop_duplicates("url", inplace=True)
crawl_df = crawl_df[(crawl_df["status"] == 200) & (crawl_df["file_path"].notna())].reset_index(drop=True)
crawl_df = crawl_df.drop(columns=["status"])
crawl_df.to_csv("./cleaned_crawl.csv", index=False)

client = chromadb.PersistentClient(path="chroma_db")
coll = client.get_or_create_collection(COLL_NAME)

if __name__ == "__main__":
    for i in tqdm(range(len(crawl_df))):
        row = crawl_df.iloc[i]
        file_path = row["file_path"]
        if not isinstance(file_path, str) or not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            document =  f.read()

        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=document)["embedding"]
        
        metadata = {
            "source_file" : file_path,
            "time": datetime.now().isoformat()
        }

        coll.upsert(
            ids=[row["url"]],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )
