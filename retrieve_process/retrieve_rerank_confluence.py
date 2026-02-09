import os
import sys
import json
import math
import time
import requests
from datetime import datetime # Tambahan untuk timestamp
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_openai import OpenAIEmbeddings
from httpx import Client, AsyncClient
from pathlib import Path
from tqdm import tqdm # Library standar untuk progress bar

# Load environment & SSL setup
load_dotenv()

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_URL_EMBED,
    LITELLM_URL,
    LITELLM_KEY,
    API_KEY_LLM,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    RETRIEVAL_RESULTS_RERANKED
)

ROUTER_INPUT_FILE = Path("llm_results.json")
RERANK_URL = f"{LITELLM_URL.rstrip('/')}/v1/rerank"

def call_reranker(query, documents, top_n=40, max_retries=3, initial_delay=2):
    if not documents: return None
    
    headers = {
        "Authorization": f"Bearer {LITELLM_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(RERANK_URL, headers=headers, json=payload, timeout=60.0, verify=False)
            if resp.status_code == 429:
                if attempt < max_retries:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"\n   ⚠️ [WAIT] Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < max_retries:
                wait_time = initial_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                return None
    return None

def extract_metadata(hit, coll_name):
    pay = hit.payload
    if coll_name == "confluence_coll":
        return {
            "chunk_id": str(hit.id),
            "page_id": pay.get("page_id"),
            "page_title": pay.get("page_title"),
            "filename": f"CONFLUENCE: {pay.get('page_title')}",
            "collection": coll_name
        }
    else:
        return {
            "chunk_id": str(hit.id),
            "filename": pay.get("source_file") or pay.get("ticket_key") or "unknown",
            "project": pay.get("parent_project_name"),
            "doc_type": pay.get("parent_doc_type_name"),
            "collection": coll_name
        }

def process_rerank_logic(q_client, embed_client, query_text, target_collection, k=20):
    # Log internal proses (tanpa newline agar tidak berantakan dengan tqdm)
    colls_to_search = [target_collection]
    if target_collection == "chunk_coll":
        colls_to_search.append("confluence_coll")

    query_vector = embed_client.embed_query(query_text)
    candidates = []
    
    for coll in colls_to_search:
        try:
            search_results = q_client.query_points(
                collection_name=coll,
                query=query_vector,
                limit=k,
                with_payload=True
            )
            for hit in search_results.points:
                content = hit.payload.get("text") or hit.payload.get("page_content") or ""
                if content:
                    candidates.append({
                        "hit": hit,
                        "content": content,
                        "vector_score": hit.score,
                        "collection": coll
                    })
        except Exception as e:
            pass

    if not candidates: return []

    docs_to_rerank = [c["content"] for c in candidates]
    rerank_resp = call_reranker(query_text, docs_to_rerank)
    
    all_weighted_results = []
    if rerank_resp and "results" in rerank_resp:
        for item in rerank_resp["results"]:
            idx = item["index"]
            rerank_score = item["relevance_score"]
            candidate = candidates[idx]
            final_weighted_score = (0.7 * candidate["vector_score"]) + (0.3 * rerank_score)
            metadata = extract_metadata(candidate["hit"], candidate["collection"])
            all_weighted_results.append({
                "metadata": metadata,
                "content": candidate["content"],
                "score_final": round(final_weighted_score, 4)
            })
    else:
        for c in candidates:
            all_weighted_results.append({
                "metadata": extract_metadata(c["hit"], c["collection"]),
                "content": c["content"],
                "score_final": round(c["vector_score"], 4)
            })

    all_weighted_results.sort(key=lambda x: x["score_final"], reverse=True)
    target_index = min(math.ceil(len(all_weighted_results) * 0.75) - 1, len(all_weighted_results) - 1)
    dynamic_threshold = all_weighted_results[target_index]["score_final"] if all_weighted_results else 0.1
    dynamic_threshold = max(0.1, dynamic_threshold) 

    formatted_results = [
        res for res in all_weighted_results
        if res["score_final"] >= dynamic_threshold
    ][:10]

    return formatted_results

def main():
    # --- UI Start ---
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*80)
    print(f"🚀 BATCH RETRIEVAL SYSTEM STARTING AT {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)

    q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, verify=False)
    embed_client = OpenAIEmbeddings(
        base_url=BASE_URL_EMBED,
        model=EMBEDDING_MODEL,
        api_key=API_KEY_LLM,
        http_client=Client(verify=False),
        check_embedding_ctx_length=False
    )

    if not ROUTER_INPUT_FILE.exists():
        print(f"❌ [ERROR] File {ROUTER_INPUT_FILE} tidak ditemukan!")
        return

    with open(ROUTER_INPUT_FILE, 'r') as f:
        router_data = json.load(f)

    total_queries = len(router_data)
    print(f"📦 Loaded {total_queries} queries from {ROUTER_INPUT_FILE}")
    print(f"🤖 Model: {RERANK_MODEL} | Collections: chunk_coll, confluence_coll, jira_items_coll")
    print("-"*80)

    output_data = {}

    # --- Progress Bar with TQDM ---
    # desc: teks di depan bar, unit: satuan proses
    pbar = tqdm(router_data, desc="Processing Queries", unit="query", dynamic_ncols=True)

    for i, item in enumerate(pbar, 1):
        query_text = item.get("question")
        target_coll = item.get("llm_response", "chunk_coll").strip().replace("'", "").replace("\"", "")
        
        # Update deskripsi bar supaya tau query mana yang lagi jalan
        pbar.set_description(f"Processing ({target_coll})")
        
        results = process_rerank_logic(q_client, embed_client, query_text, target_coll, k=20)
        output_data[query_text] = results
        
        # Jeda kestabilan
        time.sleep(0.5)

    # --- UI End ---
    print("\n" + "="*80)
    print(f"✅ FINISHED AT {datetime.now().strftime('%H:%M:%S')}")
    
    RETRIEVAL_RESULTS_RERANKED.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_RESULTS_RERANKED, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📂 Output saved to: {RETRIEVAL_RESULTS_RERANKED}")
    print("="*80)

if __name__ == "__main__":
    main()