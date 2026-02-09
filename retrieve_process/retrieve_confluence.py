import os
import sys
import json
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_openai import OpenAIEmbeddings
from httpx import Client, AsyncClient
from pathlib import Path

load_dotenv()

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_URL_EMBED,
    API_KEY_LLM,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_MODEL,
    RETRIEVAL_RESULTS_RERANKED
)

ROUTER_INPUT_FILE = Path("llm_results.json")

def extract_metadata_from_payload(hit, collection_name):
    """
    Ekstraksi metadata sesuai skema payload confluence_coll vs chunk_coll
    """
    pay = hit.payload
    
    if collection_name == "confluence_coll":
        # Skema Confluence
        return {
            "chunk_id": str(hit.id),
            "page_id": pay.get("page_id"),
            "page_title": pay.get("page_title"),
            "filename": f"CONFLUENCE: {pay.get('page_title')}",
            "collection": collection_name
        }
    else:
        # Skema Chunk (chunk_coll / jira_items_coll)
        return {
            "chunk_id": str(hit.id),
            "filename": pay.get("source_file") or pay.get("ticket_key") or "unknown",
            "project": pay.get("parent_project_name"),
            "doc_type": pay.get("parent_doc_type_name"),
            "collection": collection_name
        }

def process_query_tool_logic(q_client, embed_client, query_text, target_collection, k=5):
    """
    Logika Retrieval murni dari Tool:
    1. MMR Search dengan fetch_k = k * 4
    2. Dual-Collection jika target adalah chunk_coll
    3. TANPA Rerank & TANPA Dynamic Threshold
    """
    print(f"🔍 Processing: {query_text[:60]}... (Routing: {target_collection})")
    
    colls_to_search = [target_collection]
    if target_collection == "chunk_coll":
        colls_to_search.append("confluence_coll")

    query_vector = embed_client.embed_query(query_text)
    final_output = []

    for coll in colls_to_search:
        try:
            # Menggunakan Prefetch untuk simulasi MMR (Diversity Search)
            # Sesuai tool: fetch_k = k * 4
            search_results = q_client.query_points(
                collection_name=coll,
                prefetch=models.Prefetch(
                    query=query_vector,
                    limit=k * 4,
                ),
                query=query_vector,
                limit=k,
                with_payload=True,
            )
            
            for hit in search_results.points:
                metadata = extract_metadata_from_payload(hit, coll)
                final_output.append({
                    "metadata": metadata,
                    "content": hit.payload.get("text") or hit.payload.get("page_content") or "",
                    "score": round(hit.score, 4)
                })
        except Exception as e:
            print(f"[ERROR] Search failed in {coll}: {e}")

    # Urutkan berdasarkan skor tertinggi (karena ada gabungan koleksi)
    final_output.sort(key=lambda x: x["score"], reverse=True)
    
    # Karena k=5 per koleksi, maksimal hasil gabungan adalah 10
    return final_output[:10]

def main():
    print("="*70)
    print("BATCH RETRIEVER - TOOL LOGIC VERSION (MMR, NO THRESHOLD)")
    print("="*70)

    # Setup Clients
    q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, verify=False)
    embed_client = OpenAIEmbeddings(
        base_url=BASE_URL_EMBED,
        model=EMBEDDING_MODEL,
        api_key=API_KEY_LLM,
        http_client=Client(verify=False),
        check_embedding_ctx_length=False
    )

    if not ROUTER_INPUT_FILE.exists():
        print(f"[ERROR] Input router '{ROUTER_INPUT_FILE}' tidak ditemukan!")
        return

    with open(ROUTER_INPUT_FILE, 'r') as f:
        router_data = json.load(f)

    output_data = {}

    

    for i, item in enumerate(router_data, 1):
        query_text = item.get("question")
        # Keputusan routing murni dari hasil router sebelumnya
        target_coll = item.get("llm_response", "chunk_coll").strip().replace("'", "").replace("\"", "")

        # Eksekusi dengan logika tool (MMR)
        results = process_query_tool_logic(q_client, embed_client, query_text, target_coll, k=5)
        
        # Simpan hasil apa adanya (Top 10 gabungan atau Top 5 tunggal)
        output_data[query_text] = results
        
        if i % 5 == 0:
            print(f"Progress: {i}/{len(router_data)} queries done.")
        
        time.sleep(0.1)

        # if i == 49:
        #     break

    # Simpan ke JSON
    RETRIEVAL_RESULTS_RERANKED.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_RESULTS_RERANKED, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Selesai! Hasil retrieval disimpan di: {RETRIEVAL_RESULTS_RERANKED}")

if __name__ == "__main__":
    main()