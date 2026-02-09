import os
import json
import asyncio
import math
import requests
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from unset_proxy import unset_proxy
from truststore import inject_into_ssl
from langchain_openai import OpenAIEmbeddings
from httpx import Client, AsyncClient
from qdrant_client.models import (SearchParams)

load_dotenv()
inject_into_ssl()

# --- CONFIGURATION ---
EMBED_BASE_URL = "https://vllm-embedding-serve.prd-cml-1.apps.rpbdds03drx24lp.supporting.corp.bankmandiri.co.id"
RERANK_URL = os.getenv("RERANK_URL", "https://rerank.prd-cml-1.apps.rpbdds03drx24lp.supporting.corp.bankmandiri.co.id/v1/rerank")
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"
# FINAL_THRESHOLD dihapus karena sekarang dinamis
TASK_DESC = "Given from collection, retrieve relevant passages that answer the query user exactly same"

def get_detailed_instruct(query: str) -> str:
    return f'Instruct: {TASK_DESC}\nQuery: {query}'

def get_as_list(data):
    if data is None: return []
    if isinstance(data, list): return [d for d in data if d is not None]
    return [data]

async def call_reranker(query, documents, top_n=50):
    if not documents:
        return None
    actual_top_n = min(top_n, len(documents))
    headers = {
        "Authorization": "Bearer sk-x128-Aevgy30YChCDK63nw",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": actual_top_n,
    }
    async with AsyncClient(verify=False) as client:
        try:
            resp = await client.post(RERANK_URL, headers=headers, json=payload, timeout=60.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"❌ Rerank API Error: {e}")
            return None

async def get_chunk_values_for_filter(client, collection, field_name, search_values):
    """
    Get actual values from chunk_coll that match search_values (case-insensitive).
    Returns list of actual values that can be used in filter.
    """
    if not search_values:
        return []

    try:
        # Fetch all unique values from field (dengan limit)
        res, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=None,
            with_payload=True,
            limit=1000
        )

        # Collect unique values
        unique_values = set()
        for point in res:
            val = point.payload.get(field_name)
            if val:
                unique_values.add(val)

        # Match search_values dengan case-insensitive comparison
        search_lower = [str(s).lower() for s in search_values]
        matched_values = []
        for db_val in unique_values:
            if str(db_val).lower() in search_lower:
                matched_values.append(db_val)

        return matched_values
    except Exception as e:
        print(f"Error getting chunk values for {field_name}: {e}")
        return []

async def get_ids(client, collection, field_name, values, parent_filters=None):
    if not values: return []
    # First try with original values (case-sensitive)
    condition = models.FieldCondition(key=field_name, match=models.MatchAny(any=values))
    must_conditions = [condition]
    if parent_filters: must_conditions.extend(parent_filters)
    try:
        res, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=must_conditions),
            with_payload=False,
            limit=200
        )
        if res:
            return [str(p.id) for p in res]

        # If no results with case-sensitive, try lowercase (case-insensitive fallback)
        normalized_values = [str(v).lower() for v in values]
        condition = models.FieldCondition(key=field_name, match=models.MatchAny(any=normalized_values))
        must_conditions = [condition]
        if parent_filters: must_conditions.extend(parent_filters)
        res, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=must_conditions),
            with_payload=False,
            limit=200
        )
        return [str(p.id) for p in res]
    except Exception as e:
        print(f"Error searching {collection}: {e}")
        return []

async def process_ner_query(client, embed_client, query_text, entity, index):
    print(f"\n{'='*70}\nITERASI KE-{index}: {query_text[:60]}...\n{'='*70}")

    # --- TAHAP 1-3: HIERARCHICAL FILTER ---
    # Extract and normalize NER entities (convert to lowercase for comparison)
    proj_names = get_as_list(entity.get("project_name"))
    act_names = get_as_list(entity.get("activity_name"))
    doc_names = get_as_list(entity.get("deliverable"))

    # Normalize to lowercase
    proj_names_lower = [str(p).lower() for p in proj_names]
    act_names_lower = [str(a).lower() for a in act_names]
    doc_names_lower = [str(d).lower() for d in doc_names]

    final_must_filters = []

    # Get IDs from collections with normalized values
    p_ids = await get_ids(client, "proj_coll", "proj_name", proj_names_lower)
    if p_ids: final_must_filters.append(models.FieldCondition(key="parent_project_uuid", match=models.MatchAny(any=p_ids)))

    act_p_filter = [models.FieldCondition(key="parent_project_uuid", match=models.MatchAny(any=p_ids))] if p_ids else None
    a_ids = await get_ids(client, "activity_coll", "activity_name", act_names_lower, act_p_filter)
    if a_ids: final_must_filters.append(models.FieldCondition(key="parent_activity_uuid", match=models.MatchAny(any=a_ids)))

    doc_p_filter = []
    if p_ids: doc_p_filter.append(models.FieldCondition(key="parent_project_uuid", match=models.MatchAny(any=p_ids)))
    if a_ids: doc_p_filter.append(models.FieldCondition(key="parent_activity_uuid", match=models.MatchAny(any=a_ids)))
    d_ids = await get_ids(client, "doc_type_coll", "doc_type_name", doc_names_lower, doc_p_filter)
    if d_ids: final_must_filters.append(models.FieldCondition(key="parent_doc_type_uuid", match=models.MatchAny(any=d_ids)))

    # --- FILTERING LOGIC FOR chunk_coll ---
    # Filter berdasarkan entities yang ada di NER output dengan case-insensitive matching
    # Jika tidak ada entity, maka search kesemuanya (no filter)
    chunk_filters = []

    # Get actual values from chunk_coll with case-insensitive matching
    actual_proj_names = await get_chunk_values_for_filter(client, "chunk_coll", "parent_project_name", proj_names)
    actual_act_names = await get_chunk_values_for_filter(client, "chunk_coll", "parent_activity_name", act_names)
    actual_doc_names = await get_chunk_values_for_filter(client, "chunk_coll", "parent_doc_type_name", doc_names)

    if actual_proj_names:
        chunk_filters.append(
            models.FieldCondition(
                key="parent_project_name",
                match=models.MatchAny(any=actual_proj_names)
            )
        )
        print(f"✓ Filter by project_name: {actual_proj_names}")

    if actual_act_names:
        chunk_filters.append(
            models.FieldCondition(
                key="parent_activity_name",
                match=models.MatchAny(any=actual_act_names)
            )
        )
        print(f"✓ Filter by activity_name: {actual_act_names}")

    if actual_doc_names:
        chunk_filters.append(
            models.FieldCondition(
                key="parent_doc_type_name",
                match=models.MatchAny(any=actual_doc_names)
            )
        )
        print(f"✓ Filter by doc_type_name: {actual_doc_names}")

    if not chunk_filters:
        print("✓ No NER entities found - searching all chunks")

    # --- TAHAP 4: INITIAL VECTOR SEARCH (Top 50) ---
    formatted_query = get_detailed_instruct(query_text)
    query_vector = await embed_client.aembed_query(formatted_query)

    search_params = SearchParams(
        hnsw_ef = 300,
        exact = True,
    )

    # Use chunk_filters for chunk_coll filtering (based on parent_* fields from NER entities)
    # If no filters, search all chunks
    chunk_query_filter = None
    if chunk_filters:
        chunk_query_filter = models.Filter(must=chunk_filters)

    search_results = await client.query_points(
        collection_name="chunk_coll",
        query=query_vector,
        query_filter=chunk_query_filter,
        limit=50,
        score_threshold=0.1, # Diturunkan agar mendapatkan variasi skor yang cukup
        with_payload=True,
        search_params = search_params
    )

    if not search_results.points:
        print(f"⚠️ No results with filters. Attempting search without filters...")
        # Retry tanpa filter untuk debug
        search_results = await client.query_points(
            collection_name="chunk_coll",
            query=query_vector,
            query_filter=None,
            limit=5,
            score_threshold=0.1,
            with_payload=True,
            search_params = search_params
        )
        if search_results.points:
            print(f"📌 Sample payloads dari chunk_coll (tanpa filter):")
            for hit in search_results.points[:1]:
                print(f"  - parent_project_name: {hit.payload.get('parent_project_name')}")
                print(f"  - parent_activity_name: {hit.payload.get('parent_activity_name')}")
                print(f"  - parent_doc_type_name: {hit.payload.get('parent_doc_type_name')}")
        return []

    # --- TAHAP 5: PREPARASI DATA RERANK ---
    docs_to_rerank = []
    valid_points = []
    for hit in search_results.points:
        text_content = hit.payload.get("page_content") or hit.payload.get("text")
        if text_content:
            docs_to_rerank.append(text_content)
            valid_points.append(hit)

    # --- TAHAP 6: RERANKING & SCORING ---
    rerank_resp = await call_reranker(query_text, docs_to_rerank, top_n=len(docs_to_rerank))
    
    all_weighted_results = []
    if rerank_resp and "results" in rerank_resp:
        for item in rerank_resp["results"]:
            idx = item["index"]
            rerank_score = item["relevance_score"]
            original_hit = valid_points[idx]
            vector_score = original_hit.score
            
            # Hybrid Weighted Scoring
            final_weighted_score = (0.7 * vector_score) + (0.3 * rerank_score)
            
            pay = original_hit.payload
            all_weighted_results.append({
                "metadata": {
                    "chunk_id": str(original_hit.id),
                    "filename": pay.get("source_file"),
                },
                "content": docs_to_rerank[idx],
                "score_final": round(final_weighted_score, 4)
            })

    # --- TAHAP 7: DYNAMIC THRESHOLD CALCULATION ---
    # Urutkan skor dari tinggi ke rendah
    all_weighted_results.sort(key=lambda x: x["score_final"], reverse=True)
    
    dynamic_threshold = 0.1 # Default minimum
    if all_weighted_results:
        # Cari index yang mencakup 75% dari total data yang didapat
        # Jika ada 50 data, 75% adalah 37.5 -> kita ambil index ke-37 (data ke-38)
        target_index = min(math.ceil(len(all_weighted_results) * 0.75) - 1, len(all_weighted_results) - 1)
        raw_threshold = all_weighted_results[target_index]["score_final"]
        
        # Clamp nilai threshold antara 0.1 dan 1.0
        dynamic_threshold = max(0.1, min(1.0, raw_threshold))

    # --- TAHAP 8: FILTERING DENGAN DYNAMIC THRESHOLD ---
    formatted_results = [
        res for res in all_weighted_results 
        if res["score_final"] >= dynamic_threshold
    ][:10]  # Slicing ini memastikan maks 10, jika kurang dari 10 tetap aman

    print(f"📊 Total Candidates: {len(all_weighted_results)}")
    print(f"⚙️ Dynamic Threshold (75% coverage): {dynamic_threshold}")
    print(f"🎯 Final Chunks (Capped at 10): {len(formatted_results)}")
    
    return formatted_results

async def main():
    # ... (Bagian inisialisasi model name & embed_client tetap sama) ...
    m_info = requests.get(os.path.join(EMBED_BASE_URL, "v1/models")).json()
    m_name = m_info["data"][0]["id"]

    embed_client = OpenAIEmbeddings(
        base_url=os.path.join(EMBED_BASE_URL, "v1"),
        model=m_name,
        api_key="EMPTY",
        http_client=Client(verify=False),
        http_async_client=AsyncClient(verify=False),
        check_embedding_ctx_length=False
    )

    # PATH INPUT - Input dari router_extraction output
    input_path = '/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/batch_results_final.json'
    output_path = '/home/cdsw/retrieve_developments/retrieve_070126/output/retrieval_results_reranked.json'

    with open(input_path, 'r') as f:
        data = json.load(f)

    q_client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    output_data = {}

    try:
        # --- LOGIKA DETEKSI FORMAT INPUT ---
        queries_to_process = []

        if "results" in data and isinstance(data["results"], dict):
            print("detecting format: batch_results_final.json (New Format)")
            # Iterasi Dokumen (doc_1, doc_2, dst)
            for doc_id, doc_content in data["results"].items():
                questions = doc_content.get("questions", {})
                # Iterasi Tipe Tes (single_doc_factual, dst)
                for test_type, test_data in questions.items():
                    query_text = test_data.get("question")
                    ner_raw = test_data.get("ner_output", {})
                    
                    # Mapping: doc_type_name di file baru = deliverable di logic lama
                    entity = {
                        "project_name": ner_raw.get("project_name"),
                        "activity_name": ner_raw.get("activity_name"),
                        "deliverable": ner_raw.get("doc_type_name") # Penyesuaian kunci
                    }
                    queries_to_process.append((query_text, entity))
        else:
            print("detecting format: ner_batch_results.json (Old Format)")
            for query_text, val in data.items():
                queries_to_process.append((query_text, val.get("entity", {})))

        # --- EKSEKUSI PROSES ---
        for i, (query_text, entity) in enumerate(queries_to_process, 1):
            res = await process_ner_query(q_client, embed_client, query_text, entity, i)
            output_data[query_text] = res

        # Simpan Hasil
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ Selesai! {len(output_data)} query diproses. Hasil di {output_path}")

    finally:
        await q_client.close()

if __name__ == "__main__":
    unset_proxy()
    asyncio.run(main())