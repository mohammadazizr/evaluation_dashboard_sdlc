import os
import json
import math
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_openai import OpenAIEmbeddings
from httpx import Client, AsyncClient

load_dotenv()

# --- CONFIGURATION ---
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

RERANK_URL = f"{LITELLM_URL.rstrip('/')}/v1/rerank"
RERANK_MODEL = "cohere.rerank-v3-5:0"  # Sesuaikan dengan config LiteLLM

TASK_DESC = "Given from collection, retrieve relevant passages that answer the query user exactly same"

def get_detailed_instruct(query: str) -> str:
    return f'Instruct: {TASK_DESC}\nQuery: {query}'

def get_as_list(data):
    if data is None: return []
    if isinstance(data, list): return [d for d in data if d is not None]
    return [data]

def call_reranker(query, documents, top_n=50):
    """Call reranker API synchronously"""
    if not documents:
        return None

    actual_top_n = min(top_n, len(documents))
    headers = {
        "Authorization": f"Bearer {LITELLM_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": actual_top_n,
    }

    try:
        resp = requests.post(RERANK_URL, headers=headers, json=payload, timeout=60.0, verify=False)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] Rerank API Error: {e}")
        return None

def get_chunk_values_for_filter(client, collection, field_name, search_values):
    """
    Get actual values from chunk_coll that match search_values (case-insensitive).
    Returns list of actual values that can be used in filter.
    """
    if not search_values:
        return []

    try:
        # Scroll through collection to get unique values
        res, _ = client.scroll(
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

def get_ids(client, collection, field_name, values, parent_filters=None):
    """Get IDs from collection that match field values"""
    if not values:
        return []

    # First try with original values (case-sensitive)
    condition = models.FieldCondition(key=field_name, match=models.MatchAny(any=values))
    must_conditions = [condition]
    if parent_filters:
        must_conditions.extend(parent_filters)

    try:
        res, _ = client.scroll(
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
        if parent_filters:
            must_conditions.extend(parent_filters)

        res, _ = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=must_conditions),
            with_payload=False,
            limit=200
        )
        return [str(p.id) for p in res]
    except Exception as e:
        print(f"Error searching {collection}: {e}")
        return []

def process_query(client, embed_client, query_text, entity, index):
    """Process a single query and return retrieved results"""
    print(f"\n{'='*70}\nITERASI KE-{index}: {query_text[:60]}...\n{'='*70}")

    # --- TAHAP 1-3: HIERARCHICAL FILTER ---
    proj_names = get_as_list(entity.get("project_name"))
    act_names = get_as_list(entity.get("activity_name"))
    doc_names = get_as_list(entity.get("deliverable"))

    # Normalize to lowercase
    proj_names_lower = [str(p).lower() for p in proj_names]
    act_names_lower = [str(a).lower() for a in act_names]
    doc_names_lower = [str(d).lower() for d in doc_names]

    final_must_filters = []

    # Get IDs from collections with normalized values
    p_ids = get_ids(client, "proj_coll", "proj_name", proj_names_lower)
    if p_ids:
        final_must_filters.append(models.FieldCondition(key="parent_project_uuid", match=models.MatchAny(any=p_ids)))

    act_p_filter = [models.FieldCondition(key="parent_project_uuid", match=models.MatchAny(any=p_ids))] if p_ids else None
    a_ids = get_ids(client, "activity_coll", "activity_name", act_names_lower, act_p_filter)
    if a_ids:
        final_must_filters.append(models.FieldCondition(key="parent_activity_uuid", match=models.MatchAny(any=a_ids)))

    doc_p_filter = []
    if p_ids:
        doc_p_filter.append(models.FieldCondition(key="parent_project_uuid", match=models.MatchAny(any=p_ids)))
    if a_ids:
        doc_p_filter.append(models.FieldCondition(key="parent_activity_uuid", match=models.MatchAny(any=a_ids)))
    d_ids = get_ids(client, "doc_type_coll", "doc_type_name", doc_names_lower, doc_p_filter)
    if d_ids:
        final_must_filters.append(models.FieldCondition(key="parent_doc_type_uuid", match=models.MatchAny(any=d_ids)))

    # --- FILTERING LOGIC FOR chunk_coll ---
    chunk_filters = []

    actual_proj_names = get_chunk_values_for_filter(client, "chunk_coll", "parent_project_name", proj_names)
    actual_act_names = get_chunk_values_for_filter(client, "chunk_coll", "parent_activity_name", act_names)
    actual_doc_names = get_chunk_values_for_filter(client, "chunk_coll", "parent_doc_type_name", doc_names)

    if actual_proj_names:
        chunk_filters.append(models.FieldCondition(key="parent_project_name", match=models.MatchAny(any=actual_proj_names)))
        print(f"[OK] Filter by project_name: {actual_proj_names}")

    if actual_act_names:
        chunk_filters.append(models.FieldCondition(key="parent_activity_name", match=models.MatchAny(any=actual_act_names)))
        print(f"[OK] Filter by activity_name: {actual_act_names}")

    if actual_doc_names:
        chunk_filters.append(models.FieldCondition(key="parent_doc_type_name", match=models.MatchAny(any=actual_doc_names)))
        print(f"[OK] Filter by doc_type_name: {actual_doc_names}")

    if not chunk_filters:
        print("[OK] No NER entities found - searching all chunks")

    # --- TAHAP 4: INITIAL VECTOR SEARCH (Top 50) ---
    formatted_query = get_detailed_instruct(query_text)
    query_vector = embed_client.embed_query(formatted_query)

    chunk_query_filter = None
    if chunk_filters:
        chunk_query_filter = models.Filter(must=chunk_filters)

    search_results = client.query_points(
        collection_name="chunk_coll",
        query=query_vector,
        query_filter=chunk_query_filter,
        limit=50,
        score_threshold=0.1,
        with_payload=True,
    )

    if not search_results.points:
        print(f"[WARNING] No results with filters. Attempting search without filters...")
        search_results = client.query_points(
            collection_name="chunk_coll",
            query=query_vector,
            query_filter=None,
            limit=5,
            score_threshold=0.1,
            with_payload=True,
        )
        if search_results.points:
            print(f"[INFO] Sample payloads dari chunk_coll (tanpa filter):")
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
    rerank_resp = call_reranker(query_text, docs_to_rerank, top_n=len(docs_to_rerank))

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
    all_weighted_results.sort(key=lambda x: x["score_final"], reverse=True)

    dynamic_threshold = 0.1
    if all_weighted_results:
        target_index = min(math.ceil(len(all_weighted_results) * 0.75) - 1, len(all_weighted_results) - 1)
        raw_threshold = all_weighted_results[target_index]["score_final"]
        dynamic_threshold = max(0.1, min(1.0, raw_threshold))

    # --- TAHAP 8: FILTERING DENGAN DYNAMIC THRESHOLD ---
    formatted_results = [
        res for res in all_weighted_results
        if res["score_final"] >= dynamic_threshold
    ][:10]

    print(f"[STATS] Total Candidates: {len(all_weighted_results)}")
    print(f"[CONFIG] Dynamic Threshold (75% coverage): {dynamic_threshold}")
    print(f"[RESULT] Final Chunks (Capped at 10): {len(formatted_results)}")

    return formatted_results

def main(query_text, project_name=None, activity_name=None, doc_type_name=None):
    """
    Main function untuk memproses single query

    Args:
        query_text: Query yang ingin diproses
        project_name: Optional - project name untuk filtering
        activity_name: Optional - activity name untuk filtering
        doc_type_name: Optional - doc type name untuk filtering
    """
    print("Initializing embedding client...")

    # Initialize embedding client using LiteLLM
    try:
        if not LITELLM_URL or not LITELLM_KEY or not EMBEDDING_MODEL:
            raise ValueError("LITELLM_URL, LITELLM_KEY, or EMBEDDING_MODEL not configured")

        embed_client = OpenAIEmbeddings(
            base_url=f"{LITELLM_URL.rstrip('/')}/v1",
            model=EMBEDDING_MODEL,
            api_key=LITELLM_KEY,
            http_client=Client(verify=False),
            http_async_client=AsyncClient(verify=False),
            check_embedding_ctx_length=False
        )
        print(f"[OK] Embedding client initialized with model: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize embedding client: {e}")
        return None

    print(f"\nConnecting to Qdrant at {QDRANT_URL}...")
    try:
        q_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            verify=False
        )
        print("[OK] Connected to Qdrant successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")
        return None

    try:
        # Siapkan entity (optional filters)
        entity = {
            "project_name": project_name,
            "activity_name": activity_name,
            "deliverable": doc_type_name
        }

        # Proses query tunggal
        print(f"\n🔍 Processing single query...")
        result = process_query(q_client, embed_client, query_text, entity, 1)

        print(f"\n✅ Selesai! Hasil untuk query: '{query_text}'")
        print(f"\n📋 HASIL AKHIR:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return result

    except Exception as e:
        print(f"[ERROR] Error processing query: {e}")
        return None

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Gunakan argument dari command line
        query_text = sys.argv[1]
        project_name = sys.argv[2] if len(sys.argv) > 2 else None
        activity_name = sys.argv[3] if len(sys.argv) > 3 else None
        doc_type_name = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        # Input interaktif
        print("=" * 70)
        print("RETRIEVE SINGLE QUERY - Interaktif Mode")
        print("=" * 70)
        query_text = input("\n🔍 Masukkan query: ").strip()
        project_name = input("[INPUT] Masukkan project name (Enter untuk skip): ").strip() or None
        activity_name = input("📋 Masukkan activity name (Enter untuk skip): ").strip() or None
        doc_type_name = input("[INPUT] Masukkan doc type name (Enter untuk skip): ").strip() or None

    if not query_text:
        print("[ERROR] Query tidak boleh kosong!")
        sys.exit(1)

    main(query_text, project_name, activity_name, doc_type_name)
