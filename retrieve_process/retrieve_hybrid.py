import os
import sys
import json
import math
import time
import pickle
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_openai import OpenAIEmbeddings
from httpx import Client, AsyncClient
from pathlib import Path

load_dotenv()

# Import config from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LITELLM_URL,
    LITELLM_KEY,
    EMBEDDING_MODEL,
    QDRANT_URL,
    QDRANT_API_KEY,
    RERANK_MODEL,
    BATCH_RESULTS_FINAL_INPUT,
    RETRIEVAL_RESULTS_RERANKED,
)

# --- CONFIGURATION ---
RERANK_URL = f"{LITELLM_URL.rstrip('/')}/v1/rerank"
BM25_MODEL_PATH = Path(__file__).parent.parent / "models" / "bm25_model.pkl"

# MODE SELECTION: "hybrid_only" or "hybrid_rerank"
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")  # Default: with rerank

TASK_DESC = "Given from collection, retrieve relevant passages that answer the query user exactly same"

# Global BM25 model
bm25_model = None
vocab_dict = None


# =============================================================================
# BM25 UTILITIES
# =============================================================================

def load_bm25_model():
    """Load pre-trained BM25 model from disk."""
    global bm25_model, vocab_dict
    
    if not BM25_MODEL_PATH.exists():
        print(f"[ERROR] BM25 model not found at: {BM25_MODEL_PATH}")
        print(f"        Please run the ingest script first to train BM25 model!")
        raise FileNotFoundError(f"BM25 model not found: {BM25_MODEL_PATH}")
    
    print(f"Loading BM25 model from {BM25_MODEL_PATH}...")
    with open(BM25_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    bm25_model = model_data['bm25']
    vocab_dict = model_data['vocab']
    
    print(f"  ✅ BM25 model loaded (vocabulary size: {len(vocab_dict)})")


def generate_sparse_vector(text: str):
    """
    Generate BM25 sparse vector for query text.
    
    Args:
        text: Query text
    
    Returns:
        Dictionary with 'indices' and 'values' for Qdrant sparse vector
    """
    if not text or not text.strip():
        return {"indices": [], "values": []}
    
    if bm25_model is None or vocab_dict is None:
        print("[ERROR] BM25 model not loaded!")
        return {"indices": [], "values": []}
    
    # Tokenize query
    tokens = text.lower().split()
    
    # Get BM25 scores
    scores = bm25_model.get_scores(tokens)
    
    # Build sparse vector
    indices = []
    values = []
    
    for token in set(tokens):  # Use set to avoid duplicates
        if token in vocab_dict:
            idx = vocab_dict[token]
            score = scores[idx] if idx < len(scores) else 0.0
            
            if score > 0:  # Only include non-zero scores
                indices.append(idx)
                values.append(float(score))
    
    return {"indices": indices, "values": values}


# =============================================================================
# EXISTING UTILITIES
# =============================================================================

def get_detailed_instruct(query: str) -> str:
    return f'Instruct: {TASK_DESC}\nQuery: {query}'


def get_as_list(data):
    if data is None: return []
    if isinstance(data, list): return [d for d in data if d is not None]
    return [data]


def call_reranker(query, documents, top_n=50, max_retries=3, initial_retry_delay=1):
    """
    Call reranker API with retry logic and exponential backoff.
    """
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

    retry_delay = initial_retry_delay

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(RERANK_URL, headers=headers, json=payload, timeout=60.0, verify=False)

            if resp.status_code == 429:
                if attempt < max_retries:
                    print(f"[RATE_LIMIT] Rerank API rate limited. Retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    print(f"[ERROR] Rerank API rate limited. Max retries exceeded.")
                    return None

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"[TIMEOUT] Rerank API timeout. Retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                print(f"[ERROR] Rerank API timeout. Max retries exceeded.")
                return None

        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                print(f"[CONNECTION_ERROR] Connection error: {e}. Retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                print(f"[ERROR] Connection error: {e}. Max retries exceeded.")
                return None

        except Exception as e:
            print(f"[ERROR] Rerank API Error: {e}")
            return None

    return None


def get_chunk_values_for_filter(client, collection, field_name, search_values):
    """
    Get actual values from chunk_coll_bm25 that match search_values (case-insensitive).
    """
    if not search_values:
        return []

    try:
        res, _ = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            with_payload=True,
            limit=1000
        )

        unique_values = set()
        for point in res:
            val = point.payload.get(field_name)
            if val:
                unique_values.add(val)

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

        # Fallback: case-insensitive
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


# =============================================================================
# HYBRID SEARCH QUERY PROCESSING
# =============================================================================

def process_query(client, embed_client, query_text, entity, index, mode="hybrid_rerank"):
    """
    Process a single query using HYBRID SEARCH.
    
    Args:
        mode: "hybrid_only" or "hybrid_rerank"
            - hybrid_only: Dense + Sparse + RRF (no reranker)
            - hybrid_rerank: Dense + Sparse + RRF + Reranker
    
    Pipeline:
    1. Hierarchical filtering (project -> activity -> doc_type)
    2. Generate dense vector (semantic)
    3. Generate sparse vector (BM25 keyword)
    4. Hybrid search with Reciprocal Rank Fusion (RRF)
    5. [Optional] Reranking (if mode = "hybrid_rerank")
    6. Dynamic threshold filtering
    """
    mode_display = "HYBRID + RERANK" if mode == "hybrid_rerank" else "HYBRID ONLY (No Rerank)"
    print(f"\n{'='*70}\nITERASI KE-{index} [{mode_display}]: {query_text[:50]}...\n{'='*70}")

    # --- TAHAP 1-3: HIERARCHICAL FILTER ---
    proj_names = get_as_list(entity.get("project_name"))
    act_names = get_as_list(entity.get("activity_name"))
    doc_names = get_as_list(entity.get("deliverable"))

    proj_names_lower = [str(p).lower() for p in proj_names]
    act_names_lower = [str(a).lower() for a in act_names]
    doc_names_lower = [str(d).lower() for d in doc_names]

    final_must_filters = []

    # Get IDs from hierarchy
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

    # Build chunk filters
    chunk_filters = []
    actual_proj_names = get_chunk_values_for_filter(client, "chunk_coll_bm25", "parent_project_name", proj_names)
    actual_act_names = get_chunk_values_for_filter(client, "chunk_coll_bm25", "parent_activity_name", act_names)
    actual_doc_names = get_chunk_values_for_filter(client, "chunk_coll_bm25", "parent_doc_type_name", doc_names)

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

    chunk_query_filter = None
    if chunk_filters:
        chunk_query_filter = models.Filter(must=chunk_filters)

    # --- TAHAP 4: HYBRID SEARCH (Dense + Sparse) ---
    print(f"[HYBRID] Generating dense + sparse vectors for query...")
    
    # 4a. Generate DENSE vector (semantic)
    formatted_query = get_detailed_instruct(query_text)
    query_dense = embed_client.embed_query(formatted_query)
    
    # 4b. Generate SPARSE vector (BM25)
    query_sparse = generate_sparse_vector(query_text)
    
    print(f"  - Dense vector dim: {len(query_dense)}")
    print(f"  - Sparse vector non-zero elements: {len(query_sparse['indices'])}")

    # 4c. Execute HYBRID SEARCH with Reciprocal Rank Fusion (RRF)
    try:
        search_results = client.query_points(
            collection_name="chunk_coll_bm25",
            prefetch=[
                # Prefetch from DENSE vector
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=100,
                    filter=chunk_query_filter
                ),
                # Prefetch from SPARSE vector (BM25)
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=100,
                    filter=chunk_query_filter
                )
            ],
            # Fuse results using Reciprocal Rank Fusion
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=50,
            with_payload=True,
        )
        print(f"[HYBRID] Retrieved {len(search_results.points)} results from hybrid search")
    
    except Exception as e:
        print(f"[ERROR] Hybrid search failed: {e}")
        print(f"[FALLBACK] Retrying hybrid search without filters (simplified query)...")

        # Fallback: Retry hybrid search without filters and smaller limits
        try:
            search_results = client.query_points(
                collection_name="chunk_coll_bm25",
                prefetch=[
                    models.Prefetch(query=query_dense, using="dense", limit=30),
                    models.Prefetch(query=query_sparse, using="sparse", limit=30)
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=20,
                with_payload=True,
            )
            print(f"[FALLBACK OK] Retrieved {len(search_results.points)} results from simplified hybrid search")
        except Exception as e2:
            print(f"[FALLBACK FAILED] {str(e2)[:100]}")
            search_results = None

    if not search_results or not search_results.points:
        print(f"[WARNING] No results with filters. Attempting search without filters...")
        search_results = client.query_points(
            collection_name="chunk_coll_bm25",
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=100),
                models.Prefetch(query=query_sparse, using="sparse", limit=100)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=5,
            with_payload=True,
        )
        
        if search_results.points:
            print(f"[INFO] Sample payloads from chunk_coll_bm25 (without filters):")
            for hit in search_results.points[:1]:
                print(f"  - parent_project_name: {hit.payload.get('parent_project_name')}")
                print(f"  - parent_activity_name: {hit.payload.get('parent_activity_name')}")
                print(f"  - parent_doc_type_name: {hit.payload.get('parent_doc_type_name')}")
        return []

    # --- TAHAP 5: PREPARE RESULTS ---
    docs_to_rerank = []
    valid_points = []
    for hit in search_results.points:
        text_content = hit.payload.get("page_content") or hit.payload.get("text")
        if text_content:
            docs_to_rerank.append(text_content)
            valid_points.append(hit)

    all_weighted_results = []

    # --- MODE BRANCHING: RERANK vs NO RERANK ---
    if mode == "hybrid_rerank":
        # --- TAHAP 6a: RERANKING (Hybrid + Rerank Mode) ---
        print(f"[RERANK] Sending {len(docs_to_rerank)} documents to reranker...")
        rerank_resp = call_reranker(query_text, docs_to_rerank, top_n=len(docs_to_rerank), max_retries=3)
        
        if rerank_resp and "results" in rerank_resp:
            print(f"[RERANK] Successfully reranked {len(docs_to_rerank)} documents")
            
            for item in rerank_resp["results"]:
                idx = item["index"]
                rerank_score = item["relevance_score"]
                original_hit = valid_points[idx]
                hybrid_score = original_hit.score

                # Weighted Scoring: Hybrid + Reranker
                final_weighted_score = (0.7 * hybrid_score) + (0.3 * rerank_score)

                pay = original_hit.payload
                all_weighted_results.append({
                    "metadata": {
                        "chunk_id": str(original_hit.id),
                        "filename": pay.get("source_file"),
                    },
                    "content": docs_to_rerank[idx],
                    "score_final": round(final_weighted_score, 4),
                    "score_hybrid": round(hybrid_score, 4),
                    "score_rerank": round(rerank_score, 4)
                })
        else:
            print(f"[WARNING] Reranking failed, using hybrid scores only")
            # Fallback: use hybrid scores only
            for hit in valid_points:
                text_content = hit.payload.get("page_content") or hit.payload.get("text")
                all_weighted_results.append({
                    "metadata": {
                        "chunk_id": str(hit.id),
                        "filename": hit.payload.get("source_file"),
                    },
                    "content": text_content,
                    "score_final": round(hit.score, 4),
                    "score_hybrid": round(hit.score, 4),
                    "score_rerank": None
                })
    
    else:  # mode == "hybrid_only"
        # --- TAHAP 6b: NO RERANKING (Hybrid Only Mode) ---
        print(f"[NO RERANK] Using hybrid RRF scores directly (mode: hybrid_only)")
        
        for hit in valid_points:
            text_content = hit.payload.get("page_content") or hit.payload.get("text")
            hybrid_score = hit.score
            
            all_weighted_results.append({
                "metadata": {
                    "chunk_id": str(hit.id),
                    "filename": hit.payload.get("source_file"),
                },
                "content": text_content,
                "score_final": round(hybrid_score, 4),
                "score_hybrid": round(hybrid_score, 4),
                "score_rerank": None  # No rerank in this mode
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function for batch processing queries with HYBRID SEARCH"""
    print("=" * 70)
    print("HYBRID SEARCH RETRIEVAL")
    print("=" * 70)
    print(f"\n📌 RETRIEVAL MODE: {RETRIEVAL_MODE.upper()}")
    
    if RETRIEVAL_MODE == "hybrid_only":
        print("   → Dense + Sparse (BM25) + RRF")
        print("   → NO Reranker")
    elif RETRIEVAL_MODE == "hybrid_rerank":
        print("   → Dense + Sparse (BM25) + RRF + Reranker")
    else:
        print(f"   ⚠️  Unknown mode '{RETRIEVAL_MODE}', defaulting to 'hybrid_rerank'")
    
    print("=" * 70)

    # Load BM25 model
    try:
        load_bm25_model()
    except FileNotFoundError:
        print("\n[ERROR] Please run the ingest script first to create BM25 model!")
        return

    print("\nInitializing embedding client...")
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
        return

    print(f"\nConnecting to Qdrant at {QDRANT_URL}...")
    try:
        q_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            verify=False,
            timeout=120.0  # Increase timeout to 120 seconds for complex queries
        )
        print("[OK] Connected to Qdrant successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")
        return

    try:
        if not BATCH_RESULTS_FINAL_INPUT.exists():
            print(f"[WARNING] Input file not found: {BATCH_RESULTS_FINAL_INPUT}")
            BATCH_RESULTS_FINAL_INPUT.parent.mkdir(parents=True, exist_ok=True)
            return

        with open(BATCH_RESULTS_FINAL_INPUT, 'r') as f:
            data = json.load(f)

        output_data = {}
        queries_to_process = []

        # Detect input format
        if "results" in data and isinstance(data["results"], dict):
            print("\nDetecting format: batch_results_final.json (New Format)")
            for doc_id, doc_content in data["results"].items():
                questions = doc_content.get("questions", {})
                for test_type, test_data in questions.items():
                    query_text = test_data.get("question")
                    ner_raw = test_data.get("ner_output", {})

                    entity = {
                        "project_name": ner_raw.get("project_name"),
                        "activity_name": ner_raw.get("activity_name"),
                        "deliverable": ner_raw.get("doc_type_name")
                    }
                    queries_to_process.append((query_text, entity))
        else:
            print("\nDetecting format: ner_batch_results.json (Old Format)")
            for query_text, val in data.items():
                queries_to_process.append((query_text, val.get("entity", {})))

        # Process queries
        print(f"\n{'='*70}")
        print(f"Processing {len(queries_to_process)} queries")
        print(f"{'='*70}")
        
        for i, (query_text, entity) in enumerate(queries_to_process, 1):
            res = process_query(q_client, embed_client, query_text, entity, i, mode=RETRIEVAL_MODE)
            output_data[query_text] = res

            # Rate limiting
            if i < len(queries_to_process):
                time.sleep(0.5)

        # Save results
        RETRIEVAL_RESULTS_RERANKED.parent.mkdir(parents=True, exist_ok=True)
        with open(RETRIEVAL_RESULTS_RERANKED, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✅ DONE! {len(output_data)} queries processed")
        print(f"   Mode: {RETRIEVAL_MODE.upper()}")
        print(f"{'='*70}")
        print(f"Results saved to: {RETRIEVAL_RESULTS_RERANKED}")

    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()