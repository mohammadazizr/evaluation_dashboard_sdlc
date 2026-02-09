"""
Central configuration for paths and settings.
Supports both local development and production environments.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- BASE DIRECTORIES ---
# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Define key directories
ROUTER_EXTRACTION_DIR = PROJECT_ROOT / "router_extraction"
RETRIEVE_PROCESS_DIR = PROJECT_ROOT / "retrieve_process"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# --- OUTPUT DIRECTORIES (automatically created if don't exist) ---
ROUTER_OUTPUT_DIR = ROUTER_EXTRACTION_DIR / "output"
ROUTER_EXPORTS_DIR = ROUTER_EXTRACTION_DIR / "exports"
RETRIEVE_OUTPUT_DIR = RETRIEVE_PROCESS_DIR / "output"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output"

# Create output directories if they don't exist
for dir_path in [ROUTER_OUTPUT_DIR, ROUTER_EXPORTS_DIR, RETRIEVE_OUTPUT_DIR, EVALUATION_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# --- INPUT/OUTPUT FILE PATHS ---

# Router Extraction Paths
UNIQUE_LEVEL_NAMES_PATH = ROUTER_OUTPUT_DIR / "unique_level_names.json"
LEVEL_INIT_FILTERS_PATH = ROUTER_OUTPUT_DIR / "level_init_filters.json"
COMBINED_OUTPUT_PATH = ROUTER_OUTPUT_DIR / "combined_output.json"

PROJ_COLL_EXPORT = ROUTER_EXPORTS_DIR / "proj_coll.json"
ACTIVITY_COLL_EXPORT = ROUTER_EXPORTS_DIR / "activity_coll.json"
DOC_TYPE_COLL_EXPORT = ROUTER_EXPORTS_DIR / "doc_type_coll.json"

PROJ_ENTITY_OUTPUT = ROUTER_OUTPUT_DIR / "proj_entity_output.json"
ACT_ENTITY_OUTPUT = ROUTER_OUTPUT_DIR / "act_entity_output.json"
DOC_ENTITY_OUTPUT = ROUTER_OUTPUT_DIR / "doc_entity_output.json"

# Keyword Extraction Paths
STOPWORDS_ID_PATH = ROUTER_EXTRACTION_DIR / "utils" / "keyword_extraction" / "stopwords-id.json"
STOPWORDS_EN_PATH = ROUTER_EXTRACTION_DIR / "utils" / "keyword_extraction" / "stopwords-en.json"
EXTRACTED_KEYWORDS_PATH = ROUTER_OUTPUT_DIR / "extracted_keywords.json"
LIST_ALL_COLLECTIONS_PATH = ROUTER_OUTPUT_DIR / "list_all_collections_new.json"

# Extraction Logging Paths
KEYWORD_EXTRACTION_LOG_PATH = ROUTER_OUTPUT_DIR / "keyword_extraction_log.json"
NER_EXTRACTION_LOG_PATH = ROUTER_OUTPUT_DIR / "ner_extraction_log.json"

# Retrieval Paths
BATCH_RESULTS_FINAL_INPUT = ROUTER_OUTPUT_DIR / "batch_results_final.json"
RETRIEVAL_RESULTS_RERANKED = RETRIEVE_OUTPUT_DIR / "retrieval_results_reranked.json"

# Evaluation Paths
GROUND_TRUTH_PATH = PROJECT_ROOT / "ground_truth_060126.json"
# GROUND_TRUTH_PATH = PROJECT_ROOT / "ground_truth_edbert.json"
EVALUATION_RESULTS_PATH = EVALUATION_OUTPUT_DIR / "evaluation_results_new.json"
DASHBOARD_OUTPUT_PATH = EVALUATION_OUTPUT_DIR / "dashboard.html"

# --- ENVIRONMENT VARIABLES ---
# LiteLLM Configuration
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "")
LITELLM_MODEL_NAME = os.getenv("LITELLM_MODEL_NAME", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Rerank Configuration
RERANK_MODEL = os.getenv("RERANK_MODEL", "cohere.rerank-v3-5:0")

# --- LOGGING ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- DEBUG MODE ---
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def print_config_summary():
    """Print configuration summary for debugging."""
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Router Output Dir: {ROUTER_OUTPUT_DIR}")
    print(f"Retrieve Output Dir: {RETRIEVE_OUTPUT_DIR}")
    print(f"LiteLLM URL: {LITELLM_URL}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_config_summary()
