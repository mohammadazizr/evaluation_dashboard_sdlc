# question :  Document ORP workflow dengan judul Integrasi Pembukuan Fee Interbranch - Recurring
# collections path : output_qdrant_collections/list_all_collections.json
# 

import json
import re
# import nltk
# from nltk.corpus import stopwords
import truststore
import os
from nltk.stem import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

truststore.inject_into_ssl()

def unset_proxy():
    proxies = ['HTTPS_PROXY', 'HTTP_PROXY', 'http_proxy', 'https_proxy']
    for proxy in proxies:
        os.environ.pop(proxy, None)
    print("Proxies unset!")
    
unset_proxy()


# Ensure English stopwords are downloaded
# nltk.download('stopwords')

# --- Inisialisasi Resource ---
# Indonesia
indo_factory = StemmerFactory()
indo_stemmer = indo_factory.create_stemmer()

# Inggris
english_stemmer = SnowballStemmer("english")

def load_stopwords_from_json(file_path):
    """Safely loads stopwords from a JSON file (expects list or dict)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle if JSON is a list or a dictionary with a 'stopwords' key
            if isinstance(data, list):
                return set(data)
            return set(data.get('stopwords', []))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return set()

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import SnowballStemmer

# Inisialisasi (pastikan ini ada di luar fungsi)
indo_stemmer = StemmerFactory().create_stemmer()
english_stemmer = SnowballStemmer("english")

def clean_query(query, stopword_set):
    """
    Cleans a user query and expands acronyms with their full forms.

    Acronym expansions:
    - WMS/WFMS → workflow management system
    - ORP → operation request portal
    - LM → livin merchant
    - BOC → business operation center
    """
    # 1. Lowercase
    query = query.lower()

    # 2. Deteksi akronim untuk ekspansi di akhir
    has_wfms = bool(re.search(r'\b(wms|wfms)\b', query))
    has_orp = bool(re.search(r'\borp\b', query))
    has_lm = bool(re.search(r'\blm\b', query))
    has_boc = bool(re.search(r'\bboc\b', query))

    # 3. Hapus karakter non-alphabetic
    query = re.sub(r'[^a-zA-Z\s]', '', query)

    # 4. Tokenize dan Filter Stopwords
    words = query.split()
    filtered_words = [w for w in words if w not in stopword_set]

    # 5. Deduplikasi (preserve order)
    final_list = list(dict.fromkeys(filtered_words))

    # 6. Tambahkan ekspansi akronim di akhir (full forms)
    # if has_wfms:
    #     final_list.extend(["workflow", "management", "system"])
    # if has_orp:
    #     final_list.extend(["operation", "request", "portal"])
    # if has_lm:
    #     final_list.extend(["livin", "merchant"])
    # if has_boc:
    #     final_list.extend(["business", "operation", "center"])

    return " ".join(final_list)

def process_file_path(file_path):
    """
    Process file_path by:
    1. Removing 'ocr_result/' and the first folder after it
    2. Applying text replacements for known patterns
    3. Removing specific folders (Agile/, Waterfall/, none/)
    4. Keeping the filename

    Handles both forward slash (/) and backward slash (\) path separators.

    Examples:
    Input: "ocr_result/Operation Request Portal/Workflow Management System/Workflow For Business Operation Center (BOC)/TSD/01/output.md"
    Output: "WMS/BOC/TSD/01/output.md"

    Input: "D:\\Simplify\\rag-sdlc\\output\\ocr_result\\Operation Request Portal\\Workflow Management System\\Workflow For Business Operation Center (BOC)\\TSD\\01\\output.md"
    Output: "WMS/BOC/TSD/01/output.md"
    """
    # Normalize path: convert backslash to forward slash for consistent processing
    normalized_path = file_path.replace("\\", "/")

    # Find and extract everything after "ocr_result/"
    if "ocr_result/" in normalized_path:
        normalized_path = normalized_path.split("ocr_result/", 1)[1]

    # Split into parts
    parts = normalized_path.split("/")

    # Remove the first folder (which can be any name like "Operation Request Portal")
    if len(parts) > 1:
        parts = parts[1:]

    # Join back
    processed_path = "/".join(parts)

    # Apply text replacements
    replacements = {
        "Operation Request Portal 2025": "ORP",
        "ORP Portal": "ORP_Portal",
        "ORP Workflow": "ORP_Workflow",
        "Workflow Management System": "WMS",
        "Workflow For Business Operation Center (BOC)": "BOC",
        "Workflow For Business Operation Center": "BOC",
        "Workflow For Head Office": "Head Office",
    }

    for old_text, new_text in replacements.items():
        processed_path = processed_path.replace(old_text, new_text)

    # Remove specific folder names if they exist in the path
    folders_to_remove = ["Agile", "Waterfall", "none"]
    for folder in folders_to_remove:
        processed_path = processed_path.replace(f"{folder}/", "")

    # Clean up any double slashes that might have been created
    while "//" in processed_path:
        processed_path = processed_path.replace("//", "/")

    # Remove leading/trailing slashes
    processed_path = processed_path.strip("/")

    return processed_path


def get_qdrant_client(qdrant_url, qdrant_api_key=None):
    """Initialize and return Qdrant client."""
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        raise


def query_keywords_collection(client, collection_name="keywords_coll"):
    """
    Query all points from keywords_coll collection.
    Returns list of points with their payload.
    """
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust if needed
        )
        return points
    except Exception as e:
        print(f"Error querying collection '{collection_name}': {e}")
        raise


def analyze_keywords(user_input, logger=None):
    import sys
    import time
    from pathlib import Path

    # Import config from project root
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from config import (STOPWORDS_ID_PATH, STOPWORDS_EN_PATH, EXTRACTED_KEYWORDS_PATH,
                        QDRANT_URL, QDRANT_API_KEY)
    # LIST_ALL_COLLECTIONS_PATH is now commented out - using Qdrant keywords_coll instead
    # from config import LIST_ALL_COLLECTIONS_PATH

    start_time = time.time()

    try:
        # 1. Load both sets
        id_stops = load_stopwords_from_json(str(STOPWORDS_ID_PATH))
        en_stops = load_stopwords_from_json(str(STOPWORDS_EN_PATH))

        # 2. Combine them once (more efficient for multiple queries)
        combined_stopwords = id_stops.union(en_stops)

        # --- Execution ---
        cleaned_query = clean_query(user_input, combined_stopwords)
        print("="*50)
        print(f"Original query: {user_input}")
        print(f"Cleaned query: {cleaned_query}")
        keywords = cleaned_query.split(" ")
        print("="*50)

        # --- Using Qdrant keywords_coll (NEW) ---
        client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
        points = query_keywords_collection(client, collection_name="keywords_coll")

        if not points:
            raise ValueError("No points found in Qdrant collection 'keywords_coll'")

        # --- OLD CODE (commented out) - Previously using LIST_ALL_COLLECTIONS_PATH ---
        # with open(str(LIST_ALL_COLLECTIONS_PATH), 'r', encoding='utf-8') as f:
        #     # Parsing the JSON file into a Python dictionary
        #     data = json.load(f)
        #
        # # Filter collection dengan collection_name = "doc_type_coll"
        # doc_collection = None
        # for collection in data["collections"]:
        #     if collection.get("collection_name") == "doc_type_coll":
        #         doc_collection = collection.get("sample_points", [])
        #         break
        #
        # if doc_collection is None:
        #     raise ValueError("Collection 'doc_type_coll' not found in list_all_collections_new.json")

        kw_count = {}

        for kw in keywords:
            if kw not in kw_count.keys():
                kw_count[kw] = {"path": {}, "total_count": 0}

            # Process each point from keywords_coll
            for point in points:
                payload = point.payload
                point_keywords = payload.get("keyword", [])

                # Check if keyword matches any in the point
                if kw in [str(word).lower() for word in point_keywords]:
                    file_path = payload.get("file_path", "")

                    # Process file path
                    processed_path = process_file_path(file_path)

                    if processed_path not in kw_count[kw]["path"].keys():
                        kw_count[kw]["path"][processed_path] = 1
                        kw_count[kw]["total_count"] += 1
                    else:
                        kw_count[kw]["path"][processed_path] += 1
                        kw_count[kw]["total_count"] += 1

        # Use dictionary comprehension to filter
        cleaned_results = {
            word: data
            for word, data in kw_count.items()
            if data.get("total_count", 0) > 0
        }

        with open(str(EXTRACTED_KEYWORDS_PATH), mode="w", encoding="utf-8") as f:
            json.dump(cleaned_results, f, indent=2)

        # Log the extraction
        execution_time_ms = (time.time() - start_time) * 1000
        if logger:
            logger.log_keyword_extraction(
                question=user_input,
                cleaned_query=cleaned_query,
                keywords=cleaned_results,
                stopwords_count=len(combined_stopwords),
                execution_time_ms=execution_time_ms,
                status="success"
            )

        return cleaned_results

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        if logger:
            logger.log_keyword_extraction(
                question=user_input,
                cleaned_query="",
                keywords={},
                stopwords_count=0,
                execution_time_ms=execution_time_ms,
                status="error",
                error=str(e)
            )
        raise

if __name__ == "__main__":

    

#     extractor = EntityExtractor(
#         model_name=TEXT_MODEL_NAME,
#         api_key=API_KEY_ENV,
#         base_url=BASE_URL_ENV,
#     )
    # --- Usage ---
    # Path to your specific JSON file
    # json_path = 'embed_push_qdrant/keyword_extraction/stopwords-id.json'
    
    user_input = "Document ORP workflow dengan judul Integrasi Pembukuan Fee Interbranch - Recurring"
    analyze_keywords = analyze_keywords(user_input)
    
    # print(f"Cleaned:  {cleaned_result}")
    print(analyze_keywords)
    
    # question = "Document ORP workflow dengan judul Integrasi Pembukuan Fee Interbranch - Recurring"
