# question :  Document ORP workflow dengan judul Integrasi Pembukuan Fee Interbranch - Recurring
# collections path : output_qdrant_collections/list_all_collections.json
# 

import json
import re
# import nltk
# from nltk.corpus import stopwords
import truststore
import os

truststore.inject_into_ssl()

def unset_proxy():
    proxies = ['HTTPS_PROXY', 'HTTP_PROXY', 'http_proxy', 'https_proxy']
    for proxy in proxies:
        os.environ.pop(proxy, None)
    print("Proxies unset!")
    
unset_proxy()


# Ensure English stopwords are downloaded
# nltk.download('stopwords')




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

def clean_query(query, stopword_set):
    """
    Cleans a user query using a pre-loaded combined set of stopwords.
    """
    # 1. Lowercase and strip punctuation/numbers
    query = query.lower()
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    
    # 2. Tokenize and filter against the combined set
    words = query.split()
    cleaned_words = [w for w in words if w not in stopword_set]
    
    return " ".join(cleaned_words)

def analyze_keywords(user_input):
    
    # --- Configuration ---
    ID_STOPS_PATH = "/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/utils/keyword_extraction/stopwords-id.json"
    EN_STOPS_PATH = "/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/utils/keyword_extraction/stopwords-en.json"

    # 1. Load both sets
    id_stops = load_stopwords_from_json(ID_STOPS_PATH)
    en_stops = load_stopwords_from_json(EN_STOPS_PATH)

    # 2. Combine them once (more efficient for multiple queries)
    combined_stopwords = id_stops.union(en_stops)

    # --- Execution ---
    keywords = clean_query(user_input, combined_stopwords).split(" ")
    
    
    
    
    
    
    with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/list_all_collections.json', 'r') as f:
    # Parsing the JSON file into a Python dictionary
        data = json.load(f)
    doc_collection = data["collections"][1]["sample_points"]
    kw_count = {}
    
    for kw in keywords :
        if kw not in kw_count.keys() :
            kw_count[kw]={"path":{},"total_count":0}
        for i,item in enumerate(doc_collection):
            if kw in [str(word).lower() for word in item["payload"]["keywords"]] :
                doc_type_name = item["payload"]["doc_type_name"]
                parent_project_name = item["payload"]["parent_project_name"]
                parent_activity_name = item["payload"]["parent_activity_name"]
                doc_path = parent_project_name+"\\"+parent_activity_name+"\\"+doc_type_name
                
                if doc_path not in kw_count[kw]["path"].keys():
                    kw_count[kw]["path"][doc_path]=1
                    kw_count[kw]["total_count"]+=1
                else :
                    kw_count[kw]["path"][doc_path]+=1
                    kw_count[kw]["total_count"]+=1
    
    # Use dictionary comprehension to filter
    cleaned_results = {
        word: data 
        for word, data in kw_count.items() 
        if data.get("total_count", 0) > 0
    }
        
    with open("/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/extracted_keywords.json", mode="w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, indent=2)
                
                # kw_count[kw]+=1
    
    
    
    return cleaned_results




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
    

    print(f"Original: {user_input}")
    # print(f"Cleaned:  {cleaned_result}")
    print(analyze_keywords)
    
    # question = "Document ORP workflow dengan judul Integrasi Pembukuan Fee Interbranch - Recurring"
