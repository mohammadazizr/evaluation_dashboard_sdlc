import os
import json
import logging
import truststore
from openai import OpenAI
from dotenv import load_dotenv
import httpx

from .schemas.doc_entity_schema import DocEntityResponse
from pydantic import ValidationError

http_client = httpx.Client(verify=False)

# Opening and reading the JSON file
# with open('embed_push_qdrant/ner.json', 'r') as f:
#     # Parsing the JSON file into a Python dictionary
#     ner = json.load(f)

with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/exports/doc_type_coll.json', 'r') as f:
    # Parsing the JSON file into a Python dictionary
    data_doc_type = json.load(f)

with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/unique_level_names.json', 'r') as f:
    unique_level_names = json.load(f)
    



# Catatan: bila dilakukan di jupyterlab,
# pip install gabisa dilakukan setelah pop variable

def unset_proxy():
    proxies = ['HTTPS_PROXY', 'HTTP_PROXY', 'http_proxy', 'https_proxy']
    for proxy in proxies:
        os.environ.pop(proxy, None)
    print("Proxies unset!")

# =============================
# Environment
# =============================
load_dotenv()
truststore.inject_into_ssl()
unset_proxy()

API_KEY_ENV = os.getenv("API_KEY_ENV")
BASE_URL_ENV = os.getenv("BASE_URL_ENV")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME")

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# System Prompt
# =============================



# =============================
# Entity Extractor Class
# =============================
class EntityExtractor:
    def __init__(self, model_name: str = TEXT_MODEL_NAME, api_key: str = API_KEY_ENV, base_url: str = BASE_URL_ENV):
        if not api_key:
            raise ValueError("API key tidak tersedia")
        if not base_url:
            raise ValueError("Base URL tidak tersedia")

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        logger.info(f"Entity extractor initialized with model: {model_name}")

    def extract(self, user_question: str, proj_context: dict, act_context: dict, level_init_filters: dict = unique_level_names["doc_type_names"]) -> dict:
        
        project_parent_id = proj_context["entity"]["project_id"]
        project_parent_name = proj_context["entity"]["project_name"]
        
        activity_parent_id = act_context["entity"]["activity_id"]
        activity_parent_name = act_context["entity"]["activity_name"]
        print("activity_parent_name:",activity_parent_name)
        print("activity_parent_name[0]:", activity_parent_name[0])
        print("level_init_filters['activity_names']:", level_init_filters)
        
        # activity_name
        
        
        
        doc_type_init_filters = level_init_filters["activity_names"][activity_parent_name[0]] if activity_parent_name[0] in list(level_init_filters["activity_names"].keys()) else [item["payload"]["doc_type_name"] for item in data_doc_type if item["payload"]["parent_activity_name"]==activity_parent_name[0]]
        
        activity_context_peritem = []
        doc_type_names = ["null"]
        for i,item in enumerate(data_doc_type):
            doc_type_name = item["payload"]["doc_type_name"]
            doc_id = item["id"]
            summary = item["payload"]["summary"]
            
            parent_project_name = item["payload"]["parent_project_name"]
            parent_activity_name = item["payload"]["parent_activity_name"]
            
            
            
            childern_uuids = item["payload"]["children_chunk_uuids"]
            cont = f"""{i+1}. doc_type_name: {doc_type_name}\n
            doc_id: {doc_id}\n
            summary: {summary}\n\n"""
            # if (parent_project_name == project_parent_name) and (parent_activity_name == activity_parent_name):
            #     activity_context_peritem.append(cont)
            #     doc_type_names.append(doc_type_name)
                
            if activity_parent_name!=["null"]:
                if parent_activity_name in activity_parent_name and doc_type_name in doc_type_init_filters:
                    activity_context_peritem.append(cont)
                    doc_type_names.append(doc_type_name)
            else : 
                activity_context_peritem.append(cont)
                doc_type_names.append(doc_type_name)
                
        doc_context = "".join(activity_context_peritem)
        print("doc_type_names", doc_type_names)
        
#         with open('embed_push_qdrant/ner_4_steps/unique_level_names.json', 'r') as f:
#             unique_level_names = json.load(f)

#         doc_type_names = unique_level_names["doc_type_names"]+["null"]
        
        
        print("doc_context:",doc_context)
        print("doc_type_names:",doc_type_names)
        
        DOC_ENTITY_SYSTEM_PROMPT = f"""
you are an entity recognition and classification engine for Activities.
The user query is belong to this project : {project_parent_name}
and also belong to this activity : {activity_parent_name}
here is the context for you to classify the user query whether it belongs to which doc_type

{doc_context}

NOTE: YOU MUST ALWAYS CHOOSE AN ACTIVITY FROM THE GIVEN CONTEXT ABOVE!!! LET ME REMIND YOU THE AVAILABLE DOC_TYPE_NAMES BELOW 

{doc_type_names}

NOTE THAT : YOU CAN NAME MULTIPLE DOC_TYPE_NAME IF YOU MUST
ALSO NOTE THAT : IF YOU'RE REALLY UNSURE, YOU CAN FILL IT WITH 'null'

SPECIAL USECASE : 
- if the query mention 'Specification Document' it should be classified as 'TSD'
- if the query mention 'Requirement Document' it should be classified as 'BRD'
- if the query mention 'Functional Specification Document' it should be classified as 'FSD'

Output Format (MANDATORY)

Return only valid JSON.
No explanations.
No markdown.
No extra keys.
All value is string wrapped as list data type.

{{
  "entity": {{
    "doc_type_name": ["<DOC_TYPE_NAME_CHOSEN_FROM_THE_CONTEXT>", "<DOC_TYPE_NAME_CHOSEN_FROM_THE_CONTEXT>"],
    "doc_id": ["<DOC_ID_PROVIDED_BY_SYSTEM>","<DOC_ID_PROVIDED_BY_SYSTEM>"],
    "childern_chunk_uuids": ["<UUID_1>", "<UUID_2>"]
  }}
}}
"""
        print("after system prompt doc")
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             temperature=0.0,
#             messages=[
#                 {"role": "system", "content": DOC_ENTITY_SYSTEM_PROMPT},
#                 {"role": "user", "content": user_question},
#             ],
#         )

#         raw_output = response.choices[0].message.content.strip()

        try:
            # 3. Use Structured Outputs (.parse)
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": DOC_ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_question},
                ],
                # This ensures the LLM follows DocEntityResponse exactly
                response_format=DocEntityResponse, 
            )

            validated = response.choices[0].message.parsed
            print("response doc is done")
            
            if validated is None:
                raise ValueError("LLM refused to provide a structured response.")

            # 4. Save and return using model_dump()
            # This handles Enum-to-string conversion automatically
            output_data = validated.model_dump()
            
            with open("/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/doc_entity_output.json", mode="w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            
            return output_data

        except ValidationError as e:
            print("validation error doc")
            raise ValueError(f"Schema validation failed: {e}")



# =============================
# Example Usage
# =============================
if __name__ == "__main__":

    unset_proxy()

    extractor = EntityExtractor(
        model_name=TEXT_MODEL_NAME,
        api_key=API_KEY_ENV,
        base_url=BASE_URL_ENV,
    )

    question = (
        "Document ORP workflow dengan judul Integrasi Pembukuan Fee Interbranch - Recurring"
    )
    
    with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/proj_entity_output.json', 'r') as f:
        # Parsing the JSON file into a Python dictionary
        data_proj = json.load(f)
    
    with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/act_entity_output.json', 'r') as f:
        # Parsing the JSON file into a Python dictionary
        data_act = json.load(f)
    
    with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/level_init_filters.json', 'r') as f:
        # Parsing the JSON file into a Python dictionary
        level_init_filters = json.load(f)

    result = extractor.extract(question, data_proj, data_act, level_init_filters)

    print(json.dumps(result, indent=2))
    with open("/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/doc_entity_output.json", mode="w", encoding="utf-8") as write_file:
        json.dump(result, write_file)