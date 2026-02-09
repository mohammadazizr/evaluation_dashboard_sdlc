import os
import json
import logging
import truststore
from openai import OpenAI
from dotenv import load_dotenv
import httpx

from .schemas.activity_entity_schema import ActivityEntityResponse
from pydantic import ValidationError

http_client = httpx.Client(verify=False)

# Opening and reading the JSON file
# with open('embed_push_qdrant/ner.json', 'r') as f:
#     # Parsing the JSON file into a Python dictionary
#     ner = json.load(f)

with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/exports/activity_coll.json', 'r') as f:
    # Parsing the JSON file into a Python dictionary
    data_activity = json.load(f)

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

    def extract(self, user_question: str, proj_context: dict, level_init_filters: dict = unique_level_names["activity_names"]) -> dict:
        
        project_parent_id = proj_context["entity"]["project_id"]
        project_parent_name = proj_context["entity"]["project_name"]
        print("project_parent_name:",project_parent_name)
        
        activity_context_peritem = []
        activity_names = ["null"]
        print("project_parent_name[0]",project_parent_name[0])
        activity_init_filters = level_init_filters["project_names"][project_parent_name[0]] if project_parent_name[0]!="null" else unique_level_names["activity_names"]
        
        for i,item in enumerate(data_activity):
            activity_name = item["payload"]["activity_name"]
            activity_id = item["id"]
            summary = item["payload"]["summary"]
            parent_project_name = item["payload"]["parent_project_name"]
            childern_uuids = item["payload"]["children_doc_type_uuids"]
            cont = f"""{i+1}. activity_name: {activity_name}\n
            activity_id: {activity_id}\n
            childern_uuids: {childern_uuids}\n
            summary: {summary}\n\n"""
            if project_parent_name!=["null"]:
                if parent_project_name in project_parent_name and activity_name in activity_init_filters:
                    activity_context_peritem.append(cont)
                    activity_names.append(activity_name)
            else : 
                activity_context_peritem.append(cont)
                activity_names.append(activity_name)
                
        act_context = "".join(activity_context_peritem)
        print("activity_names:",activity_names)
        
#         with open('embed_push_qdrant/ner_4_steps/unique_level_names.json', 'r') as f:
#             unique_level_names = json.load(f)

#         activity_names = unique_level_names["activity_names"]+["null"]
        
        ACTIVITY_ENTITY_SYSTEM_PROMPT = f"""
you are an entity recognition and classification engine for Activities.
The user query is belong to this project : {project_parent_name}
here is the context for you to classify the user query whether it belongs to which Activity (especialy note for the activity_name and activity_id)
NOTE: activity_name and activity_id IS DIFFERENT!!!

{act_context}

NOTE: YOU MUST ALWAYS CHOOSE AN ACTIVITY FROM THE GIVEN CONTEXT ABOVE!!!  LET ME REMIND YOU THE AVAILABLE ACTIVITY_NAMES BELOW 

{activity_names}

NOTE THAT : YOU CAN NAME MULTIPLE ACTIVITY IF YOU MUST
ALSO NOTE THAT : IF YOU'RE REALLY UNSURE, YOU CAN FILL IT WITH 'null'

Output Format (MANDATORY)

Return only valid JSON.
No explanations.
No markdown.
No extra keys.
All value is string wrapped as list data type.

{{
  "entity": {{
    "activity_name": ["<ACTIVITY_NAME_CHOSEN_FROM_THE_CONTEXT>", "<ACTIVITY_NAME_CHOSEN_FROM_THE_CONTEXT>"],
    "activity_id": ["832b221a-3918-4ba7-86d0-306390964069", "<ACTIVITY_ID_PROVIDED_BY_SYSTEM>"],
    "childern_doc_type_uuids": ["<UUID_1>", "<UUID_2>"]
  }}
}}
"""
        
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             temperature=0.0,
#             messages=[
#                 {"role": "system", "content": ACTIVITY_ENTITY_SYSTEM_PROMPT},
#                 {"role": "user", "content": user_question},
#             ],
#         )

#         raw_output = response.choices[0].message.content.strip()

        try:
            # 3. Use Structured Outputs
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": ACTIVITY_ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_question},
                ],
                response_format=ActivityEntityResponse, # Enforces your Pydantic schema
            )

            validated = response.choices[0].message.parsed
            
            if validated is None:
                raise ValueError("LLM refused to provide a structured response.")

            # 4. Save and return
            output_path = "/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/act_entity_output.json"
            with open(output_path, mode="w", encoding="utf-8") as f:
                json.dump(validated.model_dump(), f, indent=2)
            
            return validated.model_dump()

        except ValidationError as e:
            raise ValueError(f"Output failed schema validation: {e}")



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
        "What is the activity code and author of the WFMS Workflow for Head Office Requirement Document v.0.1 from 16 Maret 2023?"
    )
    
    with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/proj_entity_output.json', 'r') as f:
        # Parsing the JSON file into a Python dictionary
        data_proj = json.load(f)

    result = extractor.extract(question, data_proj)

    print(json.dumps(result, indent=2))
    