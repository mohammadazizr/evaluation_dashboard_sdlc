import os
import json
import logging
import truststore
from openai import OpenAI
from dotenv import load_dotenv
import httpx

from schemas.proj_entity_schema import EntityResponse
from schemas.activity_entity_schema import ActivityEntityResponse
from pydantic import ValidationError

http_client = httpx.Client(verify=False)

# Opening and reading the JSON file
# with open('embed_push_qdrant/ner.json', 'r') as f:
#     # Parsing the JSON file into a Python dictionary
#     ner = json.load(f)

with open('embed_push_qdrant/ner_4_steps/exports/proj_coll.json', 'r') as f:
    # Parsing the JSON file into a Python dictionary
    data = json.load(f)
    



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

with open('embed_push_qdrant/ner_4_steps/unique_level_names.json', 'r') as f:
    unique_level_names = json.load(f)




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

    def extract(self, user_question: str, level_init_filters: dict) -> dict:
        try:
            print("list(level_init_filters['project_names'].keys()):",list(level_init_filters["project_names"].keys()))
            project_names = list(level_init_filters["project_names"].keys())+["null"] if len(list(level_init_filters["project_names"].keys()))>0 else unique_level_names["project_names"]+["null"]
            print("project_names:",project_names)
            
            context_peritem = []
            print("project_names:", project_names)
            for i,item in enumerate(data):
                project_name = item["payload"]["proj_name"]
                project_id = item["id"]
                summary = item["payload"]["summary"]
                childern_uuids = item["payload"]["children_activity_uuids"]
                cont = f"""{i+1}. project_name: {project_name}\n
                    project_id: {project_id}\n
                    childern_uuids: {childern_uuids}\n
                    summary: {summary}\n\n"""
                if project_name in project_names :
                    context_peritem.append(cont)
                    

                

            proj_context = "".join(context_peritem)
            
            

            ENTITY_SYSTEM_PROMPT = f"""
            you are an entity recognition and classification engine for the ORP and WMS system.

            here is the context for you to classify the user query whether it belongs to which projects

            {proj_context}

            NOTE: YOU MUST ALWAYS CHOOSE A PROJECT FROM THE GIVEN CONTEXT ABOVE!!! LET ME REMIND YOU THE AVAILABLE PROJECT_NAMES BELOW 

            {project_names}

            NOTE THAT : YOU CAN NAME MULTIPLE PROJECTS IF YOU MUST
            ALSO NOTE THAT : IF YOU'RE REALLY UNSURE, YOU CAN FILL IT WITH 'null'

            Output Format (MANDATORY)

            Return only valid JSON.
            No explanations.
            No markdown.
            No extra keys.
            All value is string wrapped as list data type.

            example : 
            {{
              "entity": {{
                "project_name": ["<WMS>","<ORP>"],
                "project_id": ["<PROJECT_ID_PROVIDED_BY_SYSTEM>","<PROJECT_ID_PROVIDED_BY_SYSTEM>"],
                "childern_activity_uuids": ["<UUID_1>", "<UUID_2>"]
              }}
            }}
            """

            # 1. Use the .beta.parse helper to enforce the schema at the API level
            
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_question},
                ],
                response_format=EntityResponse, # This triggers Structured Outputs
                # logprobs=True, # Must be enabled
                # top_logprobs=2
            )

            # 2. Access the pre-validated Pydantic object directly
            validated = response.choices[0].message.parsed
            # logprobs = response.choices[0].logprobs
            # # print(logprobs)
            # full_data = response.model_dump()
            # with open("embed_push_qdrant/ner_4_steps/full_api_response.json", "w") as f:
            #     json.dump(full_data, f, indent=2)
            
            if validated is None:
                # This occurs if the model refuses for safety/policy reasons
                raise ValueError("LLM refused to return structured data.")

            # 3. Save to file using model_dump()
            with open("embed_push_qdrant/ner_4_steps/proj_entity_output.json", mode="w", encoding="utf-8") as f:
                json.dump(validated.model_dump(), f, indent=2)
            
            return validated.model_dump()

        except ValidationError as e:
            # This is rare with Structured Outputs, but happens if the logic fails
            raise ValueError(f"Schema validation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise



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
    result = extractor.extract(question)

    print(json.dumps(result, indent=2))
    