import os
import sys
import json
import logging
import truststore
from openai import OpenAI
from dotenv import load_dotenv
import httpx
from pathlib import Path

from .schemas.proj_entity_schema import EntityResponse
from .schemas.activity_entity_schema import ActivityEntityResponse
import re

# Configure timeout for HTTP client (30 seconds)
http_client = httpx.Client(verify=False, timeout=30.0)

# Import config from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROJ_COLL_EXPORT, UNIQUE_LEVEL_NAMES_PATH, PROJ_ENTITY_OUTPUT

# Opening and reading the JSON file
with open(PROJ_COLL_EXPORT, 'r', encoding='utf-8') as f:
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

LITELLM_API_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_KEY", "")
LITELLM_MODEL_NAME = os.getenv("LITELLM_MODEL_NAME", "gpt-4")

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# System Prompt
# =============================

with open(UNIQUE_LEVEL_NAMES_PATH, 'r', encoding='utf-8') as f:
    unique_level_names = json.load(f)




# =============================
# Helper Functions
# =============================
def clean_json_response(response_text):
    """Remove markdown JSON formatting (```json ... ```) if present"""
    # Remove markdown code blocks
    response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
    response_text = re.sub(r'\n?```\s*$', '', response_text)
    return response_text.strip()

# =============================
# Entity Extractor Class
# =============================
class EntityExtractor:
    def __init__(self, model_name: str = LITELLM_MODEL_NAME, api_key: str = LITELLM_API_KEY, base_url: str = LITELLM_API_URL):
        if not api_key:
            raise ValueError("API key tidak tersedia")
        if not base_url:
            raise ValueError("Base URL tidak tersedia")

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{base_url.rstrip('/')}/v1" if not base_url.endswith('/v1') else base_url,
            http_client=http_client,
        )

        logger.info(f"Entity extractor initialized with model: {model_name} at {base_url}")

    def extract(self, user_question: str, level_init_filters: dict, logger_obj=None) -> dict:
        import time
        start_time = time.time()

        try:
            # print("list(level_init_filters['project_names'].keys()):",list(level_init_filters["project_names"].keys()))
            project_names = list(level_init_filters["project_names"].keys())+["null"] if len(list(level_init_filters["project_names"].keys()))>0 else unique_level_names["project_names"]+["null"]
            # print("project_names:",project_names)

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

            # Definisi pemetaan (mapping)
            mapping = {
                'ORP': 'Operation Request Portal 2025',
                'WMS': 'Workflow Management System'
            }

            project_names = [mapping.get(item, item) for item in project_names]
            print()
            print("="*40)
            print(f"List of project names to consider: {project_names}")
            print("="*40)
            print()
            
            proj_context = "".join(context_peritem)

            ENTITY_SYSTEM_PROMPT = f"""
            <persona> You are a Product Owner/Manager overseeing multiple projects. When deciding which project a user's query refers to, your methodology involves rigorous contextual analysis and evaluating the validity of various hypotheses (e.g., if Outcome A is correct, if Outcome B is correct, etc.) until you reach the most accurate and logical final decision to address the user's inquiry. </persona>

<task> Identify and determine the project names based on the user's query and the provided summary context for each project. </task>

<context> Available project_names (filtered by query keywords): {project_names}

Activities associated with each project:
- ORP: ORP Portal and ORP Workflow
- WMS: BOC and Head Office.

Summary for each project:
{proj_context} </context>

<constraints>
- YOU MUST ONLY CHOOSE PROJECT NAMES FROM THE CONTEXT PROVIDED ABOVE.
- You may identify multiple projects if the query is relevant to more than one.
- If you are genuinely unsure or find no match, fill the value with "null".
- Return ONLY valid JSON.
- No explanations or prose.
- No markdown formatting (e.g., do not use json ... ).
- Do not add extra keys outside of the provided template.
- All values must be strings wrapped in a list [] data type. </constraints>

<output-format>
JSON format. Use the following template:
{{ "entity":
{{ "project_name": ["<PROJECT_NAME_1>", "<PROJECT_NAME_2>"],
"project_id": ["<PROJECT_ID_FROM_SYSTEM>", "<PROJECT_ID_FROM_SYSTEM>"],
"children_activity_uuids": ["<UUID_1>", "<UUID_2>"] }}
}}
</output-format>
            """

            # 1. Call API with regular chat completion (with timeout)
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<user_question>{user_question}</user_question>"},
                ],
                timeout=30.0,
            )

            # 2. Get raw response and clean formatting
            raw_response = response.choices[0].message.content
            print(f"RESPONSE: {raw_response}")

            cleaned_response = clean_json_response(raw_response)
            validated = json.loads(cleaned_response)
            print("validated:", validated)
            print("="*40)
            print()

            # 3. Save to file
            PROJ_ENTITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            with open(PROJ_ENTITY_OUTPUT, mode="w", encoding="utf-8") as f:
                json.dump(validated, f, indent=2)

            # 4. Log the extraction
            execution_time_ms = (time.time() - start_time) * 1000
            if logger_obj:
                logger_obj.log_ner_extraction(
                    stage="project",
                    question=user_question,
                    input_data={"project_names": project_names, "level_init_filters": list(level_init_filters.keys())},
                    extracted_entities=validated.get("entity", {}),
                    execution_time_ms=execution_time_ms,
                    status="success",
                    metadata={"model": self.model_name}
                )

            return validated

        except (TimeoutError, httpx.TimeoutException) as e:
            execution_time_ms = (time.time() - start_time) * 1000
            if logger_obj:
                logger_obj.log_ner_extraction(
                    stage="project",
                    question=user_question,
                    input_data={},
                    extracted_entities={},
                    execution_time_ms=execution_time_ms,
                    status="error",
                    error=f"Request timeout: {str(e)}",
                    metadata={"model": self.model_name}
                )
            logger.error(f"Request timeout - LLM service took too long to respond: {e}")
            raise TimeoutError(f"LLM API timeout after 30 seconds: {e}")
        except json.JSONDecodeError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            if logger_obj:
                logger_obj.log_ner_extraction(
                    stage="project",
                    question=user_question,
                    input_data={},
                    extracted_entities={},
                    execution_time_ms=execution_time_ms,
                    status="error",
                    error=f"JSON parsing error: {str(e)}",
                    metadata={"model": self.model_name}
                )
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from API: {e}")
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            if logger_obj:
                logger_obj.log_ner_extraction(
                    stage="project",
                    question=user_question,
                    input_data={},
                    extracted_entities={},
                    execution_time_ms=execution_time_ms,
                    status="error",
                    error=str(e),
                    metadata={"model": self.model_name}
                )
            logger.error(f"Unexpected error: {e}")
            raise



# =============================
# Example Usage
# =============================
if __name__ == "__main__":

    unset_proxy()

    extractor = EntityExtractor(
        model_name=LITELLM_MODEL_NAME,
        api_key=LITELLM_API_KEY,
        base_url=LITELLM_API_URL,
    )

    question = (
        "What is the activity code and author of the WFMS Workflow for Head Office Requirement Document v.0.1 from 16 Maret 2023?"
    )
    result = extractor.extract(question)

    # print(json.dumps(result, indent=2))
    