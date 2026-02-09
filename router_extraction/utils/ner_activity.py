import os
import sys
import json
import logging
import truststore
from openai import OpenAI
from dotenv import load_dotenv
import httpx
from pathlib import Path

from .schemas.activity_entity_schema import ActivityEntityResponse
from .filter_context_helper import apply_keyword_filters
import re

# Configure timeout for HTTP client (30 seconds)
http_client = httpx.Client(verify=False, timeout=30.0)

# Import config from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ACTIVITY_COLL_EXPORT, UNIQUE_LEVEL_NAMES_PATH, ACT_ENTITY_OUTPUT, PROJ_ENTITY_OUTPUT

# Opening and reading the JSON file
with open(ACTIVITY_COLL_EXPORT, 'r', encoding='utf-8') as f:
    # Parsing the JSON file into a Python dictionary
    data_activity = json.load(f)

with open(UNIQUE_LEVEL_NAMES_PATH, 'r', encoding='utf-8') as f:
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

LITELLM_API_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_KEY", "")
LITELLM_MODEL_NAME = os.getenv("LITELLM_MODEL_NAME", "gpt-4")

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# System Prompt
# =============================



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

    def extract(self, user_question: str, proj_context: dict, level_init_filters: dict = unique_level_names["activity_names"], logger_obj=None) -> dict:
        import time
        start_time = time.time()

        try:
            project_parent_name = proj_context["entity"]["project_name"]
            print("project_parent_name:",project_parent_name)

            print("project_parent_name[0]",project_parent_name[0])

            # Get activity filters for this project from keyword extraction
            activity_init_filters = level_init_filters["project_names"].get(project_parent_name[0], []) if project_parent_name[0] != "null" else []

            logger.debug(f"[ACTIVITY] Extracted filters from keywords: {activity_init_filters}")

            # Build list of all activities for this project
            all_project_activities = []
            activity_context_dict = {}

            for i, item in enumerate(data_activity):
                activity_name = item["payload"]["activity_name"]
                parent_project_name = item["payload"]["parent_project_name"]

                # Only consider activities from the current project
                if parent_project_name == project_parent_name[0]:
                    all_project_activities.append(activity_name)

                    # Store context for later use
                    activity_id = item["id"]
                    summary = item["payload"]["summary"]
                    childern_uuids = item["payload"]["children_doc_type_uuids"]
                    cont = f"""{i+1}. activity_name: {activity_name}\n
                activity_id: {activity_id}\n
                childern_uuids: {childern_uuids}\n
                summary: {summary}\n\n"""
                    activity_context_dict[activity_name] = cont

            # Apply keyword filters to get final activity list
            activity_names = apply_keyword_filters(
                candidates=all_project_activities,
                filters=activity_init_filters if activity_init_filters else None,
                fallback_list=unique_level_names["activity_names"],
                logger=logger
            )

            logger.debug(f"[ACTIVITY] Final activity names after filtering: {activity_names}")

            # Build context with only filtered activities
            activity_context_peritem = []
            for activity_name in activity_names:
                if activity_name in activity_context_dict:
                    activity_context_peritem.append(activity_context_dict[activity_name])

            act_context = "".join(activity_context_peritem)

    #         with open('embed_push_qdrant/ner_4_steps/unique_level_names.json', 'r') as f:
    #             unique_level_names = json.load(f)

    #         activity_names = unique_level_names["activity_names"]+["null"]

            print()
            print("="*40)
            print(f"List of activity names to consider: {activity_names}")
            print("="*40)
            print()

            ACTIVITY_ENTITY_SYSTEM_PROMPT = f"""
<persona> You are a Product Owner/Manager overseeing multiple activities within the project: {project_parent_name}. When deciding which activity a user's query refers to, your methodology involves rigorous contextual analysis and evaluating the validity of various hypotheses (e.g., if Activity A is the correct intent, if Activity B is the correct intent, etc.) until you reach the most accurate and logical final decision. </persona>

<task> Identify and determine the specific Activity Names and Activity IDs based on the user's query and the provided context for each activity. </task>

<context> Available Activity Names (filtered by query keywords):
For ORP: ORP Portal and ORP Workflow
For WMS: BOC and Head Office

Activities available under the project (filtered by query keywords): {activity_names}

Detailed context for each activity: {act_context}

IMPORTANT: Note that activity_name and activity_id are distinct identifiers provided in the context. </context>

<constraints>
- YOU MUST ALWAYS CHOOSE AN ACTIVITY FROM THE GIVEN CONTEXT ABOVE.
- You may identify multiple activities if the query is relevant to more than one.
- If you are genuinely unsure or find no match, fill the value with "null".
- Return ONLY valid JSON.
- No explanations or prose.
- No markdown formatting (e.g., do not use json ... ).
- Do not add extra keys outside of the provided template.
- All values must be strings wrapped in a list [] data type. </constraints>

<output-format> JSON format. Use the following template:
{{ "entity":
{{ "activity_name": ["<ACTIVITY_NAME_CHOSEN_FROM_CONTEXT>", "<ACTIVITY_NAME_CHOSEN_FROM_CONTEXT>"],
"activity_id": ["<ACTIVITY_ID_FROM_CONTEXT>", "<ACTIVITY_ID_FROM_CONTEXT>"],
"childern_doc_type_uuids": ["<UUID_1>", "<UUID_2>"] }} }}
</output-format>
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

            # 1. Call API with regular chat completion (with timeout)
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": ACTIVITY_ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_question},
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

            # 3. Save and return
            ACT_ENTITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            with open(ACT_ENTITY_OUTPUT, mode="w", encoding="utf-8") as f:
                json.dump(validated, f, indent=2)

            # 4. Log the extraction
            execution_time_ms = (time.time() - start_time) * 1000
            if logger_obj:
                logger_obj.log_ner_extraction(
                    stage="activity",
                    question=user_question,
                    input_data={"project_name": project_parent_name, "activity_names": activity_names},
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
                    stage="activity",
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
                    stage="activity",
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

    with open(PROJ_ENTITY_OUTPUT, 'r', encoding='utf-8') as f:
        # Parsing the JSON file into a Python dictionary
        data_proj = json.load(f)

    result = extractor.extract(question, data_proj)

    # print(json.dumps(result, indent=2))
    