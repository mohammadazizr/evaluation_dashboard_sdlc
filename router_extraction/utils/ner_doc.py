import os
import sys
import json
import logging
import truststore
from openai import OpenAI
from dotenv import load_dotenv
import httpx
from pathlib import Path

from .schemas.doc_entity_schema import DocEntityResponse
from .filter_context_helper import apply_keyword_filters
import re

# Configure timeout for HTTP client (30 seconds)
http_client = httpx.Client(verify=False, timeout=30.0)

# Import config from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DOC_TYPE_COLL_EXPORT, UNIQUE_LEVEL_NAMES_PATH, DOC_ENTITY_OUTPUT, PROJ_ENTITY_OUTPUT, ACT_ENTITY_OUTPUT, LEVEL_INIT_FILTERS_PATH

# Opening and reading the JSON file
with open(DOC_TYPE_COLL_EXPORT, 'r', encoding='utf-8') as f:
    # Parsing the JSON file into a Python dictionary
    data_doc_type = json.load(f)

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

    def extract(self, user_question: str, proj_context: dict, act_context: dict, level_init_filters: dict = unique_level_names["doc_type_names"], logger_obj=None) -> dict:
        import time
        start_time = time.time()

        try:
            project_parent_name = proj_context["entity"]["project_name"]

            activity_parent_name = act_context["entity"]["activity_name"]
            print("activity_parent_name:",activity_parent_name)
            print("activity_parent_name[0]:", activity_parent_name[0])

            # Get doc type filters for this activity from keyword extraction
            doc_type_init_filters = level_init_filters["activity_names"].get(activity_parent_name[0], []) if activity_parent_name[0] != "null" else []

            logger.debug(f"[DOC_TYPE] Extracted filters from keywords: {doc_type_init_filters}")

            # Build list of all doc types for this activity
            all_activity_doc_types = []
            doc_type_context_dict = {}

            for i, item in enumerate(data_doc_type):
                doc_type_name = item["payload"]["doc_type_name"]
                parent_activity_name = item["payload"]["parent_activity_name"]

                # Only consider doc types from the current activity
                if parent_activity_name == activity_parent_name[0]:
                    all_activity_doc_types.append(doc_type_name)

                    # Store context for later use
                    doc_id = item["id"]
                    summary = item["payload"]["summary"]
                    cont = f"""{i+1}. doc_type_name: {doc_type_name}\n
                doc_id: {doc_id}\n
                summary: {summary}\n\n"""
                    doc_type_context_dict[doc_type_name] = cont

            # Apply keyword filters to get final doc type list
            doc_type_names = apply_keyword_filters(
                candidates=all_activity_doc_types,
                filters=doc_type_init_filters if doc_type_init_filters else None,
                fallback_list=unique_level_names["doc_type_names"],
                logger=logger
            )

            logger.debug(f"[DOC_TYPE] Final doc types after filtering: {doc_type_names}")

            # Build context with only filtered doc types
            activity_context_peritem = []
            for doc_type_name in doc_type_names:
                if doc_type_name in doc_type_context_dict:
                    activity_context_peritem.append(doc_type_context_dict[doc_type_name])

            doc_context = "".join(activity_context_peritem)

    #         with open('embed_push_qdrant/ner_4_steps/unique_level_names.json', 'r') as f:
    #             unique_level_names = json.load(f)

    #         doc_type_names = unique_level_names["doc_type_names"]+["null"]

            DOC_ENTITY_SYSTEM_PROMPT = f"""
<persona> You are a Product Owner/Manager overseeing documents within the project: {project_parent_name} and specifically under the activity: {activity_parent_name}. When deciding which document type a user's query refers to, your methodology involves rigorous contextual analysis and evaluating the validity of various hypotheses (e.g., whether the query refers to Technical, Functional, or Business documentation) until you reach the most accurate and logical final decision. </persona>

<task> Identify and determine the specific Document Type Names and Document IDs based on the user's query and the provided context for each document type. </task>

<context> Available Document Type Names (filtered by query keywords):
For ORP Portal: TSD, FSD, and BRD
For ORP Workflow: TSD, FSD, and BRD
For BOC: TSD and BRD
For Head Office: TSD and BRD

Available document types for this activity (filtered by query keywords): {doc_type_names}

Detailed context for each document type: {doc_context} </context>

<constraints>
- YOU MUST ALWAYS CHOOSE A DOCUMENT TYPE FROM THE GIVEN CONTEXT ABOVE.
- You may identify multiple document types if the query is relevant to more than one.
- SPECIAL MAPPING RULES:
-- If the query mentions 'Specification Document', classify it as 'TSD'.
-- If the query mentions 'Requirement Document', classify it as 'BRD'.
-- If the query mentions 'Functional Specification Document', classify it as 'FSD'.
-- If you are genuinely unsure or find no match, fill the value with "null".
- Return ONLY valid JSON.
- No explanations or prose.
- No markdown formatting (e.g., do not use json ... ).
- Do not add extra keys outside of the provided template.
- All values must be strings wrapped in a list [] data type. </constraints>

<output-format> JSON format. Use the following template:
{{ "entity":
{{ "doc_type_name": ["<DOC_TYPE_NAME_CHOSEN_FROM_CONTEXT>", "<DOC_TYPE_NAME_CHOSEN_FROM_CONTEXT>"],
"doc_id": ["<DOC_ID_FROM_CONTEXT>", "<DOC_ID_FROM_CONTEXT>"],
"childern_chunk_uuids": ["<UUID_1>", "<UUID_2>"] }} }}
</output-format>
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

            # 1. Call API with regular chat completion (with timeout)
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": DOC_ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_question},
                ],
                timeout=30.0,
            )

            # 2. Get raw response and clean formatting
            raw_response = response.choices[0].message.content
            print(f"RESPONSE: {raw_response}")
            print("response doc is done")

            cleaned_response = clean_json_response(raw_response)
            validated = json.loads(cleaned_response)
            print("validated:", validated)
            print("="*40)
            print()

            # 3. Save and return
            DOC_ENTITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            with open(DOC_ENTITY_OUTPUT, mode="w", encoding="utf-8") as f:
                json.dump(validated, f, indent=2)

            # 4. Log the extraction
            execution_time_ms = (time.time() - start_time) * 1000
            if logger_obj:
                logger_obj.log_ner_extraction(
                    stage="document",
                    question=user_question,
                    input_data={"project_name": project_parent_name, "activity_name": activity_parent_name, "doc_types": doc_type_names},
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
                    stage="document",
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
                    stage="document",
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
        "Document ORP workflow dengan judul Integrasi Pembukuan Fee Interbranch - Recurring"
    )

    with open(PROJ_ENTITY_OUTPUT, 'r', encoding='utf-8') as f:
        # Parsing the JSON file into a Python dictionary
        data_proj = json.load(f)

    with open(ACT_ENTITY_OUTPUT, 'r', encoding='utf-8') as f:
        # Parsing the JSON file into a Python dictionary
        data_act = json.load(f)

    with open(LEVEL_INIT_FILTERS_PATH, 'r', encoding='utf-8') as f:
        # Parsing the JSON file into a Python dictionary
        level_init_filters = json.load(f)

    result = extractor.extract(question, data_proj, data_act, level_init_filters)

    # print(json.dumps(result, indent=2))
    DOC_ENTITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC_ENTITY_OUTPUT, mode="w", encoding="utf-8") as write_file:
        json.dump(result, write_file)