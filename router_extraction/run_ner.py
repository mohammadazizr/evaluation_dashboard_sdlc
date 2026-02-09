import os
import re
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Setup path to project root FIRST before any imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import QDRANT_URL, QDRANT_API_KEY, ROUTER_EXPORTS_DIR, BATCH_RESULTS_FINAL_INPUT
from config import UNIQUE_LEVEL_NAMES_PATH, GROUND_TRUTH_PATH, ROUTER_OUTPUT_DIR

from utils.ner_proj import EntityExtractor as proj_extractor
from utils.ner_activity import EntityExtractor as act_extractor
from utils.ner_doc import EntityExtractor as doc_extractor
from utils.qdrant_exporter import export_collection_to_json
from utils.keyword_extraction.keyword_query_analyzer import analyze_keywords
from utils.extraction_logger import ExtractionLogger

from qdrant_client import QdrantClient
# from unset_proxy import unset_proxy

def get_hierarchical_metadata(data):
    # 1. Find the minimum total_count (greater than 0)
    counts = [val["total_count"] for val in data.values() if val["total_count"] > 0]
    if not counts:
        return {"project_names": {}, "activity_names": {}, "doc_type_names": []}
    
    min_val = min(counts)
    
    # 2. Structures to preserve relationships
    projects_to_activities = {} # {project: set(activities)}
    activities_to_docs = {}     # {activity: set(doc_types)}
    all_doc_types = set()       # set(doc_types)
    
    # 3. Process paths for words matching the min count
    for word_info in data.values():
        if word_info["total_count"] == min_val:
            for path in word_info["path"].keys():
                parts = re.split(r'[\\/]+', path)
                
                if len(parts) >= 3:
                    project, activity, doc_type = parts[0], parts[1], parts[2]
                    
                    # Map Project -> Activity
                    if project not in projects_to_activities:
                        projects_to_activities[project] = set()
                    projects_to_activities[project].add(activity)
                    
                    # Map Activity -> Doc Type
                    if activity not in activities_to_docs:
                        activities_to_docs[activity] = set()
                    activities_to_docs[activity].add(doc_type)
                    
                    # Global list of Doc Types
                    all_doc_types.add(doc_type)
            
    # 4. Convert sets to sorted lists for the final output
    return {
        "project_names": {k: sorted(list(v)) for k, v in projects_to_activities.items()},
        "activity_names": {k: sorted(list(v)) for k, v in activities_to_docs.items()},
        "doc_type_names": sorted(list(all_doc_types))
    }

# --- Execution ---
# result = get_hierarchical_metadata(data)
# import json
# print(json.dumps(result, indent=2))

def ner(question, logger=None):
    keywords = analyze_keywords(question, logger=logger)
    level_init_filters = get_hierarchical_metadata(keywords)
    level_init_filters_path = ROUTER_OUTPUT_DIR / "level_init_filters.json"
    with open(level_init_filters_path, "w", encoding="utf-8") as f:
        json.dump(level_init_filters, f, indent=2, ensure_ascii=False)

    # if len(level_init_filters["project_names"])==1 and

    proj = proj_extractor()
    entity_proj = proj.extract(question, level_init_filters, logger_obj=logger)
    print("Proj Extraction Done")
    print(entity_proj, "\n\n")

    act = act_extractor()
    entity_act = act.extract(question, entity_proj, level_init_filters, logger_obj=logger)
    print("Act Extraction Done")
    print(entity_act, "\n\n")

    doc = doc_extractor()
    entity_doc = doc.extract(question, entity_proj, entity_act, level_init_filters, logger_obj=logger)
    print("Doc Extraction Done")
    print(entity_doc, "\n\n")

    combined = {
        "project_name" : entity_proj['entity']['project_name'],
        "project_id" : entity_proj['entity']['project_id'],
        "activity_name" : entity_act['entity']['activity_name'],
        "activity_id" : entity_act['entity']['activity_id'],
        "doc_type_name" : entity_doc['entity']['doc_type_name'],
        "doc_type_id" : entity_doc['entity']['doc_id']
    }


    # Save combined JSON
    combined_output_path = ROUTER_OUTPUT_DIR / "combined_output.json"
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Combined {len(combined)} unique entries")



    return combined

def process_batch_questions(ground_truth_file, output_file, logger=None):
    """
    Process all questions from ground_truth JSON file using NER pipeline
    and output results in batch_results_final.json format
    """

    # Read ground truth file
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    questions_data = ground_truth.get("questions", {})

    # Initialize output structure
    output = {
        "metadata": {
            "execution_timestamp": datetime.now().isoformat(),
            "total_questions": len(questions_data),
            "total_successful": 0,
            "total_failed": 0,
            "execution_duration_seconds": 0,
            "batch_size": 5,
            "mode": "async_batch"
        },
        "results": {
            "doc": {
                "document_path": "path/for/doc",
                "questions": {}
            }
        },
        "summary": {
            "total_documents": 1,
            "documents_processed": 1,
            "total_questions_processed": len(questions_data),
            "total_successful": 0,
            "total_failed": 0,
            "success_rate": 0.0,
            "total_execution_time_seconds": 0
        },
        "errors": []
    }

    # Process each question
    start_time = time.time()
    successful = 0
    failed = 0

    i = 0
    for key, question_data in questions_data.items():
        q_start = time.time()

        try:
            question = question_data.get("question", "")
            question_ner = re.sub(r'\bWFMS\b', 'WMS', question, flags=re.IGNORECASE)
            test_type = question_data.get("test_type", "single_doc")
            expected_docs = question_data.get("expected_documents", [])

            # Run NER on question
            ner_result = ner(question_ner, logger=logger)

            # Build question result
            question_result = {
                "question": question,
                "test_type": test_type,
                "status": "success",
                "execution_time_ms": (time.time() - q_start) * 1000,
                "ner_output": {
                    "project_name": ner_result.get("project_name", ["null"]),
                    "project_id": ner_result.get("project_id", ["null"]),
                    "activity_name": ner_result.get("activity_name", ["null"]),
                    "activity_id": ner_result.get("activity_id", ["null"]),
                    "doc_type_name": ner_result.get("doc_type_name", ["null"]),
                    "doc_type_id": ner_result.get("doc_type_id", ["null"])
                },
                "expected_documents": expected_docs
            }

            output["results"]["doc"]["questions"][key] = question_result
            successful += 1
            print(f"[OK] {key}: SUCCESS ({question_result['execution_time_ms']:.2f}ms)")
            # i+=1
            # if i == 25:
            #     break
        except TimeoutError as e:
            failed += 1
            timeout_msg = f"LLM request timeout: {str(e)}"
            print(f"[TIMEOUT] {key}: FAILED - {timeout_msg}")
            output["errors"].append({
                "question_key": key,
                "error": timeout_msg,
                "error_type": "timeout"
            })
        except Exception as e:
            failed += 1
            print(f"[ERROR] {key}: FAILED - {str(e)}")
            output["errors"].append({
                "question_key": key,
                "error": str(e)
            })

    # Calculate total time
    total_time = time.time() - start_time

    # Count timeout errors
    timeout_errors = sum(1 for err in output["errors"] if err.get("error_type") == "timeout")

    # Add document summary
    output["results"]["doc"]["document_summary"] = {
        "total_questions": len(questions_data),
        "successful": successful,
        "failed": failed,
        "timeout_errors": timeout_errors,
        "avg_execution_time_ms": total_time * 1000 / len(questions_data) if questions_data else 0
    }

    # Update metadata and summary
    output["metadata"]["total_successful"] = successful
    output["metadata"]["total_failed"] = failed
    output["metadata"]["timeout_errors"] = timeout_errors
    output["metadata"]["execution_duration_seconds"] = total_time

    output["summary"]["total_successful"] = successful
    output["summary"]["total_failed"] = failed
    output["summary"]["timeout_errors"] = timeout_errors
    output["summary"]["success_rate"] = successful / len(questions_data) if questions_data else 0
    output["summary"]["total_execution_time_seconds"] = total_time

    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Log batch summary
    if logger:
        logger.log_batch_summary(
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_questions=len(questions_data),
            successful=successful,
            failed=failed,
            total_time_ms=total_time * 1000,
            log_type="keyword"
        )
        logger.log_batch_summary(
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_questions=len(questions_data),
            successful=successful,
            failed=failed,
            total_time_ms=total_time * 1000,
            log_type="ner"
        )

    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"  Total questions: {len(questions_data)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Timeout errors: {timeout_errors}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")

    return output

if __name__ == "__main__":
    # Load environment
    load_dotenv()

    # Initialize logger
    logger = ExtractionLogger(ROUTER_OUTPUT_DIR)
    print(f"[LOG] Extraction logger initialized. Logs will be saved to {ROUTER_OUTPUT_DIR}")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=60,
        verify=False
    )

    print("Connected to Qdrant at:", QDRANT_URL)
    print("Client info:", client)

    try:
        # Export collections
        target_collection = ["proj_coll", "activity_coll", "doc_type_coll"]
        project_names = []
        activity_names = []
        doc_type_names = []

        for collection_name in target_collection:
            export_collection_to_json(client, collection_name, str(ROUTER_EXPORTS_DIR))

            export_file = ROUTER_EXPORTS_DIR / f'{collection_name}.json'
            with open(export_file, 'r', encoding='utf-8') as f:
                data_coll = json.load(f)

            for i, item in enumerate(data_coll):
                if collection_name == "proj_coll":
                    project_names.append(item["payload"]["proj_name"])
                elif collection_name == "activity_coll":
                    activity_names.append(item["payload"]["activity_name"])
                elif collection_name == "doc_type_coll":
                    doc_type_names.append(item["payload"]["doc_type_name"])

        print()
        print("="*40)
        unique_project_names = set(project_names)
        print("unique_project_names: ", unique_project_names)
        unique_activity_names = set(activity_names)
        print("unique_activity_names: ", unique_activity_names)
        unique_doc_type_names = set(doc_type_names)
        print("unique_doc_type_names: ", unique_doc_type_names)

        print("="*40)
        print()

        unique_items = {
            "project_names": list(unique_project_names),
            "activity_names": list(unique_activity_names),
            "doc_type_names": list(unique_doc_type_names)
        }

        with open(str(UNIQUE_LEVEL_NAMES_PATH), mode="w", encoding="utf-8") as write_file:
            json.dump(unique_items, write_file)

        # Process batch from ground truth file
        ground_truth_path = str(GROUND_TRUTH_PATH)
        output_path = str(BATCH_RESULTS_FINAL_INPUT)

        process_batch_questions(ground_truth_path, output_path, logger=logger)

        # Print log file locations
        print(f"\n[LOG] Logs saved to:")
        print(f"  - Keyword Extraction: {logger.keyword_log_file}")
        print(f"  - NER Extraction: {logger.ner_log_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()