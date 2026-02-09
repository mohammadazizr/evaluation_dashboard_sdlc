"""
JSON Logger for Keyword Extraction and NER Operations
Logs all extraction activities to JSON files for tracking and debugging
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExtractionLogger:
    """Handles JSON logging for keyword extraction and NER operations"""

    def __init__(self, log_dir: Path):
        """Initialize logger with log directory"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.keyword_log_file = self.log_dir / "keyword_extraction_log.json"
        self.ner_log_file = self.log_dir / "ner_extraction_log.json"

        # Initialize log files if they don't exist
        self._initialize_log_files()

    def _initialize_log_files(self):
        """Initialize log files with empty structure"""
        for log_file in [self.keyword_log_file, self.ner_log_file]:
            if not log_file.exists():
                initial_data = {
                    "created_at": datetime.now().isoformat(),
                    "entries": []
                }
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_data, f, indent=2, ensure_ascii=False)

    def _load_log(self, log_file: Path) -> Dict[str, Any]:
        """Load existing log file"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"created_at": datetime.now().isoformat(), "entries": []}

    def _save_log(self, log_file: Path, data: Dict[str, Any]):
        """Save log file"""
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def log_keyword_extraction(self,
                              question: str,
                              cleaned_query: str,
                              keywords: Dict[str, Any],
                              stopwords_count: int,
                              execution_time_ms: float,
                              status: str = "success",
                              error: Optional[str] = None):
        """
        Log keyword extraction operation

        Args:
            question: Original question/input
            cleaned_query: Cleaned query after processing
            keywords: Extracted keywords with counts
            stopwords_count: Number of stopwords used
            execution_time_ms: Execution time in milliseconds
            status: Operation status ('success', 'error', 'partial')
            error: Error message if any
        """
        log_data = self._load_log(self.keyword_log_file)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": execution_time_ms,
            "status": status,
            "input": {
                "original_question": question,
                "cleaned_query": cleaned_query,
                "input_length": len(question)
            },
            "output": {
                "total_keywords": len(keywords),
                "keywords": keywords,
                "keywords_with_matches": sum(1 for k in keywords.values()
                                            if k.get("total_count", 0) > 0)
            },
            "processing_info": {
                "stopwords_used": stopwords_count,
                "unique_keyword_paths": sum(len(k.get("path", {}))
                                           for k in keywords.values())
            }
        }

        if error:
            entry["error"] = error

        log_data["entries"].append(entry)
        log_data["last_updated"] = datetime.now().isoformat()
        log_data["total_entries"] = len(log_data["entries"])

        self._save_log(self.keyword_log_file, log_data)
        print(f"[LOG] Keyword extraction logged: {question[:50]}... ({execution_time_ms:.2f}ms)")

    def log_ner_extraction(self,
                          stage: str,  # 'project', 'activity', 'document'
                          question: str,
                          input_data: Dict[str, Any],
                          extracted_entities: Dict[str, Any],
                          execution_time_ms: float,
                          status: str = "success",
                          error: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Log NER extraction operation

        Args:
            stage: NER stage ('project', 'activity', 'document')
            question: Original question
            input_data: Input data used for extraction
            extracted_entities: Extracted entities
            execution_time_ms: Execution time in milliseconds
            status: Operation status
            error: Error message if any
            metadata: Additional metadata (model info, confidence scores, etc.)
        """
        log_data = self._load_log(self.ner_log_file)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "execution_time_ms": execution_time_ms,
            "status": status,
            "input": {
                "question": question,
                "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else [],
                "input_summary": self._summarize_input(input_data)
            },
            "output": {
                "extracted_entities": extracted_entities,
                "entity_count": len(extracted_entities) if isinstance(extracted_entities, dict) else 1
            },
            "processing_info": {
                "stage": stage
            }
        }

        if error:
            entry["error"] = error

        if metadata:
            entry["metadata"] = metadata

        log_data["entries"].append(entry)
        log_data["last_updated"] = datetime.now().isoformat()
        log_data["total_entries"] = len(log_data["entries"])

        self._save_log(self.ner_log_file, log_data)
        print(f"[LOG] NER {stage} extraction logged ({execution_time_ms:.2f}ms)")

    def log_batch_summary(self,
                         batch_id: str,
                         total_questions: int,
                         successful: int,
                         failed: int,
                         total_time_ms: float,
                         log_type: str = "keyword"):
        """
        Log batch processing summary

        Args:
            batch_id: Batch identifier
            total_questions: Total questions processed
            successful: Number of successful extractions
            failed: Number of failed extractions
            total_time_ms: Total processing time
            log_type: Type of log ('keyword' or 'ner')
        """
        log_file = self.keyword_log_file if log_type == "keyword" else self.ner_log_file
        log_data = self._load_log(log_file)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "type": "batch_summary",
            "statistics": {
                "total_questions": total_questions,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_questions if total_questions > 0 else 0,
                "total_time_ms": total_time_ms,
                "avg_time_per_question_ms": total_time_ms / total_questions if total_questions > 0 else 0
            }
        }

        log_data["entries"].append(summary)
        log_data["last_updated"] = datetime.now().isoformat()
        log_data["total_entries"] = len(log_data["entries"])

        self._save_log(log_file, log_data)
        print(f"[LOG] Batch summary logged: {batch_id} ({successful}/{total_questions} successful)")

    def get_keyword_logs(self) -> Dict[str, Any]:
        """Get all keyword extraction logs"""
        return self._load_log(self.keyword_log_file)

    def get_ner_logs(self) -> Dict[str, Any]:
        """Get all NER extraction logs"""
        return self._load_log(self.ner_log_file)

    @staticmethod
    def _summarize_input(input_data: Any, max_length: int = 100) -> str:
        """Create a brief summary of input data"""
        if isinstance(input_data, dict):
            return str(list(input_data.keys()))[:max_length]
        elif isinstance(input_data, str):
            return input_data[:max_length]
        elif isinstance(input_data, list):
            return f"List with {len(input_data)} items"
        else:
            return str(type(input_data).__name__)
