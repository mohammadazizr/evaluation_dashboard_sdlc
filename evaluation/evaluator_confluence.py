import json
import os
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict

class RAGEvaluatorNew:
    """Evaluator untuk RAG: Presisi hanya dihitung untuk confluence_coll"""

    def __init__(self, k: int = 10):
        self.k = k
        self.results = {
            'total_questions': 0,
            'successes': 0,
            'precisions': [],
            'mrr_sum': 0.0
        }
        self.detailed_report = []

    def normalize_string(self, text: str) -> str:
        """Membersihkan string untuk perbandingan"""
        if not text: return ""
        base = os.path.basename(text).lower()
        if '.' in base:
            base = base.rsplit('.', 1)[0]
        return base.strip()

    def is_match(self, expected_list: List[str], retrieved_meta: Dict[str, Any]) -> bool:
        """Mengecek kecocokan title/filename"""
        retrieved_title = retrieved_meta.get('page_title') or retrieved_meta.get('filename') or ""
        norm_retrieved = self.normalize_string(retrieved_title)

        for expected in expected_list:
            if self.normalize_string(expected) == norm_retrieved:
                return True
        return False

    def evaluate(self, ground_truth_path: str, retrieval_results_path: str):
        # 1. Load Data
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)

        print(f"Evaluating documents from Ground Truth (Target: Confluence Only)...")

        for doc_item in gt_data:
            for q_item in doc_item.get('questions', []):
                question = q_item['question']
                expected_docs = q_item.get('source_documents', [])
                clean_q = question.strip()
                
                if clean_q not in retrieval_data:
                    continue

                self.results['total_questions'] += 1
                retrieved_chunks = retrieval_data[clean_q][:self.k]
                
                found_at_pos = -1
                correct_count = 0

                # --- LOGIKA MODIFIKASI: HANYA CEK CONFLUENCE ---
                for i, chunk in enumerate(retrieved_chunks, 1):
                    meta = chunk.get('metadata', {})
                    
                    # CEK: Hanya proses jika berasal dari confluence_coll
                    if meta.get('collection') == 'confluence_coll':
                        if self.is_match(expected_docs, meta):
                            if found_at_pos == -1:
                                found_at_pos = i
                            correct_count += 1
                    # Jika dari chunk_coll, kita abaikan (dianggap tidak relevan untuk presisi ini)

                # 3. Hitung Metrics
                # Denominator tetap len(retrieved_chunks) agar kita tahu 
                # seberapa efektif Confluence di tengah 10 chunk yang ditarik.
                precision = correct_count / len(retrieved_chunks) if retrieved_chunks else 0
                self.results['precisions'].append(precision)
                
                if found_at_pos != -1:
                    self.results['successes'] += 1
                    self.results['mrr_sum'] += (1.0 / found_at_pos)

                self.detailed_report.append({
                    "question": question,
                    "routing": doc_item.get("module"),
                    "expected": expected_docs,
                    "found_at_rank": found_at_pos,
                    "precision": round(precision, 4)
                })

    def print_summary(self):
        total = self.results['total_questions']
        if total == 0:
            print("No data evaluated.")
            return

        success_rate = (self.results['successes'] / total) * 100
        avg_precision = (sum(self.results['precisions']) / total) * 100
        mrr = (self.results['mrr_sum'] / total)

        print("\n" + "="*50)
        print("📊 CONFLUENCE-ONLY EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Questions   : {total}")
        print(f"Success Rate (R@k): {success_rate:.2f}%")
        print(f"Avg Precision     : {avg_precision:.2f}%")
        print(f"MRR               : {mrr:.4f}")
        print("="*50)

if __name__ == "__main__":
    GT_FILE = r"C:\Users\mco.mohammad\Documents\rag-sdlc-research_testing\ground_truth_r4.json"
    RESULT_FILE = r"C:\Users\mco.mohammad\Documents\rag-sdlc-research_testing\retrieve_process\output\retrieval_results_reranked.json"

    evaluator = RAGEvaluatorNew(k=10)
    
    if os.path.exists(GT_FILE) and os.path.exists(RESULT_FILE):
        evaluator.evaluate(GT_FILE, RESULT_FILE)
        evaluator.print_summary()
        
        with open("evaluation_report_confluence_only.json", "w", encoding='utf-8') as f:
            json.dump(evaluator.detailed_report, f, indent=2)
    else:
        print("Error: File input tidak ditemukan.")