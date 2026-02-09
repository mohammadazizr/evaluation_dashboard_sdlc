"""
RAG Retrieval Evaluation System - Updated Version
Evaluates retrieval performance with support for new flat ground truth format.
Handles retrieval_results_new.json format.
"""

import json
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict
import os


class RAGEvaluatorNew:
    """Evaluates RAG retrieval performance with multiple metrics - Enhanced version"""

    def __init__(self, ground_truth_path: str, k: int = 10, confidence_threshold: float = 0.7):
        """
        Initialize the evaluator
        """
        self.k = k
        self.confidence_threshold = confidence_threshold

        # Load ground truth
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)

        # Dynamic results storage to handle various test_type names (multi_docs_answer, etc.)
        self.results = defaultdict(lambda: {
            'position_counts': defaultdict(int), 
            'precisions': [], 
            'total_questions': 0, 
            'successes': 0
        })
        
        # Specific storage for no-doc
        self.results['no_doc'] = {'correct_rejections': 0, 'false_positives': 0, 'total_questions': 0}

        # Track which questions were evaluated
        self.evaluated_questions = []
        self.missing_questions = []

    def normalize_path(self, path: str) -> str:
        # 1. Ubah forward slash jadi backslash dan kecilkan huruf
        normalized = path.replace('/', '\\').lower().strip()

        # 2. Daftar prefix yang akan dihapus
        prefixes_to_remove = [
            '\\home\\cdsw\\embed_push_qdrant\\mpm\\',
            'home\\cdsw\\embed_push_qdrant\\mpm\\',
            '\\home\\cdsw\\qdrant\\mpm\\',
            'qdrant\\mpm\\',
            'mpm\\'
        ]

        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
            
        return normalized

    def extract_filename_from_metadata(self, chunk_metadata: Dict[str, Any]) -> str:
        if 'filename' in chunk_metadata:
            return self.normalize_path(chunk_metadata['filename'])
        return ""

    def evaluate_retrieval(self,
                          question: str,
                          retrieved_chunks: List[Dict[str, Any]],
                          expected_documents: List[str],
                          test_type: str) -> Dict[str, Any]:
        result = {
            'question': question,
            'test_type': test_type,
            'retrieved_count': len(retrieved_chunks),
            'expected_count': len(expected_documents)
        }

        # Normalize expected documents
        expected_docs_normalized = [self.normalize_path(doc) for doc in expected_documents]

        # Extract filenames and scores from retrieved chunks
        retrieved_docs = []
        for chunk in retrieved_chunks[:self.k]:  # Only consider top-k
            filename = self.extract_filename_from_metadata(chunk.get('metadata', {}))
            score = chunk.get('score', 0.0)
            retrieved_docs.append({'filename': filename, 'score': score})

        if test_type == 'no_doc':
            all_below_threshold = all(doc['score'] < self.confidence_threshold for doc in retrieved_docs)
            result['correct_rejection'] = all_below_threshold
            result['max_score'] = max([doc['score'] for doc in retrieved_docs]) if retrieved_docs else 0.0
            result['scores'] = [round(doc['score'], 2) for doc in retrieved_docs]

            self.results['no_doc']['total_questions'] += 1
            if all_below_threshold:
                self.results['no_doc']['correct_rejections'] += 1
            else:
                self.results['no_doc']['false_positives'] += 1

        else:
            positions_found = []
            correct_chunks = 0

            for position, doc_info in enumerate(retrieved_docs, start=1):
                if doc_info['filename'] in expected_docs_normalized:
                    positions_found.append(position)
                    correct_chunks += 1

            precision = correct_chunks / len(retrieved_docs) if len(retrieved_docs) > 0 else 0.0

            unique_retrieved_docs = set(doc['filename'] for doc in retrieved_docs if doc['filename'])
            retrieved_expected_docs = [doc for doc in expected_docs_normalized if doc in unique_retrieved_docs]
            all_docs_found = len(retrieved_expected_docs) == len(expected_docs_normalized) and len(expected_docs_normalized) > 0

            result['positions_found'] = positions_found
            result['correct_chunks'] = correct_chunks
            result['precision'] = round(precision, 4)
            result['all_docs_found'] = all_docs_found
            result['retrieved_expected_docs'] = retrieved_expected_docs
            result['missing_docs'] = [doc for doc in expected_docs_normalized if doc not in unique_retrieved_docs]

            # Update results by test type
            test_results = self.results[test_type]
            test_results['total_questions'] += 1
            test_results['precisions'].append(precision)

            if correct_chunks > 0:
                test_results['successes'] += 1

            for position in positions_found:
                test_results['position_counts'][position] += 1

        return result

    def evaluate_all(self, retrieval_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        detailed_results = []
        
        # Access the flat 'questions' structure
        questions_data = self.ground_truth.get('questions', {})

        for q_id, q_info in questions_data.items():
            question = q_info['question']
            expected_docs = q_info['expected_documents']
            test_type = q_info['test_type']

            if question not in retrieval_results:
                self.missing_questions.append({
                    'question': question,
                    'question_id': q_id,
                    'test_type': test_type
                })
                continue

            retrieved_chunks = retrieval_results[question]

            result = self.evaluate_retrieval(
                question=question,
                retrieved_chunks=retrieved_chunks,
                expected_documents=expected_docs,
                test_type=test_type
            )

            result['question_id'] = q_id
            detailed_results.append(result)
            self.evaluated_questions.append(question)

        aggregate_metrics = self.calculate_aggregate_metrics()

        return {
            'detailed_results': detailed_results,
            'aggregate_metrics': aggregate_metrics,
            'configuration': {
                'k': self.k,
                'confidence_threshold': self.confidence_threshold
            },
            'evaluation_summary': {
                'total_questions_in_ground_truth': self.count_total_questions(),
                'questions_evaluated': len(self.evaluated_questions),
                'questions_missing': len(self.missing_questions)
            }
        }

    def count_total_questions(self) -> int:
        return len(self.ground_truth.get('questions', {}))

    def calculate_aggregate_metrics(self) -> Dict[str, Any]:
        metrics = {}
        
        # All types except no_doc
        retrieval_types = [t for t in self.results.keys() if t != 'no_doc']

        for test_type in retrieval_types:
            test_results = self.results[test_type]
            if test_results['total_questions'] > 0:
                avg_precision = sum(test_results['precisions']) / len(test_results['precisions'])
                position_distribution = {str(k): v for k, v in test_results['position_counts'].items()}
                success_rate = test_results['successes'] / test_results['total_questions']

                metrics[test_type] = {
                    'total_questions': test_results['total_questions'],
                    'average_precision': round(avg_precision, 4),
                    'success_rate': round(success_rate, 4),
                    'position_distribution': position_distribution
                }

        # No-doc metrics
        no_doc_results = self.results['no_doc']
        if no_doc_results['total_questions'] > 0:
            metrics['no_doc'] = {
                'total_questions': no_doc_results['total_questions'],
                'correct_rejection_rate': round(no_doc_results['correct_rejections'] / no_doc_results['total_questions'], 4),
                'false_positive_rate': round(no_doc_results['false_positives'] / no_doc_results['total_questions'], 4),
                'correct_rejections': no_doc_results['correct_rejections'],
                'false_positives': no_doc_results['false_positives']
            }

        # Overall metrics (Retrieval)
        all_precisions = []
        total_successes = 0
        total_retrieval_q = 0

        for test_type in retrieval_types:
            all_precisions.extend(self.results[test_type]['precisions'])
            total_successes += self.results[test_type]['successes']
            total_retrieval_q += self.results[test_type]['total_questions']

        if total_retrieval_q > 0:
            metrics['overall'] = {
                'total_retrieval_questions': total_retrieval_q,
                'average_precision': round(sum(all_precisions) / len(all_precisions), 4),
                'success_rate': round(total_successes / total_retrieval_q, 4)
            }

        return metrics

    def save_results(self, results: Dict[str, Any], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_path}")

    def print_summary(self, evaluation_results: Dict[str, Any]):
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)

        summary = evaluation_results.get('evaluation_summary', {})
        print(f"\n📝 Questions:")
        print(f"   Total in ground truth: {summary.get('total_questions_in_ground_truth', 0)}")
        print(f"   ✅ Evaluated: {summary.get('questions_evaluated', 0)}")
        print(f"   ⚠️  Missing: {summary.get('questions_missing', 0)}")

        metrics = evaluation_results['aggregate_metrics']

        if 'overall' in metrics:
            print(f"\n🎯 Overall Performance:")
            print(f"   Success Rate: {metrics['overall']['success_rate']*100:.2f}%")
            print(f"   Average Precision: {metrics['overall']['average_precision']*100:.2f}%")

        print(f"\n📋 By Test Type:")
        for test_type, m in metrics.items():
            if test_type in ['overall', 'no_doc']: continue
            print(f"\n   {test_type.replace('_', ' ').title()}:")
            print(f"      Questions: {m['total_questions']}")
            print(f"      Success Rate: {m['success_rate']*100:.2f}%")
            print(f"      Avg Precision: {m['average_precision']*100:.2f}%")

        if 'no_doc' in metrics:
            m = metrics['no_doc']
            print(f"\n   No-Doc Scenarios:")
            print(f"      Questions: {m['total_questions']}")
            print(f"      Correct Rejection Rate: {m['correct_rejection_rate']*100:.2f}%")
            print(f"      False Positive Rate: {m['false_positive_rate']*100:.2f}%")

        print("\n" + "=" * 70)


def load_retrieval_results(results_path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("🔍 RAG Retrieval Evaluator - Updated Version")
    print("=" * 70)

    GROUND_TRUTH_FILE = "retrieve/ground_truth_060126.json"
    RETRIEVAL_RESULTS_FILE = "retrieve/output/retrieval_results_reranked.json"
    OUTPUT_FILE = "retrieve/output/evaluation_results_new.json"

    if len(sys.argv) > 1:
        RETRIEVAL_RESULTS_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ ERROR: Ground truth file not found: {GROUND_TRUTH_FILE}")
        sys.exit(1)

    if not os.path.exists(RETRIEVAL_RESULTS_FILE):
        print(f"❌ ERROR: Retrieval results file not found: {RETRIEVAL_RESULTS_FILE}")
        sys.exit(1)

    print(f"\n⚙️  Configuration:")
    print(f"   Ground truth: {GROUND_TRUTH_FILE}")
    print(f"   Retrieval results: {RETRIEVAL_RESULTS_FILE}")
    print(f"   Output: {OUTPUT_FILE}")

    evaluator = RAGEvaluatorNew(
        ground_truth_path=GROUND_TRUTH_FILE,
        k=10,
        confidence_threshold=0.7
    )

    print(f"   Top-k: {evaluator.k}")
    print(f"   Confidence threshold: {evaluator.confidence_threshold}")

    print(f"\n📂 Loading retrieval results...")
    retrieval_results = load_retrieval_results(RETRIEVAL_RESULTS_FILE)
    print(f"   Found {len(retrieval_results)} questions with retrieval results")

    print(f"\n🔄 Running evaluation...")
    evaluation_results = evaluator.evaluate_all(retrieval_results)

    evaluator.save_results(evaluation_results, OUTPUT_FILE)
    evaluator.print_summary(evaluation_results)

    if evaluator.missing_questions:
        print(f"\n⚠️  WARNING: {len(evaluator.missing_questions)} questions not evaluated")
        for i, missing in enumerate(evaluator.missing_questions[:5], 1):
            print(f"   {i}. [{missing['test_type']}] {missing['question'][:70]}...")

    print("\n✨ Evaluation complete!")