#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG Retrieval Evaluation Pipeline
Orchestrates the complete workflow:
1. NER (Named Entity Recognition) - router_extraction
2. Retrieval with Reranking - retrieve_process
3. Evaluation - evaluation
4. Dashboard Generation - evaluation
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path


class RAGPipeline:
    """Orchestrates the complete RAG pipeline"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.start_time = datetime.now()
        self.results = {}

    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f"🚀 {title}")
        print("=" * 80)

    def print_step(self, step_num: int, title: str):
        """Print step header"""
        print(f"\n[STEP {step_num}] {title}")
        print("-" * 80)

    def run_stage(self, stage_num: int, stage_name: str, script_path: str, working_dir: str) -> bool:
        """
        Run a pipeline stage

        Args:
            stage_num: Stage number
            stage_name: Name of the stage
            script_path: Relative path to the script
            working_dir: Working directory for the script

        Returns:
            True if successful, False otherwise
        """
        self.print_step(stage_num, stage_name)

        full_script_path = self.project_root / script_path
        full_working_dir = self.project_root / working_dir

        print(f"📁 Working directory: {full_working_dir}")
        print(f"📄 Script: {full_script_path}")
        print(f"⏱️  Started at: {datetime.now().strftime('%H:%M:%S')}")

        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(full_script_path)],
                cwd=str(full_working_dir),
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            # Print output
            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print("⚠️  Warnings/Errors:", result.stderr)

            if result.returncode == 0:
                print(f"✅ {stage_name} completed successfully")
                self.results[stage_name] = {
                    'status': 'success',
                    'returncode': result.returncode,
                    'timestamp': datetime.now().isoformat()
                }
                return True
            else:
                print(f"❌ {stage_name} failed with return code {result.returncode}")
                self.results[stage_name] = {
                    'status': 'failed',
                    'returncode': result.returncode,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
                return False

        except subprocess.TimeoutExpired:
            print(f"❌ {stage_name} timed out after 10 minutes")
            self.results[stage_name] = {
                'status': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
            return False

        except Exception as e:
            print(f"❌ Error running {stage_name}: {str(e)}")
            self.results[stage_name] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False

    def run_pipeline(self):
        """Run the complete pipeline"""
        self.print_header("RAG RETRIEVAL EVALUATION PIPELINE")
        print(f"Project root: {self.project_root}")
        print(f"Pipeline started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        all_stages = [
            # {
            #     'num': 1,
            #     'name': 'NER (Named Entity Recognition)',
            #     'script': 'router_extraction/run_ner.py',
            #     'working_dir': 'router_extraction'
            # },
            {
                'num': 2,
                'name': 'Retrieval with Reranking',
                'script': 'retrieve_process/retrieve.py',
                'working_dir': 'retrieve_process'
            },
            {
                'num': 3,
                'name': 'Evaluation',
                'script': 'evaluation/evaluator.py',
                'working_dir': 'evaluation'
            },
            {
                'num': 4,
                'name': 'Dashboard Generation',
                'script': 'evaluation/dashboard_generator.py',
                'working_dir': 'evaluation'
            }
        ]

        success_count = 0
        failed_stages = []

        for stage in all_stages:
            success = self.run_stage(
                stage['num'],
                stage['name'],
                stage['script'],
                stage['working_dir']
            )

            if success:
                success_count += 1
            else:
                failed_stages.append(stage['name'])

        # Print summary
        self.print_header("PIPELINE EXECUTION SUMMARY")

        end_time = datetime.now()
        duration = end_time - self.start_time

        print(f"\n📊 Results:")
        print(f"   Total stages: {len(all_stages)}")
        print(f"   Successful: {success_count} ✅")
        print(f"   Failed: {len(failed_stages)} ❌")
        print(f"\n⏱️  Execution time: {duration.total_seconds():.2f} seconds")
        print(f"   Started: {self.start_time.strftime('%H:%M:%S')}")
        print(f"   Ended: {end_time.strftime('%H:%M:%S')}")

        if failed_stages:
            print(f"\n⚠️  Failed stages:")
            for stage in failed_stages:
                print(f"   - {stage}")
            print(f"\n📝 Details saved in: pipeline_execution_log.json")
        else:
            print("\n🎉 All stages completed successfully!")

        # Save execution log
        self._save_execution_log(all_stages, duration)

        return success_count == len(all_stages)

    def _save_execution_log(self, all_stages, duration):
        """Save pipeline execution log"""
        log = {
            'execution_timestamp': self.start_time.isoformat(),
            'completion_timestamp': datetime.now().isoformat(),
            'total_duration_seconds': duration.total_seconds(),
            'stages': all_stages,
            'results': self.results
        }

        log_path = self.project_root / 'pipeline_execution_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        print(f"   Log file: {log_path}")


def main():
    """Main entry point"""
    try:
        pipeline = RAGPipeline()
        success = pipeline.run_pipeline()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⛔ Pipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
