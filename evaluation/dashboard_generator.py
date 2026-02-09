"""
HTML Dashboard Generator for RAG Retrieval Evaluation
Creates an interactive HTML dashboard with visualizations
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from utils.hierarchical_metrics import HierarchicalHTMLGenerator

class DashboardGenerator:
    """Generates HTML dashboard from evaluation results"""

    def __init__(self, evaluation_results: Dict[str, Any]):
        """
        Initialize dashboard generator

        Args:
            evaluation_results: Evaluation results from RAGEvaluator
        """
        self.results = evaluation_results
        self.metrics = evaluation_results.get('aggregate_metrics', {})
        self.config = evaluation_results.get('configuration', {})
        self.detailed = evaluation_results.get('detailed_results', [])
        self.hierarchical_metrics = evaluation_results.get('hierarchical_metrics', {
            'by_project': {},
            'by_activity': {},
            'by_doc_type': {}
        })

    def generate_position_chart_data(self, test_type: str) -> str:
        """
        Generate Chart.js data for position distribution

        Args:
            test_type: Type of test (single_doc, multi_doc, etc.)

        Returns:
            JavaScript code for chart data
        """
        if test_type not in self.metrics:
            return "[]"

        position_dist = self.metrics[test_type].get('position_distribution', {})

        # Create data for all positions 1-10
        positions = list(range(1, self.config.get('k', 10) + 1))
        counts = [position_dist.get(str(pos), 0) for pos in positions]

        labels = [f"Position {i}" for i in positions]

        return json.dumps({
            'labels': labels,
            'datasets': [{
                'label': 'Documents Found',
                'data': counts,
                'backgroundColor': 'rgba(54, 162, 235, 0.6)',
                'borderColor': 'rgba(54, 162, 235, 1)',
                'borderWidth': 2
            }]
        })

    def generate_precision_chart_data(self) -> str:
        """
        Generate Chart.js data for precision comparison across test types

        Returns:
            JavaScript code for chart data
        """
        test_types = ['single_doc', 'multi_doc', 'conflicting_docs', 'long_queries']
        labels = [t.replace('_', ' ').title() for t in test_types]
        precisions = [
            self.metrics.get(t, {}).get('average_precision', 0) * 100
            for t in test_types
        ]
        success_rates = [
            self.metrics.get(t, {}).get('success_rate', 0) * 100
            for t in test_types
        ]

        return json.dumps({
            'labels': labels,
            'datasets': [
                {
                    'label': 'Average Precision (%)',
                    'data': precisions,
                    'backgroundColor': 'rgba(75, 192, 192, 0.6)',
                    'borderColor': 'rgba(75, 192, 192, 1)',
                    'borderWidth': 2
                },
                {
                    'label': 'Success Rate (%)',
                    'data': success_rates,
                    'backgroundColor': 'rgba(153, 102, 255, 0.6)',
                    'borderColor': 'rgba(153, 102, 255, 1)',
                    'borderWidth': 2
                }
            ]
        })

    def context_recall_metrics(self) -> float:
        """
        Calculate context_recall: percentage of retrieved expected documents

        Returns:
            Percentage value (0-100)
        """
        total_expected_count = 0
        total_retrieved_expected = 0

        for item in self.detailed:
            total_expected_count += item.get("expected_count", 0)
            try:
                retrieved_docs = item.get("retrieved_expected_docs", [])
                if retrieved_docs:
                    total_retrieved_expected += len(retrieved_docs)
            except:
                pass

        if total_expected_count == 0:
            return 0.0

        return round((total_retrieved_expected / total_expected_count) * 100, 2)

    def mean_reciprocal_rank(self) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR): average of reciprocal ranks
        MRR measures how fast the system finds the first correct document
        Formula: 1/rank of first relevant document, averaged across all queries

        Returns:
            MRR value (0-1)
        """
        total_reciprocal_rank = 0.0
        count = 0

        for item in self.detailed:
            # Skip no_doc test cases
            if item.get("test_type") == "no_doc":
                continue

            positions_found = item.get("positions_found", [])

            # If document was found, get the first position (lowest rank)
            if positions_found:
                first_position = positions_found[0]
                reciprocal_rank = 1.0 / first_position
                total_reciprocal_rank += reciprocal_rank
                count += 1
            else:
                # If not found, reciprocal rank is 0
                count += 1

        if count == 0:
            return 0.0

        return round(total_reciprocal_rank / count, 4)

    def average_first_rank(self) -> float:
        """
        Calculate average rank of first relevant document found

        Returns:
            Average rank position (e.g., 5.2 means average first position is 5.2)
        """
        total_rank = 0.0
        count = 0

        for item in self.detailed:
            # Skip no_doc test cases
            if item.get("test_type") == "no_doc":
                continue

            positions_found = item.get("positions_found", [])

            # If document was found, get the first position (lowest rank)
            if positions_found:
                first_position = positions_found[0]
                total_rank += first_position
                count += 1

        if count == 0:
            return 0.0

        return round(total_rank / count, 2)

    def questions_with_relevant_docs(self) -> tuple:
        """
        Count questions that have relevant documents (positions_found not empty)

        Returns:
            Tuple of (questions_with_relevant_docs, total_questions)
        """
        questions_with_relevant = 0
        total_questions = 0

        for item in self.detailed:
            # Skip no_doc test cases
            if item.get("test_type") == "no_doc":
                continue

            total_questions += 1
            positions_found = item.get("positions_found", [])

            # If positions_found exists and is not empty, document was found
            if positions_found:
                questions_with_relevant += 1

        return questions_with_relevant, total_questions

    def calculate_ndcg(self, item: Dict[str, Any], k: int = None) -> float:
        """
        Calculate NDCG@K for a single query result
        NDCG = DCG / IDCG where:
        - DCG = sum(rel_i / log2(i+1)) for positions 1 to k
        - IDCG = ideal DCG (all relevant docs at top positions)

        Args:
            item: Single detailed result item
            k: Number of top positions to consider (default: retrieved_count)

        Returns:
            NDCG value (0-1)
        """
        import math

        positions_found = item.get("positions_found", [])
        retrieved_count = item.get("retrieved_count", 0)
        expected_count = item.get("expected_count", 0)

        if k is None:
            k = retrieved_count if retrieved_count > 0 else 10

        # Create relevance vector (binary: 1 if relevant, 0 otherwise)
        relevance = [0] * k
        for pos in positions_found:
            if 1 <= pos <= k:
                relevance[pos - 1] = 1

        # Calculate DCG - only count top expected_count relevant documents
        dcg = 0.0
        relevant_count = 0
        for i, rel in enumerate(relevance):
            if rel > 0 and relevant_count < expected_count:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because position starts at 1
                relevant_count += 1

        # Calculate IDCG (ideal: all relevant docs at top positions)
        idcg = 0.0
        for i in range(min(expected_count, k)):
            idcg += 1.0 / math.log2(i + 2)

        # Calculate NDCG
        if idcg == 0:
            return 0.0

        return round(min(dcg / idcg, 1.0), 4)

    def mean_ndcg(self) -> float:
        """
        Calculate Mean NDCG across all queries

        Returns:
            Average NDCG value (0-1)
        """
        total_ndcg = 0.0
        count = 0

        for item in self.detailed:
            # Skip no_doc test cases
            if item.get("test_type") == "no_doc":
                continue

            ndcg = self.calculate_ndcg(item)
            total_ndcg += ndcg
            count += 1

        if count == 0:
            return 0.0

        return round(total_ndcg / count, 4)

    def get_status_icon(self, value: float, threshold: float = 0.8) -> str:
        """
        Get status icon based on value

        Args:
            value: Metric value (0-1)
            threshold: Threshold for success

        Returns:
            HTML for status icon
        """
        if value >= threshold:
            return '<span style="color: #10b981; font-size: 24px;">✓</span>'
        elif value >= threshold - 0.1:
            return '<span style="color: #f59e0b; font-size: 24px;">⚠</span>'
        else:
            return '<span style="color: #ef4444; font-size: 24px;">✗</span>'

    def generate_hierarchical_table_html(self, hierarchy_type: str) -> str:
        """
        Generate HTML table for hierarchical metrics

        Args:
            hierarchy_type: 'by_project', 'by_activity', or 'by_doc_type'

        Returns:
            HTML string for the table
        """
        return HierarchicalHTMLGenerator.generate_table_html(
            hierarchy_type, self.hierarchical_metrics
        )

    def generate_html(self) -> str:
        """
        Generate complete HTML dashboard

        Returns:
            HTML string
        """
        overall = self.metrics.get('overall', {})
        no_doc = self.metrics.get('no_doc', {})

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Retrieval Evaluation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .header h1 {{
            color: #1f2937;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #6b7280;
            font-size: 1.1em;
        }}

        .config-info {{
            background: #f3f4f6;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: flex;
            gap: 30px;
        }}

        .config-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .config-label {{
            font-weight: 600;
            color: #4b5563;
        }}

        .config-value {{
            color: #1f2937;
            background: white;
            padding: 4px 12px;
            border-radius: 4px;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .card-title {{
            color: #6b7280;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .card-value {{
            color: #1f2937;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .card-label {{
            color: #9ca3af;
            font-size: 0.9em;
        }}

        .card-dropdown-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #667eea;
            font-size: 0.85em;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            padding: 8px 0;
            border: none;
            background: none;
            transition: color 0.2s;
        }}

        .card-dropdown-toggle:hover {{
            color: #764ba2;
        }}

        .card-dropdown-toggle .arrow {{
            transition: transform 0.3s ease;
            font-size: 0.8em;
        }}

        .card-dropdown-toggle.active .arrow {{
            transform: rotate(180deg);
        }}

        .card-dropdown-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            color: #6b7280;
            font-size: 0.85em;
            line-height: 1.6;
        }}

        .card-dropdown-content.active {{
            max-height: 500px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e5e7eb;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .chart-title {{
            color: #1f2937;
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
        }}

        .table-container {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th {{
            background: #f9fafb;
            color: #374151;
            font-weight: 600;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #e5e7eb;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
            color: #1f2937;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        .status-good {{
            color: #10b981;
            font-weight: 600;
        }}

        .status-warning {{
            color: #f59e0b;
            font-weight: 600;
        }}

        .status-poor {{
            color: #ef4444;
            font-weight: 600;
        }}

        .footer {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
            color: #6b7280;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .position-charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .hierarchy-selector {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .hierarchy-btn {{
            padding: 10px 20px;
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            color: #6b7280;
            transition: all 0.3s;
        }}

        .hierarchy-btn:hover {{
            border-color: #667eea;
            color: #667eea;
        }}

        .hierarchy-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .hierarchy-content {{
            display: none;
            animation: fadeIn 0.3s ease-in;
        }}

        .hierarchy-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}

        @media (max-width: 768px) {{
            .metrics-grid,
            .position-charts {{
                grid-template-columns: 1fr;
            }}
            .hierarchy-selector {{
                flex-direction: column;
            }}
            .hierarchy-btn {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 RAG Retrieval Evaluation Dashboard</h1>
            <p class="subtitle">Comprehensive analysis of retrieval system performance</p>
            <div class="config-info">
                <div class="config-item">
                    <span class="config-label">Top-K:</span>
                    <span class="config-value">{self.config.get('k', 10)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Confidence Threshold:</span>
                    <span class="config-value">{self.config.get('confidence_threshold', 0.7)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Generated:</span>
                    <span class="config-value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
            </div>
        </div>

        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="card">
                <div class="card-title">Overall Success Rate</div>
                <div class="card-value">{overall.get('success_rate', 0)*100:.1f}%</div>
                <div class="card-label">Questions answered correctly</div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>Overall Success Rate is the percentage of total questions that successfully found their expected documents in the retrieval results.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Average Precision</div>
                <div class="card-value">{overall.get('average_precision', 0)*100:.1f}%</div>
                <div class="card-label">Correct chunks in top-{self.config.get('k', 10)}</div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>Average Precision measures the percentage of correct expected documents in each retrieved chunk.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Precision@3</div>
                <div class="card-value">{overall.get('precision_at_3', 0)*100:.1f}%</div>
                <div class="card-label">Correct doc found in top-3</div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>Precision@3 measures the percentage of queries where at least one correct expected document is found within the top-3 retrieved results. This metric is useful for evaluating early retrieval performance.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Total Questions</div>
                <div class="card-value">{overall.get('total_retrieval_questions', 0)}</div>
                <div class="card-label">Retrieval test cases</div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>Total Questions represents the total number of test questions used in the retrieval evaluation process.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-title">No-Doc Detection</div>
                <div class="card-value">{no_doc.get('correct_rejection_rate', 0)*100:.1f}%</div>
                <div class="card-label">Correctly rejected unanswerable</div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>No-Doc Detection measures the ability to correctly identify questions that do not refer to any specific document, such as general questions that cannot be answered from the document collection.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Context Recall</div>
                <div class="card-value">{self.context_recall_metrics():.2f}%</div>
                <div class="card-label">Retrieved expected documents</div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>Context Recall Metrics measures the percentage of expected documents that were successfully retrieved across all test cases. Formula: (Total Retrieved Expected / Total Expected Count) × 100%</p>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Mean Reciprocal Rank</div>
                <div class="card-value">{self.mean_reciprocal_rank()*100:.2f}%</div>
                <div class="card-label">
                    Avg. first position: {self.average_first_rank():.1f}<br>
                    Questions with relevant docs: {self.questions_with_relevant_docs()[0]}/{self.questions_with_relevant_docs()[1]}
                </div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>Mean Reciprocal Rank (MRR) measures how quickly the system finds the first correct document. It calculates the average of reciprocal ranks (1/rank) where rank is the position of the first relevant document found. The average first position shows at which rank the system typically finds the first relevant document. Questions with relevant docs shows how many questions successfully found at least one relevant document. Higher MRR values (closer to 100%) indicate better performance.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-title">NDCG (Normalized DCG)</div>
                <div class="card-value">{self.mean_ndcg()*100:.2f}%</div>
                <div class="card-label">Ranking quality metric</div>
                <button class="card-dropdown-toggle" onclick="toggleDropdown(this)">
                    <span>View Explanation</span>
                    <span class="arrow">▼</span>
                </button>
                <div class="card-dropdown-content">
                    <p>NDCG (Normalized Discounted Cumulative Gain) measures the quality of the ranking by considering both the position and relevance of retrieved documents. It's calculated as DCG/IDCG, where DCG gives more weight to relevant documents at higher positions (discounted by log of position), and IDCG is the ideal maximum DCG. Values closer to 100% indicate better ranking quality. NDCG is particularly useful for evaluating how well the most important documents are ranked.</p>
                </div>
            </div>
        </div>

        <!-- Performance Comparison Chart -->
        <div class="metrics-grid">
            <div class="chart-container">
                <h3 class="chart-title">Performance by Test Type</h3>
                <canvas id="precisionChart"></canvas>
            </div>
        </div>

        <!-- Position Distribution Charts -->
        <h2 style="color: white; margin: 30px 0 20px 0; font-size: 1.8em;">📍 Position Distribution Analysis</h2>
        <div class="position-charts">
            <div class="chart-container">
                <h3 class="chart-title">Single Document Retrieval</h3>
                <canvas id="positionChart1"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title">Multi-Document Synthesis</h3>
                <canvas id="positionChart2"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title">Conflicting Documents</h3>
                <canvas id="positionChart3"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title">Long Queries</h3>
                <canvas id="positionChart4"></canvas>
            </div>
        </div>

        <!-- Detailed Metrics Table -->
        <div class="table-container">
            <h3 class="chart-title">Detailed Performance Metrics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Test Scenario</th>
                        <th>Questions</th>
                        <th>Success Rate</th>
                        <th>Avg Precision</th>
                        <th>Precision@3</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_table_rows()}
                </tbody>
            </table>
        </div>

        <!-- Hierarchical Metrics Section -->
        <div class="table-container">
            <h3 class="chart-title">📊 Metrics by Hierarchy Level</h3>
            <div class="hierarchy-selector">
                <button class="hierarchy-btn active" onclick="switchHierarchy('project')">By Project</button>
                <button class="hierarchy-btn" onclick="switchHierarchy('activity')">By Activity</button>
                <button class="hierarchy-btn" onclick="switchHierarchy('doc_type')">By Doc Type</button>
            </div>

            <div id="project-content" class="hierarchy-content active">
                <h4 style="margin-bottom: 15px; color: #1f2937;">Project-Level Metrics</h4>
                {self.generate_hierarchical_table_html('by_project')}
            </div>

            <div id="activity-content" class="hierarchy-content">
                <h4 style="margin-bottom: 15px; color: #1f2937;">Activity-Level Metrics</h4>
                {self.generate_hierarchical_table_html('by_activity')}
            </div>

            <div id="doc_type-content" class="hierarchy-content">
                <h4 style="margin-bottom: 15px; color: #1f2937;">Doc Type-Level Metrics</h4>
                {self.generate_hierarchical_table_html('by_doc_type')}
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>RAG Retrieval Evaluation System | Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
    </div>

    <script>
        // Toggle dropdown function for cards
        function toggleDropdown(button) {{
            const content = button.nextElementSibling;
            const isActive = button.classList.contains('active');

            if (isActive) {{
                button.classList.remove('active');
                content.classList.remove('active');
            }} else {{
                button.classList.add('active');
                content.classList.add('active');
            }}
        }}

        // Switch hierarchy view
        function switchHierarchy(hierarchyType) {{
            // Hide all content
            document.getElementById('project-content').classList.remove('active');
            document.getElementById('activity-content').classList.remove('active');
            document.getElementById('doc_type-content').classList.remove('active');

            // Remove active class from all buttons
            document.querySelectorAll('.hierarchy-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});

            // Show selected content
            const contentId = hierarchyType === 'project' ? 'project-content' :
                            hierarchyType === 'activity' ? 'activity-content' :
                            'doc_type-content';
            document.getElementById(contentId).classList.add('active');

            // Add active class to clicked button
            event.target.classList.add('active');
        }}

        // Precision Comparison Chart
        const precisionCtx = document.getElementById('precisionChart').getContext('2d');
        new Chart(precisionCtx, {{
            type: 'bar',
            data: {self.generate_precision_chart_data()},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Position Distribution Charts
        const positionCharts = [
            {{ id: 'positionChart1', type: 'single_doc' }},
            {{ id: 'positionChart2', type: 'multi_doc' }},
            {{ id: 'positionChart3', type: 'conflicting_docs' }},
            {{ id: 'positionChart4', type: 'long_queries' }}
        ];

        const positionData = {{
            'single_doc': {self.generate_position_chart_data('single_doc')},
            'multi_doc': {self.generate_position_chart_data('multi_doc')},
            'conflicting_docs': {self.generate_position_chart_data('conflicting_docs')},
            'long_queries': {self.generate_position_chart_data('long_queries')}
        }};

        positionCharts.forEach(chart => {{
            const ctx = document.getElementById(chart.id).getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: positionData[chart.type],
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{
                                stepSize: 1
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return 'Found: ' + context.parsed.y + ' times';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
        """
        return html

    def _generate_table_rows(self) -> str:
        """Generate table rows for detailed metrics"""
        rows = []

        test_types = [
            ('single_doc', 'Single Document Retrieval'),
            ('multi_doc', 'Multi-Document Synthesis'),
            ('conflicting_docs', 'Conflicting Documents'),
            ('long_queries', 'Long Query Handling'),
            ('no_doc', 'No-Document Detection')
        ]

        for test_type, label in test_types:
            if test_type not in self.metrics:
                continue

            metrics = self.metrics[test_type]

            if test_type == 'no_doc':
                success = metrics.get('correct_rejection_rate', 0)
                precision = metrics.get('correct_rejection_rate', 0)
                precision_at_3 = metrics.get('correct_rejection_rate', 0)
            else:
                success = metrics.get('success_rate', 0)
                precision = metrics.get('average_precision', 0)
                precision_at_3 = metrics.get('precision_at_3', 0)

            status_class = 'status-good' if success >= 0.8 else ('status-warning' if success >= 0.7 else 'status-poor')
            status_icon = self.get_status_icon(success, 0.8)

            rows.append(f"""
                <tr>
                    <td>{label}</td>
                    <td>{metrics.get('total_questions', 0)}</td>
                    <td class="{status_class}">{success*100:.2f}%</td>
                    <td class="{status_class}">{precision*100:.2f}%</td>
                    <td class="{status_class}">{precision_at_3*100:.2f}%</td>
                    <td>{status_icon}</td>
                </tr>
            """)

        return ''.join(rows)

    def save(self, output_path: str):
        """
        Save HTML dashboard to file

        Args:
            output_path: Path to save HTML file
        """
        html_content = self.generate_html()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Dashboard saved to: {output_path}")


def generate_dashboard(evaluation_results_path: str, output_path: str = "dashboard.html"):
    """
    Generate HTML dashboard from evaluation results JSON

    Args:
        evaluation_results_path: Path to evaluation results JSON file
        output_path: Path to save HTML dashboard
    """
    # Load evaluation results
    with open(evaluation_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Generate dashboard
    generator = DashboardGenerator(results)
    generator.save(output_path)


if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path

    # Import config from project root
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import EVALUATION_RESULTS_PATH, DASHBOARD_OUTPUT_PATH

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = str(EVALUATION_RESULTS_PATH)

    if os.path.exists(results_file):
        print(f"Generating dashboard from: {results_file}")
        generate_dashboard(results_file, str(DASHBOARD_OUTPUT_PATH))
        print("\nDashboard generated successfully!")
        print("Open 'dashboard.html' in your browser to view the results.")
    else:
        print(f"ERROR: Evaluation results file not found: {results_file}")
        print("\nUsage: python dashboard_generator.py [evaluation_results_new.json]")
