"""
Hierarchical Metrics Utilities
Shared functions for hierarchical metrics calculation and HTML generation
"""

from typing import Dict, Any, List, Tuple
from collections import defaultdict


class HierarchyExtractor:
    """Extract and manage path hierarchy (project, activity, doc_type)"""

    @staticmethod
    def extract_from_path(normalized_path: str) -> Dict[str, str]:
        """
        Extract project, activity, and doc_type from normalized path

        Args:
            normalized_path: Path in format project\\activity\\doc_type\\filename.md

        Returns:
            Dict with keys: 'project', 'activity', 'doc_type'

        Example:
            >>> path = 'wms\\boc\\tsd\\01\\output.md'
            >>> result = HierarchyExtractor.extract_from_path(path)
            >>> result == {
            ...     'project': 'wms',
            ...     'activity': 'wms\\boc',
            ...     'doc_type': 'wms\\boc\\tsd'
            ... }
            True
        """
        parts = normalized_path.split('\\')

        return {
            'project': parts[0] if len(parts) > 0 else 'unknown',
            'activity': '\\'.join(parts[:2]) if len(parts) > 1 else parts[0],
            'doc_type': '\\'.join(parts[:-1]) if len(parts) > 1 else parts[0]
        }

    @staticmethod
    def get_hierarchy_display_name(hierarchy_type: str) -> str:
        """Get display name for hierarchy type"""
        mapping = {
            'by_project': 'Project',
            'by_activity': 'Activity',
            'by_doc_type': 'Doc Type'
        }
        return mapping.get(hierarchy_type, hierarchy_type)

    @staticmethod
    def get_hierarchy_labels() -> List[Tuple[str, str]]:
        """Get all hierarchy types with their labels"""
        return [
            ('by_project', 'By Project'),
            ('by_activity', 'By Activity'),
            ('by_doc_type', 'By Doc Type')
        ]


class HierarchicalMetricsCalculator:
    """Calculate and aggregate hierarchical metrics"""

    @staticmethod
    def calculate_metrics_for_level(
        results_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a single hierarchy level

        Args:
            results_dict: Dict of {key: {position_counts, precisions, total_questions, successes}}

        Returns:
            Dict with calculated metrics for each key
        """
        metrics = {}

        for key, value in results_dict.items():
            if value['total_questions'] > 0:
                avg_precision = (
                    sum(value['precisions']) / len(value['precisions'])
                    if value['precisions']
                    else 0.0
                )
                success_rate = value['successes'] / value['total_questions']
                position_distribution = {
                    str(k): v for k, v in value['position_counts'].items()
                }

                metrics[key] = {
                    'total_questions': value['total_questions'],
                    'average_precision': round(avg_precision, 4),
                    'success_rate': round(success_rate, 4),
                    'position_distribution': position_distribution
                }

        return metrics

    @staticmethod
    def calculate_all_hierarchies(
        hierarchical_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics for all hierarchy levels

        Args:
            hierarchical_results: Full hierarchical results structure

        Returns:
            Dict with calculated metrics for all levels
        """
        hierarchical_metrics = {}

        for hierarchy_type in ['by_project', 'by_activity', 'by_doc_type']:
            results = hierarchical_results.get(hierarchy_type, {})
            hierarchical_metrics[hierarchy_type] = (
                HierarchicalMetricsCalculator.calculate_metrics_for_level(results)
            )

        return hierarchical_metrics


class HierarchicalMetricsStorage:
    """Manage storage of hierarchical metrics"""

    @staticmethod
    def initialize_storage() -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Initialize empty hierarchical results storage

        Returns:
            Empty hierarchical results structure ready for population
        """
        return {
            'by_project': defaultdict(lambda: {
                'position_counts': defaultdict(int),
                'precisions': [],
                'total_questions': 0,
                'successes': 0
            }),
            'by_activity': defaultdict(lambda: {
                'position_counts': defaultdict(int),
                'precisions': [],
                'total_questions': 0,
                'successes': 0
            }),
            'by_doc_type': defaultdict(lambda: {
                'position_counts': defaultdict(int),
                'precisions': [],
                'total_questions': 0,
                'successes': 0
            })
        }

    @staticmethod
    def update_metrics(
        hierarchical_results: Dict[str, Dict[str, Dict[str, Any]]],
        hierarchy_path: Dict[str, str],
        precision: float,
        correct_chunks: int,
        positions_found: List[int]
    ) -> None:
        """
        Update hierarchical metrics for a question

        Args:
            hierarchical_results: Storage to update
            hierarchy_path: Dict with 'project', 'activity', 'doc_type'
            precision: Precision value for this question
            correct_chunks: Number of correct chunks found
            positions_found: List of positions where documents were found
        """
        # Update by_project
        project = hierarchy_path['project']
        hierarchical_results['by_project'][project]['total_questions'] += 1
        hierarchical_results['by_project'][project]['precisions'].append(precision)
        if correct_chunks > 0:
            hierarchical_results['by_project'][project]['successes'] += 1
        for position in positions_found:
            hierarchical_results['by_project'][project]['position_counts'][position] += 1

        # Update by_activity
        activity = hierarchy_path['activity']
        hierarchical_results['by_activity'][activity]['total_questions'] += 1
        hierarchical_results['by_activity'][activity]['precisions'].append(precision)
        if correct_chunks > 0:
            hierarchical_results['by_activity'][activity]['successes'] += 1
        for position in positions_found:
            hierarchical_results['by_activity'][activity]['position_counts'][position] += 1

        # Update by_doc_type
        doc_type = hierarchy_path['doc_type']
        hierarchical_results['by_doc_type'][doc_type]['total_questions'] += 1
        hierarchical_results['by_doc_type'][doc_type]['precisions'].append(precision)
        if correct_chunks > 0:
            hierarchical_results['by_doc_type'][doc_type]['successes'] += 1
        for position in positions_found:
            hierarchical_results['by_doc_type'][doc_type]['position_counts'][position] += 1


class HierarchicalHTMLGenerator:
    """Generate HTML for hierarchical metrics display"""

    @staticmethod
    def get_status_class(success_rate: float) -> str:
        """Get CSS class based on success rate"""
        if success_rate >= 0.8:
            return 'status-good'
        elif success_rate >= 0.7:
            return 'status-warning'
        else:
            return 'status-poor'

    @staticmethod
    def get_status_icon(success_rate: float) -> str:
        """Get status icon HTML based on success rate"""
        if success_rate >= 0.8:
            return '<span style="color: #10b981; font-size: 24px;">✓</span>'
        elif success_rate >= 0.7:
            return '<span style="color: #f59e0b; font-size: 24px;">⚠</span>'
        else:
            return '<span style="color: #ef4444; font-size: 24px;">✗</span>'

    @staticmethod
    def generate_table_html(
        hierarchy_type: str,
        hierarchical_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate HTML table for hierarchy level

        Args:
            hierarchy_type: 'by_project', 'by_activity', or 'by_doc_type'
            hierarchical_metrics: Calculated metrics for this level

        Returns:
            HTML string for the table
        """
        if hierarchy_type not in hierarchical_metrics:
            return "<p>No data available</p>"

        data = hierarchical_metrics[hierarchy_type]
        if not data:
            return "<p>No data available</p>"

        rows = []
        for name, metrics in sorted(data.items()):
            success = metrics.get('success_rate', 0)
            precision = metrics.get('average_precision', 0)
            status_class = HierarchicalHTMLGenerator.get_status_class(success)
            status_icon = HierarchicalHTMLGenerator.get_status_icon(success)

            rows.append(f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{metrics.get('total_questions', 0)}</td>
                    <td class="{status_class}">{success*100:.2f}%</td>
                    <td class="{status_class}">{precision*100:.2f}%</td>
                    <td>{status_icon}</td>
                </tr>
            """)

        column_name = HierarchyExtractor.get_hierarchy_display_name(hierarchy_type)

        return f"""
            <table>
                <thead>
                    <tr>
                        <th>{column_name}</th>
                        <th>Questions</th>
                        <th>Success Rate</th>
                        <th>Avg Precision</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """

    @staticmethod
    def generate_hierarchy_buttons() -> str:
        """Generate HTML for hierarchy selector buttons"""
        buttons = []
        for type_key, label in HierarchyExtractor.get_hierarchy_labels():
            active_class = 'active' if type_key == 'by_project' else ''
            onclick_param = 'project' if type_key == 'by_project' else (
                'activity' if type_key == 'by_activity' else 'doc_type'
            )
            buttons.append(
                f'<button class="hierarchy-btn {active_class}" '
                f'onclick="switchHierarchy(\'{onclick_param}\')">{label}</button>'
            )

        return '\n'.join(buttons)

    @staticmethod
    def generate_hierarchy_content_divs(
        hierarchical_metrics: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate HTML content divs for all hierarchy levels"""
        divs = []

        hierarchy_info = [
            ('by_project', 'project', 'Project-Level Metrics'),
            ('by_activity', 'activity', 'Activity-Level Metrics'),
            ('by_doc_type', 'doc_type', 'Doc Type-Level Metrics')
        ]

        for hierarchy_type, div_id, title in hierarchy_info:
            active_class = 'active' if hierarchy_type == 'by_project' else ''
            table_html = HierarchicalHTMLGenerator.generate_table_html(
                hierarchy_type, hierarchical_metrics
            )

            divs.append(f"""
            <div id="{div_id}-content" class="hierarchy-content {active_class}">
                <h4 style="margin-bottom: 15px; color: #1f2937;">{title}</h4>
                {table_html}
            </div>
            """)

        return '\n'.join(divs)

    @staticmethod
    def get_css_styles() -> str:
        """Get CSS styles for hierarchy UI"""
        return """
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
        """

    @staticmethod
    def get_javascript_functions() -> str:
        """Get JavaScript function for hierarchy switching"""
        return """
        // Switch hierarchy view
        function switchHierarchy(hierarchyType) {
            // Hide all content
            document.getElementById('project-content').classList.remove('active');
            document.getElementById('activity-content').classList.remove('active');
            document.getElementById('doc_type-content').classList.remove('active');

            // Remove active class from all buttons
            document.querySelectorAll('.hierarchy-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected content
            const contentId = hierarchyType === 'project' ? 'project-content' :
                            hierarchyType === 'activity' ? 'activity-content' :
                            'doc_type-content';
            document.getElementById(contentId).classList.add('active');

            // Add active class to clicked button
            event.target.classList.add('active');
        }
        """
