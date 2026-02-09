"""
Streamlit Dashboard for RAG Retrieval Evaluation
Simple and clean visualization
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, Any
import sys
import os
import math

class StreamlitDashboard:
    """Streamlit dashboard generator"""
    
    def __init__(self, evaluation_results: Dict[str, Any]):
        self.results = evaluation_results
        self.metrics = evaluation_results.get('aggregate_metrics', {})
        self.config = evaluation_results.get('configuration', {})
        self.detailed = evaluation_results.get('detailed_results', [])
        self.hierarchical = evaluation_results.get('hierarchical_metrics', {})

    def context_recall(self) -> float:
        total_expected = sum(item.get("expected_count", 0) for item in self.detailed)
        total_retrieved = sum(len(item.get("retrieved_expected_docs", [])) for item in self.detailed)
        return round((total_retrieved / total_expected * 100) if total_expected > 0 else 0, 2)

    def mean_reciprocal_rank(self) -> float:
        total_rr = 0
        count = 0
        for item in self.detailed:
            if item.get("test_type") == "no_doc":
                continue
            positions = item.get("positions_found", [])
            if positions:
                total_rr += 1.0 / positions[0]
            count += 1
        return round(total_rr / count if count > 0 else 0, 4)

    def average_first_rank(self) -> float:
        """Calculate average rank of first relevant document found"""
        total_rank = 0.0
        count = 0

        for item in self.detailed:
            if item.get("test_type") == "no_doc":
                continue

            positions_found = item.get("positions_found", [])
            if positions_found:
                first_position = positions_found[0]
                total_rank += first_position
                count += 1

        return round(total_rank / count, 2) if count > 0 else 0.0

    def questions_with_relevant_docs(self) -> tuple:
        """Count questions that have relevant documents"""
        questions_with_relevant = 0
        total_questions = 0

        for item in self.detailed:
            if item.get("test_type") == "no_doc":
                continue

            total_questions += 1
            positions_found = item.get("positions_found", [])
            if positions_found:
                questions_with_relevant += 1

        return questions_with_relevant, total_questions

    def calculate_ndcg(self, item: Dict[str, Any], k: int = None) -> float:
        """
        Calculate NDCG@K for a single query result
        NDCG = DCG / IDCG where:
        - DCG = sum(1 / log2(position + 1)) for each relevant doc at its position
        - IDCG = ideal DCG (all relevant docs at top positions)
        """
        positions_found = item.get("positions_found", [])
        expected_count = item.get("expected_count", 0)
        
        if k is None:
            k = self.config.get('k', 10)

        # Calculate DCG - sum relevance / log2(position + 1) for each found document
        dcg = 0.0
        for pos in positions_found:
            if 1 <= pos <= k:
                dcg += 1.0 / math.log2(pos + 1)

        # Calculate IDCG - ideal: all relevant docs at top positions
        idcg = 0.0
        for i in range(min(expected_count, k)):
            idcg += 1.0 / math.log2(i + 2)  # i+2 because position starts at 1

        # Calculate NDCG
        if idcg == 0:
            return 0.0

        return round(min(dcg / idcg, 1.0), 4)

    def _mean_ndcg(self) -> float:
        """Calculate Mean NDCG across all queries"""
        total_ndcg = 0.0
        count = 0
        
        for item in self.detailed:
            if item.get("test_type") == "no_doc":
                continue
            
            ndcg = self.calculate_ndcg(item)
            total_ndcg += ndcg
            count += 1
        
        return round(total_ndcg / count if count > 0 else 0, 4)

    def render(self):
        st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
        
        # Custom CSS for better expander display and formula wrapping
        st.markdown("""
        <style>
        /* Make expander content area wider and scrollable */
        .streamlit-expanderContent {
            max-width: 100%;
            overflow-x: auto;
        }
        
        /* Better spacing for expander content */
        div[data-testid="stExpander"] {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-top: 8px;
        }
        
        /* Improve formula display */
        .stMarkdown p {
            overflow-x: auto;
            max-width: 100%;
        }
        
        /* Make metrics more compact */
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("📊 RAG Retrieval Evaluation Dashboard")
        st.markdown("---")
        
        # Total questions info only
        overall = self.metrics.get('overall', {})
        st.metric("Total Questions", overall.get('total_retrieval_questions', 0))
        
        st.markdown("---")
        
        # Summary metrics
        no_doc = self.metrics.get('no_doc', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Success Rate", f"{overall.get('success_rate', 0)*100:.1f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Overall Success Rate**")
                st.write("Percentage of questions that successfully found the expected documents.")
                st.write("")
                st.latex(r"\text{Success Rate} = \frac{\text{Successful Questions}}{\text{Total Questions}} \times 100\%")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>80%**: Excellent")
                st.write("- **60-80%**: Good")
                st.write("- **<60%**: Needs improvement")
            
            st.metric("Context Recall", f"{self.context_recall():.1f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Context Recall**")
                st.write("Percentage of relevant documents successfully found out of the total that should have been retrieved.")
                st.write("")
                st.latex(r"\text{Recall} = \frac{\text{Relevant Docs Found}}{\text{Total Relevant Docs}} \times 100\%")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>90%**: Excellent")
                st.write("- **70-90%**: Good")
                st.write("- **<70%**: Poor")
        
        with col2:
            st.metric("Average Precision", f"{overall.get('average_precision', 0)*100:.1f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Average Precision**")
                st.write("Measures the system's accuracy in retrieving correct documents.")
                st.write("")
                st.latex(r"\text{Precision} = \frac{\text{Correct Docs}}{\text{Total Docs Retrieved}} \times 100\%")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>80%**: High precision")
                st.write("- **60-80%**: Fair precision")
                st.write("- **<60%**: Low precision")
            
            mrr_val = self.mean_reciprocal_rank()
            avg_rank = self.average_first_rank()
            with_relevant = self.questions_with_relevant_docs()
            st.metric("MRR", f"{mrr_val*100:.1f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Mean Reciprocal Rank (MRR)**")
                st.write("Measures how quickly the system finds the first correct document.")
                st.write("")
                st.latex(r"\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}")
                st.write("")
                st.write(f"**Average Rank:** {avg_rank:.1f}")
                st.write(f"**Questions with Relevant Docs:** {with_relevant[0]}/{with_relevant[1]}")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>80%**: Excellent (docs usually in top 1-2)")
                st.write("- **50-80%**: Good (docs in top 3-5)")
                st.write("- **<50%**: Poor (docs ranked too low)")
        
        with col3:
            st.metric("Precision@3", f"{overall.get('precision_at_3', 0)*100:.1f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Precision@3**")
                st.write("Percentage of questions that find at least 1 correct document in the top 3 results.")
                st.write("")
                st.latex(r"\text{P@3} = \frac{\text{Questions with doc in top-3}}{\text{Total Questions}} \times 100\%")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>90%**: Excellent")
                st.write("- **70-90%**: Good")
                st.write("- **<70%**: Poor")
                st.write("")
                st.write("**Why it matters?** Users typically only focus on the top 3 results.")
            
            st.metric("No-Doc Detection", f"{no_doc.get('correct_rejection_rate', 0)*100:.1f}%")
            with st.expander("ℹ️ Info"):
                st.write("**No-Doc Detection**")
                st.write("The system's ability to recognize questions that cannot be answered from available documents.")
                st.write("")
                st.latex(r"\text{Rejection Rate} = \frac{\text{Correctly Rejected}}{\text{Total Irrelevant}} \times 100\%")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>90%**: Excellent")
                st.write("- **70-90%**: Good")
                st.write("- **<70%**: Poor")
                st.write("")
                st.write("**Why it matters?** Prevents the system from giving irrelevant or incorrect answers.")
        
        with col4:
            st.metric("NDCG", f"{self._mean_ndcg()*100:.1f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Normalized Discounted Cumulative Gain (NDCG)**")
                st.write("Measures ranking quality by considering both document position and relevance.")
                st.write("")
                st.latex(r"\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}")
                st.write("")
                st.write("- **DCG**: Score with penalty for lower positions")
                st.write("- **IDCG**: Ideal score (all relevant docs at the top)")
                st.write("")
                st.write("**How it works:**")
                st.write("- Document at rank 1 gets full score")
                st.write("- Documents at ranks 2-3 get reduced scores")
                st.write("- Documents at ranks 4+ get significantly lower scores")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>80%**: Excellent ranking")
                st.write("- **60-80%**: Good ranking")
                st.write("- **<60%**: Poor ranking")
        
        st.markdown("---")
        
        # Performance by test type
        st.subheader("Performance by Test Scenario")
        test_types = ['single_doc', 'multi_doc', 'conflicting_docs', 'long_queries']
        
        df_performance = pd.DataFrame([
            {
                'Test Type': t.replace('_', ' ').title(),
                'Success Rate (%)': self.metrics.get(t, {}).get('success_rate', 0) * 100,
                'Avg Precision (%)': self.metrics.get(t, {}).get('average_precision', 0) * 100
            }
            for t in test_types
        ])
        
        fig = px.bar(df_performance, x='Test Type', y=['Success Rate (%)', 'Avg Precision (%)'],
                     barmode='group', title="Performance Comparison across Scenarios")
        st.plotly_chart(fig, use_container_width=True)
        
        # Position distribution
        st.subheader("Position Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = self._position_chart('single_doc', 'Single Document')
            st.plotly_chart(fig1, use_container_width=True)
            
            fig3 = self._position_chart('conflicting_docs', 'Conflicting Documents')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig2 = self._position_chart('multi_doc', 'Multi-Document')
            st.plotly_chart(fig2, use_container_width=True)
            
            fig4 = self._position_chart('long_queries', 'Long Queries')
            st.plotly_chart(fig4, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Performance Metrics")
        df_metrics = self._create_metrics_table()
        st.dataframe(df_metrics, use_container_width=True)
        
        # Hierarchical metrics
        st.subheader("Hierarchical Metrics")
        tab1, tab2, tab3 = st.tabs(["By Project", "By Activity", "By Doc Type"])
        
        with tab1:
            df_project = self._create_hierarchical_df('by_project')
            if not df_project.empty:
                st.dataframe(df_project, use_container_width=True)
            else:
                st.info("No project-level metrics available")
        
        with tab2:
            df_activity = self._create_hierarchical_df('by_activity')
            if not df_activity.empty:
                st.dataframe(df_activity, use_container_width=True)
            else:
                st.info("No activity-level metrics available")
        
        with tab3:
            df_doc_type = self._create_hierarchical_df('by_doc_type')
            if not df_doc_type.empty:
                st.dataframe(df_doc_type, use_container_width=True)
            else:
                st.info("No doc-type-level metrics available")

    def _position_chart(self, test_type: str, title: str):
        position_dist = self.metrics.get(test_type, {}).get('position_distribution', {})
        k = self.config.get('k', 10)
        
        positions = list(range(1, k + 1))
        counts = [position_dist.get(str(pos), 0) for pos in positions]
        
        # Find the maximum count and create color list
        max_count = max(counts) if counts else 0
        colors = ['#FF4B4B' if count == max_count and count > 0 else '#1f77b4' for count in counts]
        
        fig = go.Figure(data=[go.Bar(
            x=positions, 
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
        )])
        fig.update_layout(
            title=title, 
            xaxis_title="Rank Position", 
            yaxis_title="Number of Queries",
            showlegend=False
        )
        return fig

    def _create_metrics_table(self) -> pd.DataFrame:
        test_types = [
            ('single_doc', 'Single Document'),
            ('multi_doc', 'Multi-Document'),
            ('conflicting_docs', 'Conflicting Docs'),
            ('long_queries', 'Long Queries'),
            ('no_doc', 'No-Document')
        ]
        
        data = []
        for test_type, label in test_types:
            if test_type not in self.metrics:
                continue
            
            m = self.metrics[test_type]
            data.append({
                'Test Scenario': label,
                'Total Questions': m.get('total_questions', 0),
                'Success Rate (%)': f"{m.get('success_rate', 0)*100:.1f}",
                'Avg Precision (%)': f"{m.get('average_precision', 0)*100:.1f}",
                'Precision@3 (%)': f"{m.get('precision_at_3', 0)*100:.1f}"
            })
        
        return pd.DataFrame(data)

    def _create_hierarchical_df(self, hierarchy_type: str) -> pd.DataFrame:
        data = self.hierarchical.get(hierarchy_type, {})
        if not data:
            return pd.DataFrame()
        
        rows = []
        for key, metrics in data.items():
            rows.append({
                'Category': key,
                'Total Questions': metrics.get('total_questions', 0),
                'Success Rate (%)': f"{metrics.get('success_rate', 0)*100:.1f}",
                'Avg Precision (%)': f"{metrics.get('average_precision', 0)*100:.1f}"
            })
        
        return pd.DataFrame(rows)


def load_default_results():
    """Load default evaluation results file"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import EVALUATION_RESULTS_PATH
        default_path = EVALUATION_RESULTS_PATH
    except:
        possible_paths = [
            "output/evaluation_results_new.json",
            "evaluation_results_new.json",
            "results/evaluation_results_new.json",
            "../evaluation_results_new.json",
            "data/evaluation_results_new.json"
        ]
        default_path = None
        for path in possible_paths:
            if os.path.exists(path):
                default_path = path
                break
    
    if default_path and os.path.exists(default_path):
        with open(default_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None

class ConfluenceDashboard:
    """Special Dashboard for Confluence - LM"""
    
    def __init__(self, evaluation_results: Dict[str, Any]):
        self.results = evaluation_results
        self.summary = evaluation_results.get('summary', {})
        self.total_questions = evaluation_results.get('total_questions', 0)
        self.detailed = evaluation_results.get('detailed_results', [])
    
    def _get_position_distribution(self) -> Dict[int, int]:
        """Calculate the distribution of positions for correct chunks found"""
        position_counts = {}
        
        for item in self.detailed:
            positions_found = item.get('positions_found', [])
            for pos in positions_found:
                position_counts[pos] = position_counts.get(pos, 0) + 1
        
        return position_counts
    
    def render(self):
        st.set_page_config(page_title="Confluence Evaluation Dashboard", layout="wide")
        
        # Custom CSS for better expander display
        st.markdown("""
        <style>
        .streamlit-expanderContent {
            max-width: 100%;
            overflow-x: auto;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-top: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("📊 Confluence - LM Evaluation Dashboard")
        st.markdown("---")
        
        # Total questions
        st.metric("Total Questions", self.total_questions)
        st.markdown("---")
        
        # 4 Summary metrics only
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Success Rate", f"{self.summary.get('success_rate', 0):.2f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Success Rate**")
                st.write("Percentage of questions that successfully found the expected documents.")
                st.write("")
                st.latex(r"\text{Success Rate} = \frac{\text{Successful Questions}}{\text{Total Questions}} \times 100\%")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>80%**: Excellent")
                st.write("- **60-80%**: Good")
                st.write("- **<60%**: Needs improvement")
        
        with col2:
            st.metric("Average Precision", f"{self.summary.get('avg_precision', 0):.2f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Average Precision**")
                st.write("Measures the system's accuracy in retrieving correct documents.")
                st.write("")
                st.latex(r"\text{Precision} = \frac{\text{Correct Docs}}{\text{Total Docs Retrieved}} \times 100\%")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>80%**: High precision")
                st.write("- **60-80%**: Fair precision")
                st.write("- **<60%**: Low precision")
        
        with col3:
            st.metric("Precision@3", f"{self.summary.get('avg_precision_at_3', 0):.2f}%")
            with st.expander("ℹ️ Info"):
                st.write("**Precision@3**")
                st.write("Percentage of questions that find at least 1 correct document in the top 3 results.")
                st.write("")
                st.latex(r"\text{P@3} = \frac{\text{Questions with doc in top-3}}{\text{Total Questions}} \times 100\%")
                st.write("")
                st.write("**Why it matters?** Users typically only focus on the top 3 results.")
        
        with col4:
            # Convert MRR to percentage if it's in decimal format (0-1 range)
            mrr_value = self.summary.get('mrr', 0)
            if mrr_value <= 1.0:
                mrr_display = f"{mrr_value * 100:.2f}%"
            else:
                mrr_display = f"{mrr_value:.2f}%"
            
            st.metric("MRR", mrr_display)
            with st.expander("ℹ️ Info"):
                st.write("**Mean Reciprocal Rank (MRR)**")
                st.write("Measures how quickly the system finds the first correct document.")
                st.write("")
                st.latex(r"\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}")
                st.write("")
                st.write("**Interpretation:**")
                st.write("- **>80%**: Excellent")
                st.write("- **50-80%**: Good")
                st.write("- **<50%**: Poor")
        
        st.markdown("---")
        
        # Position Distribution Chart
        st.subheader("Position Distribution of Correct Chunks")
        
        position_dist = self._get_position_distribution()
        if position_dist:
            positions = sorted(position_dist.keys())
            counts = [position_dist[pos] for pos in positions]
            
            # Find the maximum count and create color list
            max_count = max(counts) if counts else 0
            colors = ['#FF4B4B' if count == max_count and count > 0 else '#1f77b4' for count in counts]
            
            fig = go.Figure(data=[go.Bar(
                x=positions, 
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='auto',
            )])
            fig.update_layout(
                title="Distribution of Positions Where Correct Chunks Were Retrieved",
                xaxis_title="Rank Position",
                yaxis_title="Frequency",
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No position data available")

def load_confluence_results(version: str, rerank_option: str):
    """Load Confluence evaluation results based on version and rerank option"""
    rerank_suffix = "with_rerank" if rerank_option == "With Rerank" else "without_rerank"
    filename = f"evaluation_report_{version.lower()}_{rerank_suffix}.json"
    
    possible_paths = [
        filename,
        f"output/{filename}",
        f"results/{filename}",
        f"../{filename}",
        f"data/{filename}"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    return None

def render_overall_dashboard():
    """Render overall dashboard showing combined metrics from all data sources"""
    st.set_page_config(page_title="Overall Evaluation Dashboard", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .streamlit-expanderContent {
        max-width: 100%;
        overflow-x: auto;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📊 Overall Evaluation Dashboard")
    st.markdown("**Aggregated metrics across all evaluated data sources**")
    st.markdown("---")
    
    # Load all available data sources
    data_sources = {}
    
    # 1. Try to load MPM - ORP
    try:
        mpm_orp = load_default_results()
        if mpm_orp:
            data_sources['MPM - ORP'] = {
                'type': 'mpm_orp',
                'data': mpm_orp
            }
    except:
        pass
    
    # 2. Try to load all Confluence - LM variants (R4 and R5, with and without rerank)
    confluence_variants = [
        ('R4', 'Without Rerank', 'Confluence - LM (R4 w/o Rerank)'),
        ('R4', 'With Rerank', 'Confluence - LM (R4 w/ Rerank)'),
        ('R5', 'Without Rerank', 'Confluence - LM (R5 w/o Rerank)'),
        ('R5', 'With Rerank', 'Confluence - LM (R5 w/ Rerank)')
    ]
    
    for version, rerank_option, display_name in confluence_variants:
        try:
            confluence = load_confluence_results(version, rerank_option)
            if confluence:
                data_sources[display_name] = {
                    'type': 'confluence',
                    'data': confluence
                }
        except:
            pass
    
    # Check if we have any data
    if not data_sources:
        st.warning("⚠️ No evaluation data found. Please ensure evaluation results are available.")
        return
    
    # Display loaded data sources
    st.info(f"📁 **Loaded {len(data_sources)} data source(s):** {', '.join(data_sources.keys())}")
    st.markdown("---")
    
    # Calculate overall metrics using weighted average
    total_questions = 0
    total_successful = 0
    weighted_precision_sum = 0
    weighted_mrr_sum = 0
    total_top3_found = 0
    
    # Store individual metrics for breakdown table
    breakdown_data = []
    
    for source_name, source_info in data_sources.items():
        source_type = source_info['type']
        data = source_info['data']
        
        if source_type == 'mpm_orp':
            # Extract from MPM-ORP format
            overall = data.get('aggregate_metrics', {}).get('overall', {})
            questions = overall.get('total_retrieval_questions', 0)
            success_rate = overall.get('success_rate', 0)
            precision = overall.get('average_precision', 0)
            precision_3 = overall.get('precision_at_3', 0)
            
            # Calculate MRR from detailed results
            detailed = data.get('detailed_results', [])
            total_rr = 0
            count = 0
            for item in detailed:
                if item.get("test_type") == "no_doc":
                    continue
                positions = item.get("positions_found", [])
                if positions:
                    total_rr += 1.0 / positions[0]
                count += 1
            mrr = total_rr / count if count > 0 else 0
            
        elif source_type == 'confluence':
            # Extract from Confluence format
            summary = data.get('summary', {})
            questions = data.get('total_questions', 0)
            success_rate = summary.get('success_rate', 0) / 100  # Convert to decimal
            precision = summary.get('avg_precision', 0) / 100  # Convert to decimal
            precision_3 = summary.get('avg_precision_at_3', 0) / 100  # Convert to decimal
            mrr = summary.get('mrr', 0)
            if mrr > 1:  # If already in percentage
                mrr = mrr / 100
        
        # Accumulate for weighted average
        total_questions += questions
        total_successful += questions * success_rate
        weighted_precision_sum += questions * precision
        weighted_mrr_sum += questions * mrr
        total_top3_found += questions * precision_3
        
        # Store for breakdown
        breakdown_data.append({
            'Data Source': source_name,
            'Total Questions': questions,
            'Success Rate (%)': success_rate * 100,
            'Avg Precision (%)': precision * 100,
            'Precision@3 (%)': precision_3 * 100,
            'MRR (%)': mrr * 100
        })
    
    # Calculate overall weighted averages
    overall_success_rate = (total_successful / total_questions * 100) if total_questions > 0 else 0
    overall_precision = (weighted_precision_sum / total_questions * 100) if total_questions > 0 else 0
    overall_precision_3 = (total_top3_found / total_questions * 100) if total_questions > 0 else 0
    overall_mrr = (weighted_mrr_sum / total_questions * 100) if total_questions > 0 else 0
    
    # Display total questions
    st.metric("Total Questions (All Sources)", total_questions)
    st.markdown("---")
    
    # Display overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Success Rate", f"{overall_success_rate:.1f}%")
        with st.expander("ℹ️ Info"):
            st.write("**Overall Success Rate**")
            st.write("Weighted average of success rates across all data sources.")
            st.write("")
            st.write("**Formula:**")
            st.write("Σ(Success Rate × Questions) / Total Questions")
            st.write("")
            st.write("**Interpretation:**")
            st.write("- **>80%**: Excellent")
            st.write("- **60-80%**: Good")
            st.write("- **<60%**: Needs improvement")
    
    with col2:
        st.metric("Overall Precision", f"{overall_precision:.1f}%")
        with st.expander("ℹ️ Info"):
            st.write("**Overall Average Precision**")
            st.write("Weighted average of precision across all data sources.")
            st.write("")
            st.write("**Formula:**")
            st.write("Σ(Precision × Questions) / Total Questions")
            st.write("")
            st.write("**Interpretation:**")
            st.write("- **>80%**: High precision")
            st.write("- **60-80%**: Fair precision")
            st.write("- **<60%**: Low precision")
    
    with col3:
        st.metric("Overall Precision@3", f"{overall_precision_3:.1f}%")
        with st.expander("ℹ️ Info"):
            st.write("**Overall Precision@3**")
            st.write("Weighted average of Precision@3 across all data sources.")
            st.write("")
            st.write("**Formula:**")
            st.write("Σ(Precision@3 × Questions) / Total Questions")
            st.write("")
            st.write("**Interpretation:**")
            st.write("- **>90%**: Excellent")
            st.write("- **70-90%**: Good")
            st.write("- **<70%**: Poor")
    
    with col4:
        st.metric("Overall MRR", f"{overall_mrr:.1f}%")
        with st.expander("ℹ️ Info"):
            st.write("**Overall Mean Reciprocal Rank**")
            st.write("Weighted average of MRR across all data sources.")
            st.write("")
            st.write("**Formula:**")
            st.write("Σ(MRR × Questions) / Total Questions")
            st.write("")
            st.write("**Interpretation:**")
            st.write("- **>80%**: Excellent")
            st.write("- **50-80%**: Good")
            st.write("- **<50%**: Poor")
    
    st.markdown("---")
    
    # Breakdown by data source
    st.subheader("📋 Breakdown by Data Source")
    
    if breakdown_data:
        df_breakdown = pd.DataFrame(breakdown_data)
        
        # Apply styling to highlight max/min values
        def highlight_max_min(col):
            if col.name in ['Success Rate (%)', 'Avg Precision (%)', 'Precision@3 (%)', 'MRR (%)']:
                max_val = col.max()
                min_val = col.min()
                styles = []
                for val in col:
                    if val == max_val:
                        styles.append('color: #A5C89E; font-weight: bold')
                    elif val == min_val:
                        styles.append('color: #C40C0C; font-weight: bold')
                    else:
                        styles.append('')
                return styles
            return ['' for _ in col]
        
        styled_df = df_breakdown.style.apply(highlight_max_min)
        styled_df = styled_df.format({
            'Success Rate (%)': '{:.1f}',
            'Avg Precision (%)': '{:.1f}',
            'Precision@3 (%)': '{:.1f}',
            'MRR (%)': '{:.1f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Note:** Overall metrics are calculated using weighted average based on the number of questions in each data source. Only evaluated data sources are included in the calculation.")



def main():
    st.sidebar.title("Settings")
    
    # Dropdown to select data source
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Overall", "MPM - ORP", "MPM - LM", "Confluence - LM", "Jira - LM"]
    )
    
    # Handle Overall page
    if data_source == "Overall":
        render_overall_dashboard()
        return
    
    # Check if data is available
    if data_source == "MPM - LM":
        st.warning("⚠️ Data has not been evaluated yet")
        return
    elif data_source == "Jira - LM":
        st.warning("⚠️ Data has not been evaluated yet")
        return
    
    # For Confluence - LM, add R4/R5 and Rerank dropdowns
    if data_source == "Confluence - LM":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            version = st.sidebar.selectbox("Version", ["R4", "R5"])
        with col2:
            rerank_option = st.sidebar.selectbox("Rerank", ["Without Rerank", "With Rerank"])
        
        # Load Confluence results
        results = load_confluence_results(version, rerank_option)
        if not results:
            st.warning(f"⚠️ File not found: evaluation_report_{version.lower()}_{'with_rerank' if rerank_option == 'With Rerank' else 'without_rerank'}.json")
            return
        st.sidebar.success(f"✅ Loaded {version} {rerank_option}")
    else:
        # MPM - ORP (current condition)
        uploaded_file = st.sidebar.file_uploader("Upload evaluation results JSON (optional)", type=['json'])
        
        if uploaded_file:
            results = json.load(uploaded_file)
            st.sidebar.success("✅ Loaded from uploaded file")
        else:
            default_results = load_default_results()
            if default_results:
                results = default_results
                st.sidebar.info("📁 Loaded from default file")
            else:
                st.warning("⚠️ No evaluation results found. Please upload a JSON file from the sidebar.")
                st.info("Expected file location: `evaluation_results_new.json`")
                return
    
    # Render dashboard
    if data_source == "Confluence - LM":
        dashboard = ConfluenceDashboard(results)
    else:
        dashboard = StreamlitDashboard(results)
    
    dashboard.render()


if __name__ == "__main__":
    main()