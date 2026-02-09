"""
Streamlit Dashboard for RAG Retrieval Evaluation
Interactive dashboard with advanced visualizations and filtering capabilities
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import math
from typing import Dict, Any, List, Tuple


# Page configuration
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitDashboard:
    """Streamlit dashboard for RAG evaluation results"""
    
    def __init__(self, evaluation_results: Dict[str, Any]):
        """Initialize dashboard with evaluation results"""
        self.results = evaluation_results
        self.metrics = evaluation_results.get('aggregate_metrics', {})
        self.config = evaluation_results.get('configuration', {})
        self.detailed = evaluation_results.get('detailed_results', [])
        self.hierarchical_metrics = evaluation_results.get('hierarchical_metrics', {
            'by_project': {},
            'by_activity': {},
            'by_doc_type': {}
        })
    
    def calculate_context_recall(self) -> float:
        """Calculate context recall percentage"""
        total_expected_count = 0
        total_retrieved_expected = 0

        for item in self.detailed:
            total_expected_count += item.get("expected_count", 0)
            retrieved_docs = item.get("retrieved_expected_docs", [])
            if retrieved_docs:
                total_retrieved_expected += len(retrieved_docs)

        if total_expected_count == 0:
            return 0.0
        return round((total_retrieved_expected / total_expected_count) * 100, 2)
    
    def calculate_mrr(self) -> Tuple[float, float, int, int]:
        """Calculate Mean Reciprocal Rank and related metrics"""
        total_reciprocal_rank = 0.0
        total_rank = 0.0
        count = 0
        questions_with_relevant = 0

        for item in self.detailed:
            if item.get("test_type") == "no_doc":
                continue

            positions_found = item.get("positions_found", [])
            count += 1
            
            if positions_found:
                first_position = positions_found[0]
                total_reciprocal_rank += 1.0 / first_position
                total_rank += first_position
                questions_with_relevant += 1

        mrr = round(total_reciprocal_rank / count, 4) if count > 0 else 0.0
        avg_rank = round(total_rank / questions_with_relevant, 2) if questions_with_relevant > 0 else 0.0
        
        return mrr, avg_rank, questions_with_relevant, count
    
    def calculate_ndcg(self, item: Dict[str, Any], k: int = None) -> float:
        """Calculate NDCG for a single query result"""
        positions_found = item.get("positions_found", [])
        retrieved_count = item.get("retrieved_count", 0)
        expected_count = item.get("expected_count", 0)

        if k is None:
            k = retrieved_count if retrieved_count > 0 else 10

        relevance = [0] * k
        for pos in positions_found:
            if 1 <= pos <= k:
                relevance[pos - 1] = 1

        dcg = 0.0
        relevant_count = 0
        for i, rel in enumerate(relevance):
            if rel > 0 and relevant_count < expected_count:
                dcg += 1.0 / math.log2(i + 2)
                relevant_count += 1

        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(expected_count, k)))

        if idcg == 0:
            return 0.0
        return round(min(dcg / idcg, 1.0), 4)
    
    def calculate_mean_ndcg(self) -> float:
        """Calculate mean NDCG across all queries"""
        total_ndcg = 0.0
        count = 0

        for item in self.detailed:
            if item.get("test_type") == "no_doc":
                continue
            total_ndcg += self.calculate_ndcg(item)
            count += 1

        return round(total_ndcg / count, 4) if count > 0 else 0.0
    
    def get_detailed_dataframe(self) -> pd.DataFrame:
        """Convert detailed results to DataFrame"""
        data = []
        for item in self.detailed:
            data.append({
                'Query': item.get('query', 'N/A')[:100] + '...' if len(item.get('query', 'N/A')) > 100 else item.get('query', 'N/A'),
                'Test Type': item.get('test_type', 'N/A').replace('_', ' ').title(),
                'Project': item.get('project', 'N/A'),
                'Activity': item.get('activity', 'N/A'),
                'Doc Type': item.get('doc_type', 'N/A'),
                'Success': '✓' if item.get('success', False) else '✗',
                'Precision': f"{item.get('precision', 0)*100:.1f}%",
                'Positions Found': ', '.join(map(str, item.get('positions_found', []))) if item.get('positions_found') else 'None',
                'Expected Count': item.get('expected_count', 0),
                'Retrieved Count': item.get('retrieved_count', 0),
                'NDCG': f"{self.calculate_ndcg(item)*100:.1f}%"
            })
        return pd.DataFrame(data)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 RAG Retrieval Evaluation Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Comprehensive Analysis of Retrieval System Performance")
        
        # Configuration info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Top-K:** {self.config.get('k', 10)}")
        with col2:
            st.info(f"**Confidence Threshold:** {self.config.get('confidence_threshold', 0.7)}")
        with col3:
            st.info(f"**Total Questions:** {self.metrics.get('overall', {}).get('total_retrieval_questions', 0)}")
        with col4:
            st.info(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    def render_key_metrics(self):
        """Render key performance metrics"""
        st.markdown("## 🎯 Key Performance Metrics")
        
        overall = self.metrics.get('overall', {})
        no_doc = self.metrics.get('no_doc', {})
        
        # Calculate additional metrics
        context_recall = self.calculate_context_recall()
        mrr, avg_rank, questions_with_relevant, total_questions = self.calculate_mrr()
        mean_ndcg = self.calculate_mean_ndcg()
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Overall Success Rate",
                value=f"{overall.get('success_rate', 0)*100:.1f}%",
                help="Percentage of questions that successfully found their expected documents"
            )
            st.metric(
                label="Average Precision",
                value=f"{overall.get('average_precision', 0)*100:.1f}%",
                help="Percentage of correct chunks in top-K results"
            )
        
        with col2:
            st.metric(
                label="Precision@3",
                value=f"{overall.get('precision_at_3', 0)*100:.1f}%",
                help="Percentage of queries with correct doc in top-3"
            )
            st.metric(
                label="No-Doc Detection",
                value=f"{no_doc.get('correct_rejection_rate', 0)*100:.1f}%",
                help="Ability to correctly reject unanswerable questions"
            )
        
        with col3:
            st.metric(
                label="Context Recall",
                value=f"{context_recall:.2f}%",
                help="Percentage of expected documents successfully retrieved"
            )
            st.metric(
                label="Mean Reciprocal Rank",
                value=f"{mrr*100:.2f}%",
                help=f"Avg. first position: {avg_rank:.1f} | Questions with docs: {questions_with_relevant}/{total_questions}"
            )
        
        with col4:
            st.metric(
                label="NDCG Score",
                value=f"{mean_ndcg*100:.2f}%",
                help="Normalized Discounted Cumulative Gain - ranking quality metric"
            )
            st.metric(
                label="Questions with Relevant",
                value=f"{questions_with_relevant}/{total_questions}",
                help="Number of questions that found at least one relevant document"
            )
    
    def render_performance_comparison(self):
        """Render performance comparison across test types"""
        st.markdown("## 📈 Performance by Test Type")
        
        test_types = ['single_doc', 'multi_doc', 'conflicting_docs', 'long_queries']
        labels = [t.replace('_', ' ').title() for t in test_types]
        
        success_rates = [self.metrics.get(t, {}).get('success_rate', 0) * 100 for t in test_types]
        precisions = [self.metrics.get(t, {}).get('average_precision', 0) * 100 for t in test_types]
        precision_at_3 = [self.metrics.get(t, {}).get('precision_at_3', 0) * 100 for t in test_types]
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Success Rate',
            x=labels,
            y=success_rates,
            marker_color='#667eea',
            text=[f'{v:.1f}%' for v in success_rates],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Average Precision',
            x=labels,
            y=precisions,
            marker_color='#764ba2',
            text=[f'{v:.1f}%' for v in precisions],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Precision@3',
            x=labels,
            y=precision_at_3,
            marker_color='#f59e0b',
            text=[f'{v:.1f}%' for v in precision_at_3],
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            height=500,
            xaxis_title="Test Type",
            yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, 110]),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_position_distribution(self):
        """Render position distribution charts"""
        st.markdown("## 📍 Position Distribution Analysis")
        st.markdown("*Shows at which positions (ranks) the correct documents were found*")
        
        test_types = [
            ('single_doc', 'Single Document Retrieval'),
            ('multi_doc', 'Multi-Document Synthesis'),
            ('conflicting_docs', 'Conflicting Documents'),
            ('long_queries', 'Long Queries')
        ]
        
        col1, col2 = st.columns(2)
        
        for idx, (test_type, title) in enumerate(test_types):
            position_dist = self.metrics.get(test_type, {}).get('position_distribution', {})
            
            if not position_dist:
                continue
            
            positions = list(range(1, self.config.get('k', 10) + 1))
            counts = [position_dist.get(str(pos), 0) for pos in positions]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Pos {i}" for i in positions],
                    y=counts,
                    marker_color='#667eea',
                    text=counts,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Position",
                yaxis_title="Count",
                height=350,
                showlegend=False
            )
            
            if idx % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_hierarchical_metrics(self):
        """Render hierarchical metrics analysis"""
        st.markdown("## 🔍 Hierarchical Performance Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📁 By Project", "🎯 By Activity", "📄 By Document Type"])
        
        with tab1:
            self._render_hierarchy_table("by_project", "Project")
        
        with tab2:
            self._render_hierarchy_table("by_activity", "Activity")
        
        with tab3:
            self._render_hierarchy_table("by_doc_type", "Document Type")
    
    def _render_hierarchy_table(self, hierarchy_type: str, label: str):
        """Render hierarchical metrics table"""
        metrics = self.hierarchical_metrics.get(hierarchy_type, {})
        
        if not metrics:
            st.info(f"No {label.lower()} metrics available")
            return
        
        data = []
        for key, values in metrics.items():
            data.append({
                label: key,
                'Questions': values.get('total_questions', 0),
                'Success Rate': f"{values.get('success_rate', 0)*100:.1f}%",
                'Avg Precision': f"{values.get('average_precision', 0)*100:.1f}%",
                'Precision@3': f"{values.get('precision_at_3', 0)*100:.1f}%",
                'Found': values.get('total_found', 0),
                'Expected': values.get('total_expected', 0)
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visualization
        if len(df) > 0:
            fig = px.bar(
                df,
                x=label,
                y=[col for col in df.columns if '%' in col],
                title=f"Performance Metrics by {label}",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_results(self):
        """Render detailed test results"""
        st.markdown("## 📋 Detailed Test Results")
        
        df = self.get_detailed_dataframe()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_types = ['All'] + list(df['Test Type'].unique())
            selected_type = st.selectbox("Filter by Test Type", test_types)
        
        with col2:
            projects = ['All'] + list(df['Project'].unique())
            selected_project = st.selectbox("Filter by Project", projects)
        
        with col3:
            success_filter = st.selectbox("Filter by Success", ['All', 'Success ✓', 'Failed ✗'])
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['Test Type'] == selected_type]
        
        if selected_project != 'All':
            filtered_df = filtered_df[filtered_df['Project'] == selected_project]
        
        if success_filter == 'Success ✓':
            filtered_df = filtered_df[filtered_df['Success'] == '✓']
        elif success_filter == 'Failed ✗':
            filtered_df = filtered_df[filtered_df['Success'] == '✗']
        
        # Display filtered results
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filtered Results", len(filtered_df))
        with col2:
            success_count = len(filtered_df[filtered_df['Success'] == '✓'])
            st.metric("Successful", success_count)
        with col3:
            fail_count = len(filtered_df[filtered_df['Success'] == '✗'])
            st.metric("Failed", fail_count)
    
    def render_insights(self):
        """Render insights and recommendations"""
        st.markdown("## 💡 Key Insights & Recommendations")
        
        overall = self.metrics.get('overall', {})
        success_rate = overall.get('success_rate', 0)
        precision = overall.get('average_precision', 0)
        
        insights = []
        
        # Performance insights
        if success_rate >= 0.8:
            insights.append("✅ **Excellent Performance**: Success rate is above 80%. The system is performing well.")
        elif success_rate >= 0.6:
            insights.append("⚠️ **Good Performance**: Success rate is between 60-80%. There's room for improvement.")
        else:
            insights.append("❌ **Needs Improvement**: Success rate is below 60%. Consider reviewing retrieval strategies.")
        
        # Test type analysis
        test_types = ['single_doc', 'multi_doc', 'conflicting_docs', 'long_queries']
        weakest_type = min(test_types, key=lambda t: self.metrics.get(t, {}).get('success_rate', 0))
        weakest_rate = self.metrics.get(weakest_type, {}).get('success_rate', 0)
        
        insights.append(f"🎯 **Weakest Area**: {weakest_type.replace('_', ' ').title()} ({weakest_rate*100:.1f}%) needs attention.")
        
        # Position analysis
        mrr, avg_rank, _, _ = self.calculate_mrr()
        if avg_rank > 5:
            insights.append(f"📊 **Ranking Issue**: Average first position is {avg_rank:.1f}. Consider improving ranking algorithms.")
        
        # NDCG analysis
        mean_ndcg = self.calculate_mean_ndcg()
        if mean_ndcg < 0.7:
            insights.append(f"📉 **Ranking Quality**: NDCG is {mean_ndcg*100:.1f}%. Relevant documents are not ranked highly enough.")
        
        # Display insights
        for insight in insights:
            st.markdown(f"<div class='info-box'>{insight}</div>", unsafe_allow_html=True)
    
    def render_export_options(self):
        """Render export options"""
        st.markdown("## 💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export detailed results as CSV
            df = self.get_detailed_dataframe()
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Results (CSV)",
                data=csv,
                file_name=f"rag_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export full results as JSON
            json_str = json.dumps(self.results, indent=2)
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name=f"rag_evaluation_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def render(self):
        """Render complete dashboard"""
        self.render_header()
        st.markdown("---")
        
        self.render_key_metrics()
        st.markdown("---")
        
        self.render_performance_comparison()
        st.markdown("---")
        
        self.render_position_distribution()
        st.markdown("---")
        
        self.render_hierarchical_metrics()
        st.markdown("---")
        
        self.render_detailed_results()
        st.markdown("---")
        
        self.render_insights()
        st.markdown("---")
        
        self.render_export_options()


def main():
    """Main function to run the Streamlit dashboard"""
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Settings")
    st.sidebar.markdown("---")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Evaluation Results (JSON)",
        type=['json'],
        help="Upload your RAG evaluation results JSON file"
    )
    
    if uploaded_file is not None:
        try:
            # Load evaluation results
            evaluation_results = json.load(uploaded_file)
            
            st.sidebar.success("✅ File loaded successfully!")
            st.sidebar.markdown(f"**Total Questions:** {evaluation_results.get('aggregate_metrics', {}).get('overall', {}).get('total_retrieval_questions', 0)}")
            
            # Create and render dashboard
            dashboard = StreamlitDashboard(evaluation_results)
            dashboard.render()
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome screen
        st.markdown('<h1 class="main-header">📊 RAG Retrieval Evaluation Dashboard</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome! 👋
        
        This interactive dashboard helps you analyze and visualize RAG (Retrieval-Augmented Generation) evaluation results.
        
        #### Features:
        - 📈 **Performance Metrics**: Track success rates, precision, and ranking quality
        - 📍 **Position Analysis**: Understand where correct documents are found
        - 🔍 **Hierarchical Insights**: Analyze by project, activity, and document type
        - 📋 **Detailed Results**: Filter and explore individual test cases
        - 💡 **AI Insights**: Get recommendations for improvement
        - 💾 **Export Options**: Download results in CSV or JSON format
        
        #### Getting Started:
        1. Upload your evaluation results JSON file using the sidebar
        2. Explore different sections of the dashboard
        3. Use filters to drill down into specific areas
        4. Export results for further analysis
        
        ---
        
        **Ready to start?** Upload your evaluation results file in the sidebar! ⬅️
        """)
        
        # Example data structure
        with st.expander("📖 Expected JSON Structure"):
            st.code("""
{
  "aggregate_metrics": {
    "overall": {
      "success_rate": 0.85,
      "average_precision": 0.78,
      "precision_at_3": 0.82,
      ...
    },
    "single_doc": {...},
    "multi_doc": {...},
    ...
  },
  "detailed_results": [...],
  "hierarchical_metrics": {...},
  "configuration": {...}
}
            """, language="json")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 About
    This dashboard provides comprehensive analysis of RAG retrieval performance.
    
    **Version:** 2.0  
    **Last Updated:** Feb 2026
    """)


if __name__ == "__main__":
    main()