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
            with st.expander("ℹ️ What is Overall Success Rate?"):
                st.write("""
                **Overall Success Rate** adalah persentase dari total pertanyaan yang berhasil menemukan 
                dokumen yang diharapkan dalam hasil retrieval.
                
                **Formula:** (Jumlah pertanyaan berhasil / Total pertanyaan) × 100%
                
                **Interpretasi:**
                - **>80%**: Sangat baik - sistem retrieval sangat akurat
                - **60-80%**: Baik - performa memadai dengan ruang untuk perbaikan
                - **<60%**: Perlu perbaikan - banyak dokumen relevan tidak ditemukan
                """)
            
            st.metric("Context Recall", f"{self.context_recall():.1f}%")
            with st.expander("ℹ️ What is Context Recall?"):
                st.write("""
                **Context Recall** mengukur persentase dokumen yang diharapkan yang berhasil 
                di-retrieve dari semua dokumen yang seharusnya ditemukan.
                
                **Formula:** (Total dokumen relevan yang ditemukan / Total dokumen relevan yang diharapkan) × 100%
                
                **Interpretasi:**
                - **>90%**: Excellent - hampir semua dokumen relevan ditemukan
                - **70-90%**: Good - sebagian besar dokumen relevan ditemukan
                - **<70%**: Poor - banyak dokumen relevan yang terlewat
                """)
        
        with col2:
            st.metric("Average Precision", f"{overall.get('average_precision', 0)*100:.1f}%")
            with st.expander("ℹ️ What is Average Precision?"):
                st.write("""
                **Average Precision** mengukur seberapa akurat sistem dalam mengambil dokumen yang benar 
                dari semua dokumen yang di-retrieve.
                
                **Formula:** (Jumlah dokumen benar yang di-retrieve / Total dokumen yang di-retrieve) × 100%
                
                **Interpretasi:**
                - **>80%**: Sangat presisi - sedikit noise dalam hasil
                - **60-80%**: Cukup presisi - ada beberapa dokumen tidak relevan
                - **<60%**: Kurang presisi - banyak dokumen tidak relevan dalam hasil
                """)
            
            mrr_val = self.mean_reciprocal_rank()
            avg_rank = self.average_first_rank()
            with_relevant = self.questions_with_relevant_docs()
            st.metric(
                "MRR", 
                f"{mrr_val*100:.1f}%",
                delta=f"Avg rank: {avg_rank:.1f} | Found: {with_relevant[0]}/{with_relevant[1]}",
                delta_color="off"
            )
            with st.expander("ℹ️ What is MRR?"):
                st.write("""
                **Mean Reciprocal Rank (MRR)** mengukur seberapa cepat sistem menemukan 
                dokumen yang benar pertama kali.
                
                **Formula:** Average(1 / posisi dokumen relevan pertama)
                
                **Contoh:**
                - Jika dokumen relevan di posisi 1: MRR = 1/1 = 1.0 (100%)
                - Jika dokumen relevan di posisi 2: MRR = 1/2 = 0.5 (50%)
                - Jika dokumen relevan di posisi 5: MRR = 1/5 = 0.2 (20%)
                
                **Interpretasi:**
                - **>80%**: Excellent - dokumen relevan biasanya di top 1-2
                - **50-80%**: Good - dokumen relevan di top 3-5
                - **<50%**: Poor - dokumen relevan terlalu jauh di bawah
                
                **Avg rank** menunjukkan posisi rata-rata dokumen relevan pertama ditemukan.
                """)
        
        with col3:
            st.metric("Precision@3", f"{overall.get('precision_at_3', 0)*100:.1f}%")
            with st.expander("ℹ️ What is Precision@3?"):
                st.write("""
                **Precision@3** mengukur persentase pertanyaan yang berhasil menemukan 
                setidaknya satu dokumen yang benar dalam 3 hasil teratas.
                
                **Formula:** (Pertanyaan dengan dokumen benar di top-3 / Total pertanyaan) × 100%
                
                **Interpretasi:**
                - **>90%**: Excellent - hampir semua jawaban ada di top-3
                - **70-90%**: Good - sebagian besar jawaban di top-3
                - **<70%**: Poor - banyak jawaban tidak muncul di top-3
                
                **Kenapa penting?** User biasanya hanya melihat 3 hasil teratas.
                """)
            
            st.metric("No-Doc Detection", f"{no_doc.get('correct_rejection_rate', 0)*100:.1f}%")
            with st.expander("ℹ️ What is No-Doc Detection?"):
                st.write("""
                **No-Doc Detection** mengukur kemampuan sistem untuk mengenali pertanyaan 
                yang tidak bisa dijawab dari dokumen yang tersedia.
                
                **Formula:** (Pertanyaan tidak relevan yang benar ditolak / Total pertanyaan tidak relevan) × 100%
                
                **Interpretasi:**
                - **>90%**: Excellent - sangat baik mendeteksi pertanyaan di luar konteks
                - **70-90%**: Good - cukup baik menghindari jawaban yang salah
                - **<70%**: Poor - sering memberikan jawaban untuk pertanyaan yang tidak relevan
                
                **Kenapa penting?** Mencegah sistem memberikan jawaban yang salah atau tidak relevan.
                """)
        
        with col4:
            st.metric("NDCG", f"{self._mean_ndcg()*100:.1f}%")
            with st.expander("ℹ️ What is NDCG?"):
                st.write("""
                **Normalized Discounted Cumulative Gain (NDCG)** mengukur kualitas ranking 
                dengan mempertimbangkan posisi dan relevansi dokumen.
                
                **Formula:** DCG / IDCG
                - **DCG**: Skor kumulatif dengan penalty untuk posisi lebih rendah
                - **IDCG**: Skor ideal (semua dokumen relevan di posisi teratas)
                
                **Cara kerja:**
                - Dokumen relevan di posisi 1 mendapat skor penuh
                - Dokumen relevan di posisi 2-3 mendapat skor dikurangi
                - Dokumen relevan di posisi 4+ mendapat skor lebih kecil lagi
                
                **Interpretasi:**
                - **>80%**: Excellent - ranking sangat optimal
                - **60-80%**: Good - ranking cukup baik
                - **<60%**: Poor - dokumen relevan terlalu tersebar
                
                **Kenapa penting?** Tidak cukup hanya menemukan dokumen yang benar, 
                tetapi harus di posisi yang tepat (semakin atas semakin baik).
                """)
        
        st.markdown("---")
        
        # Performance by test type
        st.subheader("Performance by Test Type")
        test_types = ['single_doc', 'multi_doc', 'conflicting_docs', 'long_queries']
        
        df_performance = pd.DataFrame([
            {
                'Test Type': t.replace('_', ' ').title(),
                'Success Rate': self.metrics.get(t, {}).get('success_rate', 0) * 100,
                'Avg Precision': self.metrics.get(t, {}).get('average_precision', 0) * 100
            }
            for t in test_types
        ])
        
        fig = px.bar(df_performance, x='Test Type', y=['Success Rate', 'Avg Precision'],
                     barmode='group', title="Performance Comparison")
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
        
        fig = go.Figure(data=[go.Bar(x=positions, y=counts)])
        fig.update_layout(title=title, xaxis_title="Position", yaxis_title="Count")
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
                'Questions': m.get('total_questions', 0),
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
                'Questions': metrics.get('total_questions', 0),
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


def main():
    st.sidebar.title("Settings")
    
    # Try to load default results
    default_results = load_default_results()
    
    uploaded_file = st.sidebar.file_uploader("Upload evaluation results JSON (optional)", type=['json'])
    
    if uploaded_file:
        results = json.load(uploaded_file)
        st.sidebar.success("✅ Loaded from uploaded file")
    elif default_results:
        results = default_results
        st.sidebar.info("📁 Loaded from default file")
    else:
        st.warning("⚠️ No evaluation results found. Please upload a JSON file from the sidebar.")
        st.info("Expected file location: `evaluation_results_new.json`")
        return
    
    # Render dashboard
    dashboard = StreamlitDashboard(results)
    dashboard.render()


if __name__ == "__main__":
    main()