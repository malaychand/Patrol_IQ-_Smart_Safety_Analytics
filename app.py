"""
PatrolIQ - Smart Safety Analytics Platform
Simple Streamlit Application
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import json

# ==============================
# Configuration
# ==============================
st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics",
    page_icon="🚨",
    layout="wide"
)

RESULTS_PATH = "Patrool_results/"

# ==============================
# Helper Functions
# ==============================
def load_csv(filename):
    """Load CSV file from results directory"""
    path = os.path.join(RESULTS_PATH, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None
    return None

def load_json(filename):
    """Load JSON file from results directory"""
    path = os.path.join(RESULTS_PATH, filename)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None
    return None

def show_image(filename, caption=None):
    """Display image from results directory"""
    path = os.path.join(RESULTS_PATH, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {filename}")

# ==============================
# Sidebar
# ==============================
st.sidebar.title("🚨 PatrolIQ")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 EDA Results", 
        "🗺️ Geographic Hotspots",
        "⏰ Temporal Patterns",
        "🔬 Dimensionality Reduction",
        "📈 Model Performance",
        "🧪 MLflow Tracking"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 System Status")

# Check results folder
if os.path.exists(RESULTS_PATH):
    files = os.listdir(RESULTS_PATH)
    st.sidebar.success(f"✓ Results folder found")
    st.sidebar.info(f"📄 {len(files)} files available")
else:
    st.sidebar.error("✗ Results folder not found")

# ==============================
# Page 1: Home
# ==============================
if page == "🏠 Home":
    
    st.title("🚨 PatrolIQ - Smart Safety Analytics Platform")

    st.markdown("""
    ## 🎯 Mission
    Build a comprehensive urban safety intelligence platform that leverages unsupervised machine learning 
    to analyze crime patterns and optimize police resource allocation.

    Imagine working as a **crime intelligence analyst** helping law enforcement answer critical questions using data.
    """)

    st.markdown("""
    ### Critical Questions Answered
    - 📍 **Where should we patrol tonight?**
    - 🏘️ **Which neighborhoods need more resources?**
    - ⏰ **When do most crimes occur?**
    """)

    st.markdown("---")

    # Business Use Cases
    st.subheader("🏙️ Business Use Cases")

    use_col1, use_col2 = st.columns(2)

    with use_col1:
        st.markdown("**Police Departments**")
        st.markdown("""
        - Optimize patrol routes
        - Identify high-risk areas
        - Predict crime patterns
        - Improve response time
        """)

        st.markdown("**City Administration**")
        st.markdown("""
        - Data-driven urban planning
        - Strategic surveillance placement
        - Budget allocation insights
        - District crime monitoring
        """)

    with use_col2:
        st.markdown("**Analytics Firms**")
        st.markdown("""
        - Multi-city crime intelligence
        - Predictive policing solutions
        - Safety benchmarking
        - Stakeholder reports
        """)

        st.markdown("**Emergency Response**")
        st.markdown("""
        - Risk-based call prioritization
        - Optimized ambulance deployment
        - Multi-agency coordination
        - Real-time situational awareness
        """)

    st.markdown("---")
    
    # Project Overview
    st.subheader("📌 Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Project Title**")
        st.info("PatrolIQ - Smart Safety Analytics Platform")
        
        st.markdown("**Domain**")
        st.info("Public Safety and Urban Analytics")

    with col2:
        st.markdown("**Skills Applied**")
        st.markdown("""
        - Python
        - Machine Learning
        - Unsupervised Learning
        - Clustering Algorithms
        - Dimensionality Reduction
        - Geographic Data Analysis
        - Data Visualization
        - MLflow
        - Streamlit Cloud Deployment
        """)

    st.markdown("---")

    # Key Statistics
    st.subheader("📊 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Size", "500,000", "Crime Records")
    col2.metric("Features", "22+", "Engineered")
    col3.metric("Crime Types", "33", "Categories")
    col4.metric("ML Models", "5+", "Algorithms")

    st.markdown("---")

    # Technical Approach
    st.subheader("🔬 Technical Approach")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🗺️ Geographic Clustering**")
        st.markdown("- K-Means")
        st.markdown("- DBSCAN")
        st.markdown("- Hierarchical Clustering")
        st.info("Goal: Identify crime hotspots")

    with col2:
        st.markdown("**⏰ Temporal Pattern Analysis**")
        st.markdown("- Hourly crime trends")
        st.markdown("- Weekly patterns")
        st.markdown("- Seasonal cycles")
        st.info("Goal: Optimize patrol timing")

    with col3:
        st.markdown("**🔬 Dimensionality Reduction**")
        st.markdown("- PCA (70%+ variance)")
        st.markdown("- t-SNE / UMAP")
        st.markdown("- Feature importance ranking")
        st.info("Goal: Visualize complex crime patterns")

    st.markdown("---")


    # Technologies
    st.subheader("🛠️ Technologies Used")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("**Machine Learning**")
        st.markdown("- Scikit-learn")
        st.markdown("- NumPy / Pandas")

    with tech_col2:
        st.markdown("**Visualization**")
        st.markdown("- Plotly")
        st.markdown("- Matplotlib / Seaborn")

    with tech_col3:
        st.markdown("**Deployment & Tracking**")
        st.markdown("- Streamlit Cloud")
        st.markdown("- MLflow")
        st.markdown("- DagsHub")

# ==============================
# Page 2: EDA Results
# ==============================
elif page == "📊 EDA Results":
    
    st.title("📊 Exploratory Data Analysis Results")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Missing Values", 
        "Crime Types", 
        "Arrest Analysis",
        "Temporal Patterns",
        "Geographic Analysis"
    ])
    
    with tab1:
        st.subheader("Missing Values Analysis")
        missing_df = load_csv("missing_values_summary.csv")
        if missing_df is not None:
            st.dataframe(missing_df, use_container_width=True)
    
    with tab2:
        st.subheader("Crime Type Distribution")
        show_image("02_crime_type_analysis.png")
        
        crime_counts = load_csv("crime_type_counts.csv")
        if crime_counts is not None:
            st.dataframe(crime_counts.head(20), use_container_width=True)
    
    with tab3:
        st.subheader("Arrest & Domestic Incidents")
        show_image("03_arrest_domestic_analysis.png")
        
        arrest_rates = load_csv("arrest_rates_by_crime.csv")
        if arrest_rates is not None:
            st.dataframe(arrest_rates.head(20), use_container_width=True)
    
    with tab4:
        st.subheader("Temporal Crime Patterns")
        show_image("04_temporal_patterns.png")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly = load_csv("temporal_hourly.csv")
            if hourly is not None:
                st.markdown("**Hourly Distribution**")
                st.dataframe(hourly, use_container_width=True)
        
        with col2:
            daily = load_csv("temporal_daily.csv")
            if daily is not None:
                st.markdown("**Daily Distribution**")
                st.dataframe(daily, use_container_width=True)
    
    with tab5:
        st.subheader("Geographic Distribution")
        show_image("05_geographic_analysis.png")
        
        districts = load_csv("district_counts.csv")
        if districts is not None:
            st.dataframe(districts.head(20), use_container_width=True)
    
    st.markdown("---")
    
    # Additional EDA
    st.subheader("Additional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Correlation Analysis**")
        show_image("06_correlation_analysis.png")
    
    with col2:
        st.markdown("**Crime Descriptions**")
        show_image("07_descriptions_analysis.png")
    
    st.markdown("---")
    st.markdown("**Feature Engineering**")
    show_image("08_feature_engineering.png")

# ==============================
# Page 3: Geographic Hotspots
# ==============================
elif page == "🗺️ Geographic Hotspots":
    
    st.title("🗺️ Geographic Crime Hotspots")
    
    # Algorithm Comparison
    st.subheader("⚖️ Algorithm Performance Comparison")
    
    comparison_df = load_csv("geo_clustering_comparison.csv")
    
    if comparison_df is not None:
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(
                comparison_df,
                x='Algorithm',
                y='Silhouette Score',
                title="Silhouette Score",
                text='Silhouette Score',
                color='Algorithm'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                comparison_df,
                x='Algorithm',
                y='Davies-Bouldin Score',
                title="Davies-Bouldin Score",
                text='Davies-Bouldin Score',
                color='Algorithm'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.dataframe(comparison_df, use_container_width=True, height=300)
        
        # Best algorithm
        best_algo = comparison_df.loc[comparison_df['Silhouette Score'].idxmax(), 'Algorithm']
        st.success(f"🏆 **Best Algorithm**: {best_algo}")
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📍 Cluster Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["KMeans", "DBSCAN", "Hierarchical", "Comparison"])
    
    with tab1:
        st.markdown("### KMeans Clustering (K=6)")
        show_image("geo_kmeans_clusters.png")
        st.markdown("**Elbow Analysis**")
        show_image("geo_kmeans_elbow_analysis.png")
    
    with tab2:
        st.markdown("### DBSCAN Clustering")
        show_image("geo_dbscan_clusters.png")
    
    with tab3:
        st.markdown("### Hierarchical Clustering")
        col1, col2 = st.columns(2)
        with col1:
            show_image("geo_hierarchical_clusters.png")
        with col2:
            show_image("geo_hierarchical_dendrogram.png")
    
    with tab4:
        st.markdown("### Performance Comparison")
        show_image("geo_clustering_comparison.png")

# ==============================
# Page 4: Temporal Patterns
# ==============================
elif page == "⏰ Temporal Patterns":
    
    st.title("⏰ Temporal Crime Patterns")
    
    st.info("Temporal analysis identifies when crimes occur to optimize patrol schedules.")
    
    # Main visualization
    st.subheader("📈 Crime Patterns by Time")
    show_image("temporal_crime_patterns.png")
    
    st.markdown("---")
    
    # Elbow analysis
    st.subheader("🔍 Optimal Cluster Selection")
    show_image("temporal_kmeans_elbow.png")
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Temporal Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    hourly = load_csv("temporal_hourly.csv")
    daily = load_csv("temporal_daily.csv")
    monthly = load_csv("temporal_monthly.csv")
    seasonal = load_csv("temporal_seasonal.csv")
    
    if hourly is not None:
        peak_hour = hourly.loc[hourly['Crime_Count'].idxmax(), 'Hour']
        col1.metric("Peak Hour", f"{int(peak_hour)}:00")
    
    if daily is not None:
        peak_day = daily.loc[daily['Crime_Count'].idxmax(), 'Day']
        col2.metric("Peak Day", peak_day)
    
    if monthly is not None:
        peak_month = monthly.loc[monthly['Crime_Count'].idxmax(), 'Month']
        col3.metric("Peak Month", int(peak_month))
    
    if seasonal is not None:
        peak_season = seasonal.loc[seasonal['Crime_Count'].idxmax(), 'Season']
        col4.metric("Peak Season", peak_season)
    
    # Data tables
    st.markdown("---")
    st.subheader("📋 Detailed Breakdown")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Hourly", "Daily", "Monthly", "Seasonal"])
    
    with tab1:
        if hourly is not None:
            st.dataframe(hourly, use_container_width=True)
    
    with tab2:
        if daily is not None:
            st.dataframe(daily, use_container_width=True)
    
    with tab3:
        if monthly is not None:
            st.dataframe(monthly, use_container_width=True)
    
    with tab4:
        if seasonal is not None:
            st.dataframe(seasonal, use_container_width=True)

# ==============================
# Page 5: Dimensionality Reduction
# ==============================
elif page == "🔬 Dimensionality Reduction":
    
    st.title("🔬 Dimensionality Reduction Analysis")
    
    st.info("Reduce 22+ features to 2-3 dimensions while preserving patterns.")
    
    # PCA Analysis
    st.subheader("📊 PCA - Principal Component Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Variance Explained", 
        "Feature Importance", 
        "2D Projection", 
        "3D Projection"
    ])
    
    with tab1:
        show_image("pca_variance_explained.png")
        
        summary = load_json("model_summary.json")
        if summary and "Dimensionality Reduction" in summary:
            pca_info = summary["Dimensionality Reduction"].get("PCA", {})
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Variance (3 comp)", f"{pca_info.get('variance_explained', 0):.2%}")
            col2.metric("70% Variance", f"{pca_info.get('components_for_70pct', 'N/A')} comp")
            col3.metric("80% Variance", f"{pca_info.get('components_for_80pct', 'N/A')} comp")
    
    with tab2:
        show_image("pca_feature_importance.png")
        
        loadings = load_csv("pca_feature_loadings.csv")
        if loadings is not None:
            st.dataframe(loadings, use_container_width=True)
    
    with tab3:
        show_image("pca_2d_projection.png")
    
    with tab4:
        show_image("pca_3d_projection.png")
    
    st.markdown("---")
    
    # t-SNE Analysis
    st.subheader("🎨 t-SNE Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**t-SNE (Perplexity=30)**")
        show_image("tsne_perplexity_30.png")
    
    with col2:
        st.markdown("**t-SNE (Perplexity=50)**")
        show_image("tsne_perplexity_50.png")

# ==============================
# Page 6: Model Performance
# ==============================
elif page == "📈 Model Performance":
    
    st.title("📈 Model Performance & Metrics")
    
    summary = load_json("model_summary.json")
    
    if summary:
        
        # Geographic Clustering
        st.subheader("🗺️ Geographic Clustering Performance")
        
        if "Geographic Clustering" in summary:
            geo = summary["Geographic Clustering"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**KMeans**")
                st.metric("Clusters", geo["KMeans"].get("n_clusters", "N/A"))
                st.metric("Silhouette", f"{geo['KMeans'].get('silhouette_score', 0):.4f}")
                st.metric("Davies-Bouldin", f"{geo['KMeans'].get('davies_bouldin_score', 0):.4f}")
            
            with col2:
                st.markdown("**DBSCAN**")
                st.metric("Clusters", geo["DBSCAN"].get("n_clusters", "N/A"))
                sil = geo["DBSCAN"].get("silhouette_score")
                st.metric("Silhouette", f"{sil:.4f}" if sil else "N/A")
                st.metric("Epsilon", f"{geo['DBSCAN'].get('eps', 0):.2f}")
            
            with col3:
                st.markdown("**Hierarchical**")
                st.metric("Clusters", geo["Hierarchical"].get("n_clusters", "N/A"))
                st.metric("Silhouette", f"{geo['Hierarchical'].get('silhouette_score', 0):.4f}")
                st.metric("Davies-Bouldin", f"{geo['Hierarchical'].get('davies_bouldin_score', 0):.4f}")
            
            best_algo = geo.get("Best Algorithm", "N/A")
            st.success(f"🏆 **Best Algorithm**: {best_algo}")
        
        st.markdown("---")
        
        # Temporal Clustering
        st.subheader("⏰ Temporal Clustering Performance")
        
        if "Temporal Clustering" in summary:
            temp = summary["Temporal Clustering"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Algorithm", temp.get("Algorithm", "N/A"))
            col2.metric("Clusters", temp.get("n_clusters", "N/A"))
            col3.metric("Silhouette", f"{temp.get('silhouette_score', 0):.4f}")
            col4.metric("Davies-Bouldin", f"{temp.get('davies_bouldin_score', 0):.4f}")
        
        st.markdown("---")
        
        # Dimensionality Reduction
        st.subheader("🔬 Dimensionality Reduction Performance")
        
        if "Dimensionality Reduction" in summary:
            dr = summary["Dimensionality Reduction"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**PCA**")
                pca = dr.get("PCA", {})
                st.metric("Components", pca.get("n_components", "N/A"))
                st.metric("Variance Explained", f"{pca.get('variance_explained', 0):.2%}")
                st.metric("Components for 70%", pca.get("components_for_70pct", "N/A"))
            
            with col2:
                st.markdown("**t-SNE**")
                tsne = dr.get("t-SNE", {})
                st.metric("Components", tsne.get("n_components", "N/A"))
                st.metric("Perplexity", tsne.get("perplexity", "N/A"))
        
        st.markdown("---")
        
        # Download buttons
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Summary (JSON)",
                data=json.dumps(summary, indent=4),
                file_name="model_summary.json",
                mime="application/json"
            )
        
        with col2:
            comparison_df = load_csv("geo_clustering_comparison.csv")
            if comparison_df is not None:
                st.download_button(
                    label="📊 Download Comparison (CSV)",
                    data=comparison_df.to_csv(index=False),
                    file_name="clustering_comparison.csv",
                    mime="text/csv"
                )
    
    else:
        st.error("Model summary not found. Please run the analysis first.")

# ==============================
# Page 7: MLflow Tracking
# ==============================
elif page == "🧪 MLflow Tracking":
    
    st.title("🧪 MLflow Experiment Tracking")
    
    st.info("All experiments are tracked using MLflow for reproducibility and version control.")
    
    # DagsHub link
    st.subheader("🌐 DagsHub Repository")
    
    st.markdown("""
    **Repository Details:**
    - Owner: `malaychand`
    - Repo: `Patrol_IQ-_Smart_Safety_Analytics`
    - Platform: DagsHub
    """)
    
    st.markdown("""
    ### 📊 View MLflow Dashboard:
    [🔗 Open MLflow Tracking UI](https://dagshub.com/malaychand/Patrol_IQ-_Smart_Safety_Analytics.mlflow)
    """)
    
    st.markdown("---")
    
    # What's tracked
    st.subheader("📝 Tracked Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Parameters Logged**")
        st.markdown("- Algorithm type")
        st.markdown("- Number of clusters (K)")
        st.markdown("- Distance metrics")
        st.markdown("- Sample sizes")
        st.markdown("- Random seeds")
    
    with col2:
        st.markdown("**Metrics Logged**")
        st.markdown("- Silhouette scores")
        st.markdown("- Davies-Bouldin index")
        st.markdown("- Calinski-Harabasz score")
        st.markdown("- Variance explained")
        st.markdown("- Training time")
    
    st.markdown("---")
    
    # Local MLflow
    st.subheader("💻 Local MLflow Setup")
    
    st.code("mlflow ui", language="bash")
    
    st.info("Run the above command to start local MLflow UI at http://localhost:5000")
    
    # Summary stats
    st.markdown("---")
    st.subheader("📊 Experiment Statistics")
    
    summary = load_json("model_summary.json")
    
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Experiments", "6+")
        col2.metric("Geographic Models", "3")
        col3.metric("Temporal Models", "1")
        col4.metric("DR Techniques", "2")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>PatrolIQ - Smart Safety Analytics Platform</b></p>
    <p>Built with Streamlit | Powered by Machine Learning & MLflow</p>
</div>
""", unsafe_allow_html=True)