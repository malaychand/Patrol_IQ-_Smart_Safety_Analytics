import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import mlflow
import mlflow.sklearn

# ==========================================================
# Paths
# ==========================================================
DATA_PATH = r"\data\processed\feature_engineered_crimes.csv"
MLFLOW_URI = "mlruns"
MODEL_NAME = "Crime_Clustering_Model"

# ==========================================================
# Load Data
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

df = load_data()

# ==========================================================
# Map numeric crime codes to names
# ==========================================================
crime_type_mapping = {
    1: "THEFT",
    2: "BATTERY",
    3: "ASSAULT",
    4: "ROBBERY",
    5: "BURGLARY",
    6: "NARCOTICS",
    7: "CRIMINAL DAMAGE",
    8: "OTHER OFFENSE",
    9: "MOTOR VEHICLE THEFT",
    10: "DECEPTIVE PRACTICE",
    # Extend with all PrimaryType_Code values
}
if 'PrimaryType_Code' in df.columns:
    df['PrimaryType_Label'] = df['PrimaryType_Code'].map(crime_type_mapping)

# ==========================================================
# Cluster labels mapping
# ==========================================================
cluster_mapping = {
    '0': "Cluster A – Residential Area Crimes",
    '1': "Cluster B – Commercial Area Crimes",
    '2': "Cluster C – Nightlife / Entertainment Crimes",
    '3': "Cluster D – Industrial / Warehouse Crimes",
    '4': "Cluster E – Mixed Low-Density Crimes"
}

# ==========================================================
# Load MLflow model
# ==========================================================
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
    return model

best_model = load_model()

# ==========================================================
# Streamlit Layout
# ==========================================================
st.set_page_config(page_title="PatrolIQ - Crime Dashboard", layout="wide")
st.title("🚨 PatrolIQ - Smart Safety Analytics Dashboard")
st.markdown("""
**Mission:** Analyze crime patterns to answer:
- Where should we patrol tonight?
- Which neighborhoods need more resources?
- When do most crimes occur?
""")

page = st.sidebar.selectbox("Select Page", 
                            ["Overview", "Geographic Hotspots", "Temporal Patterns", 
                             "Cluster Visualization", "Model Monitoring"])

# ==========================================================
# Page 1: Overview
# ==========================================================
if page == "Overview":
    st.header("📊 Dataset Overview")
    
    # Sample Records
    st.subheader("Sample Records")
    st.dataframe(df.head(10))

    # Column Information
    st.subheader("Columns Info")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Data Type": df.dtypes.values
    })
    st.dataframe(col_info, height=300)

    # Crime Type Distribution
    st.subheader("Crime Type Distribution")
    crime_counts = df['PrimaryType_Label'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']

    fig = px.bar(crime_counts, x='Crime Type', y='Count', title="Crime Type Counts", text='Count',
                 color='Crime Type', color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Description:** Number of crimes by type. Shows which crimes are most common in the city.")

# ==========================================================
# Page 2: Geographic Hotspots
# ==========================================================
elif page == "Geographic Hotspots":
    st.header("🗺 Geographic Crime Hotspots")

    # Slider for sample size
    map_sample_size = st.sidebar.slider("Number of points to display on map", 1000, 50000, 5000, step=1000)

    # Cluster Map with colors
    features = ['Latitude_norm', 'Longitude_norm', 'CrimeSeverity', 'PrimaryType_Code']
    X_features = df[features].dropna().copy()
    X_features = pd.get_dummies(X_features, columns=['PrimaryType_Code'])

    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_features)

    # Predict clusters
    labels = best_model.predict(X_pca) if hasattr(best_model, 'predict') else best_model.fit_predict(X_pca)
    df_geo = df.loc[df[['Latitude_norm', 'Longitude_norm']].dropna().index].copy()
    df_geo['Cluster'] = labels.astype(str)
    df_geo['Cluster_Label'] = df_geo['Cluster'].map(cluster_mapping)
    df_geo_sample = df_geo.sample(n=min(map_sample_size, len(df_geo)), random_state=42)

    fig = px.scatter_mapbox(df_geo_sample, lat='Latitude_norm', lon='Longitude_norm', color='Cluster_Label',
                            hover_name='PrimaryType_Label', hover_data=['CrimeSeverity'], zoom=10, height=600,
                            color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Description:** Each color represents a cluster of similar crimes. Hover to see crime type and severity.")
    
    # Hotspot Recommendations
    st.subheader("🔹 Hotspot Recommendations")
    recent_crimes = df.dropna(subset=['Date'])
    recent_crimes = recent_crimes[recent_crimes['Date'] >= (recent_crimes['Date'].max() - pd.Timedelta(days=7))]
    top_blocks = recent_crimes['Block'].value_counts().head(5)
    st.markdown("**Where should we patrol tonight?**")
    st.table(top_blocks.reset_index().rename(columns={'index':'Block','Block':'Crime Count'}))

    top_districts = df['District'].value_counts().head(5)
    st.markdown("**Which neighborhoods need more resources?**")
    st.table(top_districts.reset_index().rename(columns={'index':'District','District':'Crime Count'}))

    peak_hour = df['Hour'].value_counts().idxmax()
    peak_day = df['DayOfWeek'].value_counts().idxmax()
    peak_month = df['Month'].value_counts().idxmax()
    st.markdown("**When do most crimes occur?**")
    st.markdown(f"- **Peak Hour:** {peak_hour}:00")
    st.markdown(f"- **Peak Day:** {peak_day}")
    st.markdown(f"- **Peak Month:** {peak_month}")

# ==========================================================
# Page 3: Temporal Patterns
# ==========================================================
elif page == "Temporal Patterns":
    st.header("⏰ Temporal Crime Patterns")

    df_time = df.dropna(subset=['Date'])
    df_time['YearMonth'] = df_time['Date'].dt.to_period('M')
    
    # Crime count over months/years
    crime_by_month = df_time.groupby('YearMonth').size().reset_index(name='CrimeCount')
    crime_by_month['YearMonth'] = crime_by_month['YearMonth'].astype(str)
    fig = px.line(crime_by_month, x='YearMonth', y='CrimeCount', title="Crime Count Over Time",
                  markers=True, line_shape='linear', color_discrete_sequence=['crimson'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Description:** Shows monthly crime trends.")

    # Hourly distribution
    st.subheader("Crimes by Hour")
    fig = px.histogram(df_time, x='Hour', nbins=24, title="Hourly Crime Distribution",
                       color_discrete_sequence=['darkorange'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Description:** Highlights peak hours for crimes.")

    # Day of Week
    st.subheader("Crimes by Day of Week")
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_counts = df['DayOfWeek'].value_counts().reindex(day_order)
    fig = px.bar(x=day_counts.index, y=day_counts.values, labels={'x':'Day','y':'Crime Count'},
                 title="Crimes by Day of Week", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Description:** Shows weekly crime patterns.")

    # Month / Season / Weekend
    st.subheader("Crimes by Month")
    month_counts = df['Month'].value_counts().sort_index()
    fig = px.bar(x=month_counts.index, y=month_counts.values, labels={'x':'Month','y':'Crime Count'},
                 title="Crimes by Month", color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crimes by Season")
    if 'Season' in df.columns:
        season_counts = df['Season'].value_counts()
        fig = px.bar(x=season_counts.index, y=season_counts.values, labels={'x':'Season','y':'Crime Count'},
                     title="Crimes by Season", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crimes on Weekend vs Weekday")
    if 'Weekend' in df.columns:
        weekend_counts = df['Weekend'].value_counts()
        fig = px.bar(x=weekend_counts.index, y=weekend_counts.values, labels={'x':'Weekend','y':'Crime Count'},
                     title="Weekend vs Weekday Crimes", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Page 4: Cluster Visualization
# ==========================================================
elif page == "Cluster Visualization":
    st.header("🧮 PCA 2D Cluster Visualization")

    features = ['Latitude_norm', 'Longitude_norm', 'CrimeSeverity', 'PrimaryType_Code']
    X_full = df[features].dropna().copy()
    X_full = pd.get_dummies(X_full, columns=['PrimaryType_Code'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full)

    labels = best_model.predict(X_pca) if hasattr(best_model, 'predict') else best_model.fit_predict(X_pca)
    df_pca = pd.DataFrame({
        'PC1': X_pca[:,0],
        'PC2': X_pca[:,1],
        'Cluster': labels.astype(str)
    })
    df_pca['Cluster_Label'] = df_pca['Cluster'].map(cluster_mapping)

    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster_Label',
                     title="PCA 2D Cluster Plot",
                     hover_data=['Cluster_Label'], color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Description:** Visual representation of crimes in 2D PCA space. Each color is a cluster with a meaningful label.")

# ==========================================================
# Page 5: Model Monitoring
# ==========================================================
elif page == "Model Monitoring":
    st.header("📈 MLflow Model Monitoring")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Crime_Clustering")

    if experiment:
        st.write("Experiment Name:", experiment.name)
        st.write("Experiment ID:", experiment.experiment_id)
        runs = client.search_runs(experiment.experiment_id, order_by=["metrics.silhouette_score DESC"])
        if runs:
            run = runs[0]
            metrics = pd.DataFrame(list(run.data.metrics.items()), columns=['Metric', 'Value'])
            params = pd.DataFrame(list(run.data.params.items()), columns=['Parameter', 'Value'])

            st.subheader("Best Run Details")
            st.markdown(f"**Run ID:** {run.info.run_id}")
            st.markdown(f"**Silhouette Score:** {run.data.metrics['silhouette_score']}")

            st.subheader("Logged Parameters")
            st.dataframe(params, use_container_width=True)

            st.subheader("Logged Metrics")
            st.dataframe(metrics, use_container_width=True)
        else:
            st.write("No runs found.")
    else:
        st.write("Experiment not found.")
