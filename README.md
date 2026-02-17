# 🚓 PatrolIQ – Smart Safety Analytics Platform

PatrolIQ is an end-to-end **urban safety intelligence platform** that uses unsupervised machine learning to analyze **500,000+ crime records** from the Chicago Police Department.

The platform identifies **crime hotspots**, analyzes **temporal crime patterns**, performs **dimensionality reduction**, and provides actionable insights for **police patrol optimization** and **public safety planning**.

Built using **Python, Streamlit, and modern data science workflows**, PatrolIQ demonstrates how machine learning can support safer cities through data-driven decision-making.

---

## 🔗 Project Links

- **DagsHub Repository:**  
  https://dagshub.com/malaychand/Patrol_IQ-_Smart_Safety_Analytics

- **MLflow Experiment Dashboard:**  
  https://dagshub.com/malaychand/Patrol_IQ-_Smart_Safety_Analytics.mlflow/#/experiments

---

## 🚀 Key Features

- Crime Hotspot Detection using:
  - K-Means
  - DBSCAN
  - Hierarchical Clustering
- Temporal Crime Pattern Analysis
- Dimensionality Reduction using PCA and t-SNE
- Analysis of **500,000 real-world crime records**
- Interactive visualizations
- Streamlit-based web application
- Advanced feature engineering from 22+ variables
- Data-driven insights for public safety

---

## 🎯 Project Objective

To build an intelligent crime analytics platform that enables law enforcement and city administrators to:

- Identify high-risk areas
- Optimize patrol routes
- Understand crime patterns over time
- Allocate resources efficiently

---

## 📂 Dataset

**Source:** Chicago Police Department Crime Dataset  
- Full dataset: **7.8 million records (~1.7 GB)**
- Project sample: **500,000 most recent records**

### Key Fields
- Primary crime type
- Location description
- Latitude & Longitude
- Arrest flag
- Domestic violence flag
- Date & time
- District, ward, community area

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Clustering | K-Means, DBSCAN, Hierarchical |
| Dimensionality Reduction | PCA, t-SNE |
| Visualization | Matplotlib, Plotly |
| Web App | Streamlit |
| Experiment Tracking | MLflow |
| Model Hosting | DagsHub |

---

## 📊 Project Workflow

### 1. Data Preprocessing
- Handle missing values
- Normalize geographic coordinates
- Extract time-based features
- Encode categorical variables

### 2. Exploratory Data Analysis
- Crime distribution by type
- Geographic hotspot maps
- Temporal trend analysis
- Arrest and domestic crime correlations

### 3. Feature Engineering
- Hour, day, month, season features
- Crime severity scoring
- Geographic binning
- Encoded categorical variables

---

## 🤖 Unsupervised Learning

### Geographic Clustering
Three clustering algorithms were applied:

**K-Means**
- Tested clusters from 2 to 8
- Selected best K using silhouette score

**DBSCAN**
- Tested multiple eps values
- Identified dense crime zones
- Detected noise/outliers

**Hierarchical Clustering**
- Tested cluster sizes from 3 to 8
- Generated dendrogram for zone hierarchy

**Evaluation Metrics**
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score

---

### Temporal Pattern Clustering
- K-Means on:
  - Hour
  - Day of week
  - Month
  - Season
- Identified distinct crime-time profiles

---

## 🌀 Dimensionality Reduction

### PCA (Principal Component Analysis)
- Reduced high-dimensional features
- Extracted main crime-driving factors
- Visualized in 2D and 3D

### t-SNE
- Generated 2D visual cluster maps
- Revealed hidden patterns in crime data

---

## 📈 Key Results

### Geographic Clustering
- Multiple crime hotspots identified
- DBSCAN achieved best silhouette score

### Temporal Clustering
- Distinct patterns:
  - Late-night crimes
  - Midday incidents
  - Weekend spikes

### PCA Insights
Top contributing features:
- Location coordinates
- Time of day
- Month/season

---

## 🏙️ Business Use Cases

### Police Departments
- Optimize patrol routes
- Reduce response times
- Identify high-risk zones

### City Administration
- Data-driven safety planning
- Strategic CCTV placement
- Budget justification

### Emergency Response
- Priority-based dispatching
- Resource optimization
- Multi-agency coordination

---

## 🖥️ Streamlit Application

The project includes a **multi-page Streamlit web app** featuring:

- Crime hotspot visualization
- Temporal pattern dashboards
- PCA and t-SNE visualizations
- Model performance comparisons

---

## ⚙️ How to Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/patroliq.git
cd patroliq
