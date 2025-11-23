# **PatrolIQ – Smart Safety Analytics Platform**

PatrolIQ is an end-to-end **urban safety intelligence platform** that uses unsupervised machine learning to analyze 500,000+ crime records from the Chicago Police Department. The platform identifies **crime hotspots**, analyzes **temporal crime patterns**, performs **dimensionality reduction**, and provides actionable insights for **police patrol optimization** and **public safety planning**.

Built with **Python, Streamlit, MLflow, and modern data science workflows**, PatrolIQ brings data-driven crime analysis to life.

---

## 🚀 **Key Features**
- **Crime Hotspot Detection** using K-Means, DBSCAN, and Hierarchical Clustering  
- **Temporal Crime Pattern Analysis** (hour/day/month trends)  
- **Dimensionality Reduction** using PCA and t-SNE/UMAP  
- **500,000 Chicago Crime Records** processed for real-time analysis  
- **Geospatial Visualization** with interactive maps  
- **MLflow Tracking & Model Registry** for experiment management  
- **Streamlit Web App** deployed on Streamlit Cloud  
- **Advanced Feature Engineering** from 22+ crime and location variables  
- **Business Insights** for Police, City Administration, and Emergency Response Teams  

---

## 🎯 **Objective**
Build an intelligent urban safety analytics tool that empowers law enforcement agencies to make **data-driven decisions** by identifying crime hotspots, optimizing patrol routes, and revealing hidden crime patterns across Chicago.

---

## 📂 **Dataset**
- Source: *Chicago Police Department Crime Dataset*  
- Full dataset: **7.8 million records (~1.7GB)**  
- Project sample: **500,000 most recent records**  
- Key Fields:
  - Crime type  
  - Location description  
  - Latitude/Longitude  
  - Arrest & domestic violence flag  
  - Date/Time  
  - District, Ward, Community Area  

---

## 🛠️ **Tech Stack**
| Component | Technology |
|----------|------------|
| Programming | Python |
| Libraries | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, UMAP-learn |
| Unsupervised ML | K-Means, DBSCAN, Hierarchical Clustering |
| Dimensionality Reduction | PCA, t-SNE, UMAP |
| Experiment Tracking | MLflow |
| Deployment | Streamlit Cloud |
| Visualization | Folium, Plotly, Altair |
| Geospatial Tools | Shapely, Geopy |

---

## 📊 **Project Workflow**

### **1. Data Preprocessing**
- Handle missing or inconsistent values  
- Normalize geographical coordinates  
- Extract time features (hour, day, month, season)  
- Encode categorical features  
- Validate data quality  

### **2. Exploratory Data Analysis**
- Crime frequency distribution  
- Hotspot visualization using heatmaps  
- Trend analysis by hour/day/month  
- Arrest and domestic violence correlations  

### **3. Feature Engineering**
- Temporal features (weekend flag, seasons, hour bins)  
- Crime severity scoring  
- Geo-binning and coordinate clustering  
- District categorization  

### **4. Unsupervised Learning**

#### 🟦 **Geographic Clustering**
- K-Means (5–10 clusters)  
- DBSCAN for density-based hotspots  
- Hierarchical clustering with dendrogram  
- Evaluation metrics: *Silhouette Score, Davies-Bouldin Index*  

#### 🕒 **Temporal Clustering**
- K-Means using hour/day/month  
- Identifies 3–5 temporal crime profiles  
- Late-night crimes, peak hours, seasonal spikes  

---

## 🌀 **Dimensionality Reduction**
- **PCA**: Reduce 22+ features → 2–3 components  
  - Target: **70%+ variance explanation**  
- **t-SNE/UMAP** for 2D visualization of crime clusters  
- Identify underlying patterns in high-dimensional space  

---

## 📁 **MLflow Tracking**
Tracks:
- Clustering parameters  
- PCA variance  
- Silhouette & Davies-Bouldin scores  
- Best-performing models  
- Model Registry for deployment  

---

## 🏙️ **Business Use Cases**

### **Police Departments**
- 60%+ faster response time  
- Optimized patrol assignments  
- High-risk area identification  

### **City Administration**
- Better urban safety planning  
- Data-backed CCTV & lighting placement  
- Budget justification  

### **Emergency Response Teams**
- Resource optimization  
- Risk-aware dispatching  
- Multi-agency coordination  

