# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to the cleaned dataset from Step 1
DATA_PATH = "data/processed/cleaned_crimes.csv"

# ---------- LOAD DATA ----------
print("🔹 Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded successfully — shape: {df.shape}")
print("\nColumns:", df.columns.tolist())

# ---------- BASIC STATS ----------
print("\n🔹 Basic Statistical Summary:")
print(df.describe(include='all').T)

# ---------- 1️⃣ CRIME TYPE DISTRIBUTION ----------
print("\n🔹 Top 10 Most Common Crime Types:")
top_types = df['Primary Type'].value_counts().head(10)
print(top_types)

plt.figure(figsize=(10,5))
sns.barplot(x=top_types.values, y=top_types.index, palette="viridis")
plt.title("Top 10 Most Common Crime Types")
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")
plt.tight_layout()
plt.savefig("data/eda/top_crime_types.png")
plt.close()

# ---------- 2️⃣ GEOGRAPHIC DISTRIBUTION ----------
print("\n🔹 Checking geographic data (latitude, longitude)...")

# Convert to numeric safely
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# Drop missing or invalid coordinates
geo_df = df.dropna(subset=['Latitude', 'Longitude'])

if len(geo_df) > 5000:
    geo_sample = geo_df.sample(5000, random_state=42)
else:
    geo_sample = geo_df

plt.figure(figsize=(8, 6))
sns.scatterplot(data=geo_sample, x='Longitude', y='Latitude', alpha=0.3, s=10)
plt.title("Crime Incidents — Geographic Distribution (Sample)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
os.makedirs("data/eda", exist_ok=True)
plt.savefig("data/eda/geo_heatmap.png")
plt.close()

# ---------- 3️⃣ TEMPORAL ANALYSIS ----------
print("\n🔹 Temporal analysis...")

# Crimes by Hour
plt.figure(figsize=(8,4))
sns.countplot(x='Hour', data=df, palette="coolwarm")
plt.title("Crimes by Hour of Day")
plt.tight_layout()
plt.savefig("data/eda/crimes_by_hour.png")
plt.close()

# Crimes by Day of Week
plt.figure(figsize=(8,4))
sns.countplot(x='DayOfWeek', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title("Crimes by Day of Week")
plt.tight_layout()
plt.savefig("data/eda/crimes_by_dayofweek.png")
plt.close()

# Crimes by Month
plt.figure(figsize=(8,4))
month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
sns.countplot(x='Month', data=df, order=month_order, palette="plasma")
plt.title("Crimes by Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/eda/crimes_by_month.png")
plt.close()

# ---------- 4️⃣ ARREST & DOMESTIC CORRELATIONS ----------
print("\n🔹 Arrest & Domestic incident patterns...")
plt.figure(figsize=(6,4))
sns.countplot(x='Arrest', data=df)
plt.title("Arrest vs Non-Arrest")
plt.tight_layout()
plt.savefig("data/eda/arrest_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(x='Domestic', data=df)
plt.title("Domestic vs Non-Domestic Crimes")
plt.tight_layout()
plt.savefig("data/eda/domestic_distribution.png")
plt.close()

# Cross-tab for deeper insight
cross_tab = pd.crosstab(df['Primary Type'], df['Arrest'])
print("\n🔹 Arrest rates by crime type (top 10):")
print((cross_tab.div(cross_tab.sum(1), axis=0).sort_values(by=True, ascending=False).head(10)))

# ---------- 5️⃣ SAVE SUMMARY ----------
summary_stats = {
    "Total Crimes": len(df),
    "Unique Crime Types": df['Primary Type'].nunique(),
    "Arrest Rate (%)": round(df['Arrest'].value_counts(normalize=True).get(True, 0)*100, 2),
    "Domestic Crimes (%)": round(df['Domestic'].value_counts(normalize=True).get(True, 0)*100, 2)
}

summary_df = pd.DataFrame(summary_stats, index=[0])
os.makedirs("data/eda", exist_ok=True)
summary_df.to_csv("data/eda/summary_stats.csv", index=False)

print("\n✅ EDA Completed Successfully!")
print("📊 Charts saved in: data/eda/")
print("📄 Summary saved as: data/eda/summary_stats.csv")
