import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# ===================================================================
#  CONFIG
# ===================================================================
INPUT_FILE = "data/processed/feature_engineered_crimes.csv"
OUTPUT_DIR = "outputs/temporal_analysis"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
#  LOAD DATA
# ===================================================================
def load_data():
    print("📌 Loading dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if 'Date' not in df.columns:
        raise Exception("❌ 'Date' column missing in dataset")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Temporal Features
    df['Hour'] = df['Date'].dt.hour
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()

    print("✔ Temporal features created.")
    return df


# ===================================================================
# 1️⃣ K-MEANS TEMPORAL CLUSTERING
# ===================================================================
def run_kmeans(df):
    print("📌 Running KMeans...")

    X = df[['Hour', 'Month']].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['TemporalCluster'] = kmeans.fit_predict(X_scaled)

    # Save cluster plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df['Hour'], y=df['Month'],
        hue=df['TemporalCluster'], palette='tab10'
    )
    plt.title("Temporal Crime Clusters (Hour vs Month)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.savefig(f"{OUTPUT_DIR}/temporal_clusters.png")
    plt.close()

    print("✔ Saved: temporal_clusters.png")
    return df


# ===================================================================
# 2️⃣ HOURLY CRIME PATTERN
# ===================================================================
def hourly_pattern(df):
    hourly_counts = df['Hour'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=hourly_counts.index, y=hourly_counts.values)
    plt.title("Hourly Crime Frequency")
    plt.xlabel("Hour of Day")
    plt.ylabel("Crime Count")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/hourly_pattern.png")
    plt.close()

    print("✔ Saved: hourly_pattern.png")
    return hourly_counts


# ===================================================================
# 3️⃣ MONTHLY / SEASONAL CRIME PATTERN
# ===================================================================
def monthly_pattern(df):
    monthly_counts = df['Month'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_counts.index, y=monthly_counts.values)
    plt.title("Monthly Crime Frequency")
    plt.xlabel("Month")
    plt.ylabel("Crime Count")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/monthly_pattern.png")
    plt.close()

    print("✔ Saved: monthly_pattern.png")
    return monthly_counts


# ===================================================================
# 4️⃣ CRIME PROFILE HEATMAP (Hour vs DayOfWeek)
# ===================================================================
def crime_profile_heatmap(df):
    pivot = df.pivot_table(
        index='DayOfWeek',
        columns='Hour',
        values='Location',
        aggfunc='count'
    ).fillna(0)

    # Order weekdays
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(order)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="Reds")
    plt.title("Crime Profile Heatmap (Day vs Hour)")
    plt.savefig(f"{OUTPUT_DIR}/crime_profiles.png")
    plt.close()

    print("✔ Saved: crime_profiles.png")


# ===================================================================
# 5️⃣ SUMMARY GENERATION
# ===================================================================
def generate_summary(hourly, monthly):
    print("📌 Creating summary...")

    peak_hour = hourly.idxmax()
    peak_hour_count = hourly.max()

    peak_month = monthly.idxmax()
    peak_month_count = monthly.max()

    summary = f"""
==================== TEMPORAL CRIME SUMMARY ====================

⭐ Peak Crime Hour: {peak_hour}:00  
   → Total incidents: {peak_hour_count}

⭐ Peak Crime Month: {peak_month}  
   → Total incidents: {peak_month_count}

⭐ Insights:
- High-risk time periods identified
- Seasonal crime variations detected
- Temporal clusters created for crime pattern profiling
- Heatmap reveals day–hour behavior patterns

All visuals saved in:
{OUTPUT_DIR}/
==============================================================
"""

    with open(f"{OUTPUT_DIR}/temporal_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("✔ Saved: temporal_summary.txt")


# ===================================================================
# MAIN EXECUTION
# ===================================================================
def main():
    print("\n🔍 Step 5: Temporal Pattern Analysis Started\n")

    df = load_data()
    df = run_kmeans(df)

    hourly = hourly_pattern(df)
    monthly = monthly_pattern(df)
    crime_profile_heatmap(df)

    generate_summary(hourly, monthly)

    print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
    print(f"📁 Check outputs in: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
