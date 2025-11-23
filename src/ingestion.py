# src/ingestion.py
import pandas as pd
import os

# ---------- PATHS ----------
DATA_RAW = r"data/raw/crimes_dataset.csv"        # path to raw CSV
SAMPLE_OUT = r"data/samples/chicago_500k.csv"    # 500k sample
PROCESSED_OUT = r"data/processed/cleaned_crimes.csv"

TARGET_SAMPLE = 500_000
CHUNKSIZE = 100_000  # load 100k rows at a time


# ---------- STEP 1A: SAMPLE RECENT RECORDS ----------
def sample_recent():
    print("🔹 Reading dataset in chunks...")
    chunks = []
    total_rows = 0

    # Read CSV file in chunks to handle large file sizes
    for chunk in pd.read_csv(DATA_RAW, low_memory=False, chunksize=CHUNKSIZE):
        total_rows += len(chunk)
        chunks.append(chunk)
        print(f"  Loaded {total_rows} rows so far...")
        if total_rows >= TARGET_SAMPLE * 2:  # limit how much we load
            break

    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded total rows: {len(df)}")

    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        raise ValueError("❌ 'Date' column not found in CSV.")

    print("🔹 Converting Date column...")

    # Try known date formats for faster conversion
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
            if df['Date'].notna().mean() > 0.5:  # if >50% parsed
                break
        except Exception:
            continue

    # Drop invalid dates and sort by most recent
    df = df.dropna(subset=['Date']).sort_values('Date', ascending=False)

    # Take most recent 500k rows
    sample_df = df.head(TARGET_SAMPLE)

    # Save sampled data
    os.makedirs(os.path.dirname(SAMPLE_OUT), exist_ok=True)
    sample_df.to_csv(SAMPLE_OUT, index=False)
    print(f"✅ Saved sample: {SAMPLE_OUT} ({len(sample_df)} rows)")

    return sample_df


# ---------- STEP 1B: CLEANING & FEATURE ENGINEERING ----------
def clean_and_process(df):
    print("🔹 Cleaning data...")
    df = df.fillna("Unknown")

    # Extract temporal features
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()

    # Basic data quality check
    print("🔹 Missing values per column:")
    print(df.isnull().sum().to_string())

    # Save cleaned dataset
    os.makedirs(os.path.dirname(PROCESSED_OUT), exist_ok=True)
    df.to_csv(PROCESSED_OUT, index=False)
    print(f"✅ Saved cleaned data: {PROCESSED_OUT}")


# ---------- MAIN ----------
if __name__ == "__main__":
    df_sample = sample_recent()
    clean_and_process(df_sample)
