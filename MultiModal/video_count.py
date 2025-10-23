import os
import pandas as pd
from tqdm import tqdm

# --- User settings ---
csv_path = "/home/manik/Downloads/FakeAVCeleb_v1.2/meta_data.csv"  # Path to your CSV file
root_dir = "/home/manik/Downloads/FakeAVCeleb_v1.2/"               # Root directory for videos
csv_path = "/home/manik/Downloads/FakeAVCeleb_v1.2/meta_data_processed.csv"  # Filtered CSV

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Keep only rows with processed videos ---
print("\n🔍 Checking for missing processed videos...\n")

keep_rows = []
for row in tqdm(df.itertuples(index=False), total=len(df), desc="Checking"):
    orig_path = os.path.join(
        root_dir,
        str(row.type),
        str(row.race),
        str(row.gender),
        str(row.source),
        str(row.path)
    )
    base, ext = os.path.splitext(orig_path)
    processed_path = f"{base}_roi{ext}"

    if os.path.exists(processed_path):
        keep_rows.append(True)
    else:
        keep_rows.append(False)
        print(f"❌ Missing processed video:\n   {processed_path}\n")

# --- Filter and save ---
# filtered_df = df[pd.Series(keep_rows)]
# filtered_df.to_csv(output_csv, index=False)

# print(f"✅ Done. Total kept videos: {len(filtered_df)} / {len(df)}")
# print(f"📄 Filtered CSV saved to: {output_csv}")
