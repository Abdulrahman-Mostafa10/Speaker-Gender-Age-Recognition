import pandas as pd
import os
import glob

base_dir = "data_processed"
stats_dir = "stats"
labels = ["Male_Twenties", "Female_Twenties", "Male_Fifties", "Female_Fifties"]

if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

for label in labels:
    folder_path = os.path.join(base_dir, label)
    
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist. Skipping...")
        continue
    
    csv_files = glob.glob(os.path.join(folder_path, "batch_*_features.csv"))
    
    if not csv_files:
        print(f"No CSV files matching 'batch_*_features.csv' found in {folder_path}. Skipping...")
        continue
    
    combined_df = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    output_filename = f"{label.lower()}_features.csv"
    output_path = os.path.join(stats_dir, output_filename)
    
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to {output_path} with {len(combined_df)} entries.")