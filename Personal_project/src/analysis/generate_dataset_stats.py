import os
import pandas as pd
import numpy as np

# Adjust path to project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.load_data import load_raw_cdc

TABLES_DIR = "paper/generated_tables"
os.makedirs(TABLES_DIR, exist_ok=True)

def generate_dataset_stats_table():
    print("Generating Dataset Statistics Table...")
    
    # Load data
    df, _ = load_raw_cdc()
    
    # Calculate stats
    stats = []
    
    # 1. Total samples
    total_samples = len(df)
    
    # 2. Class balance
    # Assuming target is usually the first column or named 'Diabetes_binary'
    target_col = 'Diabetes_binary'
    if target_col not in df.columns:
        target_col = df.columns[0]
        
    class_counts = df[target_col].value_counts()
    neg_count = class_counts.get(0, 0)
    pos_count = class_counts.get(1, 0)
    neg_perc = (neg_count / total_samples) * 100
    pos_perc = (pos_count / total_samples) * 100
    
    # 3. Feature types (simplified)
    # We know there are 21 features + 1 target = 22 cols usually.
    # Let's just list a few key features stats.
    
    # Create a summary table content
    # We will create a small table with: Feature Name | Type | Mean | Std | Min | Max
    # But for the paper, a "Dataset Characteristics" table is better.
    
    # Table 1: Dataset Characteristics
    # Row 1: Total Samples
    # Row 2: Features
    # Row 3: Class 0 (Healthy)
    # Row 4: Class 1 (Diabetic)
    # Row 5: Missing Values
    
    table_content = r"""
\begin{table}[htbp]
\caption{Dataset Characteristics (CDC Diabetes Health Indicators)}
\begin{center}
\begin{tabular}{|l|c|}
\hline
\textbf{Characteristic} & \textbf{Value} \\
\hline
Total Samples & %d \\
Total Features & %d \\
\hline
Class 0 (Non-Diabetic) & %d (%.1f\%%) \\
Class 1 (Diabetic) & %d (%.1f\%%) \\
Class Ratio & 1 : %.1f \\
\hline
Missing Values & 0 (Imputed) \\
Feature Types & Numerical & Ordinal \\
\hline
\end{tabular}
\label{tab:dataset_stats}
\end{center}
\end{table}
""" % (
    total_samples, 
    len(df.columns) - 1, # Exclude target
    neg_count, neg_perc,
    pos_count, pos_perc,
    neg_count / pos_count if pos_count > 0 else 0
)

    save_path = os.path.join(TABLES_DIR, "table_dataset_stats.tex")
    with open(save_path, "w") as f:
        f.write(table_content)
    
    print(f"Saved dataset stats table to {save_path}")

if __name__ == "__main__":
    generate_dataset_stats_table()
