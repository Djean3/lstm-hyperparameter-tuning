# 3. utils/summary_logger.py - summarize model_runs.csv into a single averaged row
import os
import csv
import pandas as pd
import numpy as np

def summarize_model_runs(filename, summary_filename=None):
    df = pd.read_csv(filename)

    # Use best val_loss per replicate
    if "replicate" in df.columns and "val_loss" in df.columns:
        df = df.sort_values("val_loss", ascending=True).drop_duplicates(subset=["replicate"], keep="first")
        print(f"✅ Summarizing best val_loss per replicate: {len(df)} rows")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    summary = df[numeric_cols].mean().to_dict()
    for col in non_numeric_cols:
        try:
            summary[col] = df[col].mode().iloc[0]
        except:
            summary[col] = "N/A"

    summary["total_epochs"] = df['epoch'].nunique() if 'epoch' in df.columns else 0
    summary["total_trials"] = df['trial_id'].nunique() if 'trial_id' in df.columns else 1
    summary["filename"] = filename

    if summary_filename is None:
        summary_filename = filename.replace(".csv", "_summary.csv")

    pd.DataFrame([summary]).to_csv(summary_filename, index=False)
    return summary