import json
import os
import random
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
LABELED_DATA_FILE = "labeled_main_dataset.json"
ERROR_RATES_FILE = "value_error_rates.json"
OUTPUT_CSV = "group_level_prevalences12.csv"

NUM_GROUPS = 50  
RANDOM_SEED = 42

# Original full list (used for correct index mapping)
FEATURES_ALL = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
    'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
    'Universalism: objectivity'
]

# Noise-free, validated features only (TPR > FPR)
FEATURES_APPROVED = [
    'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement', 'Power: resources',
    'Security: societal', 'Conformity: interpersonal', 'Benevolence: dependability',
    'Universalism: concern', 'Universalism: tolerance', 'Universalism: objectivity'
]

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading files...")
with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
    labeled_data = json.load(f)

with open(ERROR_RATES_FILE, "r", encoding="utf-8") as f:
    error_rates = json.load(f)

# Shuffle data
random.seed(RANDOM_SEED)
random.shuffle(labeled_data)

# ==========================================
# 2. DEFINE GROUPS
# ==========================================
print(f"Dividing {len(labeled_data)} pairs into {NUM_GROUPS} random groups...")
groups = np.array_split(labeled_data, NUM_GROUPS)

# ==========================================
# 3. QUANTIFICATION PIPELINE (ACC FORMULA)
# ==========================================
def run_acc_correction(raw_prevalence, tpr, fpr):
    denominator = tpr - fpr
    if denominator == 0:
        return raw_prevalence
    corrected = (raw_prevalence - fpr) / denominator
    return float(np.clip(corrected, 0.0, 1.0))

group_rows = []

print("Running noise-free quantification...")
for group_idx, group in enumerate(groups):
    group_size = len(group)
    row_data = {
        "group_id": group_idx + 1,
        "group_size": group_size
    }
    
    for feature in FEATURES_APPROVED:
        # Match index in original 20-class output
        feat_idx = FEATURES_ALL.index(feature)
        
        tpr = error_rates[feature]["tpr"]
        fpr = error_rates[feature]["fpr"]
        
        # Extract predictions
        op_preds = [item["op_context"]["predicted_binary"][feat_idx] for item in group]
        win_preds = [item["winning_rebuttal"]["predicted_binary"][feat_idx] for item in group]
        lose_preds = [item["losing_rebuttal"]["predicted_binary"][feat_idx] for item in group]
        
        # Calculate raw prevalences
        op_raw_prev = sum(op_preds) / group_size
        win_raw_prev = sum(win_preds) / group_size
        lose_raw_prev = sum(lose_preds) / group_size
        
        # Apply corrections
        op_corr_prev = run_acc_correction(op_raw_prev, tpr, fpr)
        win_corr_prev = run_acc_correction(win_raw_prev, tpr, fpr)
        lose_corr_prev = run_acc_correction(lose_raw_prev, tpr, fpr)
        
        safe_feature_name = feature.lower().replace(":", "").replace(" ", "_").replace("-", "_")
        
        row_data[f"op_{safe_feature_name}_raw"] = op_raw_prev
        row_data[f"op_{safe_feature_name}_corrected"] = op_corr_prev
        row_data[f"win_{safe_feature_name}_raw"] = win_raw_prev
        row_data[f"win_{safe_feature_name}_corrected"] = win_corr_prev
        row_data[f"lose_{safe_feature_name}_raw"] = lose_raw_prev
        row_data[f"lose_{safe_feature_name}_corrected"] = lose_corr_prev
        
    group_rows.append(row_data)

# ==========================================
# 4. SAVE GROUP-LEVEL CSV
# ==========================================
df_output = pd.DataFrame(group_rows)
df_output.to_csv(OUTPUT_CSV, index=False)
print(f"Quantification completed successfully. Output saved to: {OUTPUT_CSV}")
