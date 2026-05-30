import os
import pandas as pd
import numpy as np
from scipy import stats

CSV_FILE = "group_level_prevalences12.csv"

# Map the 12 validated values to the 4 Higher-Order Foci
FOCI_MAPPING = {
    "Openness to Change": [
        'Self-direction: action', 'Stimulation', 'Hedonism'
    ],
    "Self-Enhancement": [
        'Achievement', 'Power: resources'
    ],
    "Conservation": [
        'Security: societal', 'Conformity: rules', 'Conformity: interpersonal'
    ],
    "Self-Transcendence": [
        'Benevolence: dependability', 'Universalism: concern', 'Universalism: tolerance', 'Universalism: objectivity'
    ]
}

FEATURES_APPROVED = [
    'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement', 'Power: resources',
    'Security: societal', 'Conformity: rules', 'Conformity: interpersonal', 'Benevolence: dependability',
    'Universalism: concern', 'Universalism: tolerance', 'Universalism: objectivity'
]

def get_col(prefix, feature):
    safe = feature.lower().replace(":", "").replace(" ", "_").replace("-", "_")
    return f"{prefix}_{safe}_corrected"

# ==========================================
# 1. LOAD DATA
# ==========================================
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Could not find CSV file: {CSV_FILE}.")

df = pd.read_csv(CSV_FILE)

# ==========================================
# 2. TEST RQ3: VALUE-MATCHING VS. VALUE-PIVOTING
# ==========================================
print("\n" + "="*65)
print("  TEST 1: VALUE-MATCHING (SYMMETRY) VS. VALUE-PIVOTING (PIVOT)")
print("="*65)

dist_win_list = []
dist_lose_list = []

for _, row in df.iterrows():
    op_vector = []
    win_vector = []
    lose_vector = []
    
    for focus, features in FOCI_MAPPING.items():
        op_vector.append(np.mean([row[get_col("op", f)] for f in features]))
        win_vector.append(np.mean([row[get_col("win", f)] for f in features]))
        lose_vector.append(np.mean([row[get_col("lose", f)] for f in features]))
        
    op_vector = np.array(op_vector)
    win_vector = np.array(win_vector)
    lose_vector = np.array(lose_vector)
    
    # Calculate Euclidean distances
    dist_win = np.linalg.norm(op_vector - win_vector)
    dist_lose = np.linalg.norm(op_vector - lose_vector)
    
    dist_win_list.append(dist_win)
    dist_lose_list.append(dist_lose)

mean_dist_win = np.mean(dist_win_list)
mean_dist_lose = np.mean(dist_lose_list)

# Paired t-test
t_stat, p_val = stats.ttest_rel(dist_win_list, dist_lose_list)

print(f"Mean Distance (OP to Winning Rebuttals): {mean_dist_win:.4f}")
print(f"Mean Distance (OP to Losing Rebuttals):  {mean_dist_lose:.4f}")
print(f"Paired t-test result: t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")

if p_val < 0.05:
    print("\nResult: STATISTICALLY SIGNIFICANT")
    if mean_dist_win < mean_dist_lose:
        print("  -> Winning rebuttals are significantly CLOSER to OPs.")
        print("  -> Conclusion supports VALUE-MATCHING (Symmetry) as the dominant mechanism.")
    else:
        print("  -> Winning rebuttals are significantly FURTHER from OPs.")
        print("  -> Conclusion supports VALUE-PIVOTING (Reframing/Pivoting) as the dominant mechanism.")
else:
    print("\nResult: NOT STATISTICALLY SIGNIFICANT")
    print("  -> No statistical difference between matching and pivoting distances at the group level.")

# ==========================================
# 3. TEST RQ2: INDIVIDUAL VALUE PREDICTORS
# ==========================================
print("\n" + "="*65)
print("  TEST 2: INDIVIDUAL VALUE PREVALENCE COMPARISON (RQ2)")
print("="*65)
print(f"{'VALUE CATEGORY':<30} | {'MEAN WIN':<10} | {'MEAN LOSE':<10} | {'P-VALUE':<10} | {'RESULT'}")
print("-" * 85)

significant_predictors = []

for feature in FEATURES_APPROVED:
    win_cols = df[get_col("win", feature)]
    lose_cols = df[get_col("lose", feature)]
    
    mean_win = win_cols.mean()
    mean_lose = lose_cols.mean()
    
    t_stat_f, p_val_f = stats.ttest_rel(win_cols, lose_cols)
    
    status = " "
    if p_val_f < 0.05:
        if mean_win > mean_lose:
            status = "SIG (POSITIVE PREDICTOR)"
            significant_predictors.append((feature, "Positive", p_val_f))
        else:
            status = "SIG (NEGATIVE PREDICTOR)"
            significant_predictors.append((feature, "Negative", p_val_f))
            
    print(f"{feature:<30} | {mean_win:<10.4f} | {mean_lose:<10.4f} | {p_val_f:<10.4e} | {status}")

# ==========================================
# 4. SUMMARY REPORT
# ==========================================
print("\n" + "="*65)
print("  METHODOLOGICAL SUMMARY FOR YOUR THESIS")
print("="*65)
print(f"1. Total Groups Analyzed: {len(df)}")
if len(significant_predictors) > 0:
    print("2. Significant Value Predictors Detected:")
    for pred, direction, p in significant_predictors:
        print(f"   - {pred:<28} ({direction:>8}) | p = {p:.4e}")
else:
    print("2. Significant Value Predictors Detected: None")
print("="*65 + "\n")