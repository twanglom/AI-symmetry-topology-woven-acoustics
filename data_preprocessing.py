import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# USER SETTINGS — change these before running
# ============================================================
SYMMETRY = "P2"   # Options: "P2", "P4", "PM", "PMM"
DATA_DIR  = "DATA"
# ============================================================

input_dir  = f"{DATA_DIR}/{SYMMETRY}"
output_dir = f"{DATA_DIR}/{SYMMETRY}/output"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# ============================
# Step 1: Load full labeled dataset (300 samples)
# Columns: P00–P99 (binary woven pattern) + JCA transport parameters + alpha_avg + alpha_max
# ============================
df = pd.read_csv(f"{input_dir}/labeled_results_objectives.csv")

# ============================
# Step 2: Split — hold-out test set (10% = 30 samples)
# Fixed random_state=42 for reproducibility
# ============================
train_full, test = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    shuffle=True,
)

test.to_csv(f"{output_dir}/test.csv", index=False)
print(f"Saved test.csv  : {len(test)} samples")

# ============================
# Step 3: Generate labeled training subsets at varying fractions
# Used for the label-scarcity experiment (Section 4.1.2 of the paper)
# ============================
ratios = {
    "10":  0.10,
    "20":  0.20,
    "30":  0.30,
    "50":  0.50,
    "80":  0.80,
    "100": 1.00,
}

N = len(train_full)

for name, r in ratios.items():
    n_samples = int(N * r)
    subset = train_full.sample(n=n_samples, replace=False, random_state=42)
    out_path = f"{output_dir}/train_{name}.csv"
    subset.to_csv(out_path, index=False)
    print(f"Saved train_{name}.csv : {n_samples} samples")

print("\nAll CSV files generated successfully.")
