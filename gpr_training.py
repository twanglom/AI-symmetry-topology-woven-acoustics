"""
Train GPR on Convolutional Autoencoder Latent Space (PyEGRO)
=============================================================
Workflow:
  1. Load pre-trained Semi-ConvAE
  2. Encode labeled data into the latent space
  3. Train a GPR surrogate model on the latent vectors
  4. Evaluate on held-out test set and save results

Dependencies:
  pip install PyEGRO  (https://twanglom.github.io/PyEGRO/)
  pip install torch numpy pandas scikit-learn
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch

from semi_convae import ConvAE_SemiSupervised

# ============================================================
# USER SETTINGS — change these before running
# ============================================================
SYMMETRY    = "P2"   # Options: "P2", "P4", "PM", "PMM"
DATA_DIR    = "DATA"
TRAIN_FRAC  = "100"  # Labeled fraction: "10","20","30","50","80","100"
TARGET_COL  = "alpha_avg"

# GPR training hyperparameters
NUM_ITERATIONS = 1000
KERNEL         = "rbf"
LEARNING_RATE  = 0.01
PATIENCE       = 20
# ============================================================


# ============================================================
# Encoder loading
# ============================================================
def load_encoder(model_path, device="cpu"):
    """
    Load the trained Semi-ConvAE encoder from a checkpoint.

    Returns
    -------
    encode_fn : callable
        Function mapping (N, D) numpy array -> (N, bottleneck) latent array.
    config : dict
        Model configuration (size, bottleneck, dropout, architecture).
    scalers : dict or None
        Normalization scalers saved during training.
    """
    model_dir  = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    arch = config.get("architecture", "mlp")
    if arch != "convolutional":
        raise ValueError(
            f"Expected 'convolutional' architecture, got '{arch}'."
        )

    model = ConvAE_SemiSupervised(
        size=config["size"],
        bottleneck=config["bottleneck"],
        dropout=config["dropout"],
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"Loaded model  : {model_path}")
    print(f"Architecture  : {arch.upper()}, bottleneck={config['bottleneck']}")

    # Load normalization scalers
    scalers = None
    scalers_path = os.path.join(model_dir, "scalers.pkl")
    if os.path.exists(scalers_path):
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        print(f"Loaded scalers: {scalers_path}")

    @torch.no_grad()
    def encode_fn(X):
        """Encode (N, D) numpy array to (N, bottleneck) latent vectors."""
        X = np.asarray(X, dtype=np.float32)
        z = model.encode(torch.from_numpy(X).to(device))
        return z.cpu().numpy()

    return encode_fn, config, scalers


# ============================================================
# Data loading
# ============================================================
def load_data(csv_train, csv_test, target_col, exclude_cols=None):
    """Load train and test CSVs; return features and targets as numpy arrays."""
    df_train = pd.read_csv(csv_train)
    df_test  = pd.read_csv(csv_test)

    exclude = {target_col}
    if exclude_cols:
        exclude.update(exclude_cols)
    feature_cols = [c for c in df_train.columns if c not in exclude]

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train[target_col].values.reshape(-1, 1).astype(np.float32)
    X_test  = df_test[feature_cols].values.astype(np.float32)
    y_test  = df_test[target_col].values.reshape(-1, 1).astype(np.float32)

    print(f"\nTrain: {X_train.shape[0]} samples x {X_train.shape[1]} features")
    print(f"Test : {X_test.shape[0]}  samples x {X_test.shape[1]}  features")
    print(f"y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"y_test  range: [{y_test.min():.4f},  {y_test.max():.4f}]")

    return X_train, y_train, X_test, y_test, feature_cols


# ============================================================
# Latent bounds
# ============================================================
def load_latent_bounds(latent_dir, bottleneck_dim, Z_train=None):
    """
    Load percentile-based latent bounds from latent_ranges.csv.
    Falls back to empirical bounds from Z_train if file is missing.
    Bounds are used to constrain the GA search space during optimization.
    """
    ranges_path = os.path.join(latent_dir, "latent_ranges.csv")
    try:
        df = pd.read_csv(ranges_path, index_col=0)
        bounds = np.column_stack([df["p0.5"].values, df["p99.5"].values])
        print(f"Loaded latent bounds: {ranges_path}")
    except Exception:
        if Z_train is not None:
            bounds = np.column_stack([
                np.percentile(Z_train, 0.5,  axis=0),
                np.percentile(Z_train, 99.5, axis=0),
            ])
            print("Using empirical bounds from training data (0.5th–99.5th percentile).")
        else:
            bounds = np.tile([-5.0, 5.0], (bottleneck_dim, 1))
            print("Using default bounds [-5, 5].")
    return bounds


# ============================================================
# Main training function
# ============================================================
def train_gpr(
    csv_train,
    csv_test,
    model_path,
    target_col   = "alpha_avg",
    output_dir   = "results_gpr_latent",
    device       = None,
    num_iterations = 1000,
    kernel         = "rbf",
    learning_rate  = 0.01,
    patience       = 20,
):
    """
    Train a GPR surrogate model on the ConvAE latent space.

    Steps
    -----
    1. Load data from CSV files.
    2. Load pre-trained Semi-ConvAE encoder.
    3. Encode patterns into the latent space.
    4. Train GPR with PyEGRO on latent vectors.
    5. Save results and evaluation plots.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("GPR TRAINING ON AUTOENCODER LATENT SPACE")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Train data : {csv_train}")
    print(f"Test data  : {csv_test}")
    print(f"AE model   : {model_path}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data (binary patterns as features; exclude JCA parameters)
    jca_cols = ["porosity", "tortuosity", "thermal_length",
                "viscous_length", "permeability", "alpha_max"]
    X_train, y_train, X_test, y_test, _ = load_data(
        csv_train, csv_test, target_col, exclude_cols=jca_cols
    )

    # 2. Load encoder
    encoder, config, scalers = load_encoder(model_path, device)

    # 3. Apply Y-normalization if scalers are available
    if scalers is not None and scalers.get("scaler_X") is not None:
        X_train = scalers["scaler_X"].transform(X_train)
        X_test  = scalers["scaler_X"].transform(X_test)
        print("Applied X normalization from training scalers.")
    else:
        print("No X normalization applied (binary inputs do not require scaling).")

    # 4. Encode to latent space
    Z_train = encoder(X_train)
    Z_test  = encoder(X_test)
    print(f"\nEncoded: {X_train.shape} -> {Z_train.shape}")

    # 5. Load latent bounds
    model_dir     = os.path.dirname(model_path)
    bottleneck_dim = config["bottleneck"]
    latent_names  = [f"z{i+1}" for i in range(bottleneck_dim)]
    bounds        = load_latent_bounds(model_dir, bottleneck_dim, Z_train)

    # 6. Train GPR with PyEGRO
    from PyEGRO.meta.gpr import MetaTraining
    from PyEGRO.meta.gpr.visualization import visualize_gpr

    print("\n" + "=" * 60)
    print("TRAINING GPR")
    print("=" * 60)

    meta = MetaTraining(
        num_iterations  = num_iterations,
        prefer_gpu      = (device == "cuda"),
        show_progress   = True,
        show_hardware_info = False,
        show_model_info    = False,
        output_dir         = output_dir,
        kernel             = kernel,
        learning_rate      = learning_rate,
        patience           = patience,
    )

    meta.train(
        X=Z_train, y=y_train,
        X_test=Z_test, y_test=y_test,
        feature_names=latent_names,
    )

    visualize_gpr(
        meta=meta,
        X_train=Z_train, y_train=y_train,
        X_test=Z_test,   y_test=y_test,
        variable_names=latent_names,
        bounds=bounds,
        savefig=True,
    )

    print(f"\nResults saved in: {output_dir}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    sym        = SYMMETRY
    output_dir = f"{DATA_DIR}/{sym}/output/GPR_Opt"
    model_path = f"{DATA_DIR}/{sym}/output/SemiConvAE_Opt/conv_semi_supervised_ae_model.pth"

    train_gpr(
        csv_train      = f"{DATA_DIR}/{sym}/output/train_{TRAIN_FRAC}.csv",
        csv_test       = f"{DATA_DIR}/{sym}/output/test.csv",
        model_path     = model_path,
        target_col     = TARGET_COL,
        output_dir     = output_dir,
        num_iterations = NUM_ITERATIONS,
        kernel         = KERNEL,
        learning_rate  = LEARNING_RATE,
        patience       = PATIENCE,
    )
