import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# ============================================================
# USER SETTINGS — change these before running
# ============================================================
SYMMETRY = "P2"   # Options: "P2", "P4", "PM", "PMM"
DATA_DIR = "DATA"

# Model
BOTTLENECK   = 10
DROPOUT      = 0.2

# Training
EPOCHS          = 5000
BATCH_LABELED   = 32
BATCH_UNLABELED = 128
LR              = 5e-4
WEIGHT_DECAY    = 1e-3

# Loss weights (see Section 3.4.3 in paper)
RECON_WEIGHT = 1.0    # lambda_r: labeled reconstruction loss
REG_WEIGHT   = 10.0   # lambda_p: acoustic property prediction loss
UNSUP_WEIGHT = 3.0    # lambda_u: unlabeled reconstruction loss

TARGET_COL   = "alpha_avg"
# ============================================================

# ====================== Utils ======================
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ====================== Convolutional Model ======================
class ConvAE_SemiSupervised(nn.Module):
    """
    Semi-supervised Convolutional Autoencoder with regression head.
    Works with NxN matrices (N = size).
    ✅ Fixed: Uses F.interpolate for exact size control
    """
    def __init__(self, size=10, bottleneck=10, dropout=0.2):
        super().__init__()
        self.size = size
        self.bottleneck = bottleneck
        self.config = {
            'size': size,
            'bottleneck': bottleneck,
            'dropout': dropout,
            'architecture': 'convolutional'
        }
        
        # Encoder: 3 convolutional layers designed for 10x10 binary woven patterns.
        # Two stride-2 convolutions progressively downsample the spatial dimension:
        # Conv1 (stride=1): 10x10 - extract local interlacing features at full resolution
        # Conv2 (stride=2): 10x10 -> 5x5, capture mid-scale structural patterns
        # Conv3 (stride=2):  5x5  -> 3x3, capture global symmetry features
        # The resulting 3x3x128 feature map is then projected to the bottleneck.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        # Compute actual feature map size via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, size, size)
            encoded = self.encoder(dummy)
            self.encoded_size = encoded.shape[2]
            self.encoded_channels = encoded.shape[1]

        print(f"  Encoder path: 1 -> 32 -> 64 -> 128")
        print(f"  Encoded feature map: {self.encoded_channels} x {self.encoded_size} x {self.encoded_size}")
        
        # Bottleneck (fully-connected projection)
        self.flatten_dim = self.encoded_channels * self.encoded_size * self.encoded_size
        print(f"  Flatten dimension: {self.flatten_dim}")
        
        self.fc_encode = nn.Linear(self.flatten_dim, bottleneck)
        self.fc_decode = nn.Linear(bottleneck, self.flatten_dim)
        
        # Decoder: mirrors the encoder using transposed convolutions.
        # 3x3 -> 5x5 -> 10x10, followed by bilinear interpolation to ensure exact output size.
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )
        
        # Verify decoder output shape with a dummy pass
        with torch.no_grad():
            dummy_encoded = torch.zeros(1, self.encoded_channels, 
                                        self.encoded_size, self.encoded_size)
            decoded = self.decoder_conv(dummy_encoded)
            print(f"  Decoder raw output: {decoded.shape[2:]} (before resize)")
            print(f"  Will resize to: ({size}, {size})")
        
        # Regression head
        self.reg_head = nn.Linear(bottleneck, 1)
        
        print(f"  Bottleneck: {self.flatten_dim} -> {bottleneck} -> {self.flatten_dim}")
        print(f"  Regression head: {bottleneck} -> 1")
        
    def encode(self, x):
        """
        Encode input to latent space
        x: (batch, size*size) or (batch, 1, size, size)
        returns: z (batch, bottleneck)
        """
        if x.dim() == 2:
            x = x.view(-1, 1, self.size, self.size)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        return z
    
    def decode(self, z):
        """
        Decode latent to reconstruction
        z: (batch, bottleneck)
        returns: logits (batch, size*size)
        """
        h = self.fc_decode(z)
        h = h.view(-1, self.encoded_channels, self.encoded_size, self.encoded_size)
        out = self.decoder_conv(h)
        
        # Resize to exact (size x size) using bilinear interpolation
        if out.shape[2:] != (self.size, self.size):
            out = F.interpolate(out, size=(self.size, self.size), 
                              mode='bilinear', align_corners=False)
        
        # Flatten: (batch, 1, size, size) -> (batch, size*size)
        out = out.view(-1, self.size * self.size)
        return out
    
    def forward(self, x):
        """
        Forward pass
        x: (batch, size*size) or (batch, size, size)
        returns: 
            - logits: (batch, size*size)
            - z: (batch, bottleneck)
            - y_pred: (batch, 1)
        """
        z = self.encode(x)
        logits = self.decode(z)
        y_pred = self.reg_head(z)
        return logits, z, y_pred
    
    def get_config(self):
        return self.config.copy()

# ====================== Save model + config ======================
def save_model_with_config(model, save_dir, model_filename="conv_semi_supervised_ae_model.pth"):
    _ensure_dir(save_dir)
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    config_path = os.path.join(save_dir, "config.json")
    config = model.get_config()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n✔ Model saved: {model_path}")
    print(f"✔ Config saved: {config_path}")
    print("\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    return model_path, config_path

# ====================== Generic Datasets ======================
class SupervisedWoven(Dataset):
    """
    X: shape (N, size, size) or (N, size*size)
    y: shape (N,)
    """
    def __init__(self, X, y, size):
        self.size = size
        X = X.astype(np.float32)
        if X.ndim == 3:
            assert X.shape[1:] == (size, size), f"X must be (N,{size},{size})"
            X = X.reshape(-1, size*size)
        elif X.ndim == 2:
            assert X.shape[1] == size*size, f"X must have {size*size} features"
        else:
            raise ValueError("X must be 2D or 3D array")
        self.x = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class UnsupervisedWoven(Dataset):
    """
    X: shape (N, size, size) or (N, size*size)
    """
    def __init__(self, X, size):
        self.size = size
        X = X.astype(np.float32)
        if X.ndim == 3:
            assert X.shape[1:] == (size, size), f"X must be (N,{size},{size})"
            X = X.reshape(-1, size*size)
        elif X.ndim == 2:
            assert X.shape[1] == size*size, f"X must have {size*size} features"
        else:
            raise ValueError("X must be 2D or 3D array")
        self.x = torch.from_numpy(X)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]

# ====================== Plotting ======================
def set_journal_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def _subsample_xy(y, step):
    y = np.asarray(y)
    x = np.arange(1, len(y)+1)
    if step is None or step <= 1 or len(y) <= 1:
        return x, y
    idx = np.arange(0, len(y), step)
    if idx[-1] != len(y)-1:
        idx = np.append(idx, len(y)-1)
    return x[idx], y[idx]

def plot_loss(tr_hist, val_hist, save_dir, fname_base="fig1_total_loss",
              figsize=(3.5, 2.6), step=None):
    _ensure_dir(save_dir)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x_tr, y_tr = _subsample_xy(tr_hist, step)
    x_va, y_va = _subsample_xy(val_hist, step)
    ax.plot(x_tr, y_tr, '-k', label="Train")
    ax.plot(x_va, y_va, '--r', label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    png = os.path.join(save_dir, f"{fname_base}.png")
    fig.savefig(png)
    plt.close(fig)
    return png

def plot_recon_loss(tr_hist, val_hist, save_dir, fname_base="fig2_recon_loss",
                    figsize=(3.5, 2.6), step=None):
    _ensure_dir(save_dir)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x_tr, y_tr = _subsample_xy(tr_hist, step)
    x_va, y_va = _subsample_xy(val_hist, step)
    ax.plot(x_tr, y_tr, '-g', label="Train Recon")
    ax.plot(x_va, y_va, '--', color='purple', label="Val Recon")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss (BCE)")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    png = os.path.join(save_dir, f"{fname_base}.png")
    fig.savefig(png)
    plt.close(fig)
    return png

def plot_reg_loss(tr_hist, val_hist, save_dir, fname_base="fig3_reg_loss",
                  figsize=(3.5, 2.6), step=None):
    _ensure_dir(save_dir)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x_tr, y_tr = _subsample_xy(tr_hist, step)
    x_va, y_va = _subsample_xy(val_hist, step)
    ax.plot(x_tr, y_tr, '-b', label="Train Reg")
    ax.plot(x_va, y_va, '--', color='orange', label="Val Reg")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Regression Loss (MSE)")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    png = os.path.join(save_dir, f"{fname_base}.png")
    fig.savefig(png)
    plt.close(fig)
    return png

def plot_accuracy(acc_hist, save_dir, fname_base="fig4_test_accuracy",
                  figsize=(3.5, 2.6), step=None):
    _ensure_dir(save_dir)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x_ac, y_ac = _subsample_xy(acc_hist, step)
    ax.plot(x_ac, y_ac, 'royalblue', marker=None, markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linewidth=0.3, alpha=0.4)
    png = os.path.join(save_dir, f"{fname_base}.png")
    fig.savefig(png)
    plt.close(fig)
    return png

def save_triplet_panel(fname, gt, pr, diff):
    fig = plt.figure(figsize=(6,2), dpi=200)
    mats = [(gt, "GT"), (pr, "Recon"), (diff, "Diff")]
    diff_cmap = ListedColormap(["#ffffff", "#d94143"])
    for i, (m, t) in enumerate(mats, start=1):
        ax = fig.add_subplot(1,3,i, aspect='equal')
        if t == "Diff":
            ax.imshow(m, cmap=diff_cmap, vmin=0, vmax=1, interpolation='nearest')
        else:
            ax.imshow(m, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(t, fontsize=8, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

# ====================== Evaluation ======================


@torch.no_grad()
def evaluate_ae(model, loader, device="cpu", threshold=0.5, max_show=6,
                save_dir=None, prefix="test", size=4):
    model.eval()
    shown = 0
    acc_sum = reg_loss_sum = n_tot = 0
    if save_dir is not None:
        _ensure_dir(save_dir)

    for x, y_true in loader:
        x = x.to(device)
        y_true = y_true.to(device)

        logits, z, y_pred = model(x)
        probs = torch.sigmoid(logits)
        x_recon = (probs > threshold).float()

        b = x.size(0)  # multiply by batch size for correct mean computation
        
        acc_sum += (x_recon == x).float().mean().item() * b
        reg_loss_sum += F.mse_loss(y_pred.squeeze(), y_true, reduction="sum").item()
        n_tot += b

        to_show = min(x.size(0), max_show - shown)
        for i in range(to_show):
            gt = x[i].view(size, size).cpu().numpy().astype(int)
            pr = x_recon[i].view(size, size).cpu().numpy().astype(int)
            diff = (gt != pr).astype(int)
            if shown + i < max_show:
                print(f"\n=== Sample #{shown+i+1} ({prefix}) ===")
                print("GT:");      print(gt)
                print("Recon:");   print(pr)
                print(f"y_true: {y_true[i].item():.4f}, y_pred: {y_pred[i].item():.4f}")
                if save_dir is not None:
                    fname = os.path.join(save_dir, f"{prefix}_sample_{shown+i+1:02d}_panel.png")
                    save_triplet_panel(fname, gt, pr, diff)
        shown += to_show
        if shown >= max_show: break

    return acc_sum/max(n_tot,1), reg_loss_sum/max(n_tot,1)

@torch.no_grad()
def collect_latents(model, loader, device="cpu", max_batches=None):
    model.eval()
    zs = []
    for bidx, batch in enumerate(loader):
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            x = batch
        x = x.to(device)
        _, z, _ = model(x)
        zs.append(z.detach().cpu().numpy())
        if max_batches is not None and (bidx + 1) >= max_batches:
            break
    if len(zs) == 0:
        return np.empty((0, model.config['bottleneck']))
    return np.concatenate(zs, axis=0)

# ====================== Data Loading ======================

def load_labeled_data(csv_path, size=None, y_col="y", normalize_y=True,
                      feature_cols=None, feature_prefix=None):
    print(f"Loading labeled data from: {csv_path}")
    df = pd.read_csv(csv_path)

    X = df[feature_cols].values.astype(np.float32).reshape(-1, size, size)
    y = df[y_col].values.astype(np.float32)
    print(f"  Loaded: {len(X):,} patterns ({size}x{size}) with labels")
    print(f"  y range (raw): [{y.min():.3f}, {y.max():.3f}]")

    scaler_y = None
    if normalize_y:
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        print(f"  y range (normalized): [{y.min():.3f}, {y.max():.3f}]")
    return X, y, scaler_y, size

def load_unlabeled_data(csv_path, size=None, feature_cols=None):
    print(f"Loading unlabeled data from: {csv_path}")
    df = pd.read_csv(csv_path)

    X = df[feature_cols].values.astype(np.float32).reshape(-1, size, size)
    print(f"  Loaded: {len(X):,} patterns ({size}x{size}) without labels")
    return X, size

def split_labeled_data(X, y, test_size=0.15, val_size=0.10, seed=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, shuffle=True
    )
    print(f"\nSplit -> train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ====================== Main ======================
def main():
    # ======= Config (set USER SETTINGS at the top of this file) =======
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fixed: woven pattern is always 10x10
    size            = 10

    bottleneck      = BOTTLENECK
    dropout         = DROPOUT
    epochs          = EPOCHS
    batch_labeled   = BATCH_LABELED
    batch_unlabeled = BATCH_UNLABELED
    lr              = LR
    weight_decay    = WEIGHT_DECAY
    recon_weight    = RECON_WEIGHT
    reg_weight      = REG_WEIGHT
    unsup_weight    = UNSUP_WEIGHT
    y_col           = TARGET_COL

    save_dir      = f"{DATA_DIR}/{SYMMETRY}/output/SemiConvAE_Opt"
    csv_labeled   = f"{DATA_DIR}/{SYMMETRY}/output/train_100.csv"
    csv_unlabeled = f"{DATA_DIR}/{SYMMETRY}/unlabeled_10000.csv"

    # Column selection: binary pattern pixels P00-P99
    feature_cols = [f'P{i}{j}' for i in range(size) for j in range(size)]

    print("=" * 60)
    print("Convolutional Semi-Supervised Autoencoder Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Matrix size: {size}x{size}")
    print(f"Bottleneck: {bottleneck} | Dropout: {dropout}")
    print(f"Architecture: Convolutional (Conv2d + ConvTranspose2d)")
    print(f"Data: labeled={csv_labeled}, unlabeled={csv_unlabeled}")
    print(f"Save dir: {save_dir}")
    print("=" * 60 + "\n")

    _ensure_dir(save_dir)

    # ===== Load & split =====
    X_labeled, y_labeled, scaler_y, size = load_labeled_data(
        csv_labeled, size=size, y_col=y_col, normalize_y=True,
        feature_cols=feature_cols
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_labeled_data(
        X_labeled, y_labeled, test_size=0.15, val_size=0.10, seed=42
    )
    X_unlabeled, size = load_unlabeled_data(
        csv_unlabeled, size=size, feature_cols=feature_cols
    )

    print(f"{'='*60}")
    print("Final Data Summary:")
    print(f"{'='*60}")
    print(f"  Labeled Train:      {len(X_train):>8,} samples")
    print(f"  Labeled Val:        {len(X_val):>8,} samples")
    print(f"  Labeled Test:       {len(X_test):>8,} samples")
    print(f"  Unlabeled:          {len(X_unlabeled):>8,} samples")
    print(f"  {'-'*56}")
    print(f"  Total Training:   {len(X_train) + len(X_unlabeled):>8,} samples")
    print(f"  Unlabeled Ratio: {len(X_unlabeled)/max(len(X_train),1):.1f}x labeled")
    print(f"{'='*60}\n")

    # ===== Dataloaders =====
    dl_tr_labeled = DataLoader(
        SupervisedWoven(X_train, y_train, size),
        batch_size=batch_labeled, shuffle=True, drop_last=True
    )
    dl_val = DataLoader(
        SupervisedWoven(X_val, y_val, size),
        batch_size=batch_labeled, shuffle=False
    )
    dl_te = DataLoader(
        SupervisedWoven(X_test, y_test, size),
        batch_size=batch_labeled, shuffle=False
    )
    dl_unlabeled = DataLoader(
        UnsupervisedWoven(X_unlabeled, size),
        batch_size=batch_unlabeled, shuffle=True, drop_last=True
    )

    # ===== Model =====
    model = ConvAE_SemiSupervised(
        size=size, bottleneck=bottleneck, dropout=dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)

    print(f"\nConvolutional Model created..")
    print(f"  Architecture: Conv2d -> Bottleneck({bottleneck}) -> ConvTranspose2d")
    print(f"  Encoder channels: 1 -> 32 -> 64 -> 128")
    print(f"  Dropout: {dropout}")

    # ===== Training =====
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    tr_total_hist, tr_recon_hist, tr_reg_hist = [], [], []
    val_total_hist, val_recon_hist, val_reg_hist = [], [], []
    acc_hist = []

    # Per-epoch metrics log (saved to CSV at the end of training)
    metrics_log = []

    unlabeled_iter = iter(dl_unlabeled)

    for ep in range(1, epochs+1):
        model.train()
        total_sum = recon_sum = reg_sum = n = 0

        for x, y_batch in dl_tr_labeled:
            x = x.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            # Supervised (labeled) branch
            logits, z, y_pred = model(x)
            recon_loss_labeled = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
            reg_loss = F.mse_loss(y_pred, y_batch, reduction="mean")

            # Unsupervised (unlabeled) branch
            try:
                x_unlabeled = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(dl_unlabeled)
                x_unlabeled = next(unlabeled_iter)
            x_unlabeled = x_unlabeled.to(device)
            logits_unsup, _, _ = model(x_unlabeled)
            recon_loss_unlabeled = F.binary_cross_entropy_with_logits(
                logits_unsup, x_unlabeled, reduction="mean"
            )

            total_loss = (recon_weight * recon_loss_labeled +
                          reg_weight * reg_loss +
                          unsup_weight * recon_loss_unlabeled)

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            b = x.size(0)
            total_sum += total_loss.item() * b
            recon_sum += recon_loss_labeled.item() * b
            reg_sum += reg_loss.item() * b
            n += b

        tr_total_hist.append(total_sum / n)
        tr_recon_hist.append(recon_sum / n)
        tr_reg_hist.append(reg_sum / n)

        # validation
        model.eval()
        with torch.no_grad():
            val_total_sum = val_recon_sum = val_reg_sum = m = 0
            for xv, yv in dl_val:
                xv = xv.to(device)
                yv = yv.to(device).unsqueeze(1)
                lv, _, yv_pred = model(xv)
                recon_v = F.binary_cross_entropy_with_logits(lv, xv, reduction="mean")
                reg_v = F.mse_loss(yv_pred, yv, reduction="mean")
                total_v = recon_weight * recon_v + reg_weight * reg_v

                b = xv.size(0)
                val_total_sum += total_v.item() * b
                val_recon_sum += recon_v.item() * b
                val_reg_sum += reg_v.item() * b
                m += b

            val_total_hist.append(val_total_sum / m)
            val_recon_hist.append(val_recon_sum / m)
            val_reg_hist.append(val_reg_sum / m)

        # Evaluate reconstruction accuracy every 10 epochs
        if ep % 10 == 0 or ep == 1:
            test_acc, _ = evaluate_ae(
                model, dl_te, device=device, threshold=0.5,
                max_show=0, save_dir=None, prefix=f"ep{ep}", size=size
            )
            acc_hist.append(test_acc)
            print(f"epoch {ep:04d} | train_loss {tr_total_hist[-1]:.4f} | "
                  f"val_loss {val_total_hist[-1]:.4f} | "
                  f"recon {tr_recon_hist[-1]:.4f} | reg {tr_reg_hist[-1]:.4f} | "
                  f"test_acc {test_acc:.4f}")
        else:
            acc_hist.append(None)

        # Append metrics for this epoch
        metrics_log.append({
            "epoch": ep,
            "train_total_loss": tr_total_hist[-1],
            "train_recon_loss": tr_recon_hist[-1],
            "train_reg_loss":   tr_reg_hist[-1],
            "val_total_loss":   val_total_hist[-1],
            "val_recon_loss":   val_recon_hist[-1],
            "val_reg_loss":     val_reg_hist[-1],
            "test_accuracy":    acc_hist[-1],  # None if not evaluated this epoch
        })

    # Save full training log as CSV
    log_df = pd.DataFrame(metrics_log)
    log_csv_path = os.path.join(save_dir, "training_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    print(f"✔ Training log saved: {log_csv_path}")

    # Save model weights, config, and scalers
    save_model_with_config(model, save_dir)
    if scaler_y is not None:
        scaler_path = os.path.join(save_dir, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler_X': None, 'scaler_y': scaler_y}, f)
        print(f"✔ Scalers saved: {scaler_path}")

        # update config with normalization info
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['normalization'] = {
            'y_normalized': True,
            'y_mean': float(scaler_y.mean_[0]),
            'y_std': float(scaler_y.scale_[0])
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✔ Config updated with normalization info")

    # Final evaluation on the held-out test set
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    test_acc, test_reg = evaluate_ae(
        model, dl_te, device=device, threshold=0.5,
        max_show=6, save_dir=save_dir, prefix="test_final", size=size
    )
    print(f"\n✔ Test accuracy: {test_acc:.4f}")
    print(f"✔ Test regression loss: {test_reg:.4f}")

    # Save final test metrics to CSV
    final_test_csv = os.path.join(save_dir, "final_test_metrics.csv")
    pd.DataFrame([{
        "test_accuracy": test_acc,
        "test_regression_loss": test_reg
    }]).to_csv(final_test_csv, index=False)
    print(f"✔ Final test metrics saved: {final_test_csv}")

    # Export latent vectors and percentile bounds for the optimizer
    Z_tr = collect_latents(model, dl_tr_labeled, device=device)
    latent_csv_path = os.path.join(save_dir, "latent_space.csv")
    pd.DataFrame(Z_tr, columns=[f"z{i+1}" for i in range(model.config['bottleneck'])]).to_csv(latent_csv_path, index=False)

    ranges_df = pd.DataFrame({
        "min": Z_tr.min(axis=0),
        "max": Z_tr.max(axis=0),
        "p0.5": np.percentile(Z_tr, 0.5, axis=0),
        "p99.5": np.percentile(Z_tr, 99.5, axis=0)
    }, index=[f"z{i+1}" for i in range(model.config['bottleneck'])])
    ranges_csv_path = os.path.join(save_dir, "latent_ranges.csv")
    ranges_df.to_csv(ranges_csv_path)

    print(f"\n✔ Latent space: {latent_csv_path}")
    print(f"✔ Latent ranges: {ranges_csv_path}")

    # Generate training curve plots
    set_journal_style()
    plot_loss(tr_total_hist, val_total_hist, save_dir, step=5)
    plot_recon_loss(tr_recon_hist, val_recon_hist, save_dir, step=5)
    plot_reg_loss(tr_reg_hist, val_reg_hist, save_dir, step=5)
    acc_hist_filtered = [a for a in acc_hist if a is not None]
    if len(acc_hist_filtered) > 0:
        plot_accuracy(acc_hist_filtered, save_dir, fname_base="fig4_test_accuracy", step=1)
        print(f"\n✔ Created test accuracy plot")

    print("\n✅ Training completed successfully!")
    print(f"All outputs saved in: {save_dir}")

if __name__ == "__main__":
    main()