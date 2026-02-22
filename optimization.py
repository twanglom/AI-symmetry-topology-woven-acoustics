"""
GA Topology Optimization on Conv-AE Latent Space
=================================================
Workflow:
  1. Load pre-trained Semi-ConvAE and GPR surrogate models.
  2. Run a Genetic Algorithm (GA) in the compact latent space.
  3. Robustness test: N independent runs to quantify stability.
  4. Uncertainty quantification: GPR predictive mean and sigma.
  5. Decode the optimal latent vector to a binary woven pattern.

Dependencies:
  pip install PyEGRO   (https://twanglom.github.io/PyEGRO/)
  pip install pymoo torch numpy pandas scikit-learn matplotlib
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR
from semi_convae import ConvAE_SemiSupervised

# ============================================================
# USER SETTINGS — change these before running
# ============================================================
SYMMETRY    = "P2"   # Options: "P2", "P4", "PM", "PMM"
DATA_DIR    = "DATA"

POPULATION_SIZE    = 100
N_GENERATIONS      = 55
N_ROBUSTNESS_RUNS  = 30
SEED               = 42
CHECKPOINT_GENS    = [0, 10, 20, 40, 50]
USE_DECODER        = True
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_MODEL_PATH      = f"{DATA_DIR}/{SYMMETRY}/output/SemiConvAE/conv_semi_supervised_ae_model.pth"
AE_DIR             = os.path.dirname(AE_MODEL_PATH)
LATENT_RANGES_PATH = os.path.join(AE_DIR, "latent_ranges.csv")
GPR_MODEL_DIR      = f"{DATA_DIR}/{SYMMETRY}/output/GPR"
TRAIN_LATENT_PATH  = f"{DATA_DIR}/{SYMMETRY}/output/SemiConvAE_Opt/train_latent_vectors.npy"
OPT_RESULTS_DIR    = f"{DATA_DIR}/{SYMMETRY}/output/Optimized_Result_UQ"


# ============================================================
# Utility
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_latent_bounds(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Latent ranges not found: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    xl = df["p0.5"].values.astype(float)
    xu = df["p99.5"].values.astype(float)
    return xl, xu


# ============================================================
# Model loading
# ============================================================
def load_gpr(model_dir, prefer_gpu=True):
    gpr = DeviceAgnosticGPR(prefer_gpu=prefer_gpu)
    gpr.load_model(model_dir=model_dir)
    return gpr


def make_gpr_predict_fn(gpr):
    def predict(z):
        y_pred, _ = gpr.predict(z.reshape(1, -1))
        return float(y_pred[0, 0])
    return predict


def load_convae(model_path, device="cpu"):
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    with open(config_path) as f:
        config = json.load(f)
    model = ConvAE_SemiSupervised(
        size=config["size"],
        bottleneck=config["bottleneck"],
        dropout=config["dropout"],
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    print(f"Loaded ConvAE: bottleneck={config['bottleneck']}")
    return model


@torch.no_grad()
def decode_from_latent(model, z, device, threshold=0.5):
    """Map a latent vector to logits, probability map, and binary pattern."""
    size   = model.size
    z_t    = torch.from_numpy(z.reshape(1, -1)).float().to(device)
    logits = model.decode(z_t)
    probs  = torch.sigmoid(logits)
    binary = (probs > threshold).float()
    return (
        logits.view(size, size).cpu().numpy(),
        probs.view(size, size).cpu().numpy(),
        binary.view(size, size).cpu().numpy().astype(int),
    )


# ============================================================
# Visualization helpers
# ============================================================
def plot_convergence(res, save_path):
    gens = np.arange(1, len(res.history) + 1)
    best = [-algo.opt.get("F")[0] for algo in res.history]
    plt.figure(figsize=(6, 4))
    plt.plot(gens, best, "b-", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Broadband Absorption ($\\bar{\\alpha}$)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def save_checkpoints(res, model, device, base_dir, checkpoints):
    ckpt_dir = os.path.join(base_dir, "checkpoints_history")
    ensure_dir(ckpt_dir)
    max_gen = len(res.history)
    for g in checkpoints:
        if g >= max_gen:
            continue
        algo   = res.history[g]
        best_z = algo.opt.get("X")[0]
        raw_f  = algo.opt.get("F")[0]
        score  = float(-raw_f[0] if hasattr(raw_f, "__len__") else -raw_f)
        _, probs, _ = decode_from_latent(model, best_z, device)
        plt.figure(figsize=(3.5, 3))
        im = plt.imshow(probs, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        plt.title(f"Gen {g}, Pred {score:.3f}", fontsize=12)
        plt.xticks([]); plt.yticks([])
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, f"gen_{g:03d}_score_{score:.4f}.png"), dpi=150)
        plt.close()
        np.save(os.path.join(ckpt_dir, f"gen_{g:03d}.npy"), best_z)
    print("Checkpoints saved.")


def visualize_evolution(res, model, device, save_path, gens_to_show):
    max_gen    = len(res.history)
    valid_gens = sorted(set([g for g in gens_to_show if g < max_gen] + [max_gen - 1]))
    n_cols     = len(valid_gens)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.2))
    if n_cols == 1:
        axes = [axes]
    for ax, g in zip(axes, valid_gens):
        best_z  = res.history[g].opt.get("X")[0]
        raw_f   = res.history[g].opt.get("F")[0]
        score   = float(-raw_f[0] if hasattr(raw_f, "__len__") else -raw_f)
        _, probs, _ = decode_from_latent(model, best_z, device)
        im = ax.imshow(probs, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Gen {g}, Pred {score:.3f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Evolution of Woven Topology (Probability Map)", y=0.96, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_final_triplet(logits, probs, binary, out_path, title="Optimization Result"):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    im0 = axes[0].imshow(logits, cmap="coolwarm", interpolation="nearest")
    axes[0].set_title("Logits"); axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(probs, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Probability"); axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(binary, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Binary Pattern"); axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)




# ============================================================
# Pymoo problem definition
# ============================================================
class LatentGPRProblem(ElementwiseProblem):
    """Minimize negative GPR prediction in the latent space (= maximize absorption)."""
    def __init__(self, xl, xu, gpr_predict_fn):
        super().__init__(n_var=len(xl), n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.gpr_predict_fn = gpr_predict_fn

    def _evaluate(self, z, out, *args, **kwargs):
        out["F"] = -self.gpr_predict_fn(z)


# ============================================================
# Robustness test
# ============================================================
def run_robustness_test(problem, termination, save_dir, n_runs):
    """
    Run the GA n_runs times with different seeds and report
    mean ± std of the final best fitness.
    """
    print(f"\nROBUSTNESS TEST — {n_runs} independent runs")
    all_histories, final_results = [], []

    for i in range(n_runs):
        algo = GA(pop_size=POPULATION_SIZE, eliminate_duplicates=True, seed=i + 100)
        res  = minimize(problem, algo, termination, verbose=False, save_history=True)
        curve = [float(-a.opt.get("F")[0][0]) for a in res.history]
        all_histories.append(curve)
        final_results.append(curve[-1])
        print(f"  Run {i+1:02d}/{n_runs}: {curve[-1]:.10f}")

    all_histories = np.array(all_histories)
    mean_val = np.mean(final_results)
    std_val  = np.std(final_results)
    print(f"\nMean: {mean_val:.10f}  Std: {std_val:.10f}")

    # Plot
    gens       = np.arange(1, all_histories.shape[1] + 1)
    mean_curve = all_histories.mean(axis=0)
    std_curve  = all_histories.std(axis=0)
    plt.figure(figsize=(6, 4))
    for h in all_histories:
        plt.plot(gens, h, color="blue", alpha=0.1, linewidth=1)
    plt.plot(gens, mean_curve, color="blue", linewidth=2, label="Mean")
    plt.fill_between(gens, mean_curve - 1.96 * std_curve,
                     mean_curve + 1.96 * std_curve, color="blue", alpha=0.1, label="95% CI")
    plt.xlabel("Generation")
    plt.ylabel("Broadband Absorption ($\\bar{\\alpha}$)")
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "robustness_convergence.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")

    return mean_val, std_val, final_results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("GA INVERSE DESIGN + UNCERTAINTY QUANTIFICATION")
    print(f"Symmetry group : {SYMMETRY}")
    print(f"Device         : {DEVICE}")
    print("=" * 60)

    ensure_dir(OPT_RESULTS_DIR)

    # 1. Load latent bounds
    xl, xu = load_latent_bounds(LATENT_RANGES_PATH)
    print(f"Latent dimension: {len(xl)}")

    # 2. Load GPR surrogate
    gpr_loader  = load_gpr(GPR_MODEL_DIR, prefer_gpu=(DEVICE == "cuda"))
    gpr_predict = make_gpr_predict_fn(gpr_loader)

    # 3. Load ConvAE decoder (for visualization)
    model_ae = None
    if USE_DECODER:
        try:
            model_ae = load_convae(AE_MODEL_PATH, DEVICE)
        except Exception as e:
            print(f"Warning: ConvAE not loaded ({e}). Visualization disabled.")

    # 4. Define optimization problem
    problem     = LatentGPRProblem(xl, xu, gpr_predict)
    termination = get_termination("n_gen", N_GENERATIONS)

    # 5. Robustness test (N runs)
    mean_opt, std_opt, all_runs = run_robustness_test(
        problem, termination, OPT_RESULTS_DIR, N_ROBUSTNESS_RUNS
    )

    # 6. Final visualization run
    print(f"\nFinal visualization run (seed={SEED})...")
    algorithm = GA(pop_size=POPULATION_SIZE, eliminate_duplicates=True, seed=SEED)
    res = minimize(problem, algorithm, termination,
                   seed=SEED, verbose=True, save_history=True)
    best_z = res.X

    # 7. Uncertainty quantification
    pred_mean, pred_std = gpr_loader.predict(best_z.reshape(1, -1))
    mu    = float(pred_mean[0, 0])
    sigma = float(pred_std[0, 0])
    print(f"\nPredicted mean (mu)  : {mu:.6f}")
    print(f"Predictive std (sigma): {sigma:.6f}")
    print(f"95% CI               : [{mu - 1.96*sigma:.6f}, {mu + 1.96*sigma:.6f}]")

    # 8. Save results
    np.save(os.path.join(OPT_RESULTS_DIR, "best_latent_z.npy"), best_z)
    with open(os.path.join(OPT_RESULTS_DIR, "optimization_report.txt"), "w") as f:
        f.write(f"Symmetry: {SYMMETRY}\n\n")
        f.write(f"Robustness (N={N_ROBUSTNESS_RUNS})\n")
        f.write(f"  Mean: {mean_opt:.10f}\n  Std:  {std_opt:.10f}\n\n")
        f.write(f"Best candidate\n")
        f.write(f"  Mu:    {mu:.10f}\n  Sigma: {sigma:.10f}\n")
        f.write(f"  95% CI: [{mu-1.96*sigma:.10f}, {mu+1.96*sigma:.10f}]\n\n")
        f.write(f"Best latent vector: {best_z.tolist()}\n")

    # 9. Visualizations
    if model_ae is not None:
        plot_convergence(res, os.path.join(OPT_RESULTS_DIR, "convergence_plot.png"))
        save_checkpoints(res, model_ae, DEVICE, OPT_RESULTS_DIR, CHECKPOINT_GENS)
        visualize_evolution(res, model_ae, DEVICE,
                            os.path.join(OPT_RESULTS_DIR, "evolution_strip.png"),
                            CHECKPOINT_GENS)
        logits, probs, binary = decode_from_latent(model_ae, best_z, DEVICE)
        save_final_triplet(logits, probs, binary,
                           os.path.join(OPT_RESULTS_DIR, "best_design_triplet.png"))
        np.save(os.path.join(OPT_RESULTS_DIR, "best_pattern_binary.npy"), binary)

    print(f"\nAll results saved to: {OPT_RESULTS_DIR}")


if __name__ == "__main__":
    main()
