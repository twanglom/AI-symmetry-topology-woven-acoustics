# AI-Assisted Symmetry-Informed Topology Optimization of Woven Materials for Broadband Sound Absorption

**Thanasak Wanglomklang, Frédéric Gillot, Sébastien Besset, Sonia Mahmoudi**


---

## Overview

This repository provides the code and data accompanying the paper. We propose a symmetry-informed, metamodel-based framework for the inverse design of woven acoustic materials. The pipeline consists of three stages:

1. **Constraint-aware pattern generation** — binary woven patterns are sampled under crystallographic wallpaper group symmetries (p2, p4, pm, pmm), hang-on manufacturability constraints, and topological equivalence rules.
2. **Semi-supervised Convolutional Autoencoder (Semi-ConvAE)** — compresses 10×10 binary topologies into a compact 10-dimensional latent space using both labeled (300 FEM-simulated) and unlabeled (10,000) patterns.
3. **Gaussian Process Regression (GPR) + Genetic Algorithm (GA)** — a GPR surrogate is trained on the latent space and used to guide a GA toward woven patterns with maximized broadband sound absorption.

---

## Repository Structure

```
├── data_preprocessing.py      # Train/test split and labeled subset generation
├── semi_convae.py             # Semi-supervised ConvAE model definition and training
├── gpr_training.py            # GPR surrogate training on the latent space
├── optimization.py            # GA topology optimization + uncertainty quantification
├── requirements.txt           # Python dependencies
└── DATA/
    ├── P2/
    │   ├── labeled_results_objectives.csv   # 300 labeled samples (binary patterns + JCA parameters + alpha)
    │   └── unlabeled_10000.csv              # 10,000 unlabeled feasible patterns
    ├── P4/  (same structure)
    ├── PM/  (same structure)
    └── PMM/ (same structure)
```

---

## Data Description

### `labeled_results_objectives.csv` (300 rows per symmetry group)

| Column group | Columns | Description |
|---|---|---|
| Binary pattern | `P00` – `P99` | 10×10 binary woven topology (0 = warp over weft, 1 = weft over warp) |
| Transport parameters | `porosity`, `tortuosity`, `thermal_length`, `viscous_length`, `permeability` | JCA macroscopic parameters from COMSOL FEM |
| Acoustic output | `alpha_avg`, `alpha_max` | Broadband-averaged and peak sound absorption coefficient |

### `unlabeled_10000.csv` (10,000 rows per symmetry group)

Columns `P00`–`P99` only. These are feasible woven patterns satisfying symmetry and hang-on constraints, sub-sampled from a full pool of 1,000,000 patterns and used for unsupervised reconstruction training.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<username>/symmetry-topology-woven-acoustics.git
cd symmetry-topology-woven-acoustics

# Install dependencies
pip install -r requirements.txt

# Install PyEGRO (GPR library, developed by the authors)
pip install PyEGRO
```

---

## Usage

Run the scripts in the following order. Before running each script, set the `SYMMETRY` variable at the top of the file to one of `"P2"`, `"P4"`, `"PM"`, `"PMM"`.

### Step 1 — Data preprocessing

```bash
python data_preprocessing.py
```

Splits the 300 labeled samples into a held-out test set (30 samples) and training subsets at six fractions (10%, 20%, 30%, 50%, 80%, 100%). Outputs are saved to `DATA/{SYMMETRY}/output/`.

### Step 2 — Train the Semi-ConvAE

```bash
python semi_convae.py
```

Trains the semi-supervised convolutional autoencoder. Key hyperparameters are set at the top of the `main()` function:

| Parameter | Value | Description |
|---|---|---|
| `bottleneck` | 10 | Latent space dimensionality |
| `dropout` | 0.2 | Dropout rate |
| `epochs` | 5000 | Training epochs |
| `reg_weight` | 10.0 | Weight for supervised prediction loss (λ_p) |
| `unsup_weight` | 3.0 | Weight for unlabeled reconstruction loss (λ_u) |

Trained model, config, scalers, and latent vectors are saved to `DATA/{SYMMETRY}/output/SemiConvAE_Opt/`.

### Step 3 — Train the GPR surrogate

```bash
python gpr_training.py
```

Encodes the training data into the latent space and fits a GPR model with an RBF kernel using PyEGRO. Results are saved to `DATA/{SYMMETRY}/output/GPR_Opt/`.

### Step 4 — Topology optimization

```bash
python optimization.py
```

Runs the GA in the latent space to maximize broadband absorption. Performs 30 independent runs for robustness quantification. Outputs include:
- `best_pattern_binary.npy` — optimal 10×10 binary woven pattern
- `optimization_report.txt` — mean ± std across runs and GPR uncertainty
- Convergence plots and topology evolution visualizations

---

## Computational Resources

Training was performed on the Newton HPC cluster at École Centrale de Lyon using an NVIDIA Quadro P4000 GPU.

---

## Requirements

```
torch>=2.0
numpy
pandas
scikit-learn
matplotlib
pymoo
PyEGRO   # pip install PyEGRO  (https://twanglom.github.io/PyEGRO/)
```

Full list: see `requirements.txt`.

---

## Full Dataset

The complete dataset is available from the corresponding author upon reasonable request due to file size. COMSOL `.mph` simulation files are also available upon request.

---

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Thanasak Wanglomklang — thanasak.wanglomklang@ec-lyon.fr  
Laboratoire de Tribologie et Dynamique des Systèmes (LTDS), École Centrale de Lyon