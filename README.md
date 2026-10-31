# Robust Wi-Fi Fingerprint-Based Indoor Localization Under Access-Point Loss and RSSI Noise

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Research](https://img.shields.io/badge/Research-Reproducible-success)
![Status](https://img.shields.io/badge/Status-Reviewer%20Revision%20Round%202-green)

A reproducible machine-learning study investigating the robustness of
Wi-Fi fingerprint-based indoor localization under **access-point (AP) loss**
and **RSSI measurement noise**.

The study evaluates classical and neural machine-learning models under
controlled Wi-Fi degradation and investigates whether **AP-dropout and
RSSI-noise augmentation during training** can improve robustness.

The experiments use the **UJIIndoorLoc** benchmark and evaluate both:

- **Building-floor classification**
- **Coordinate localization**

The original primary experimental pipeline was frozen before the
reviewer-requested post-hoc analyses. The additional controls reported in
Notebooks 11–13 do not retroactively alter the original model-selection
decisions.

---

# Research Overview

Wi-Fi fingerprint-based indoor localization can achieve strong performance
under clean conditions, but real-world Wi-Fi environments are dynamic.

Access points may:

- disappear,
- become temporarily unavailable,
- change signal strength,
- experience interference,
- or produce noisy RSSI measurements.

A model that performs well on clean fingerprints may therefore experience
substantial degradation after deployment.

This project systematically evaluates these effects and investigates
robustness-oriented training strategies.

---

# Main Research Questions

**RQ1.** How does increasing Wi-Fi access-point loss affect indoor
localization performance?

**RQ2.** How does RSSI measurement noise affect indoor localization
performance?

**RQ3.** Can AP-dropout and RSSI-noise augmentation during training improve
robustness?

**RQ4.** Which evaluated machine-learning models maintain stronger
performance under degraded Wi-Fi conditions?

---

# Research Hypothesis

> Models trained using simulated access-point loss and RSSI perturbation
> will exhibit lower performance degradation under corrupted Wi-Fi
> conditions than models trained only on clean Wi-Fi fingerprints.

---

# Key Findings

The experiments show that **access-point loss is substantially more
disruptive than independent Gaussian RSSI noise alone**.

Robustness-oriented training improves performance particularly when the
Wi-Fi environment becomes heavily degraded.

<p align="center">
  <img
    src="results/figures/08_final_severe_classification.png"
    alt="Final severe-condition classification comparison"
    width="700"
  >
</p>

<p align="center">
  <em>
    Final held-out classification performance under
    50% detected-AP loss and RSSI noise σ = 10 dB.
  </em>
</p>

---

# Primary Frozen Held-Out Results

## Classification

Under the pre-specified severe condition:

**50% detected-AP loss + Gaussian RSSI noise σ = 10 dB**

| Training strategy | Accuracy |
|---|---:|
| Clean-trained MLP | **68.53%** |
| AP-dropout-trained MLP | **74.24%** |
| Improvement | **+5.71 percentage points** |

The paired hierarchical bootstrap produced:

**95% CI: +4.41 to +7.06 percentage points**

The interval excludes zero under the specified benchmark corruption process.

### Clean-condition trade-off

| Training strategy | Accuracy |
|---|---:|
| Clean-trained MLP | **91.09%** |
| AP-dropout-trained MLP | **90.55%** |
| Difference | **−0.54 percentage points** |

The clean-condition difference had a 95% bootstrap confidence interval of:

**−1.80 to +0.63 percentage points**

The interval includes zero, so the experiment does not provide clear
evidence of a meaningful clean-condition classification penalty.

---

# Primary Frozen Localization Results

For coordinate localization, the selected robustness strategy used
**combined AP-dropout and RSSI-noise augmentation**.

Under the severe condition:

**50% detected-AP loss + Gaussian RSSI noise σ = 10 dB**

| Training strategy | Mean localization error |
|---|---:|
| Clean-trained MLP | **30.1632** |
| Combined-augmentation MLP | **15.2405** |
| Absolute reduction | **14.9227** |
| Relative reduction | **≈ 49.5%** |

> Localization errors are reported in **dataset coordinate units**.
> No unsupported conversion to physical meters is assumed.

<p align="center">
  <img
    src="results/figures/08_final_severe_localization.png"
    alt="Final severe-condition localization comparison"
    width="700"
  >
</p>

<p align="center">
  <em>
    Mean localization error under severe combined degradation.
  </em>
</p>

---

# Robustness Across the Primary Evaluation Grid

The primary robustness evaluation contains:

- **6 AP-loss levels**
- **4 RSSI-noise levels**
- **24 combined corruption conditions**
- **5 stochastic corruption seeds for each non-clean condition**

AP-loss levels:

```text
0%, 10%, 20%, 30%, 40%, 50%
```

RSSI Gaussian noise:

```text
σ = 0, 2, 5, 10 dB
```

## Classification

Across all 24 conditions:

| Training regime | Mean accuracy | Mean Macro F1 | Minimum accuracy | Maximum accuracy |
|---|---:|---:|---:|---:|
| Clean | 0.8218 | 0.8085 | 0.6853 | 0.9122 |
| AP Dropout | **0.8483** | **0.8361** | **0.7424** | 0.9055 |

## Localization

Across all 24 conditions:

| Training regime | Mean error | Median-condition error | Minimum error | Maximum error |
|---|---:|---:|---:|---:|
| Clean | 18.5953 | 16.9085 | 12.0538 | 30.1632 |
| Combined augmentation | **11.9318** | **11.6042** | **10.1136** | **15.2405** |

---

# Robustness Heatmap

<p align="center">
  <img
    src="results/figures/08_final_accuracy_improvement_heatmap.png"
    alt="Accuracy improvement heatmap"
    width="760"
  >
</p>

<p align="center">
  <em>
    Improvement from robustness-oriented training across
    AP-loss and RSSI-noise conditions.
  </em>
</p>

Additional figures are available in:

```text
results/figures/
```

---

# Reviewer-Requested Robustness Controls

The revision process introduced additional analyses to test specific threats
to the interpretation of the original frozen study.

These analyses are **post-hoc sensitivity controls**. They strengthen the
internal-validity assessment but do not replace the original frozen primary
evaluation.

## Equal-Volume Repeated-Clean Control

The original robust MLP regimes contained the clean training set plus two
corrupted copies. Notebook 11 tests whether the observed robustness gain can
be explained by increased training volume alone.

A fixed early-stopping subset is created **before augmentation**, preventing
clean and corrupted variants of the same fingerprint from appearing on
opposite sides of the stopping split.

Under the severe condition:

| MLP classification regime | Accuracy |
|---|---:|
| Repeated clean | 0.6857 |
| AP dropout | **0.7343** |
| Improvement | **+4.86 percentage points** |

For localization:

| MLP localization regime | Mean error |
|---|---:|
| Repeated clean | 32.7550 |
| Combined augmentation | **16.3305** |
| Reduction | **16.4245 coordinate units** |

These controls show that simple repetition of clean training examples did not
reproduce the robustness benefit.

They do not establish corruption exposure as the sole causal mechanism,
because augmentation may also change optimization trajectories and effective
sample diversity.

---

# Random-Forest Augmentation Control

Notebook 12 evaluates whether corruption-aware augmentation benefits extend
beyond the MLP architecture.

Under severe corruption:

| RF classification regime | Accuracy |
|---|---:|
| Repeated clean | 0.6506 |
| AP dropout | **0.7624** |
| Improvement | **+11.18 percentage points** |

For localization:

| RF localization regime | Mean error |
|---|---:|
| Repeated clean | 38.7053 |
| Combined augmentation | **17.5568** |
| Reduction | **21.1485 coordinate units** |

The direction of benefit therefore extends to Random Forest and is not
specific to the neural architecture.

For the RF reviewer controls:

```text
n_estimators = 300
random_state = 42
bootstrap = default scikit-learn behavior
n_jobs = -1
```

Concatenating augmented copies retains the original labels and therefore
preserves the original class distribution.

No additional class-stratified RF sampling or class reweighting is applied.

---

# Twenty-Seed Severe-Condition Sensitivity Analysis

Notebook 13 increases severe-condition stochastic evaluation from five seeds
to twenty:

```text
42–61
```

Because the same 1,111 held-out fingerprint identities are evaluated under
every corruption seed, the repeated evaluations are not treated as
independent observations.

A **5,000-replicate crossed two-way cluster bootstrap** separately resamples:

1. corruption-seed identities, and
2. fingerprint identities.

The same resampled fingerprint identities are retained across all selected
seeds within each replicate.

## MLP

| Task | Equal-volume baseline | Robust | Effect | 95% CI |
|---|---:|---:|---:|---:|
| Classification | 0.6867 | 0.7387 | +0.0520 | [0.0405, 0.0644] |
| Localization | 32.5340 | 16.2535 | 16.2806 | [15.2554, 17.3044] |

## Random Forest

| Task | Equal-volume baseline | Robust | Effect | 95% CI |
|---|---:|---:|---:|---:|
| Classification | 0.6519 | 0.7644 | +0.1125 | [0.0996, 0.1257] |
| Localization | 38.7541 | 17.6802 | 21.0739 | [19.7566, 22.4384] |

These confidence intervals quantify uncertainty associated with fingerprint
sampling and stochastic corruption realizations **conditional on the fixed
UJIIndoorLoc test set and the specified synthetic corruption process**.

They are not confidence intervals over independently sampled buildings,
devices, users, or real-world deployments.

---

# Structured Global AP-Outage Analysis

Notebook 13 additionally evaluates a structured AP-loss mechanism in which
the same selected AP identities are removed across fingerprints for each
corruption seed.

This provides a shared-infrastructure-failure sensitivity test in addition
to independent per-fingerprint AP dropout.

Global AP-outage levels:

```text
10%, 30%, 50%
```

The strongest nominal global outage produced approximately:

```text
49.04% effective loss of originally observed AP evidence
```

At this level, MLP robustness-oriented training produced:

| RSSI noise | Classification gain | Localization-error reduction |
|---|---:|---:|
| σ = 0 dB | +5.42 percentage points | 17.8805 coordinate units |
| σ = 10 dB | +4.36 percentage points | 17.9774 coordinate units |

This structured operator is more realistic than independent scan-level AP
loss for a shared infrastructure outage, but it still does not reproduce:

- physical AP coverage geometry,
- power/network topology,
- outage duration,
- temporal trajectories,
- motion dependence,
- or device-specific scanning behavior.

---

# Diagnostic Visualizations

The reviewer-control analysis also includes:

- normalized building-floor confusion matrices,
- spatial localization-error scatter plots,
- MLP severe-condition diagnostics,
- Random-Forest severe-condition outputs,
- and machine-readable per-condition result files.

Diagnostic plots generated from a single corruption realization are labeled
as such and should not be interpreted as repeated-seed averages.

Aggregate conclusions are based on the repeated-seed analyses.

---

# Dataset

This study uses the **UJIIndoorLoc** Wi-Fi fingerprinting benchmark.

| Characteristic | Value |
|---|---:|
| Official training fingerprints | 19,937 |
| Held-out final-test fingerprints | 1,111 |
| Wi-Fi AP features | 520 |
| Buildings | 3 |
| Floors | 5 |
| Building-floor classes | 13 |
| Undetected AP proportion | 96.54% |
| Average observed APs per fingerprint | ≈ 18 |
| Median observed APs per fingerprint | 17 |
| Weakest observed training RSSI | −104 dBm |

The original dataset represents undetected access points using:

```text
100
```

For model input, this project converts undetected APs to:

```text
-110 dBm
```

A Boolean observation mask is retained before replacement.

**Important:** the observation mask is used only to determine which AP
measurements are eligible for corruption. It is **not supplied as an input
feature** to KNN, Random Forest, or MLP.

---

# Experimental Design

```text
                    UJIIndoorLoc
                         │
                         ▼
                Dataset Exploration
                         │
                         ▼
                   Preprocessing
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
     Building-Floor            Coordinate
      Classification          Localization
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                 Clean Baselines
                KNN / RF / MLP
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       AP Loss       RSSI Noise      Combined
      Evaluation      Evaluation     Degradation
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                 Robust Training
             AP / Noise / Combined
                         │
                         ▼
                 Development-Based
                  Model Selection
                         │
                         ▼
                Frozen Held-Out Test
                         │
                         ▼
                Primary Bootstrap
                         │
                         ▼
              Reviewer Controls 11–13
```

---

# Machine-Learning Models

## K-Nearest Neighbors

```text
n_neighbors = 5
weights = distance
```

KNN provides a distance-based fingerprint-matching baseline.

---

## Random Forest

```text
n_estimators = 300
random_state = 42
```

Random Forest provides a nonlinear ensemble baseline.

Reviewer-control RF runs additionally use parallel CPU execution with:

```text
n_jobs = -1
```

---

## Multilayer Perceptron

Architecture:

```text
Input
  ↓
128 hidden units
  ↓
64 hidden units
  ↓
Output
```

Configuration:

```text
Activation: ReLU
Optimizer: Adam
Learning rate: 0.001
Learning-rate schedule: constant
Batch size: 200
L2 regularization alpha: 1e-4
Adam beta1: 0.9
Adam beta2: 0.999
Adam epsilon: 1e-8
Shuffle: enabled
Random state: 42
Maximum iterations: 300
Early-stopping patience: 15
Custom initializer: none
Custom learning-rate scheduler: none
```

Model sizes:

```text
MLP classifier: 75,789 trainable parameters
MLP regressor:  75,074 trainable parameters
```

---

# Corruption Models

## Access-Point Loss

For each Wi-Fi fingerprint, a fraction of the **currently observed APs** is
randomly removed.

Tested AP-loss fractions:

```text
0%, 10%, 20%, 30%, 40%, 50%
```

Dropped APs are represented as:

```text
-110 dBm
```

Already-missing APs are not selected again.

The primary AP-loss operator samples removal independently for each
fingerprint.

---

## RSSI Noise

Gaussian perturbation is applied to surviving observed RSSI measurements:

```text
RSSI_noisy = RSSI + ε
```

where:

```text
ε ~ N(0, σ²)
```

and:

```text
σ ∈ {0, 2, 5, 10} dB
```

Observed noisy values are clipped to:

```text
[-104, 0] dBm
```

Missing or dropped APs remain at:

```text
-110 dBm
```

In combined conditions:

1. AP loss is applied first.
2. RSSI noise is added only to surviving observed APs.

Corruption is generated in the original dBm space before the
training-fitted scaler is applied.

---

# Threshold-Crossing Fading Extension

The primary Gaussian RSSI-noise model deliberately separates amplitude
perturbation from AP visibility loss.

Consequently, surviving observed RSSI measurements are clipped to the
observed range and Gaussian noise alone cannot make an observed AP become
missing.

A deployment-oriented extension could simulate threshold-crossing fading by:

1. applying RSSI noise or fading in dBm space,
2. applying a receiver-sensitivity threshold after perturbation,
3. converting sufficiently weakened values below that threshold to the
   missing-AP representation,
4. and updating the observation mask accordingly.

This mechanism was not included in the primary study because the experiment
intentionally separated RSSI-amplitude perturbation from AP-visibility loss
for interpretability.

It is a priority for future deployment-oriented robustness evaluation.

---

# Robust Training

Three MLP augmentation strategies were investigated.

## AP-Dropout Augmentation

Training fingerprints are augmented with simulated AP-loss conditions.

## RSSI-Noise Augmentation

Training fingerprints are augmented with Gaussian RSSI perturbations.

## Combined Augmentation

Both AP dropout and RSSI noise are introduced during training.

The primary development-set ablation selected:

```text
Classification:
AP-dropout augmentation

Localization:
Combined AP-dropout + RSSI-noise augmentation
```

The original robust regimes contained one clean copy plus two corrupted
training copies.

The reviewer-requested equal-volume experiment separately controls for this
increased training volume.

---

# Early-Stopping Scope

For the **original frozen MLP experiment**, the internal 10% early-stopping
fraction was drawn after concatenation of the augmented training data.

Therefore, clean and corrupted variants of the same underlying fingerprint
could in principle appear on opposite sides of the internal stopping split.

This does **not** contaminate:

- the independent development subset, or
- the held-out final test set.

However, the original stopping criterion may be optimistic.

The reviewer-control reruns remove this issue by creating a fixed
early-stopping subset **before augmentation** and augmenting only the
remaining model-fitting fingerprints.

Positive robustness gains persist under this leakage-resistant procedure.

---

# Evaluation Metrics

## Classification

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Confusion Matrix
```

Macro averaging gives equal importance to each building-floor class.

## Localization

```text
Coordinate MAE
Coordinate RMSE
Mean Euclidean localization error
Median localization error
90th-percentile localization error
```

The Euclidean error for sample `i` is:

```text
e_i = sqrt(
    (x_pred - x_true)^2 +
    (y_pred - y_true)^2
)
```

Localization results are reported in **dataset coordinate units** because the
underlying coordinate reference system was not independently verified and
converted to metric ground distance.

---

# Statistical Analysis

## Primary analysis

Each non-clean corruption condition in the frozen primary study uses
corruption seeds:

```text
42, 43, 44, 45, 46
```

The severe-condition primary comparison uses a paired hierarchical bootstrap
with:

```text
5,000 replicates
```

## Reviewer-control analysis

The severe-condition sensitivity analysis uses:

```text
20 corruption seeds: 42–61
```

and a crossed two-way cluster bootstrap with:

```text
5,000 replicates
```

Bootstrap seeds:

```text
Classification: 23042
Localization:   23043
```

The crossed procedure separately resamples seed identities and fingerprint
identities while preserving the same selected fingerprint identities across
the selected corruption seeds.

---

# Reproducibility and Machine-Readable Results

Machine-readable numerical outputs are stored under:

```text
results/csv/
```

Reviewer-control outputs include files for:

- equal-volume MLP controls,
- Random-Forest augmentation controls,
- 20-seed severe-condition sensitivity analysis,
- structured global AP-outage analysis,
- crossed-bootstrap summaries,
- reproducibility metadata,
- training-time summaries,
- and additional second-round statistical artifacts.

Key currently reported artifacts include:

```text
results/csv/13_crossed_cluster_bootstrap_summary.csv
```

Additional detailed CSVs generated during the second-round revision should
remain in `results/csv/` and be referenced by their exact filenames in the
final manuscript Data Availability section.

Figures are stored under:

```text
results/figures/
```

Publication-ready manuscript artifacts are stored under:

```text
paper/
```

---

# Repository Structure

```text
wifi-localization-research/
│
├── README.md
├── requirements.txt
├── research_plan.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── trainingData.csv
│   │   └── validationData.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_clean_baselines.ipynb
│   ├── 04_ap_loss_experiments.ipynb
│   ├── 05_rssi_noise_experiments.ipynb
│   ├── 06_robust_training.ipynb
│   ├── 07_final_analysis.ipynb
│   ├── 08_final_evaluation.ipynb
│   ├── 09_statistical_analysis.ipynb
│   ├── 10_final_figures_and_tables.ipynb
│   ├── 11_equal_volume_control.ipynb
│   ├── 12_random_forest_robust_training.ipynb
│   └── 13_extended_robustness_analysis.ipynb
│
├── models/
│
├── results/
│   ├── csv/
│   └── figures/
│
├── paper/
│   ├── figures/
│   ├── tables/
│   └── manuscript artifacts
│
└── src/
```

---

# Notebook Workflow

Notebooks **01–10** represent the original frozen primary workflow.

Notebooks **11–13** contain reviewer-requested post-hoc controls.

| Notebook | Purpose |
|---|---|
| `01_dataset_exploration.ipynb` | Dataset exploration, sparsity analysis, and quality checks |
| `02_preprocessing.ipynb` | Train/development/test preparation and scaling |
| `03_clean_baselines.ipynb` | KNN, Random Forest, and MLP clean baselines |
| `04_ap_loss_experiments.ipynb` | Controlled AP-loss robustness |
| `05_rssi_noise_experiments.ipynb` | Controlled RSSI-noise robustness |
| `06_robust_training.ipynb` | Corruption-aware MLP training and augmentation ablation |
| `07_final_analysis.ipynb` | Development analysis and model-selection summaries |
| `08_final_evaluation.ipynb` | Frozen held-out test evaluation |
| `09_statistical_analysis.ipynb` | Primary bootstrap analysis |
| `10_final_figures_and_tables.ipynb` | Publication figures and tables |
| `11_equal_volume_control.ipynb` | Equal-volume and leakage-resistant MLP control |
| `12_random_forest_robust_training.ipynb` | Random-Forest augmentation control |
| `13_extended_robustness_analysis.ipynb` | 20-seed analysis, crossed bootstrap, global outages, and diagnostics |

## Primary frozen workflow

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
```

## Reviewer-requested post-hoc workflow

```text
11 → 12 → 13
```

---

# Reproducing the Experiments

## 1. Clone the repository

```bash
git clone https://github.com/me-nwa/robust-wifi-indoor-localization.git
cd robust-wifi-indoor-localization
```

After the second-round reproducibility release has been published, the exact
revision can be checked out with:

```bash
git checkout v1.2-feedback-round-two
```

## 2. Create the Python environment

The experiments were run using Python 3.11.

### macOS / Linux

```bash
python3.11 -m venv .envden702-wifi
source .envden702-wifi/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bash
py -3.11 -m venv .envden702-wifi
.envden702-wifi\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Obtain UJIIndoorLoc

Download the official UJIIndoorLoc dataset from the UCI Machine Learning
Repository.

Place:

```text
trainingData.csv
validationData.csv
```

under:

```text
data/raw/
```

so that:

```text
data/raw/
├── trainingData.csv
└── validationData.csv
```

The supplied UJIIndoorLoc validation file is used in this project as the
held-out final test set and must not be used for model selection.

## 4. Run the original frozen workflow

Execute:

```text
notebooks/01_dataset_exploration.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_clean_baselines.ipynb
notebooks/04_ap_loss_experiments.ipynb
notebooks/05_rssi_noise_experiments.ipynb
notebooks/06_robust_training.ipynb
notebooks/07_final_analysis.ipynb
notebooks/08_final_evaluation.ipynb
notebooks/09_statistical_analysis.ipynb
notebooks/10_final_figures_and_tables.ipynb
```

in numerical order.

## 5. Run reviewer-requested controls

After the primary workflow, execute:

```text
notebooks/11_equal_volume_control.ipynb
notebooks/12_random_forest_robust_training.ipynb
notebooks/13_extended_robustness_analysis.ipynb
```

These notebooks reproduce:

- equal-volume repeated-clean controls,
- leakage-resistant early stopping,
- Random-Forest augmentation,
- 20-seed severe-condition analysis,
- crossed two-way cluster bootstrap,
- structured global AP outages,
- confusion matrices,
- spatial localization diagnostics,
- and second-round reproducibility outputs.

---

# Reproducibility Protocol

## Training/development separation

The official UJIIndoorLoc training file is divided into:

```text
Model-training subset: 15,949 fingerprints
Development subset:     3,988 fingerprints
```

The split is stratified by combined building-floor class.

## Final test set

The supplied validation file contains:

```text
1,111 fingerprints
```

and is treated as the held-out final test set.

## Scaling

Feature scaling is fitted using training data only.

Corruption is applied in dBm space **before** the already-fitted scaling
transform.

## Model selection

Primary model and augmentation selection is performed using the development
subset.

## Final evaluation

Primary final models are frozen before held-out testing.

Reviewer-requested controls are clearly labeled as post-hoc sensitivity
analyses and do not retroactively change the original model selection.

---

# Limitations

The study has several important limitations.

- Evaluation is based on the single UJIIndoorLoc benchmark.
- The primary AP-loss model uses independent per-fingerprint removal.
- The primary Gaussian RSSI model uses independent zero-mean perturbations.
- AP-specific failure probabilities estimated from longitudinal
  infrastructure data are not modeled.
- Device-dependent RSSI offsets are not modeled.
- Threshold-crossing fading is not represented in the primary noise model.
- Motion-conditioned fading is not modeled.
- Spatially and temporally correlated physical outages are only partially
  approximated by the structured global AP-outage control.
- Development results come from a random row-level split of the training
  collection and may therefore be optimistic.
- The original frozen MLP early-stopping split could contain clean and
  corrupted variants of the same underlying fingerprint on opposite sides
  of the stopping split.
- The reviewer-control reruns remove this early-stopping issue, but the
  original primary models are retained for reproducibility.
- Localization error remains in dataset coordinate units rather than verified
  metric distance.
- The study evaluates benchmark-level predictive robustness rather than
  route-level navigation performance, latency, energy use, or real-world
  deployment reliability.

## Priorities for future robustness modeling

The highest-priority extensions are:

1. **AP-specific failure probabilities** estimated from longitudinal network
   logs.
2. **Device-dependent RSSI offsets** across hardware platforms.
3. **Threshold-crossing fading** that allows weak signals to become
   undetected.
4. **Spatially and temporally correlated outages** based on real
   infrastructure topology.
5. **Motion-conditioned fading and trajectory-level evaluation**.

Replication on an additional independently collected indoor-localization
dataset is also required before claiming cross-dataset generality.

---

# Future Work

Potential extensions include:

- validation on additional indoor-localization benchmarks,
- longitudinal Wi-Fi fingerprint collection,
- AP-specific reliability modeling,
- device-independent robustness evaluation,
- threshold-crossing fading,
- reliability-aware AP weighting,
- uncertainty-aware localization,
- calibrated prediction under corruption,
- domain adaptation across buildings,
- hybrid Wi-Fi / BLE / inertial localization,
- trajectory-level robustness evaluation,
- and real-time mobile deployment studies.

---

# Academic Context

This repository accompanies the study:

> **Robust Wi-Fi Fingerprint-Based Indoor Localization Under Access-Point Loss and RSSI Noise**

The work was prepared as an original PhD-level research project for
**DEN702 — Advanced Research Communication** at **Abu Dhabi University**.

---

# Research Paper

The manuscript reports:

- problem formulation,
- related work and research gap,
- mathematical corruption models,
- machine-learning methodology,
- corruption-aware augmentation,
- baseline experiments,
- ablation analysis,
- held-out evaluation,
- bootstrap statistical analysis,
- reviewer-requested robustness controls,
- diagnostic visualizations,
- limitations,
- reproducibility information,
- and future research directions.

The manuscript source is maintained separately in Overleaf.

---

# Citation

Until formal publication details are available:

```bibtex
@unpublished{aamena2026robust,
  title  = {Robust Wi-Fi Fingerprint-Based Indoor Localization
            Under Access-Point Loss and RSSI Noise},
  author = {Mohamed, Aamena },
  year   = {2026},
  note   = {PhD research project, Abu Dhabi University}
}
```

---

# Repository Versioning

Development for the second reviewer round is performed on:

```text
feedback-round-two
```

After all second-round reviewer changes are finalized, the exact
reproducibility version will be archived under:

```text
v1.2-feedback-round-two
```

The tag should be created only after the manuscript, notebooks, results,
README, and reviewer-response package are finalized.

Example final commands:

```bash
git add .
git commit -m "Finalize reviewer feedback round two revision"
git push

git tag -a v1.2-feedback-round-two \
  -m "Final revision addressing reviewer feedback round two"

git push origin v1.2-feedback-round-two
```

---

# Data Availability

The study uses the publicly available **UJIIndoorLoc** benchmark.

The raw benchmark should be obtained from its official distribution rather
than unnecessarily redistributed in this repository.

Place the original files under:

```text
data/raw/
```

Processed data can then be regenerated through the documented notebook
workflow.

---

# Code Availability

The public repository is:

https://github.com/me-nwa/robust-wifi-indoor-localization

The repository contains the code required for:

- preprocessing,
- baseline training,
- corruption generation,
- robustness evaluation,
- MLP augmentation,
- Random-Forest augmentation controls,
- held-out evaluation,
- primary statistical analysis,
- crossed two-way cluster bootstrap,
- structured global AP-outage evaluation,
- diagnostic figures,
- and publication artifacts.

The final second-round reproducibility snapshot will be identified by the
release tag:

```text
v1.2-feedback-round-two
```

---

# Responsible Reproducibility

The held-out UJIIndoorLoc test results must not be treated as a continuing
development benchmark.

The original final evaluation was frozen after model selection.

Reviewer-requested controls were subsequently added to test specific threats
to interpretation and are labeled as post-hoc sensitivity analyses.

Any future method changes should be treated as **new experiments** rather
than used to retroactively modify the original frozen primary results.

---

# License

A software license should be selected before final public archival if one has
not already been added.

Common options for academic research software include:

```text
MIT License
Apache License 2.0
BSD 3-Clause License
```

The repository license applies only to original code and does not override
the terms associated with third-party datasets or external resources.

---

# Contact

**Author:** Aamena Mohamed   
**Program:** PhD in Intelligent Systems Engineering  
**Institution:** Abu Dhabi University  
**Location:** Abu Dhabi, United Arab Emirates  
**Email:** 1109499@students.adu.ac.ae

---

# Project Status

```text
Dataset preparation                         ✅ Complete
Clean baseline experiments                  ✅ Complete
AP-loss experiments                         ✅ Complete
RSSI-noise experiments                      ✅ Complete
Primary corruption-aware training           ✅ Complete
Frozen held-out final evaluation            ✅ Complete
Primary bootstrap analysis                  ✅ Complete
Equal-volume reviewer control               ✅ Complete
Random-Forest augmentation control          ✅ Complete
20-seed sensitivity analysis                ✅ Complete
Crossed two-way cluster bootstrap           ✅ Complete
Structured global AP-outage analysis        ✅ Complete
Confusion-matrix diagnostics                ✅ Complete
Spatial localization diagnostics            ✅ Complete
Second-round reproducibility revision       🔄 In progress
Final v1.2 release tag                      ⏳ Pending
```

**Primary experimental pipeline status: FROZEN**

No additional tuning of the original frozen models should be performed using
the held-out test results.