<div align="center">

# Robust Wi-Fi Fingerprint Indoor Localization

### Under Access-Point Loss and RSSI Noise

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn 1.9](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset: UJIIndoorLoc](https://img.shields.io/badge/Dataset-UJIIndoorLoc-4C8BF5)](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
[![Paper: under review](https://img.shields.io/badge/Paper-under%20review-8A2BE2)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**How do Wi-Fi localization models fail when access points disappear, and can simple
corruption-aware training make them degrade gracefully?**

[Key results](#key-results) · [Method](#method-at-a-glance) · [Quick start](#quick-start) ·
[Notebooks](#notebook-guide) · [Reproducibility](#reproducibility-protocol) · [Citation](#citation)

</div>

---

## Overview

Wi-Fi fingerprinting localizes a device from the signal strengths (RSSI) of nearby access points (APs).
It works well on clean benchmark data, but in real buildings APs go offline and readings fluctuate.
This repository contains the complete, auditable pipeline behind our study of those two failure modes
on the public **UJIIndoorLoc** benchmark (3 buildings, 13 building-floor classes, 520 APs):

- **Two tasks:** building-floor classification and 2-D coordinate localization.
- **Three model families:** K-nearest neighbors, Random Forest and a multilayer perceptron (MLP).
- **A 6 × 4 corruption grid:** loss of 0–50 % of the APs *actually observed* in each fingerprint, crossed
  with Gaussian RSSI noise of σ = 0, 2, 5 and 10 dB.
- **Corruption-aware training:** AP-dropout, RSSI-noise and combined augmentation.
- **A frozen held-out test set** (1,111 fingerprints), never used for model selection, with paired-bootstrap
  confidence intervals.

> [!IMPORTANT]
> **Main finding.** Losing access points is far more damaging than RSSI noise. Training on fingerprints with
> randomly dropped APs recovers a large part of the lost performance, and the benefit is not explained by the
> extra training data.

<p align="center">
  <img src="results/figures/08_final_severe_classification.png" alt="Severe-condition classification: clean vs AP-dropout training" width="48%">
  &nbsp;
  <img src="results/figures/08_final_severe_localization.png" alt="Severe-condition localization: clean vs combined augmentation" width="48%">
</p>
<p align="center"><sub>Held-out performance under the severe condition: 50 % detected-AP loss plus σ = 10 dB RSSI noise.</sub></p>

---

## Key results

### Primary analysis (frozen held-out test set)

Severe condition: **50 % detected-AP loss + σ = 10 dB RSSI noise**, five corruption seeds.

| Task | Clean-trained MLP | Robust MLP | Improvement | 95 % CI |
|---|---:|---:|---:|---:|
| Building-floor accuracy (AP-dropout) | 68.53 % | **74.24 %** | **+5.71 pp** | [+4.41, +7.06] pp |
| Mean localization error (combined) | 30.16 | **15.24** | **−49.5 %** | [14.21, 15.63] units reduction |
| Clean-input accuracy (AP-dropout) | 91.09 % | 90.55 % | −0.54 pp | [−1.80, +0.63] pp |

Across all 24 grid conditions, AP-dropout training raised mean accuracy from 0.822 to 0.848, and combined
augmentation cut mean localization error from 18.60 to 11.93 coordinate units.

### Reviewer-requested controls (post-hoc)

These analyses were added during peer review, after the primary test evaluation was frozen. They test
specific threats to interpretation and were **not** used to change model selection.

| Question | Control | Result (severe condition) |
|---|---|---|
| Is the gain just more training data? | Repeated-clean model trained on the **same number of instances** (43,062) | AP-dropout still **+4.86 pp** accuracy; combined still **−16.42 units (−50.1 %)** error |
| Is it specific to the MLP? | Same augmentation applied to **Random Forest** | **+11.18 pp** accuracy; **−21.15 units (−54.6 %)** error |
| Are five seeds enough? | **20 corruption seeds** (42–61) | MLP **+5.20 pp** / **−16.28 units**; RF **+11.25 pp** / **−21.07 units** |
| Does it hold for shared outages? | **Global AP outage**: the same APs removed from every fingerprint | At 50 % outage: **+5.42 pp** (no noise), **+4.36 pp** (σ = 10 dB) for the MLP |

Confidence intervals for the 20-seed analysis are reported in the paper (Table 10).

<p align="center">
  <img src="results/figures/11_equal_volume_severe_classification.png" alt="Equal-volume control: classification" width="48%">
  &nbsp;
  <img src="results/figures/12_rf_severe_classification.png" alt="Random Forest augmentation: classification" width="48%">
</p>
<p align="center"><sub>Left: equal-volume control (MLP). Right: Random Forest with augmentation.</sub></p>

> [!NOTE]
> Localization errors are in **dataset coordinate units**, the planar coordinates supplied with UJIIndoorLoc.
> They have not been converted to verified meters, so a 16-unit reduction is not a 16 m improvement.

---

## Method at a glance

```mermaid
flowchart LR
    A[UJIIndoorLoc<br/>19,937 train · 1,111 test] --> B[Preprocessing<br/>100 → −110 dBm<br/>MinMax on train only]
    B --> C[Stratified split<br/>15,949 train · 3,988 dev]
    C --> D[Clean baselines<br/>KNN · RF · MLP]
    D --> E[Corruption grid<br/>6 AP-loss × 4 noise levels]
    E --> F[Robust training<br/>AP-dropout · noise · combined]
    F --> G[Select on dev only]
    G --> H[Frozen held-out test<br/>+ paired bootstrap]
    H -.post-hoc.-> I[Reviewer controls<br/>equal volume · RF · 20 seeds · global outage]
```

**Corruption operators** (applied in dBm space, before scaling)

| Operator | What it does |
|---|---|
| **AP loss** | Removes a fraction *p* of the APs *observed* in each fingerprint (at least one when *p* > 0) and sets them to −110 dBm. With ~18 observed APs per fingerprint, 50 % loss removes about 9 of them. |
| **RSSI noise** | Adds N(0, σ²) to surviving observed APs only, clipped to [−104, 0] dBm. |
| **Global AP outage** | Removes the same randomly chosen AP identities from every fingerprint, simulating a shared infrastructure failure. |

**Models**

| Model | Configuration |
|---|---|
| KNN | k = 5, distance weighting |
| Random Forest | 300 trees, random state 42 |
| MLP | 128 → 64 ReLU units; Adam (lr 0.001, batch 200, L2 1e-4); early stopping (patience 15, max 300 epochs); 75,789 / 75,074 parameters (classifier / regressor) |

The observation mask is used only to decide which APs may be corrupted; it is never a model input.

---

## Quick start

**Requirements:** Python 3.11. The exact package versions are pinned in `requirements.txt`.

```bash
git clone https://github.com/me-nwa/robust-wifi-indoor-localization.git
cd robust-wifi-indoor-localization

python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Get the data.** Download UJIIndoorLoc from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) and place the two files here
(the raw data is not redistributed):

```text
data/raw/
├── trainingData.csv
└── validationData.csv
```

**Run.** Execute the notebooks in numerical order with Jupyter:

```bash
jupyter lab
```

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10      primary study
11 → 12 → 13                                          reviewer-requested controls
```

Notebooks 11–13 reuse the processed data and models produced by Notebooks 02, 03 and 07.

---

## Notebook guide

| Notebook | Stage | Purpose |
|---|---|---|
| `01_dataset_exploration` | Data | Sparsity, AP visibility and label structure |
| `02_preprocessing` | Data | Missing-AP encoding, stratified train/dev split, scaling |
| `03_clean_baselines` | Baselines | KNN, Random Forest and MLP on clean input |
| `04_ap_loss_experiments` | Degradation | Detected-AP loss from 0 % to 50 % |
| `05_rssi_noise_experiments` | Degradation | Gaussian RSSI noise from σ = 0 to 10 dB |
| `06_robust_training` | Degradation | Combined AP-loss × RSSI-noise grid |
| `07_final_analysis` | Robust training | Augmentation regimes and development-set ablation |
| `08_final_evaluation` | Final test | Frozen held-out evaluation over the full grid |
| `09_statistical_analysis` | Statistics | Paired bootstrap for the severe condition |
| `10_final_figures_and_tables` | Publication | Paper figures and LaTeX tables |
| `11_equal_volume_control` | Review | Equal-volume repeated-clean control (MLP) |
| `12_random_forest_robust_training` | Review | Augmentation applied to Random Forest |
| `13_extended_robustness_analysis` | Review | 20 seeds, global AP outages, confusion matrices, spatial error maps |

---

## Repository structure

```text
robust-wifi-indoor-localization/
├── data/
│   ├── raw/                 # UJIIndoorLoc CSVs (download separately)
│   └── processed/           # regenerated by Notebook 02
├── models/                  # trained models (regenerated by the notebooks)
├── notebooks/               # 01–13, run in order
├── results/
│   ├── csv/                 # every table, per-seed result and metadata file
│   └── figures/             # exploratory and diagnostic plots
├── paper/
│   ├── figures/             # publication figures (PDF)
│   └── tables/              # LaTeX tables
├── requirements.txt
└── research_plan.md
```

---

## Reproducibility protocol

- **No test leakage in model selection.** The official validation file is the final test set. Models and
  augmentation regimes were selected on a stratified development split of the training file only.
- **Training-only preprocessing.** The RSSI scaler and coordinate scaler are fitted on the training subset.
- **Fixed seeds.** Split and models use random state 42. Corruption seeds are 42–46 (primary analysis) and 42–61
  (20-seed analysis). Augmentation seeds are 142 and 242.
- **Leakage-safe early stopping in the controls.** Notebook 11 fixes the early-stopping subset *before*
  augmentation, so a fingerprint and its corrupted copies never sit on opposite sides of the split.
- **Recorded provenance.** Training times, software versions and platform details are saved in
  `results/csv/11_training_summary.csv`, `results/csv/12_rf_training_summary.csv` and
  `results/csv/13_reproducibility_metadata.json`.
- **Post-hoc disclosure.** Notebooks 11–13 were run after the frozen test evaluation. They are sensitivity
  analyses, not new confirmatory tests. Further tuning against the same 1,111-fingerprint test set is discouraged;
  new hypotheses should be checked on new data.

---

## Limitations

- **Single benchmark.** The study uses only UJIIndoorLoc; replication on independent datasets is future work.
- **Synthetic failures.** Corruptions are simulated. Spatial AP coverage, network topology, device offsets,
  temporal drift and fading below receiver sensitivity are not modelled.
- **Conditional uncertainty.** Confidence intervals describe uncertainty within this test set and corruption
  process, not across buildings, devices or real deployments.
- **Units.** Errors are in dataset coordinate units, not verified meters.

---

## Citation

The accompanying paper is currently under review. Citation details will be added here once it is published.
Until then, please reference this repository by its URL:
`https://github.com/me-nwa/robust-wifi-indoor-localization`.

**Releases.** `v1.0-den702-paper` is the original submission. Tag the reviewed version as well, so each paper
version maps to an exact commit.

**Dataset.** If you use UJIIndoorLoc, please also cite: J. Torres-Sospedra *et al.*, "UJIIndoorLoc: A new
multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems,"
*IPIN 2014*, pp. 261–270, doi:[10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492).

---

## License

The code in this repository is released under the [MIT License](LICENSE). The UJIIndoorLoc dataset is subject to
its own terms and is not redistributed here.

## Contact

Questions and issues: please open a [GitHub issue](https://github.com/me-nwa/robust-wifi-indoor-localization/issues).

<sub>Prepared as part of DEN702 — Advanced Research Communication, Abu Dhabi University.</sub>
