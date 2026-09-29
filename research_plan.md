# DEN702 Research Paper

## Working Title
Robust Wi-Fi Fingerprint-Based Indoor Localization Under Access-Point Loss and RSSI Noise

## Research Problem
Wi-Fi fingerprint-based indoor localization can achieve high localization accuracy under stable conditions. However, real indoor Wi-Fi environments are dynamic. Access points may become unavailable and RSSI measurements may fluctuate because of environmental changes, human movement, device differences, and interference.

## Research Questions

RQ1: How does increasing Wi-Fi access-point loss affect indoor localization performance?

RQ2: How does RSSI measurement noise affect indoor localization performance?

RQ3: Can AP-dropout and RSSI-noise augmentation during training improve model robustness?

RQ4: Which machine-learning model maintains the best performance under degraded Wi-Fi conditions?

## Hypothesis
Models trained using simulated access-point loss and RSSI perturbation will exhibit lower performance degradation under corrupted Wi-Fi conditions than models trained only on clean Wi-Fi fingerprints.

## Baseline Models
- K-Nearest Neighbors
- Random Forest
- Multi-Layer Perceptron

## Evaluation Metrics

Classification:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Localization:
- Mean Absolute Error
- Root Mean Squared Error
- Median Localization Error
- 90th Percentile Error

## Planned Experiments

1. Clean baseline
2. 10% AP loss
3. 20% AP loss
4. 30% AP loss
5. 40% AP loss
6. 50% AP loss
7. RSSI noise ±2 dBm
8. RSSI noise ±5 dBm
9. RSSI noise ±10 dBm
10. Combined AP loss and RSSI noise
11. AP-dropout training
12. RSSI-noise augmentation
13. Combined robust training
14. Ablation analysis
