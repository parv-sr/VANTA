# RN baseline freeze and final preprocessing results

The exact frozen reference is VANTA-RN v0.1 at Git commit dfa7ff55dc980cfc1093c1a0a6de831f53e51703. EX05 does not change VANTA_RN/model.py or its preprocessing code.

Ames train-fitted numeric preprocessing experiment:
- Validation RMSE (seeds 301/302): asinh 21,040.058; clip_z5 21,079.503; standard 21,127.174; winsor_01_99 21,717.873. Selected asinh on validation only.
- Confirmation (seeds 303/304/305): asinh R² 0.915762±0.010408, RMSE 24,550.434±4,467.074, MAE 14,433.241±1,348.173. Standard R² 0.891797±0.028313, RMSE 27,411.838±3,080.412, MAE 16,888.708±3,856.665.
- Paired asinh-minus-standard RMSE: seed 303 -7,927.496; seed 304 +374.796; seed 305 -1,031.513. Most improvement from seed 303; do not claim universal benefit.
- Earlier preprocessing asinh changed numeric tokens (Q/K), numeric attention values and direct values simultaneously. EX05-A isolates *attention-value-only* asinh under standard input preprocessing, so its effect is not mathematically equivalent.
