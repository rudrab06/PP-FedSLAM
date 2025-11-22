# **PP-FedSLAM**

### Privacy-Preserving Federated Learning for Visual Odometry

Two-stream RGB-D + GRU model with Differential Privacy, Reliability Weighting, and Trimmed-Mean Federated Aggregation.

---

# 📌 Overview

PP-FedSLAM is a federated learning pipeline for Visual Odometry (VO) using RGB-D images.
It preserves privacy using Differential Privacy, handles non-IID clients using Reliability + Trimmed-Mean aggregation, and models motion using a Two-Stream ResNet18 + GRU architecture with 6D rotations and geodesic loss.

---

# ✨ Key Features

* Two-Stream ResNet18 (RGB + Depth)
* GRU for temporal motion modeling
* Converts each frame pair into 12-channel input
* 6D rotation representation
* Geodesic rotation loss
* Per-client Differential Privacy (clipping + Gaussian noise)
* Reliability-aware + Trimmed-Mean aggregation
* ATE & RPE evaluation using `evo`

---

# ⚙️ Installation (uv Environment)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv vo_venv
source vo_venv/bin/activate
uv pip install -r requirements.txt
uv pip install evo --upgrade
```

---

# 📥 Download TUM RGB-D Dataset (Small 0.4 MB)

Download:

[https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz](https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz)

Extract:

```bash
mkdir -p tum_data
tar -xvzf rgbd_dataset_freiburg1_xyz.tgz -C tum_data/
```

---

# 🔗 Generate associations.txt

```bash
python3 generate_associations.py
```

Creates:

```
tum_data/rgbd_dataset_freiburg1_xyz/associations.txt
```

---

# 🏃 Run PP-FedSLAM Training

```bash
python3 run_vo_experiment.py
```

Outputs:

```
estimated_trajectory_split.txt
ground_truth_split.txt
```

---

# 📊 Evaluating Trajectories

### ATE

```bash
evo_ape tum ground_truth_split.txt estimated_trajectory_split.txt -a --save_results ate_results.zip
```

### RPE

```bash
evo_rpe tum ground_truth_split.txt estimated_trajectory_split.txt -r trans_part --save_results rpe_results.zip
```

---

# 📈 ATE/RPE Analysis Scripts

Copy `run_analysis.py` into:

```
ate_results/run_analysis.py
rpe_results/run_analysis.py
```

Run:

### ATE analysis

```bash
cd ate_results
python3 run_analysis.py
```

### RPE analysis

```bash
cd ../rpe_results
python3 run_analysis.py
```

Generates:

* ATE vs Distance plots
* RPE vs Time plots
* Histograms
* Boxplots
* CSV statistics

---

# 🔧 Configuration Highlights 

| Parameter           | Value    |
| ------------------- | -------- |
| **FL_ROUNDS**       | 50       |
| CLIENTS_PER_ROUND   | 3        |
| NUM_CLIENTS         | 5        |
| DP_CLIPPING_NORM    | 1e-2     |
| DP_NOISE_MULTIPLIER | 0.1      |
| RELIABILITY_SCORE   | 0.9      |
| SEQUENCE_LEN        | 5        |
| IMG_SIZE            | 224      |
| BACKBONE            | ResNet18 |

---

# 🤝 Contributors

| Student             | Contribution                                                                                   |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| **Rudra Baunk**     | Coding, debugging, full FL pipeline, DP integration, experiments, plotting, ATE/RPE evaluation |
| **Nikunj Indoriya** | Dataset setup, preprocessing, evaluation scripts, plots, documentation support                 |

---

# 📚 Citation

```
R. Baunk, N. Indoriya,
"PP-FedSLAM: Privacy-Preserving Federated Learning for Visual Odometry with Reliability-Aware Aggregation", 2025.
```

---

# 🧭 License (MIT)

```
MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy...
```
