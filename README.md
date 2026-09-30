# Metaminds · Adaptive Sensor Fault Detection Engine
[![Track](https://img.shields.io/badge/IEEE%20CIS-Track%2004-blueviolet)](https://github.com/srividyamushti/Metaminds)
[![Auditor](https://img.shields.io/badge/Autonomous%20AI%20Auditor-Verified-brightgreen)](https://github.com/srividyamushti/Metaminds)
[![Latency](https://img.shields.io/badge/Edge%20Latency-0.038ms-cyan)](https://metaminds-flax.vercel.app/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An edge-optimized fault detection framework utilizing **Multi-Objective Evolutionary Computation (Genetic Algorithms)** to discover Pareto-optimal model architectures that balance classification robustness, sub-millisecond inference latency, and memory complexity under non-stationary environmental distribution shifts.

---

## 1. IEEE CIS Track 04 Domain Alignment
In industrial cyber-physical systems and IoT sensor arrays, environmental fluctuations (ambient temperature swings, mechanical wear) introduce gradual **calibration drift**. Static machine learning models frequently mistake this baseline shift for catastrophic hardware failure, generating costly false alarms.

This project implements an **Evolutionary Computation pipeline** directly aligned with the core pillars of the **IEEE Computational Intelligence Society (CIS)**:
- **Evolutionary Hyperparameter Search:** Multi-objective genetic algorithm searching across tree depth and minimum sample splits.
- **Pareto Optimality:** Scalarized fitness balancing $F_1$-score against edge execution latency ($\tau$) and parameter complexity ($N_{\text{nodes}}$).
- **Sub-Millisecond Inference:** Fully exportable decision tree logic executing in $<0.05\text{ ms}$ on edge microcontrollers.

---

## 2. Mathematical Formulation
The optimization objective is formulated as a multi-objective maximization problem:

$$\max_{\theta \in \Theta} \; \Phi(\theta) = \Big[ F_1(\theta) - \lambda_1 \cdot N_{\text{nodes}}(\theta) - \lambda_2 \cdot \tau_{\text{inf}}(\theta) \Big]$$

Where:
- $\theta = [\text{max\_depth}, \text{min\_samples\_split}]$ represents the chromosome candidate.
- $F_1(\theta)$: Harmonic mean of precision and recall evaluated under drifted test data.
- $N_{\text{nodes}}(\theta)$: Total decision node count (proxy for memory footprint).
- $\tau_{\text{inf}}(\theta)$: Per-sample execution latency in milliseconds.
- $\lambda_1 = 0.003, \; \lambda_2 = 0.02$: Complexity and latency penalty coefficients.

---

## 3. System Architecture & AST Modularity

```text
├── data/
│   └── sensor_data.csv        # Baseline & drift benchmark telemetry
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_generator.py      # Multi-axis sensor stream & drift injection
│   ├── model.py               # Lightweight decision tree & latency profiler
│   └── optimizer.py           # Multi-objective genetic algorithm (IEEE CIS)
├── tests/
│   ├── __init__.py            # Package initialization
│   ├── test_model.py          # Model unit tests & boundary checks
│   └── test_pipeline.py       # Parameterized sweeps & stress test fixtures
├── .gitignore                 # Zero-secret credential exclusion
├── requirements.txt           # Cloud runtime dependencies
├── index.html                 # Production Vercel edge dashboard
└── app.py                     # Interactive Streamlit dashboard
