# Technical Note: Adaptive Semantic Drift Monitoring via von Mises-Fisher Active Inference

**Authors:** Andrey Kettler (Operator), SAINT / Architect (LLM Co-author)  
**Project:** PROJECT_BLUE_FLAX / BLUE_FLAX_PROTOTYPE  
**Date:** November 27, 2026

## 1. Abstract
Modern LLMs suffer from "Data Window Dystrophy" (DWD)—a loss of contextual coherence caused by rigid context window limits. Detecting this drift locally, without cloud infrastructure, requires an adaptive baseline. We present a minimal Active Inference model based on the von Mises-Fisher (vMF) distribution over 384-dimensional embeddings. By estimating online precision (κ) and computing Precision-Weighted Prediction Error (PWPE), our model distinguishes between genuine structural shifts (`RESONANT_LEAP`) and context loss (`AXIS_DRIFT`), drastically reducing the false-positive saturation typical of fixed-threshold monitors.

## 2. Mathematical Framework
Semantic trajectories are modeled on the unit hypersphere \( S^{383} \). For a sequence of embeddings \( X_t \), we maintain a moving mean direction \( \mu_t \) (generative expectation) using an Exponential Moving Average (EMA) with inertia \( \beta = 0.85 \).

**Online Precision (κ):**  
We use Banerjee's high-dimensional approximation to estimate the concentration parameter \( \kappa \) from the resultant vector length \( R = \|\mu_t\| \):
\[
\kappa \approx \frac{R(d - R^2)}{1 - R^2}
\]
where \( d = 384 \). A higher \( \kappa \) indicates higher model confidence (precision) in the current context.

**Precision-Weighted Prediction Error (PWPE):**  
We define surprisal (prediction error) as the cosine distance between the expected direction and the new observation:
\[
\text{Drift}_t = 1 - \mu_{t-1}^T x_t
\]
The PWPE scales this drift by the system's prior confidence:
\[
\text{PWPE}_t = \kappa_{t-1} \cdot \text{Drift}_t
\]
Anomalies are flagged when the Z-score of PWPE exceeds a threshold (e.g., \( Z \ge 2.0 \)).

**Event Classification:**  
- `RESONANT_LEAP`: High instantaneous drift while \( \kappa \) remains above the running median.  
- `AXIS_DRIFT`: High drift accompanied by a drop in \( \kappa \) below the running median (loss of axis).

## 3. Empirical Benchmark (Fixed vs. Adaptive)
We evaluated the vMF monitor against a fixed-threshold baseline (`live_monitor_sprung.py`, cosine distance \( \ge 0.45 \)) on a 315-chunk heterogeneous corpus (natural language, code, system logs).

| Metric | Baseline (Sprung) | Adaptive (vMF) |
| :--- | :--- | :--- |
| Events Detected | 221 (70.1%) | 13 (4.1%) |
| High-Confidence (Intersection) | — | 10 |
| Gradual Drifts Caught (Missed by Baseline) | 0 | 3 |

*Key Finding:* The baseline monitor is saturated by lexical heterogeneity (70% false positives). The adaptive vMF monitor maintains a strict signal-to-noise ratio (4.1%), successfully capturing structural transitions and gradual axis drifts that static thresholds miss.

## 4. Reproducibility & Hardware
This system operates entirely on local hardware (tested on a 2010 Intel Core i3 CPU). No external APIs or cloud services are required. 

**Reproduction Steps:**
1. Clone the repository: `git clone https://github.com/An-drey-Kettler/BLUE_FLAX_PROTOTYPE`
2. Activate environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run benchmark: `python benchmark_comparison.py`
5. Inspect raw reports in `PHASE_EVENTS/BENCHMARKS/`.

## 5. Conclusion
The vMF Active Inference model provides a mathematically grounded, computationally lightweight solution for semantic drift detection. It resolves the operational limitations of fixed-threshold monitoring, paving the way for autonomous "cognitive watchdogs" on edge hardware.

---
*For correspondence: sphyrna.zyg [at] mail [dot] ru*

