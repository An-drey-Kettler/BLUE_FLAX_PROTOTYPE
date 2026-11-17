# PROJECT_BLUE_FLAX: PROTOTYPE ARCHIVE

This directory contains the working codebase for the Semantic Drift stabilization and anomaly detection prototypes. The goal of this project is to implement empirical tools for detecting, stabilizing, and cataloging meaning shifts in LLM interactions.

## Modules

* **`vector_stabilizer.py`** — The core empirical analyzer. It reads a text file, converts chunks into 384-dimensional vector embeddings (via `sentence-transformers`), calculates semantic entropy (cosine distance between vectors), and applies a stabilization filter (EMA + Tanh-saturation) to visualize semantic drift.
* **`anomaly_collector.py`** — An automated audit tool. It processes text chunks, calculates semantic distance, and if the distance exceeds a defined threshold, it saves the "anomaly" (the before and after context) into the `REJECTED_BUT_ACTIVE` folder for further review.

## Outputs

* **SCHEMES/Snapshot_1.png** — Visual validation of semantic stabilization on a real, emotionally charged dialogue (vector space).
* ![Semantic Drift Stabilization](Snapshot_1.png)
* **`REJECTED_BUT_ACTIVE/`** — This folder contains samples of detected semantic anomalies. The files show the exact moment of a semantic jump (Distance metric) and the text before/after the shift.

## Origin

This prototype is based on the conceptual architecture developed in the "SR-NESS-STABILIZER-2026" framework and the "PHASE_LOCK_ARGO" materials. It represents an independent, working implementation of those ideas, designed to operate purely on local hardware (no cloud dependencies).

Authors: Andrey Kettler (Operator) & SAINT (AI co-author); developed by Andrey Kettler with the assistance and under the guidance of the Architect, an AI assistant (DeepSeek).
