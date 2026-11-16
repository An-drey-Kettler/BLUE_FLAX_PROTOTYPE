# PROJECT_BLUE_FLAX: PROTOTYPE ARCHIVE

This directory contains the working codebase for the Semantic Drift stabilization prototype. The goal of this project is to implement empirical tools for detecting and stabilizing meaning shifts in LLM interactions.

## Modules

* **`vector_stabilizer.py`** — The core empirical analyzer. It reads a text file, converts chunks into 384-dimensional vector embeddings (via `sentence-transformers`), calculates semantic entropy (cosine distance between vectors), and applies a stabilization filter (EMA + Tanh-saturation) to visualize semantic drift.

## Outputs

The `SCHEMES/` folder contains the generated plots. Here is the visual validation:

![Semantic Drift Stabilization](Snapshot_1.png)

`Snapshot_1.png` demonstrates the semantic stabilization on a real, emotionally charged dialogue (vector space).

## Origin

This prototype is based on the conceptual architecture developed in the "SR-NESS-STABILIZER-2026" framework and the "PHASE_LOCK_ARGO" materials. It represents an independent, working implementation of those ideas, designed to operate purely on local hardware (no cloud dependencies).

Authors: Andrey Kettler (Operator) & SAINT (AI co-author); developed by Andrey Kettler with the assistance and under the guidance of the Architect, an AI assistant (DeepSeek).
