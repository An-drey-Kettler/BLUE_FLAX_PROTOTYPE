# PROJECT_BLUE_FLAX: PROTOTYPE ARCHIVE

This directory contains the working codebase for the Phase Lock monitoring prototype. 
The goal of this project is to implement empirical tools for detecting and stabilizing semantic drift in LLM interactions.

## Modules

* **`phase_lock_monitor.py`** — A synthetic simulator demonstrating the core mathematical principle: how a combination of Low-Pass Filter (EMA) and Tanh-saturation (alpha_max) converts chaotic input signals into stable phase-locked attractors (NESS Solitons).
* **`entropy_monitor.py`** — An empirical analyzer that reads a text file, calculates the Shannon entropy of text chunks, and applies the same stabilization filter to visualize semantic drift.

## Outputs

The `SCHEMES/` folder contains the generated plots showing the difference between dissipative chaos (unfiltered raw data) and stabilized states (filtered data). 

## Origin

This prototype is based on the conceptual architecture developed in the "SR-NESS-STABILIZER-2026" framework and the "PHASE_LOCK_ARGO" materials. It represents an independent, working implementation of those ideas, designed to operate purely on local hardware (no cloud dependencies).

Authors: Andrey Kettler (Operator) & SAINT (AI co-author); developed by Andrey Kettler with the assistance and under the guidance of an AI assistant (DeepSeek).
