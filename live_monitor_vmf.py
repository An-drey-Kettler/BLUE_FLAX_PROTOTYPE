#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_monitor_vmf.py — Adaptive Real-Time Semantic Radar (vMF / Active Inference)
PROJECT_BLUE_FLAX | Authors: SAINT (Architect) & Beacon-3 for Operator Andrey Kettler
Companion to live_monitor_sprung.py (baseline fixed-threshold detector).
This module does NOT replace sprung; it runs alongside it.
"""

import os
import re
import sys
import argparse
from collections import deque
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# NUMERICAL STABILITY: Banerjee high-dimensional kappa approximation
# kappa = (R * (d - R^2)) / (1 - R^2), R = ||mean direction|| in [0, 1]
# ---------------------------------------------------------------------------
KAPPA_MAX = 1.0e6
R_CLIP_MAX = 0.9999
R_CLIP_MIN = 1.0e-4
DENOM_FLOOR = 1.0e-6


def kappa_banerjee_safe(R, d=384):
    """Numerically safe Banerjee kappa. Never divides by zero, never inf."""
    R = float(np.clip(R, R_CLIP_MIN, R_CLIP_MAX))
    denom = max(1.0 - R * R, DENOM_FLOOR)
    kappa = (R * (d - R * R)) / denom
    return float(min(kappa, KAPPA_MAX))


# ---------------------------------------------------------------------------
# Line handling (shared logic with sprung: formatting-noise filter)
# ---------------------------------------------------------------------------
_NOISE_RE = re.compile(r'^[\s=*_\-~#<>|/\\]+$')


def is_noise_line(line):
    """Skip separator/formatting lines and lines without alnum chars."""
    s = line.strip()
    if not s:
        return True
    if _NOISE_RE.match(s):
        return True
    if not re.search(r'[0-9A-Za-zА-Яа-я]', s):
        return True
    return False


# ---------------------------------------------------------------------------
# CLI COMPATIBILITY: mirror sprung's interface
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Adaptive vMF semantic radar (companion to live_monitor_sprung.py)"
    )
    p.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Path to text file, or '-' for stdin (live stream)",
    )
    p.add_argument(
        "--encoder",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name (same default as sprung)",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.85,
        help="EMA decay for running mean direction",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Chunks to accumulate before emitting alarms",
    )
    p.add_argument(
        "--z-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold on PWPE for event export",
    )
    p.add_argument(
        "--window",
        type=int,
        default=500,
        help="Rolling history window size for Z-score and median kappa",
    )
    p.add_argument(
        "--report",
        default="PHASE_EVENTS/live_vmf_report.txt",
        help="Report output path (compare with sprung logs)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-line console output",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Core online state
# ---------------------------------------------------------------------------
class VMFMonitor:
    def __init__(self, encoder_name, beta=0.85, warmup=5, z_threshold=2.0, window=500):
        self.model = SentenceTransformer(encoder_name)
        self.d = self.model.get_embedding_dimension()
        self.beta = beta
        self.warmup = warmup
        self.z_threshold = z_threshold
        self.running_vec = None
        self.mu = None
        self.kappa_prev = None
        self.n = 0
        self.pwpe_hist = deque(maxlen=window)
        self.kappa_hist = deque(maxlen=window)

    def _unit(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def step(self, text):
        """Process one chunk. Returns dict with drift, kappa, pwpe, z, class."""
        x = self._unit(self.model.encode([text])[0])
        self.n += 1

        if self.running_vec is None:
            self.running_vec = x.copy()
            self.mu = self._unit(self.running_vec)
            R = float(np.linalg.norm(self.running_vec))
            self.kappa_prev = kappa_banerjee_safe(R, self.d)
            return dict(
                idx=self.n,
                drift=0.0,
                kappa=self.kappa_prev,
                pwpe=0.0,
                z=0.0,
                cls="INIT",
            )

        drift = 1.0 - float(np.dot(self.mu, x))
        pwpe = self.kappa_prev * drift  # Precision-Weighted Prediction Error

        # Update generative model (EMA on hypersphere)
        self.running_vec = self.beta * self.running_vec + (1.0 - self.beta) * x
        R = float(np.linalg.norm(self.running_vec))
        self.mu = self._unit(self.running_vec)
        kappa = kappa_banerjee_safe(R, self.d)

        # Evaluate Z-score after warmup against prior history
        z = 0.0
        cls = "WARMUP" if self.n <= self.warmup else "NORMAL"
        if self.n > self.warmup and len(self.pwpe_hist) >= 3:
            arr = np.array(self.pwpe_hist)
            m, s = float(arr.mean()), float(arr.std()) + 1e-8
            z = (pwpe - m) / s
            if z >= self.z_threshold:
                k_med = float(np.median(self.kappa_hist)) if self.kappa_hist else kappa
                cls = "RESONANT_LEAP" if kappa >= k_med else "AXIS_DRIFT"

        self.kappa_prev = kappa
        self.kappa_hist.append(kappa)
        if self.n > self.warmup:
            self.pwpe_hist.append(pwpe)

        return dict(
            idx=self.n,
            drift=drift,
            kappa=kappa,
            pwpe=pwpe,
            z=z,
            cls=cls,
        )


# ---------------------------------------------------------------------------
# REPORT FORMAT: aligned with sprung logs for direct comparison
# ---------------------------------------------------------------------------
def export_report(events, report_path, encoder_name, warmup, z_threshold):
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== LIVE VMF SEMANTIC RADAR REPORT ===\n")
        f.write(f"Encoder: {encoder_name}\n")
        f.write(f"Warmup chunks: {warmup} | Z-threshold: {z_threshold}\n")
        f.write(f"Total events: {len(events)}\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"{'IDX':>5} | {'CLASS':<14} | {'DRIFT':>8} | {'KAPPA':>10} | {'PWPE':>10} | {'Z':>7}\n"
        )
        f.write("-" * 70 + "\n")
        for e in events:
            f.write(
                f"{e['idx']:>5} | {e['cls']:<14} | {e['drift']:>8.4f} | "
                f"{e['kappa']:>10.2f} | {e['pwpe']:>10.4f} | {e['z']:>7.2f}\n"
            )
        f.write("-" * 70 + "\n")
        f.write(
            "Note: fields mirror live_monitor_sprung.py event columns "
            "(idx, class, drift, kappa, pwpe, z).\n"
        )
    print(f"[*] Report written: {report_path} ({len(events)} events)", flush=True)


# ---------------------------------------------------------------------------
# Main loop: file or stdin stream
# ---------------------------------------------------------------------------
def run(args):
    mon = VMFMonitor(
        args.encoder,
        beta=args.beta,
        warmup=args.warmup,
        z_threshold=args.z_threshold,
        window=args.window,
    )
    events = []
    stream = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    try:
        for raw in stream:
            line = raw.rstrip("\n")
            if is_noise_line(line):
                continue
            res = mon.step(line)
            if not args.quiet:
                print(
                    f"[{res['idx']:>4}] {res['cls']:<14} drift={res['drift']:.4f} "
                    f"kappa={res['kappa']:.2f} pwpe={res['pwpe']:.4f} z={res['z']:.2f}",
                    flush=True,
                )
            if res["cls"] in ("RESONANT_LEAP", "AXIS_DRIFT"):
                events.append(res)
    finally:
        if stream is not sys.stdin:
            stream.close()
    export_report(events, args.report, args.encoder, args.warmup, args.z_threshold)


if __name__ == "__main__":
    run(parse_args())

