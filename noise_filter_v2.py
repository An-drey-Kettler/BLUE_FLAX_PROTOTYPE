#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_filter_v2.py — Adaptive noise suppression based on vMF precision (κ).
PROJECT_BLUE_FLAX | Architect & Operator
"""
import re
import numpy as np
from live_monitor_vmf import VMFMonitor, is_noise_line, smart_chunk_text

# ===== НАСТРОЙКИ =====
ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"
KAPPA_LOW_RATIO = 0.8  # Порог: если κ < 80% от медианы, фильтр агрессивнее

def adaptive_filter(chunks, encoder_name=ENCODER):
    mon = VMFMonitor(encoder_name, beta=0.85, warmup=5, z_threshold=2.0, window=500, z_mode="global")
    
    # Первый проход: собираем все κ
    print("[*] Pass 1: Collecting Kappa statistics...")
    kappas = []
    for ch in chunks:
        res = mon.step(ch)
        kappas.append(res['kappa'])
    
    # Считаем медиану (без первых 5 warmup)
    median_kappa = np.median(kappas[5:])
    dynamic_threshold = median_kappa * KAPPA_LOW_RATIO
    print(f"[*] Median Kappa: {median_kappa:.1f}")
    print(f"[*] Dynamic Threshold (80%): {dynamic_threshold:.1f}")
    
    # Второй проход: фильтруем
    print("[*] Pass 2: Filtering...")
    # Пересоздаем монитор, чтобы не тащить состояние из первого прохода
    mon = VMFMonitor(encoder_name, beta=0.85, warmup=5, z_threshold=2.0, window=500, z_mode="global")
    
    filtered_chunks = []
    filtered_indices = []
    
    for i, ch in enumerate(chunks):
        res = mon.step(ch)
        
        # 1. Базовый фильтр
        if is_noise_line(ch):
            filtered_indices.append(i)
            continue
            
        # 2. Адаптивный фильтр: если κ ниже динамического порога И чанк короткий/подозрительный
        if res['kappa'] < dynamic_threshold and len(ch.strip()) < 30:
            filtered_indices.append(i)
            continue
            
        filtered_chunks.append(ch)
        
    return filtered_chunks, filtered_indices, kappas, median_kappa

if __name__ == "__main__":
    print("[*] Loading test text...")
    with open("How_can_we_help_Gavin_Wood.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = smart_chunk_text(text)
    print(f"[*] Total chunks: {len(chunks)}")
    
    filtered, indices, kappas, med_k = adaptive_filter(chunks)
    
    print(f"\n[*] Filtered out: {len(indices)} chunks ({len(indices)/len(chunks)*100:.1f}%)")
    print(f"[*] Remaining: {len(filtered)} chunks")
    
    print("\n[+] Filtered indices (first 10):")
    for idx in indices[:10]:
        print(f"    Chunk {idx:03d}: {chunks[idx][:60]}...")

