import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


def smart_chunk_text(text, max_chars=350, min_chars=60):
    sentences = re.split(r'(?<=[.!?…])\s+|\n+', text.strip())
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk)
            current_chunk = sentence
    if len(current_chunk) >= min_chars:
        chunks.append(current_chunk)
    return chunks


def compute_vmf_active_inference(embeddings, beta=0.85):
    """
    Минимальная модель Active Inference на единичной гиперсфере (von Mises-Fisher):
    - mu: генеративное ожидание (скользящее среднее направление темы)
    - kappa: онлайн-оценка концентрации (precision) по аппроксимации Банерджи (d=384)
    - raw_drift: косинусная ошибка предсказания (1 - mu^T * x)
    - pwpe: взвешенная по точности ошибка предсказания (kappa * raw_drift)
    """
    n, d = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = embeddings / norms

    raw_drift = np.zeros(n)
    kappa_series = np.zeros(n)
    pwpe_series = np.zeros(n)

    # Инициализация без искусственного нуля (по первому чанку)
    running_vec = X[0].copy()
    mu = running_vec / np.linalg.norm(running_vec)
    R_init = 0.85
    kappa_series[0] = (R_init * (d - R_init ** 2)) / (1.0 - R_init ** 2)

    for i in range(1, n):
        x = X[i]
        # 1. Ошибка предсказания (Surprisal без констант)
        drift = 1.0 - float(np.dot(mu, x))
        raw_drift[i] = drift

        # 2. Взвешенная по точности ошибка предсказания (Precision-Weighted Prediction Error)
        pwpe_series[i] = kappa_series[i - 1] * drift

        # 3. Обновление генеративной модели (EMA на сфере) и концентрации kappa
        running_vec = beta * running_vec + (1.0 - beta) * x
        R = float(np.linalg.norm(running_vec))
        R = np.clip(R, 1e-4, 0.9999)

        mu = running_vec / R
        kappa = (R * (d - R ** 2)) / (1.0 - R ** 2)
        kappa_series[i] = kappa

    return raw_drift, kappa_series, pwpe_series


def export_vmf_events(chunks, raw_drift, kappa, pwpe, z_threshold=2.0, output_dir="PHASE_EVENTS"):
    """
    Экспортирует аномалии по Z-оценке взвешенной по точности ошибки предсказания (PWPE).
    """
    os.makedirs(output_dir, exist_ok=True)
    # Считаем статистику начиная с 5-го шага (после прогрева)
    mean_pwpe = np.mean(pwpe[5:])
    std_pwpe = np.std(pwpe[5:]) + 1e-8
    z_scores = (pwpe - mean_pwpe) / std_pwpe

    events = []
    for i in range(5, len(chunks)):
        if z_scores[i] >= z_threshold:
            events.append((i, raw_drift[i], kappa[i], pwpe[i], z_scores[i]))

    report_path = os.path.join(output_dir, "vmf_active_inference_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== VON MISES-FISHER ACTIVE INFERENCE REPORT (v4) ===\n")
        f.write(f"Total Chunks: {len(chunks)} | Detected Surprisal Events (Z >= {z_threshold}): {len(events)}\n")
        f.write(f"Mean Precision (kappa): {np.mean(kappa[5:]):.2f} | Mean PWPE: {mean_pwpe:.2f}\n")
        f.write("=" * 70 + "\n\n")

        for idx, drift, k_val, p_val, z_val in events:
            f.write(f"[Event #{idx:03d}] Z-Score: {z_val:.2f} | PWPE: {p_val:.2f} | Drift: {drift:.4f} | Kappa: {k_val:.1f}\n")
            f.write(f"  CONTEXT: {chunks[idx - 1]}\n")
            f.write(f"  TRIGGER: {chunks[idx]}\n\n")

    print(f"[*] Exported {len(events)} vMF surprisal events into '{report_path}'")


def run_vmf_analysis(filename="test_text.txt", max_chars=350, beta=0.85):
    print("[*] Loading multilingual embedding model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print(f"[*] Chunking {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = smart_chunk_text(text, max_chars=max_chars)
    print(f"[*] Total chunks: {len(chunks)}")

    print("[*] Encoding chunks into 384D hypersphere...")
    embeddings = model.encode(chunks)

    print("[*] Computing von Mises-Fisher precision (kappa) & PWPE...")
    raw_drift, kappa, pwpe = compute_vmf_active_inference(embeddings, beta=beta)

    export_vmf_events(chunks, raw_drift, kappa, pwpe)

    # Визуализация Active Inference: Precision (kappa) и Precision-Weighted Prediction Error
    t = np.arange(1, len(chunks))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t, kappa[1:], color='royalblue', linewidth=2, label='Online Precision κ (vMF Concentration)')
    ax1.set_title('von Mises-Fisher Active Inference: Online Precision (κ)')
    ax1.set_ylabel('Concentration κ')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)

    mean_p = np.mean(pwpe[5:])
    thresh_p = mean_p + 2.0 * np.std(pwpe[5:])
    ax2.plot(t, pwpe[1:], color='darkorange', linewidth=1.8, label='Precision-Weighted Prediction Error (κ · Drift)')
    ax2.axhline(y=thresh_p, color='red', linestyle='--', alpha=0.7, label='Surprisal Threshold (+2σ)')
    ax2.set_title('Precision-Weighted Prediction Error (Surprisal)')
    ax2.set_xlabel('Semantic Chunks')
    ax2.set_ylabel('PWPE (Surprisal)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_vmf_analysis('test_text.txt')
