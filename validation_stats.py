# validation_stats.py
# Run this script to verify the statistical significance of the AXIS_DRIFT detector.
# Data source: output from vector_stabilizer_v3.py (multilingual 384D encoder).

import math
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def calculate_stats(group_axis_drift, group_normal):
    """
    group_axis_drift: smoothed drift values for AXIS_DRIFT events
    group_normal: smoothed drift values for normal context
    """
    mean1, mean2 = np.mean(group_axis_drift), np.mean(group_normal)
    std1, std2 = np.std(group_axis_drift, ddof=1), np.std(group_normal, ddof=1)
    n1, n2 = len(group_axis_drift), len(group_normal)

    if HAS_SCIPY:
        t_stat, p_value = stats.ttest_ind(group_axis_drift, group_normal, equal_var=False)
    else:
        t_stat = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
        p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    print(f"Group 1 (AXIS_DRIFT, n={n1}): mean={mean1:.4f}, std={std1:.4f}")
    print(f"Group 2 (Normal, n={n2}): mean={mean2:.4f}, std={std2:.4f}")
    print(f"Welch t-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Cohen's d: {cohens_d:.2f}")

    return t_stat, p_value, cohens_d


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import vector_stabilizer_v3 as v3

    print("[*] Loading test_text.txt and running validation statistics...")
    with open("test_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = v3.smart_chunk_text(text)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(chunks, normalize_embeddings=True)

    _, attractor_drift = v3.calculate_semantic_metrics(embeddings)
    _, stab_attractor = v3.stabilize_signal(attractor_drift, beta=0.85, alpha_max=0.45)

    axis_threshold = 0.36
    group_axis_drift = stab_attractor[stab_attractor >= axis_threshold]
    group_normal = stab_attractor[stab_attractor < axis_threshold]

    calculate_stats(group_axis_drift, group_normal)
