# validation_stats.py
# Run this to verify the statistical significance of the AXIS_DRIFT detector.
# Data source: output from vector_stabilizer_v3.py (or v4_vmf).

import numpy as np
from scipy import stats

def calculate_stats(group_axis_drift, group_normal):
    """
    group_axis_drift: list of smoothed drift values for AXIS_DRIFT events
    group_normal: list of smoothed drift values for normal context
    """
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(group_axis_drift, group_normal, equal_var=False)
    
    # Cohen's d (standardized mean difference)
    mean1, mean2 = np.mean(group_axis_drift), np.mean(group_normal)
    std1, std2 = np.std(group_axis_drift, ddof=1), np.std(group_normal, ddof=1)
    n1, n2 = len(group_axis_drift), len(group_normal)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    print(f"Group 1 (AXIS_DRIFT, n={n1}): mean={mean1:.4f}, std={std1:.4f}")
    print(f"Group 2 (Normal, n={n2}): mean={mean2:.4f}, std={std2:.4f}")
    print(f"Welch t-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Cohen's d: {cohens_d:.2f}")

    return t_stat, p_value, cohens_d

# Example data (replace with actual values from your run)
# axis_drift_vals = [...]
# normal_vals = [...]
# calculate_stats(axis_drift_vals, normal_vals)
