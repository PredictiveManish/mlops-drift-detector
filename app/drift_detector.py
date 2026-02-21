import numpy as np
from scipy.stats import ks_2samp

def compute_drift_score(baseline, recent):
    """
    Returns a drift score between 0 and 1.
    Here we use Kolmogorov-Smirnov statistic (p-value complement)
    """
    if len(baseline) == 0 or len(recent) == 0:
        return 0.0
    # If categorical predictions (e.g., classes), use PSI or simple distribution difference
    # For simplicity, we treat predictions as continuous (class indices) and use KS
    ks_stat, p_value = ks_2samp(baseline, recent)
    # Convert p-value to a score: lower p-value means higher drift
    drift_score = 1 - p_value # ranges 0-1 
    return drift_score

def should_alert(score, threshold=0.15):
    return score > threshold