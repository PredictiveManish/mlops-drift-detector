import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import logging

def compute_drift_score(baseline, recent, method='psi'):
    """
    Returns a drift score between 0 and 1.
    
    Methods:
    - 'psi': Population Stability Index (best for categorical)
    - 'ks': Kolmogorov-Smirnov (for continuous)
    - 'chi2': Chi-square test (for categorical)
    """
    if len(baseline) == 0 or len(recent) == 0:
        return 0.0
    
    if method == 'psi':
        return compute_psi(baseline, recent)
    elif method == 'chi2':
        return compute_chi2_drift(baseline, recent)
    else:
        # Default to KS test
        ks_stat, p_value = ks_2samp(baseline, recent)
        # Convert to 0-1 scale where higher means more drift
        drift_score = 1 - p_value
        return min(max(drift_score, 0), 1)  # Clamp between 0-1

def compute_psi(baseline, recent, bins=3):
    """
    Population Stability Index
    PSI < 0.1: no significant drift
    PSI 0.1-0.2: moderate drift
    PSI > 0.2: significant drift
    """
    # For categorical predictions (0,1,2), we can use the classes as bins
    unique_classes = sorted(set(baseline + recent))
    
    # If we have all 3 classes, use them as bins
    if len(unique_classes) <= 3:
        # Calculate distributions
        baseline_counts = np.bincount(baseline, minlength=3)
        recent_counts = np.bincount(recent, minlength=3)
        
        # Convert to percentages
        baseline_pct = baseline_counts / len(baseline)
        recent_pct = recent_counts / len(recent)
        
        # Add small epsilon to avoid division by zero
        baseline_pct = np.clip(baseline_pct, 0.001, 0.999)
        recent_pct = np.clip(recent_pct, 0.001, 0.999)
        
        # Calculate PSI
        psi = np.sum((recent_pct - baseline_pct) * np.log(recent_pct / baseline_pct))
        
        # Normalize PSI to 0-1 scale (typical PSI > 0.2 is significant)
        # So we'll map PSI 0-0.2 to 0-1 scale
        normalized_psi = min(psi / 0.2, 1.0)
        
        logging.info(f"PSI: {psi:.3f}, Normalized: {normalized_psi:.3f}")
        return normalized_psi
    
    else:
        # Fallback to KS test if we have more classes
        ks_stat, p_value = ks_2samp(baseline, recent)
        return 1 - p_value

def compute_chi2_drift(baseline, recent):
    """
    Chi-square test for categorical drift detection
    """
    # Create contingency table
    unique_classes = sorted(set(baseline + recent))
    
    # Count frequencies
    baseline_counts = [baseline.count(c) for c in unique_classes]
    recent_counts = [recent.count(c) for c in unique_classes]
    
    # Create contingency table
    contingency = np.array([baseline_counts, recent_counts])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Convert p-value to drift score (lower p-value = more drift)
    drift_score = 1 - p_value
    
    logging.info(f"Chi2: {chi2:.3f}, p-value: {p_value:.3f}, Drift score: {drift_score:.3f}")
    return drift_score

def should_alert(score, threshold=0.15):
    """Determine if drift score exceeds threshold"""
    return score > threshold