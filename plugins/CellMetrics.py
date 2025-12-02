"""
CellMetrics.py - High-performance cell quality metrics for Phy plugins

This module provides Numba JIT-compiled functions for fast analysis of:
- ISI violations
- Amplitude variations
- Waveform variations
- Spike quality metrics

All functions are optimized for multi-core CPUs and can be imported by any plugin.
"""

import numpy as np
import logging

logger = logging.getLogger('phy')

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    logger.info("CellMetrics: Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warn("CellMetrics: Numba not available, using NumPy fallback (slower)")
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ========== CORE METRIC FUNCTIONS ==========

@jit(nopython=True, fastmath=True, cache=True)
def compute_amplitude_variance_fast(spike_amps, window_start, window_end):
    """
    Ultra-fast amplitude variance calculation using Numba JIT.
    
    Parameters
    ----------
    spike_amps : ndarray
        Array of spike amplitudes (float64)
    window_start : int
        Start index of window
    window_end : int
        End index of window (exclusive)
    
    Returns
    -------
    float
        Standard deviation of amplitudes in window
    """
    window_vals = spike_amps[window_start:window_end]
    if len(window_vals) <= 1:
        return 0.0
    
    mean_val = np.mean(window_vals)
    variance = 0.0
    for val in window_vals:
        variance += (val - mean_val) ** 2
    return np.sqrt(variance / len(window_vals))


@jit(nopython=True, fastmath=True, cache=True)
def compute_waveform_variance_fast(waveforms, window_start, window_end):
    """
    Ultra-fast waveform variance using L2 norm with Numba JIT.
    
    Calculates the standard deviation of L2 norms of centered waveforms.
    This is much faster than correlation distance and still effective.
    
    Parameters
    ----------
    waveforms : ndarray
        2D array of waveforms (n_spikes, n_features), float64
    window_start : int
        Start index of window
    window_end : int
        End index of window (exclusive)
    
    Returns
    -------
    float
        Standard deviation of waveform L2 norms (0.0 if no waveform features)
    """
    # Check if waveforms have features
    if waveforms.shape[1] == 0:
        return 0.0
    
    window_waves = waveforms[window_start:window_end]
    n_waves = window_end - window_start
    
    if n_waves <= 1:
        return 0.0
    
    # Calculate mean waveform
    mean_wave = np.zeros(window_waves.shape[1])
    for i in range(n_waves):
        for j in range(window_waves.shape[1]):
            mean_wave[j] += window_waves[i, j]
    mean_wave /= n_waves
    
    # Calculate L2 norms of centered waveforms
    norms = np.zeros(n_waves)
    for i in range(n_waves):
        norm_sq = 0.0
        for j in range(window_waves.shape[1]):
            diff = window_waves[i, j] - mean_wave[j]
            norm_sq += diff * diff
        norms[i] = np.sqrt(norm_sq)
    
    # Return std of norms
    mean_norm = np.mean(norms)
    variance = 0.0
    for norm in norms:
        variance += (norm - mean_norm) ** 2
    return np.sqrt(variance / n_waves)


@jit(nopython=True, fastmath=True, cache=True)
def compute_isi_array(spike_times):
    """
    Compute inter-spike intervals (ISIs) for spike train.
    
    Parameters
    ----------
    spike_times : ndarray
        Array of spike times in seconds (float64)
    
    Returns
    -------
    isi_prev : ndarray
        ISI before each spike
    isi_next : ndarray
        ISI after each spike
    """
    n_spikes = len(spike_times)
    isi_prev = np.zeros(n_spikes)
    isi_next = np.zeros(n_spikes)
    
    # Previous ISI
    isi_prev[0] = spike_times[0] - (spike_times[0] - 1.0)
    for i in range(1, n_spikes):
        isi_prev[i] = spike_times[i] - spike_times[i-1]
    
    # Next ISI
    for i in range(n_spikes - 1):
        isi_next[i] = spike_times[i+1] - spike_times[i]
    isi_next[n_spikes-1] = 1.0
    
    return isi_prev, isi_next


# ========== ISI VIOLATION ANALYSIS ==========

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def analyze_isi_violations_numba(spike_times, spike_amps, waveforms, 
                                  isi_threshold, amp_factor, wave_threshold):
    """
    Numba-optimized ISI violation analysis with MULTI-FEATURE SCORING.
    
    Uses a robust scoring system that combines all three features:
    1. ISI violation severity (how short the ISI is)
    2. Amplitude changes (normalized score)
    3. Waveform changes (normalized score)
    
    A spike is marked suspicious if:
    - It has an ISI violation AND
    - The combined evidence score exceeds threshold
    
    This is more robust than simple OR logic.
    
    Parameters
    ----------
    spike_times : ndarray
        Spike times in seconds (float64)
    spike_amps : ndarray
        Spike amplitudes (float64)
    waveforms : ndarray
        Waveform features (n_spikes, n_features), float64
    isi_threshold : float
        ISI threshold in seconds (e.g., 0.0015 for 1.5ms)
    amp_factor : float
        Multiplier for amplitude std threshold (e.g., 2.5)
    wave_threshold : float
        Threshold for waveform variance (e.g., 0.2)
    
    Returns
    -------
    suspicious : ndarray
        Boolean array marking suspicious spikes
    """
    n_spikes = len(spike_times)
    suspicious = np.zeros(n_spikes, dtype=np.bool_)
    
    # Pre-calculate thresholds
    amp_std = np.std(spike_amps)
    amp_thresh = amp_std * amp_factor
    
    # Calculate ISIs
    isi_prev, isi_next = compute_isi_array(spike_times)
    
    # Parallel loop over spikes
    for i in prange(n_spikes):
        # Only check spikes with ISI violations
        if isi_prev[i] < isi_threshold or isi_next[i] < isi_threshold:
            # Define window
            window_start = max(0, i - 1)
            window_end = min(n_spikes, i + 2)
            
            # Calculate all metrics
            amp_var = compute_amplitude_variance_fast(spike_amps, window_start, window_end)
            wave_var = compute_waveform_variance_fast(waveforms, window_start, window_end)
            
            # MULTI-FEATURE SCORING SYSTEM
            # Each feature contributes to a total evidence score
            evidence_score = 0.0
            
            # 1. ISI violation severity (0-1 score based on how short)
            min_isi = min(isi_prev[i], isi_next[i])
            isi_score = 1.0 - (min_isi / isi_threshold)  # Closer to 0 = higher score
            evidence_score += max(0.0, isi_score)
            
            # 2. Amplitude evidence (normalized 0-1)
            if amp_var > amp_thresh:
                amp_score = min(1.0, amp_var / amp_thresh - 1.0)
                evidence_score += amp_score
            
            # 3. Waveform evidence (normalized 0-1)
            if wave_var > wave_threshold:
                wave_score = min(1.0, wave_var / wave_threshold - 1.0)
                evidence_score += wave_score
            
            # Mark as suspicious if combined evidence is strong
            # RELAXED threshold: 1.8 (was 1.5)
            # Requires stronger evidence: ISI violation + significant amplitude/waveform changes
            # This reduces false positives on good clusters
            if evidence_score >= 1.8:
                suspicious[i] = True
    
    return suspicious
@jit(nopython=True, fastmath=True, cache=True)
def count_isi_violations(spike_times, isi_threshold=0.0015, min_isi=0.0):
    """
    Count ISI violations in spike train.
    
    Parameters
    ----------
    spike_times : ndarray
        Spike times in seconds (float64)
    isi_threshold : float
        ISI threshold in seconds (default: 1.5ms)
    min_isi : float
        Minimum ISI (refractory period), default 0
    
    Returns
    -------
    n_violations : int
        Number of ISI violations
    violation_rate : float
        Fraction of spikes with violations
    """
    if len(spike_times) <= 1:
        return 0, 0.0
    
    isi_prev, isi_next = compute_isi_array(spike_times)
    
    violations = 0
    for i in range(len(spike_times)):
        if isi_prev[i] < isi_threshold or isi_next[i] < isi_threshold:
            if min_isi == 0.0 or isi_prev[i] >= min_isi:
                violations += 1
    
    return violations, float(violations) / float(len(spike_times))


# ========== AMPLITUDE ANALYSIS ==========

@jit(nopython=True, fastmath=True, cache=True)
def compute_amplitude_cutoff(spike_amps, num_bins=100):
    """
    Estimate amplitude cutoff using histogram method.
    
    Estimates the fraction of missing spikes based on amplitude distribution.
    
    Parameters
    ----------
    spike_amps : ndarray
        Spike amplitudes (float64)
    num_bins : int
        Number of histogram bins
    
    Returns
    -------
    cutoff : float
        Estimated fraction of missing spikes (0-1)
    """
    if len(spike_amps) < 10:
        return 0.0
    
    # Create histogram
    min_amp = np.min(spike_amps)
    max_amp = np.max(spike_amps)
    bin_width = (max_amp - min_amp) / num_bins
    
    hist = np.zeros(num_bins)
    for amp in spike_amps:
        bin_idx = int((amp - min_amp) / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        hist[bin_idx] += 1
    
    # Find peak
    peak_idx = 0
    peak_val = hist[0]
    for i in range(num_bins):
        if hist[i] > peak_val:
            peak_val = hist[i]
            peak_idx = i
    
    # Count missing spikes (bins before peak)
    missing = 0.0
    for i in range(peak_idx):
        missing += (peak_val - hist[i])
    
    total = float(len(spike_amps))
    cutoff = missing / (total + missing) if (total + missing) > 0 else 0.0
    
    return min(cutoff, 1.0)


# ========== HELPER FUNCTIONS FOR PLUGIN USE ==========

def prepare_spike_data(spike_times, spike_amps, waveforms):
    """
    Prepare spike data for Numba functions (ensure float64).
    
    Parameters
    ----------
    spike_times : ndarray
        Spike times
    spike_amps : ndarray
        Spike amplitudes
    waveforms : ndarray or None
        Waveform features (can be None if not available)
    
    Returns
    -------
    spike_times_f64 : ndarray
        Spike times as float64
    spike_amps_f64 : ndarray
        Amplitudes as float64
    waveforms_f64 : ndarray
        Waveforms as float64 (empty 2D array if waveforms is None)
    """
    spike_times_f64 = np.asarray(spike_times, dtype=np.float64)
    spike_amps_f64 = np.asarray(spike_amps, dtype=np.float64)
    
    # Handle None waveforms by creating an empty 2D array
    if waveforms is None or (isinstance(waveforms, np.ndarray) and waveforms.size == 0):
        waveforms_f64 = np.zeros((len(spike_times), 0), dtype=np.float64)
    else:
        waveforms_f64 = np.asarray(waveforms, dtype=np.float64)
        # Ensure it's 2D
        if waveforms_f64.ndim == 1:
            waveforms_f64 = waveforms_f64.reshape(-1, 1)
    
    return spike_times_f64, spike_amps_f64, waveforms_f64


def analyze_cluster_quality(spike_times, spike_amps, waveforms,
                            isi_threshold=0.0015, amp_factor=2.5, wave_threshold=0.2):
    """
    High-level function to analyze cluster quality metrics.
    
    Parameters
    ----------
    spike_times : ndarray
        Spike times in seconds
    spike_amps : ndarray
        Spike amplitudes
    waveforms : ndarray
        Waveform features (n_spikes, n_features)
    isi_threshold : float
        ISI threshold in seconds (default: 1.5ms)
    amp_factor : float
        Amplitude variance threshold multiplier
    wave_threshold : float
        Waveform variance threshold
    
    Returns
    -------
    results : dict
        Dictionary with quality metrics:
        - 'suspicious_spikes': boolean array
        - 'n_suspicious': number of suspicious spikes
        - 'suspicious_fraction': fraction of suspicious spikes
        - 'isi_violations': number of ISI violations
        - 'isi_violation_rate': fraction of violations
        - 'amplitude_cutoff': estimated missing spikes fraction
    """
    # Prepare data
    spike_times_f64, spike_amps_f64, waveforms_f64 = prepare_spike_data(
        spike_times, spike_amps, waveforms
    )
    
    # Analyze ISI violations with waveform/amplitude checks
    suspicious = analyze_isi_violations_numba(
        spike_times_f64, spike_amps_f64, waveforms_f64,
        isi_threshold, amp_factor, wave_threshold
    )
    
    # Count ISI violations
    n_violations, violation_rate = count_isi_violations(spike_times_f64, isi_threshold)
    
    # Compute amplitude cutoff
    amp_cutoff = compute_amplitude_cutoff(spike_amps_f64)
    
    return {
        'suspicious_spikes': suspicious,
        'n_suspicious': int(np.sum(suspicious)),
        'suspicious_fraction': float(np.sum(suspicious)) / float(len(suspicious)) if len(suspicious) > 0 else 0.0,
        'isi_violations': n_violations,
        'isi_violation_rate': violation_rate,
        'amplitude_cutoff': amp_cutoff,
    }


# ========== MODULE INFO ==========

def get_numba_info():
    """Get information about Numba availability and settings."""
    info = {
        'numba_available': NUMBA_AVAILABLE,
        'parallel_enabled': NUMBA_AVAILABLE,
        'fastmath_enabled': NUMBA_AVAILABLE,
        'cache_enabled': NUMBA_AVAILABLE,
    }
    return info


if __name__ == '__main__':
    # Test the module
    print("CellMetrics Module Test")
    print("=" * 50)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    
    # Create test data
    np.random.seed(42)
    n_spikes = 1000
    spike_times = np.sort(np.random.rand(n_spikes) * 100).astype(np.float64)
    spike_amps = np.random.randn(n_spikes).astype(np.float64) + 1.0
    waveforms = np.random.randn(n_spikes, 50).astype(np.float64)
    
    # Test analysis
    print("\nTesting analyze_cluster_quality...")
    results = analyze_cluster_quality(spike_times, spike_amps, waveforms)
    
    print(f"Suspicious spikes: {results['n_suspicious']} ({results['suspicious_fraction']*100:.1f}%)")
    print(f"ISI violations: {results['isi_violations']} ({results['isi_violation_rate']*100:.1f}%)")
    print(f"Amplitude cutoff: {results['amplitude_cutoff']:.3f}")
    
    print("\nModule test complete!")
