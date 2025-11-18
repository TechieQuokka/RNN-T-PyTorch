"""
Common audio utility functions for pre-analysis module.
"""
import numpy as np
import librosa
from typing import Tuple, Optional


def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to mono if necessary.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 22050 Hz)

    Returns:
        audio: Audio signal as numpy array
        sr: Sample rate
    """
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr


def get_frequency_bands(sr: int) -> dict:
    """
    Get frequency band ranges based on sample rate.

    Args:
        sr: Sample rate

    Returns:
        Dictionary with frequency band ranges in Hz
    """
    nyquist = sr / 2

    return {
        'low': (20, min(200, nyquist)),      # Bass/Drum
        'mid': (200, min(4000, nyquist)),    # Voice
        'high': (4000, min(20000, nyquist))  # Cymbal/HiHat
    }


def hz_to_bin(hz: float, n_fft: int, sr: int) -> int:
    """
    Convert frequency in Hz to FFT bin index.

    Args:
        hz: Frequency in Hz
        n_fft: FFT size
        sr: Sample rate

    Returns:
        FFT bin index
    """
    return int(hz * n_fft / sr)


def db_to_amplitude(db: float) -> float:
    """Convert dB to amplitude."""
    return 10 ** (db / 20)


def amplitude_to_db(amplitude: float, ref: float = 1.0) -> float:
    """Convert amplitude to dB."""
    return 20 * np.log10(np.maximum(amplitude, 1e-10) / ref)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Input audio signal
        target_db: Target level in dB (default: -20 dB)

    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    current_db = amplitude_to_db(rms)

    # Calculate gain needed
    gain_db = target_db - current_db
    gain = db_to_amplitude(gain_db)

    # Apply gain
    normalized = audio * gain

    # Prevent clipping
    peak = np.abs(normalized).max()
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)

    return normalized


def detect_voice_activity(audio: np.ndarray, sr: int,
                         top_db: float = 20.0) -> np.ndarray:
    """
    Detect voice activity using energy-based VAD.

    Args:
        audio: Input audio signal
        sr: Sample rate
        top_db: Threshold in dB below peak (default: 20 dB)

    Returns:
        Array of (start, end) sample indices for voice segments
    """
    return librosa.effects.split(audio, top_db=top_db)


def calculate_spectral_energy(stft: np.ndarray,
                              freq_range: Tuple[int, int]) -> float:
    """
    Calculate energy in specific frequency range.

    Args:
        stft: STFT magnitude (frequency bins x time frames)
        freq_range: Tuple of (start_bin, end_bin)

    Returns:
        Energy as float value
    """
    start_bin, end_bin = freq_range
    band_energy = np.sum(stft[start_bin:end_bin, :] ** 2)
    return float(band_energy)


def detect_spikes(signal: np.ndarray, threshold: float = 3.0) -> bool:
    """
    Detect sudden spikes in signal (for impulse noise detection).

    Args:
        signal: Input signal
        threshold: Number of standard deviations for spike detection

    Returns:
        True if spikes detected, False otherwise
    """
    # Calculate z-score
    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-10:
        return False

    z_scores = np.abs((signal - mean) / std)

    # Count spikes
    spike_count = np.sum(z_scores > threshold)
    spike_ratio = spike_count / len(signal)

    # Consider it impulse noise if > 1% of frames have spikes
    return spike_ratio > 0.01
