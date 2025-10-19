"""Post-Processing Module for final quality control."""

import numpy as np
import scipy.signal as signal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Post-processing for final quality control and optimization.

    Features:
    - Noise gate
    - Loudness normalization
    - High-pass filter
    - De-essing
    """

    def __init__(
        self,
        noise_gate_threshold: float = -40.0,
        target_lufs: float = -16.0,
        highpass_cutoff: float = 80.0,
        apply_deessing: bool = True,
        deess_frequency: float = 6000.0,
        deess_threshold: float = -20.0,
    ):
        """
        Initialize post-processor.

        Args:
            noise_gate_threshold: Noise gate threshold in dB
            target_lufs: Target loudness in LUFS
            highpass_cutoff: High-pass filter cutoff in Hz
            apply_deessing: Whether to apply de-essing
            deess_frequency: De-essing frequency in Hz
            deess_threshold: De-essing threshold in dB
        """
        self.noise_gate_threshold = noise_gate_threshold
        self.target_lufs = target_lufs
        self.highpass_cutoff = highpass_cutoff
        self.apply_deessing = apply_deessing
        self.deess_frequency = deess_frequency
        self.deess_threshold = deess_threshold

    def apply_noise_gate(
        self,
        audio: np.ndarray,
        threshold: Optional[float] = None,
        sr: int = 44100
    ) -> np.ndarray:
        """
        Apply noise gate to remove silence and low-energy segments.

        Args:
            audio: Audio waveform
            threshold: Threshold in dB (uses self.noise_gate_threshold if None)
            sr: Sample rate

        Returns:
            Gated audio
        """
        if threshold is None:
            threshold = self.noise_gate_threshold

        # Convert threshold from dB to linear
        threshold_linear = 10 ** (threshold / 20)

        # Calculate envelope
        window_size = int(0.01 * sr)  # 10ms window
        envelope = np.abs(audio)

        # Smooth envelope
        kernel = np.ones(window_size) / window_size
        envelope_smooth = np.convolve(envelope, kernel, mode='same')

        # Create gate mask
        gate_mask = envelope_smooth > threshold_linear

        # Apply gate with smooth transitions
        audio_gated = audio * gate_mask

        logger.info(f"Applied noise gate with threshold {threshold}dB")
        return audio_gated

    def normalize_loudness(
        self,
        audio: np.ndarray,
        sr: int,
        target_lufs: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize audio loudness to target LUFS.

        Args:
            audio: Audio waveform
            sr: Sample rate
            target_lufs: Target loudness in LUFS (uses self.target_lufs if None)

        Returns:
            Normalized audio
        """
        if target_lufs is None:
            target_lufs = self.target_lufs

        try:
            import pyloudnorm as pyln  # type: ignore[import-untyped]

            # Measure loudness
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)

            # Normalize to target
            audio_normalized = pyln.normalize.loudness(audio, loudness, target_lufs)

            logger.info(f"Normalized loudness from {loudness:.2f} to {target_lufs:.2f} LUFS")
            return audio_normalized

        except ImportError:
            logger.warning("pyloudnorm not installed. Using peak normalization instead.")
            # Fallback to peak normalization
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio * (0.95 / peak)
            return audio

    def apply_highpass_filter(
        self,
        audio: np.ndarray,
        sr: int,
        cutoff: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove sub-vocal frequencies.

        Args:
            audio: Audio waveform
            sr: Sample rate
            cutoff: Cutoff frequency in Hz (uses self.highpass_cutoff if None)

        Returns:
            Filtered audio
        """
        if cutoff is None:
            cutoff = self.highpass_cutoff

        # Design high-pass filter
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist

        # Butterworth filter, order 4
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)

        # Apply filter
        audio_filtered = signal.filtfilt(b, a, audio)

        logger.info(f"Applied high-pass filter at {cutoff}Hz")
        return audio_filtered

    def apply_deessing(
        self,
        audio: np.ndarray,
        sr: int,
        frequency: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply de-essing to reduce harsh sibilance.

        Args:
            audio: Audio waveform
            sr: Sample rate
            frequency: De-essing frequency in Hz
            threshold: Threshold in dB

        Returns:
            De-essed audio
        """
        if frequency is None:
            frequency = self.deess_frequency
        if threshold is None:
            threshold = self.deess_threshold

        # Design band-pass filter around sibilance frequency
        nyquist = sr / 2
        low_freq = (frequency - 1000) / nyquist
        high_freq = (frequency + 1000) / nyquist

        # Band-pass filter to extract sibilance
        b, a = signal.butter(2, [low_freq, high_freq], btype='band', analog=False)
        sibilance = signal.filtfilt(b, a, audio)

        # Calculate envelope of sibilance
        window_size = int(0.005 * sr)  # 5ms window
        envelope = np.abs(sibilance)
        kernel = np.ones(window_size) / window_size
        envelope_smooth = np.convolve(envelope, kernel, mode='same')

        # Convert threshold to linear
        threshold_linear = 10 ** (threshold / 20)

        # Create compression mask
        compression_ratio = 3.0  # 3:1 compression
        mask = np.ones_like(envelope_smooth)
        above_threshold = envelope_smooth > threshold_linear
        mask[above_threshold] = threshold_linear / (envelope_smooth[above_threshold] * compression_ratio)

        # Apply compression to sibilance band
        sibilance_compressed = sibilance * mask

        # Reconstruct audio
        audio_deessed = audio - sibilance + sibilance_compressed

        logger.info(f"Applied de-essing at {frequency}Hz")
        return audio_deessed

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        apply_noise_gate: bool = True,
        apply_normalization: bool = True,
        apply_highpass: bool = True,
        apply_deess: bool = True,
    ) -> np.ndarray:
        """
        Complete post-processing pipeline.

        Args:
            audio: Audio waveform
            sr: Sample rate
            apply_noise_gate: Whether to apply noise gate
            apply_normalization: Whether to normalize loudness
            apply_highpass: Whether to apply high-pass filter
            apply_deess: Whether to apply de-essing

        Returns:
            Processed audio
        """
        logger.info("Starting post-processing")

        # High-pass filter (remove sub-vocal frequencies)
        if apply_highpass:
            audio = self.apply_highpass_filter(audio, sr)

        # De-essing (reduce harsh sibilance)
        if apply_deess and self.apply_deessing:
            audio = self.apply_deessing(audio, sr)

        # Noise gate (remove silence)
        if apply_noise_gate:
            audio = self.apply_noise_gate(audio, sr=sr)

        # Loudness normalization (final level adjustment)
        if apply_normalization:
            audio = self.normalize_loudness(audio, sr)

        logger.info("Post-processing complete")
        return audio
