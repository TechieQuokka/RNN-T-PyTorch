"""Audio Input Module - File ingestion and validation."""

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import librosa  # type: ignore[import-untyped]
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioQualityMetrics:
    """Audio quality pre-assessment metrics."""

    duration: float  # seconds
    sample_rate: int
    channels: int
    bit_depth: int
    rms_level: float  # Root mean square level
    dynamic_range: float  # dB
    dc_offset: float
    clipping_detected: bool


class AudioInput:
    """
    Audio file ingestion and validation.

    Supported formats: WAV, MP3, FLAC, M4A/AAC
    """

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}

    def __init__(self, target_sr: int = 44100):
        """
        Initialize AudioInput module.

        Args:
            target_sr: Target sample rate for normalization
        """
        self.target_sr = target_sr

    def validate_format(self, file_path: str) -> bool:
        """
        Validate if file format is supported.

        Args:
            file_path: Path to audio file

        Returns:
            True if format is supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        return True

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform with sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        self.validate_format(file_path)

        try:
            # librosa handles most formats through audioread/soundfile
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            logger.info(f"Loaded audio: {file_path} ({sr}Hz, shape={audio.shape})")
            return audio, sr

        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise RuntimeError(f"Audio loading failed: {e}")

    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo to mono.

        Args:
            audio: Audio array (channels, samples) or (samples,)

        Returns:
            Mono audio array (samples,)
        """
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # Average channels
            return np.mean(audio, axis=0)
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")

    def normalize_sample_rate(
        self, audio: np.ndarray, orig_sr: int, target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Audio waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate (default: self.target_sr)

        Returns:
            Resampled audio
        """
        if target_sr is None:
            target_sr = self.target_sr

        if orig_sr == target_sr:
            return audio

        logger.info(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def normalize_amplitude(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Normalize audio amplitude to target peak level.

        Args:
            audio: Audio waveform
            target_peak: Target peak amplitude (0.0 - 1.0)

        Returns:
            Normalized audio
        """
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio

    def analyze_quality(self, audio: np.ndarray, sr: int) -> AudioQualityMetrics:
        """
        Pre-assess audio quality.

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            AudioQualityMetrics object
        """
        # Duration
        duration = len(audio) / sr

        # Channels (should be mono at this point)
        channels = 1 if audio.ndim == 1 else audio.shape[0]

        # RMS level
        rms = np.sqrt(np.mean(audio ** 2))

        # Dynamic range (dB)
        peak = np.abs(audio).max()
        if rms > 0:
            dynamic_range = 20 * np.log10(peak / rms)
        else:
            dynamic_range = 0.0

        # DC offset
        dc_offset = np.mean(audio)

        # Clipping detection (samples at max amplitude)
        clipping_threshold = 0.99
        clipping_detected = np.any(np.abs(audio) > clipping_threshold)

        metrics = AudioQualityMetrics(
            duration=duration,
            sample_rate=sr,
            channels=channels,
            bit_depth=16,  # Assume 16-bit after loading
            rms_level=float(rms),
            dynamic_range=float(dynamic_range),
            dc_offset=float(dc_offset),
            clipping_detected=bool(clipping_detected),
        )

        logger.info(f"Quality metrics: duration={duration:.2f}s, RMS={rms:.4f}, "
                   f"dynamic_range={dynamic_range:.2f}dB, clipping={clipping_detected}")

        return metrics

    def preprocess(
        self,
        file_path: str,
        convert_mono: bool = True,
        normalize_sr: bool = True,
        normalize_amp: bool = True,
    ) -> Tuple[np.ndarray, int, AudioQualityMetrics]:
        """
        Complete preprocessing pipeline.

        Args:
            file_path: Path to audio file
            convert_mono: Convert to mono if True
            normalize_sr: Resample to target_sr if True
            normalize_amp: Normalize amplitude if True

        Returns:
            Tuple of (processed_audio, sample_rate, quality_metrics)
        """
        # Load audio
        audio, sr = self.load_audio(file_path)

        # Convert to mono
        if convert_mono:
            audio = self.convert_to_mono(audio)

        # Normalize sample rate
        if normalize_sr:
            audio = self.normalize_sample_rate(audio, sr, self.target_sr)
            sr = self.target_sr

        # Normalize amplitude
        if normalize_amp:
            audio = self.normalize_amplitude(audio)

        # Analyze quality
        metrics = self.analyze_quality(audio, sr)

        # Validation warnings
        if metrics.duration < 1.0:
            logger.warning(f"Very short audio: {metrics.duration:.2f}s")

        if metrics.clipping_detected:
            logger.warning("Clipping detected in input audio")

        if abs(metrics.dc_offset) > 0.01:
            logger.warning(f"Significant DC offset detected: {metrics.dc_offset:.4f}")

        return audio, sr, metrics
