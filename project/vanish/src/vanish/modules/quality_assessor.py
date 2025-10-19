"""Quality Assessment Module for evaluating output quality."""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for quality metrics."""

    snr: Optional[float] = None  # Signal-to-Noise Ratio (dB)
    pesq: Optional[float] = None  # Perceptual Evaluation of Speech Quality
    stoi: Optional[float] = None  # Short-Time Objective Intelligibility
    spectral_convergence: Optional[float] = None

    def meets_targets(
        self,
        target_snr: float = 20.0,
        target_pesq: float = 3.5,
        target_stoi: float = 0.9
    ) -> bool:
        """Check if metrics meet target thresholds."""
        checks = []

        if self.snr is not None:
            checks.append(self.snr >= target_snr)

        if self.pesq is not None:
            checks.append(self.pesq >= target_pesq)

        if self.stoi is not None:
            checks.append(self.stoi >= target_stoi)

        return all(checks) if checks else False

    def __str__(self) -> str:
        """String representation of metrics."""
        lines = ["Quality Metrics:"]

        if self.snr is not None:
            lines.append(f"  SNR: {self.snr:.2f} dB")

        if self.pesq is not None:
            lines.append(f"  PESQ: {self.pesq:.2f}")

        if self.stoi is not None:
            lines.append(f"  STOI: {self.stoi:.3f}")

        if self.spectral_convergence is not None:
            lines.append(f"  Spectral Convergence: {self.spectral_convergence:.3f}")

        return "\n".join(lines)


class QualityAssessor:
    """
    Quality assessment for audio processing results.

    Metrics:
    - SNR (Signal-to-Noise Ratio)
    - PESQ (Perceptual Evaluation of Speech Quality)
    - STOI (Short-Time Objective Intelligibility)
    """

    def __init__(
        self,
        calculate_snr: bool = True,
        calculate_pesq: bool = True,
        calculate_stoi: bool = True,
    ):
        """
        Initialize quality assessor.

        Args:
            calculate_snr: Whether to calculate SNR
            calculate_pesq: Whether to calculate PESQ
            calculate_stoi: Whether to calculate STOI
        """
        self.calculate_snr = calculate_snr
        self.calculate_pesq = calculate_pesq
        self.calculate_stoi = calculate_stoi

    def calculate_signal_to_noise_ratio(
        self,
        clean: np.ndarray,
        noisy: np.ndarray
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR).

        Args:
            clean: Clean signal
            noisy: Noisy signal

        Returns:
            SNR in dB
        """
        # Noise is the difference
        noise = noisy - clean

        # Calculate power
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)

        # Avoid division by zero
        if noise_power < 1e-10:
            return 100.0  # Very high SNR

        # SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)

        return float(snr)

    def calculate_pesq_score(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
        sr: int
    ) -> float:
        """
        Calculate PESQ (Perceptual Evaluation of Speech Quality).

        Args:
            reference: Reference signal
            degraded: Degraded signal
            sr: Sample rate

        Returns:
            PESQ score (1.0 - 4.5)
        """
        try:
            from pesq import pesq  # type: ignore[import-untyped]

            # PESQ requires specific sample rates (8kHz or 16kHz)
            if sr not in [8000, 16000]:
                import librosa  # type: ignore[import-untyped]
                target_sr = 16000
                logger.info(f"Resampling for PESQ: {sr}Hz -> {target_sr}Hz")
                reference = librosa.resample(reference, orig_sr=sr, target_sr=target_sr)
                degraded = librosa.resample(degraded, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            # Calculate PESQ
            mode = 'wb' if sr == 16000 else 'nb'  # wideband or narrowband
            score = pesq(sr, reference, degraded, mode)

            return float(score)

        except ImportError:
            logger.warning("PESQ library not installed. Skipping PESQ calculation.")
            return None
        except Exception as e:
            logger.error(f"PESQ calculation failed: {e}")
            return None

    def calculate_stoi_score(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        sr: int
    ) -> float:
        """
        Calculate STOI (Short-Time Objective Intelligibility).

        Args:
            clean: Clean signal
            noisy: Noisy signal
            sr: Sample rate

        Returns:
            STOI score (0.0 - 1.0)
        """
        try:
            from pystoi import stoi  # type: ignore[import-untyped]

            # Calculate STOI
            score = stoi(clean, noisy, sr, extended=False)

            return float(score)

        except ImportError:
            logger.warning("pystoi library not installed. Skipping STOI calculation.")
            return None
        except Exception as e:
            logger.error(f"STOI calculation failed: {e}")
            return None

    def calculate_spectral_convergence(
        self,
        reference: np.ndarray,
        processed: np.ndarray
    ) -> float:
        """
        Calculate spectral convergence (frequency domain accuracy).

        Args:
            reference: Reference signal
            processed: Processed signal

        Returns:
            Spectral convergence score
        """
        # Compute STFT
        ref_stft = np.fft.rfft(reference)
        proc_stft = np.fft.rfft(processed)

        # Ensure same length
        min_len = min(len(ref_stft), len(proc_stft))
        ref_stft = ref_stft[:min_len]
        proc_stft = proc_stft[:min_len]

        # Calculate spectral convergence
        numerator = np.linalg.norm(np.abs(ref_stft) - np.abs(proc_stft))
        denominator = np.linalg.norm(np.abs(ref_stft))

        if denominator < 1e-10:
            return 0.0

        convergence = numerator / denominator

        return float(convergence)

    def assess(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        sr: int
    ) -> QualityMetrics:
        """
        Perform complete quality assessment.

        Args:
            clean: Clean/processed signal
            noisy: Noisy/original signal
            sr: Sample rate

        Returns:
            QualityMetrics object
        """
        logger.info("Starting quality assessment")

        metrics = QualityMetrics()

        # Ensure same length
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        # Calculate SNR
        if self.calculate_snr:
            try:
                metrics.snr = self.calculate_signal_to_noise_ratio(clean, noisy)
                logger.info(f"SNR: {metrics.snr:.2f} dB")
            except Exception as e:
                logger.error(f"SNR calculation failed: {e}")

        # Calculate PESQ
        if self.calculate_pesq:
            pesq_score = self.calculate_pesq_score(noisy, clean, sr)
            if pesq_score is not None:
                metrics.pesq = pesq_score
                logger.info(f"PESQ: {metrics.pesq:.2f}")

        # Calculate STOI
        if self.calculate_stoi:
            stoi_score = self.calculate_stoi_score(noisy, clean, sr)
            if stoi_score is not None:
                metrics.stoi = stoi_score
                logger.info(f"STOI: {metrics.stoi:.3f}")

        # Calculate spectral convergence
        try:
            metrics.spectral_convergence = self.calculate_spectral_convergence(noisy, clean)
            logger.info(f"Spectral Convergence: {metrics.spectral_convergence:.3f}")
        except Exception as e:
            logger.error(f"Spectral convergence calculation failed: {e}")

        logger.info("Quality assessment complete")
        return metrics

    def generate_report(self, metrics: QualityMetrics) -> str:
        """
        Generate quality assessment report.

        Args:
            metrics: QualityMetrics object

        Returns:
            Report string
        """
        return str(metrics)
