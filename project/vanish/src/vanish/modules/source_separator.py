"""Source Separation Module using Demucs."""

import torch
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SeparatedAudio:
    """Container for separated audio sources."""

    vocals: np.ndarray
    background: np.ndarray
    separation_quality: float  # 0.0 - 1.0


class SourceSeparator:
    """
    Source separation using Demucs v4 (Hybrid Transformer).

    Separates speech (vocals) from background noise/music.
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cuda",
        shifts: int = 1,
        overlap: float = 0.25,
        segment: int = 10,
    ):
        """
        Initialize Demucs source separator.

        Args:
            model_name: Model to use (htdemucs, htdemucs_ft, etc.)
            device: Device for processing (cuda/cpu)
            shifts: Number of random shifts for augmentation
            overlap: Overlap between segments (0.0-1.0)
            segment: Segment length in seconds
        """
        self.model_name = model_name
        self.device = device
        self.shifts = shifts
        self.overlap = overlap
        self.segment = segment
        self.model = None

    def load_model(self) -> None:
        """Load Demucs model."""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        try:
            from demucs.pretrained import get_model  # type: ignore[import-untyped]
            from demucs.apply import apply_model  # type: ignore[import-untyped]

            logger.info(f"Loading Demucs model: {self.model_name}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.apply_model = apply_model

            logger.info(f"Model loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                "Demucs not installed. Install with: pip install demucs"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def separate(self, audio: np.ndarray, sr: int) -> SeparatedAudio:
        """
        Separate vocals from background.

        Args:
            audio: Audio waveform (mono or stereo)
            sr: Sample rate

        Returns:
            SeparatedAudio object with vocals and background
        """
        if self.model is None:
            self.load_model()

        # Ensure model sample rate matches
        if sr != self.model.samplerate:
            logger.warning(
                f"Sample rate mismatch: audio={sr}Hz, model={self.model.samplerate}Hz. "
                "Resampling recommended before separation."
            )

        # Convert to torch tensor
        if audio.ndim == 1:
            # Mono to stereo (Demucs expects stereo)
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)  # [2, samples]
        else:
            audio_tensor = torch.from_numpy(audio).float()

        # Add batch dimension [1, channels, samples]
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)

        logger.info(f"Separating audio (shape={audio_tensor.shape})")

        try:
            with torch.no_grad():
                # Apply Demucs separation
                sources = self.apply_model(
                    self.model,
                    audio_tensor,
                    shifts=self.shifts,
                    overlap=self.overlap,
                    segment=self.segment,
                    device=self.device,
                )

            # Sources shape: [1, num_sources, channels, samples]
            # Demucs sources order: drums, bass, other, vocals
            sources = sources.cpu().numpy()[0]  # Remove batch dim: [num_sources, channels, samples]

            # Extract vocals and background
            vocals_idx = self.model.sources.index('vocals')
            vocals = sources[vocals_idx]  # [channels, samples]

            # Background = sum of all other sources
            background_indices = [i for i in range(len(sources)) if i != vocals_idx]
            background = np.sum(sources[background_indices], axis=0)  # [channels, samples]

            # Convert to mono if needed
            if vocals.shape[0] > 1:
                vocals = np.mean(vocals, axis=0)
            else:
                vocals = vocals[0]

            if background.shape[0] > 1:
                background = np.mean(background, axis=0)
            else:
                background = background[0]

            # Calculate separation quality
            quality = self.calculate_separation_quality(vocals, background)

            logger.info(f"Separation complete. Quality score: {quality:.3f}")

            return SeparatedAudio(
                vocals=vocals,
                background=background,
                separation_quality=quality
            )

        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise RuntimeError(f"Source separation failed: {e}")

    def calculate_separation_quality(
        self, vocals: np.ndarray, background: np.ndarray
    ) -> float:
        """
        Estimate separation quality using SNR-based metric.

        Args:
            vocals: Separated vocals
            background: Separated background

        Returns:
            Quality score (0.0 - 1.0)
        """
        # Calculate RMS of vocals and background
        vocals_rms = np.sqrt(np.mean(vocals ** 2))
        background_rms = np.sqrt(np.mean(background ** 2))

        # Avoid division by zero
        if background_rms < 1e-8:
            return 1.0

        # SNR in dB
        snr_db = 20 * np.log10(vocals_rms / background_rms) if vocals_rms > 0 else -60

        # Map SNR to 0-1 scale (assume -10dB to 30dB range)
        # Good separation: > 20dB
        quality = np.clip((snr_db + 10) / 40, 0.0, 1.0)

        return float(quality)

    def extract_vocals(self, separated: SeparatedAudio) -> np.ndarray:
        """
        Extract vocals from separated audio.

        Args:
            separated: SeparatedAudio object

        Returns:
            Vocals waveform
        """
        return separated.vocals

    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded model."""
        if self.model is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.model.samplerate,
            "sources": self.model.sources,
            "shifts": self.shifts,
            "overlap": self.overlap,
            "segment": self.segment,
        }
