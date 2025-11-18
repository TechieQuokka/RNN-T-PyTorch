"""
Pre-Analysis module for audio preprocessing pipeline.

Analyzes input audio characteristics and generates optimized
configuration for downstream processing modules.
"""
import numpy as np
import librosa
from typing import Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_utils import (
    load_audio, get_frequency_bands, hz_to_bin,
    calculate_spectral_energy, detect_spikes, detect_voice_activity
)
from config.analysis_config import (
    PreAnalysisResult, NoiseTypes, VolumeProfile, AnalysisConfig
)


class PreAnalyzer:
    """Audio pre-analysis engine."""

    def __init__(self, config: AnalysisConfig = None):
        """
        Initialize PreAnalyzer.

        Args:
            config: Analysis configuration (uses defaults if None)
        """
        self.config = config or AnalysisConfig()

    def analyze(self, audio_path: str) -> PreAnalysisResult:
        """
        Perform complete pre-analysis on audio file.

        Args:
            audio_path: Path to input audio file

        Returns:
            PreAnalysisResult with all analysis metrics
        """
        # Load audio
        audio, sr = load_audio(audio_path, sr=self.config.sample_rate)

        # Run all analyses
        bgm_intensity = self.analyze_bgm_intensity(audio, sr)
        noise_types = self.classify_noise_type(audio, sr)
        voice_bandwidth = self.estimate_voice_bandwidth(audio, sr)
        volume_profile = self.analyze_volume_profile(audio, sr)

        # Create result object
        result = PreAnalysisResult(
            bgm_intensity=bgm_intensity,
            noise_types=noise_types,
            voice_bandwidth=voice_bandwidth,
            volume_profile=volume_profile
        )

        return result

    def analyze_bgm_intensity(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure BGM (background music) intensity ratio.

        Analyzes frequency spectrum to estimate the ratio of BGM vs voice.
        Higher values indicate more prominent background music.

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            BGM intensity ratio (0.0 ~ 1.0)
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.config.n_fft,
                           hop_length=self.config.hop_length)
        magnitude = np.abs(stft)

        # Get frequency bands
        bands = get_frequency_bands(sr)

        # Convert Hz to bin indices
        low_start = hz_to_bin(bands['low'][0], self.config.n_fft, sr)
        low_end = hz_to_bin(bands['low'][1], self.config.n_fft, sr)
        mid_start = hz_to_bin(bands['mid'][0], self.config.n_fft, sr)
        mid_end = hz_to_bin(bands['mid'][1], self.config.n_fft, sr)
        high_start = hz_to_bin(bands['high'][0], self.config.n_fft, sr)
        high_end = hz_to_bin(bands['high'][1], self.config.n_fft, sr)

        # Calculate energy in each band
        low_energy = calculate_spectral_energy(magnitude, (low_start, low_end))
        mid_energy = calculate_spectral_energy(magnitude, (mid_start, mid_end))
        high_energy = calculate_spectral_energy(magnitude, (high_start, high_end))

        total_energy = low_energy + mid_energy + high_energy

        # Avoid division by zero
        if total_energy < 1e-10:
            return 0.0

        # BGM ratio = (low + high) / total
        # Voice is mostly in mid frequencies
        bgm_energy = low_energy + high_energy
        bgm_ratio = bgm_energy / total_energy

        # Clamp to [0, 1]
        return float(np.clip(bgm_ratio, 0.0, 1.0))

    def classify_noise_type(self, audio: np.ndarray, sr: int) -> NoiseTypes:
        """
        Classify types of noise present in audio.

        Detects:
        - Static noise (AC, fan, constant background)
        - Impulse noise (clicks, pops, sudden bursts)
        - Howling (feedback, resonance)

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            NoiseTypes object with detection flags
        """
        # 1. Static Noise Detection using Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio,
                                                     n_fft=self.config.n_fft,
                                                     hop_length=self.config.hop_length)[0]
        # High flatness (>0.5) indicates noise-like signal
        static_noise = float(np.mean(flatness)) > 0.5

        # 2. Impulse Noise Detection using Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio,
                                                 frame_length=self.config.n_fft,
                                                 hop_length=self.config.hop_length)[0]
        impulse_noise = detect_spikes(zcr, threshold=self.config.spike_threshold)

        # 3. Howling Detection using Pitch Tracking
        # Howling shows sustained pitch at unusual frequencies
        try:
            # Use pyin for better pitch detection
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7')   # ~2093 Hz
            )

            # Remove NaN values
            valid_f0 = f0[~np.isnan(f0)]

            if len(valid_f0) > 0:
                # Check for sustained tones (howling characteristic)
                # Compute variance of pitch - low variance = sustained tone
                pitch_variance = np.var(valid_f0)
                # High voiced probability with low variance = potential howling
                avg_voiced_prob = np.mean(voiced_probs[~np.isnan(f0)])
                howling = (pitch_variance < 100) and (avg_voiced_prob > 0.8)
            else:
                howling = False

        except Exception:
            # If pitch detection fails, assume no howling
            howling = False

        return NoiseTypes(
            static=static_noise,
            impulse=impulse_noise,
            howling=howling
        )

    def estimate_voice_bandwidth(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate the effective bandwidth of voice content.

        Measures the frequency range containing significant voice energy.
        Narrow bandwidth (<3kHz) indicates telephone/compressed quality.
        Wide bandwidth (>5kHz) indicates high-quality recording.

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Bandwidth in Hz
        """
        # Detect voice activity regions
        intervals = detect_voice_activity(audio, sr, top_db=self.config.vad_top_db)

        if len(intervals) == 0:
            # No voice detected, return a default
            return 3000.0

        # Extract voice segments
        voice_segments = []
        for start, end in intervals:
            voice_segments.append(audio[start:end])

        # Concatenate all voice segments
        if len(voice_segments) > 0:
            voice_audio = np.concatenate(voice_segments)
        else:
            voice_audio = audio

        # Compute spectral rolloff (frequency below which 95% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(
            y=voice_audio,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            roll_percent=0.95
        )[0]

        # Take 95th percentile as representative bandwidth
        bandwidth = float(np.percentile(rolloff, 95))

        return bandwidth

    def analyze_volume_profile(self, audio: np.ndarray, sr: int) -> VolumeProfile:
        """
        Analyze volume characteristics of audio.

        Measures:
        - Average volume (RMS)
        - Peak volume
        - Dynamic range (difference between loud and quiet parts)

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            VolumeProfile with volume statistics
        """
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio,
                                 frame_length=self.config.n_fft,
                                 hop_length=self.config.hop_length)[0]

        # Convert to dB (reference = max RMS)
        rms_max = np.max(rms)
        if rms_max < 1e-10:
            # Silence or near-silence
            return VolumeProfile(
                avg_db=-60.0,
                peak_db=-60.0,
                dynamic_range=0.0
            )

        rms_db = librosa.amplitude_to_db(rms, ref=rms_max)

        # Calculate statistics
        avg_volume = float(np.mean(rms_db))
        peak_volume = float(np.max(rms_db))  # This will be 0.0 dB (reference)

        # Dynamic range: difference between peak and 10th percentile (quiet parts)
        quiet_level = float(np.percentile(rms_db, 10))
        dynamic_range = peak_volume - quiet_level

        return VolumeProfile(
            avg_db=avg_volume,
            peak_db=peak_volume,
            dynamic_range=dynamic_range
        )


def main():
    """Example usage of PreAnalyzer."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Audio Pre-Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-saves to same directory as .json)
  python pre_analyzer.py input.wav

  # Specify custom output path
  python pre_analyzer.py input.wav --output-config results/config.json

  # Use different sample rate
  python pre_analyzer.py input.wav --sr 44100

Output:
  - Analysis summary (printed to console)
  - Pipeline config JSON file (auto-saved or custom path)

Analysis Metrics:
  1. BGM Intensity: Ratio of background music vs voice
  2. Noise Types: Static/Impulse/Howling detection
  3. Voice Bandwidth: Frequency range of voice content
  4. Volume Profile: Average/Peak volume and dynamic range
        """)
    parser.add_argument('input', type=str, help='Input audio file path')
    parser.add_argument('--sr', type=int, default=22050,
                       help='Sample rate (default: 22050)')
    parser.add_argument('--output-config', type=str, default=None,
                       help='Save pipeline config to JSON file (default: same as input with .json extension)')

    args = parser.parse_args()

    # Create analyzer
    config = AnalysisConfig(sample_rate=args.sr)
    analyzer = PreAnalyzer(config=config)

    # Analyze
    print(f"\nAnalyzing: {args.input}")
    result = analyzer.analyze(args.input)

    # Print summary
    result.print_summary()

    # Auto-generate output path if not specified
    if args.output_config:
        output_path = args.output_config
    else:
        # Same location as input, change extension to .json
        input_path = Path(args.input)
        output_path = input_path.with_suffix('.json')

    # Save config
    pipeline_config = result.to_pipeline_config()
    with open(output_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    print(f"Pipeline configuration saved to: {output_path}\n")


if __name__ == '__main__':
    main()
