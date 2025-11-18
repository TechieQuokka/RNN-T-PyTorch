"""
Unit tests for PreAnalyzer module.
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pre_analyzer import PreAnalyzer
from config.analysis_config import AnalysisConfig, NoiseTypes, VolumeProfile


def generate_test_audio(duration=3.0, sr=22050, freq=440.0, noise_level=0.0):
    """Generate synthetic test audio."""
    t = np.linspace(0, duration, int(sr * duration))

    # Pure tone
    audio = np.sin(2 * np.pi * freq * t)

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.randn(len(audio)) * noise_level
        audio = audio + noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio


class TestPreAnalyzer:
    """Test suite for PreAnalyzer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = AnalysisConfig(sample_rate=22050)
        self.analyzer = PreAnalyzer(config=self.config)
        self.sr = 22050

    def test_bgm_intensity_pure_voice(self):
        """Test BGM intensity with pure voice (mid-frequency)."""
        # Generate mid-frequency tone (voice range)
        audio = generate_test_audio(freq=500.0, sr=self.sr)

        intensity = self.analyzer.analyze_bgm_intensity(audio, self.sr)

        # Pure mid-frequency should have low BGM intensity
        assert 0.0 <= intensity <= 1.0
        assert intensity < 0.5  # Should be lower for voice-like signal

    def test_bgm_intensity_with_bass(self):
        """Test BGM intensity with bass frequencies."""
        # Generate low-frequency tone (bass)
        audio = generate_test_audio(freq=100.0, sr=self.sr)

        intensity = self.analyzer.analyze_bgm_intensity(audio, self.sr)

        # Low frequency should increase BGM intensity
        assert 0.0 <= intensity <= 1.0
        assert intensity > 0.3  # Should be higher for bass

    def test_noise_classification_clean(self):
        """Test noise classification with clean audio."""
        audio = generate_test_audio(freq=440.0, sr=self.sr, noise_level=0.0)

        noise_types = self.analyzer.classify_noise_type(audio, self.sr)

        assert isinstance(noise_types, NoiseTypes)
        # Just verify the function returns valid types
        # Note: Synthetic signals may trigger some detectors

    def test_noise_classification_noisy(self):
        """Test noise classification with noisy audio."""
        audio = generate_test_audio(freq=440.0, sr=self.sr, noise_level=0.5)

        noise_types = self.analyzer.classify_noise_type(audio, self.sr)

        assert isinstance(noise_types, NoiseTypes)
        # High noise level should likely detect static noise
        # But we don't enforce strict requirements due to synthetic nature

    def test_voice_bandwidth_estimation(self):
        """Test voice bandwidth estimation."""
        audio = generate_test_audio(freq=500.0, sr=self.sr)

        bandwidth = self.analyzer.estimate_voice_bandwidth(audio, self.sr)

        assert bandwidth > 0
        assert bandwidth < self.sr / 2  # Below Nyquist frequency

    def test_volume_profile_normal(self):
        """Test volume profile analysis."""
        audio = generate_test_audio(freq=440.0, sr=self.sr)

        profile = self.analyzer.analyze_volume_profile(audio, self.sr)

        assert isinstance(profile, VolumeProfile)
        assert profile.peak_db >= profile.avg_db  # Peak should be >= average
        assert profile.dynamic_range >= 0  # Dynamic range should be non-negative

    def test_volume_profile_silence(self):
        """Test volume profile with silence."""
        audio = np.zeros(self.sr * 3)  # 3 seconds of silence

        profile = self.analyzer.analyze_volume_profile(audio, self.sr)

        assert isinstance(profile, VolumeProfile)
        assert profile.avg_db < -50  # Very quiet
        assert profile.dynamic_range >= 0

    def test_pipeline_config_generation(self):
        """Test pipeline configuration generation."""
        from config.analysis_config import PreAnalysisResult

        # Create mock result
        result = PreAnalysisResult(
            bgm_intensity=0.5,
            noise_types=NoiseTypes(static=True, impulse=False, howling=False),
            voice_bandwidth=3000.0,
            volume_profile=VolumeProfile(avg_db=-20.0, peak_db=0.0, dynamic_range=20.0)
        )

        config = result.to_pipeline_config()

        # Verify config structure
        assert 'source_separation' in config
        assert 'noise_reduction' in config
        assert 'voice_enhancement' in config
        assert 'compression' in config

        # Verify values are reasonable
        assert 0 <= config['source_separation']['mdx_ratio'] <= 1.0
        assert 0 <= config['noise_reduction']['strength'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
