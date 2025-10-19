"""Tests for audio input module."""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

from vanish.modules.audio_input import AudioInput


@pytest.fixture
def audio_input():
    """Create AudioInput instance."""
    return AudioInput(target_sr=44100)


@pytest.fixture
def temp_audio_file():
    """Create temporary audio file."""
    # Generate 1 second of sine wave
    sr = 44100
    duration = 1.0
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_validate_format_valid(audio_input, temp_audio_file):
    """Test format validation with valid file."""
    assert audio_input.validate_format(temp_audio_file) is True


def test_validate_format_invalid(audio_input):
    """Test format validation with invalid file."""
    with pytest.raises(FileNotFoundError):
        audio_input.validate_format('nonexistent.wav')

    with tempfile.NamedTemporaryFile(suffix='.xyz') as f:
        with pytest.raises(ValueError, match="Unsupported format"):
            audio_input.validate_format(f.name)


def test_load_audio(audio_input, temp_audio_file):
    """Test audio loading."""
    audio, sr = audio_input.load_audio(temp_audio_file)

    assert isinstance(audio, np.ndarray)
    assert sr == 44100
    assert len(audio) > 0


def test_convert_to_mono(audio_input):
    """Test stereo to mono conversion."""
    # Create stereo audio
    stereo = np.random.randn(2, 1000)

    mono = audio_input.convert_to_mono(stereo)

    assert mono.ndim == 1
    assert len(mono) == 1000

    # Test already mono
    mono_input = np.random.randn(1000)
    mono_output = audio_input.convert_to_mono(mono_input)
    assert np.array_equal(mono_input, mono_output)


def test_normalize_sample_rate(audio_input):
    """Test sample rate normalization."""
    audio = np.random.randn(16000)

    # Resample from 16kHz to 44.1kHz
    resampled = audio_input.normalize_sample_rate(audio, 16000, 44100)

    expected_len = int(len(audio) * 44100 / 16000)
    assert abs(len(resampled) - expected_len) < 10  # Allow small difference


def test_normalize_amplitude(audio_input):
    """Test amplitude normalization."""
    audio = np.random.randn(1000) * 2  # Audio with peak > 1

    normalized = audio_input.normalize_amplitude(audio, target_peak=0.9)

    assert np.abs(normalized).max() <= 0.91  # Small tolerance


def test_analyze_quality(audio_input):
    """Test quality analysis."""
    audio = np.random.randn(44100)  # 1 second
    sr = 44100

    metrics = audio_input.analyze_quality(audio, sr)

    assert metrics.duration == pytest.approx(1.0, rel=0.01)
    assert metrics.sample_rate == 44100
    assert metrics.channels == 1
    assert metrics.rms_level > 0
    assert metrics.dynamic_range > 0


def test_preprocess_pipeline(audio_input, temp_audio_file):
    """Test complete preprocessing pipeline."""
    audio, sr, metrics = audio_input.preprocess(temp_audio_file)

    assert isinstance(audio, np.ndarray)
    assert sr == 44100
    assert metrics.duration > 0
    assert audio.ndim == 1  # Should be mono
    assert np.abs(audio).max() <= 1.0  # Should be normalized
