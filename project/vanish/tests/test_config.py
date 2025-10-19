"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from vanish.config import PipelineConfig, DemucsConfig, ResembleConfig


def test_default_config():
    """Test default configuration creation."""
    config = PipelineConfig()

    assert config.device in ['cuda', 'cpu']
    assert config.target_sr == 44100
    assert config.channels == 1
    assert config.enhancement_mode == 'resemble'


def test_config_validation():
    """Test configuration validation."""
    config = PipelineConfig()

    # Should not raise
    config.validate()

    # Test invalid sample rate
    config.target_sr = 99999
    with pytest.raises(ValueError, match="Unsupported sample rate"):
        config.validate()


def test_config_yaml_roundtrip():
    """Test saving and loading configuration from YAML."""
    config = PipelineConfig()
    config.demucs.shifts = 2
    config.resemble.denoiser_run_steps = 50

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.to_yaml(f.name)
        temp_path = f.name

    try:
        # Load configuration
        loaded_config = PipelineConfig.from_yaml(temp_path)

        # Verify values
        assert loaded_config.demucs.shifts == 2
        assert loaded_config.resemble.denoiser_run_steps == 50

    finally:
        Path(temp_path).unlink()


def test_nested_config_creation():
    """Test nested configuration objects."""
    config = PipelineConfig()

    assert isinstance(config.demucs, DemucsConfig)
    assert isinstance(config.resemble, ResembleConfig)

    assert config.demucs.model == 'htdemucs'
    assert config.resemble.denoiser_run_steps == 30
