"""Configuration management for Vanish pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml
import torch
from pathlib import Path


@dataclass
class DemucsConfig:
    """Demucs source separation configuration."""

    model: str = "htdemucs"  # Hybrid Transformer Demucs
    stems: int = 2  # vocals + background
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    shifts: int = 1  # inference augmentation
    overlap: float = 0.25  # overlap between chunks
    segment: int = 10  # segment length in seconds

    # RTX 3060 12GB optimization
    batch_size: int = 1
    num_workers: int = 4


@dataclass
class ResembleConfig:
    """Resemble-Enhance configuration."""

    model: str = "resemble-enhance"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    denoiser_run_steps: int = 30  # quality vs speed tradeoff
    enhance_run_steps: int = 30
    solver: str = "midpoint"  # ODE solver
    nfe: int = 64  # number of function evaluations

    # RTX 3060 12GB optimization
    chunk_size: int = 44100 * 10  # 10 seconds chunks


@dataclass
class VoiceFixerConfig:
    """VoiceFixer fallback configuration."""

    model: str = "voicefixer"
    mode: int = 2  # 0: 24kHz, 1: 44.1kHz, 2: auto
    cuda: bool = torch.cuda.is_available()


@dataclass
class PostProcessConfig:
    """Post-processing configuration."""

    noise_gate_threshold: float = -40.0  # dB
    target_lufs: float = -16.0  # loudness target
    highpass_cutoff: float = 80.0  # Hz
    apply_deessing: bool = True
    deess_frequency: float = 6000.0  # Hz
    deess_threshold: float = -20.0  # dB


@dataclass
class QualityConfig:
    """Quality assessment configuration."""

    calculate_snr: bool = True
    calculate_pesq: bool = True
    calculate_stoi: bool = True
    target_snr: float = 20.0  # dB
    target_pesq: float = 3.5
    target_stoi: float = 0.9


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Audio processing settings
    target_sr: int = 44100  # 44.1kHz sample rate
    channels: int = 1  # mono output

    # Enhancement mode
    enhancement_mode: Literal["resemble", "voicefixer", "both"] = "resemble"

    # Module configurations
    demucs: DemucsConfig = field(default_factory=DemucsConfig)
    resemble: ResembleConfig = field(default_factory=ResembleConfig)
    voicefixer: VoiceFixerConfig = field(default_factory=VoiceFixerConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Output settings
    output_format: str = "wav"
    output_bitdepth: int = 16  # 16 or 24 bit

    # Processing options
    save_intermediate: bool = False
    intermediate_dir: Optional[Path] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create nested configs
        demucs_config = DemucsConfig(**config_dict.get('demucs', {}))
        resemble_config = ResembleConfig(**config_dict.get('resemble', {}))
        voicefixer_config = VoiceFixerConfig(**config_dict.get('voicefixer', {}))
        postprocess_config = PostProcessConfig(**config_dict.get('postprocess', {}))
        quality_config = QualityConfig(**config_dict.get('quality', {}))

        # Remove nested configs from main dict
        for key in ['demucs', 'resemble', 'voicefixer', 'postprocess', 'quality']:
            config_dict.pop(key, None)

        return cls(
            **config_dict,
            demucs=demucs_config,
            resemble=resemble_config,
            voicefixer=voicefixer_config,
            postprocess=postprocess_config,
            quality=quality_config
        )

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'device': self.device,
            'target_sr': self.target_sr,
            'channels': self.channels,
            'enhancement_mode': self.enhancement_mode,
            'output_format': self.output_format,
            'output_bitdepth': self.output_bitdepth,
            'save_intermediate': self.save_intermediate,
            'demucs': self.demucs.__dict__,
            'resemble': self.resemble.__dict__,
            'voicefixer': self.voicefixer.__dict__,
            'postprocess': self.postprocess.__dict__,
            'quality': self.quality.__dict__,
        }

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """Validate configuration settings."""
        # Check device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")

        # Check sample rate
        if self.target_sr not in [16000, 22050, 44100, 48000]:
            raise ValueError(f"Unsupported sample rate: {self.target_sr}")

        # Check bit depth
        if self.output_bitdepth not in [16, 24, 32]:
            raise ValueError(f"Unsupported bit depth: {self.output_bitdepth}")

        # Check VRAM for RTX 3060 12GB
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_mem < 6:
                raise ValueError(f"Insufficient GPU memory: {gpu_mem:.1f}GB (minimum 6GB)")
