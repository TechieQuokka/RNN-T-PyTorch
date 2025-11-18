"""
Configuration models for pre-analysis module.
"""
from typing import Dict
from pydantic import BaseModel, Field


class NoiseTypes(BaseModel):
    """Detected noise types."""
    static: bool = Field(False, description="Static noise (AC, fan)")
    impulse: bool = Field(False, description="Impulse noise (clicks, pops)")
    howling: bool = Field(False, description="Howling/feedback")


class VolumeProfile(BaseModel):
    """Volume characteristics of audio."""
    avg_db: float = Field(..., description="Average volume in dB")
    peak_db: float = Field(..., description="Peak volume in dB")
    dynamic_range: float = Field(..., description="Dynamic range in dB")


class PreAnalysisResult(BaseModel):
    """Complete pre-analysis results."""
    bgm_intensity: float = Field(..., ge=0.0, le=1.0,
                                description="BGM intensity ratio (0.0 ~ 1.0)")
    noise_types: NoiseTypes = Field(..., description="Detected noise types")
    voice_bandwidth: float = Field(..., gt=0,
                                  description="Voice bandwidth in Hz")
    volume_profile: VolumeProfile = Field(..., description="Volume characteristics")

    def to_pipeline_config(self) -> Dict:
        """
        Generate optimized configuration for downstream pipeline modules.

        Returns:
            Dictionary with module-specific configurations
        """
        config = {
            'source_separation': self._get_separation_config(),
            'noise_reduction': self._get_noise_reduction_config(),
            'voice_enhancement': self._get_enhancement_config(),
            'compression': self._get_compression_config()
        }
        return config

    def _get_separation_config(self) -> Dict:
        """Generate source separation configuration."""
        # Higher BGM → increase Demucs ratio (better bass handling)
        if self.bgm_intensity > 0.6:
            return {
                'mdx_ratio': 0.60,
                'demucs_ratio': 0.40,
                'reason': 'High BGM intensity detected'
            }
        elif self.bgm_intensity < 0.3:
            return {
                'mdx_ratio': 1.0,
                'demucs_ratio': 0.0,
                'reason': 'Low BGM intensity - MDX-Net only'
            }
        else:
            return {
                'mdx_ratio': 0.65,
                'demucs_ratio': 0.35,
                'reason': 'Moderate BGM intensity'
            }

    def _get_noise_reduction_config(self) -> Dict:
        """Generate noise reduction configuration."""
        # Static noise → aggressive filtering
        if self.noise_types.static:
            strength = 0.8
            reason = 'Static noise detected'
        else:
            strength = 0.5
            reason = 'No significant static noise'

        config = {
            'strength': strength,
            'reason': reason
        }

        # Additional processing for impulse noise
        if self.noise_types.impulse:
            config['apply_gate'] = True
            config['gate_threshold'] = -40  # dB

        # Additional processing for howling
        if self.noise_types.howling:
            config['apply_notch_filter'] = True

        return config

    def _get_enhancement_config(self) -> Dict:
        """Generate voice enhancement configuration."""
        # Narrow bandwidth → aggressive enhancement
        if self.voice_bandwidth < 3000:
            return {
                'strength': 1.0,
                'reason': f'Narrow bandwidth ({self.voice_bandwidth:.0f} Hz)'
            }
        elif self.voice_bandwidth > 5000:
            return {
                'strength': 0.5,
                'reason': f'Wide bandwidth ({self.voice_bandwidth:.0f} Hz) - avoid over-processing'
            }
        else:
            return {
                'strength': 0.75,
                'reason': f'Moderate bandwidth ({self.voice_bandwidth:.0f} Hz)'
            }

    def _get_compression_config(self) -> Dict:
        """Generate dynamic range compression configuration."""
        # Large dynamic range → stronger compression
        if self.volume_profile.dynamic_range > 25:
            return {
                'ratio': 3.0,
                'threshold': -20,  # dB
                'attack': 5,       # ms
                'release': 50,     # ms
                'reason': f'Large dynamic range ({self.volume_profile.dynamic_range:.1f} dB)'
            }
        else:
            return {
                'ratio': 2.0,
                'threshold': -15,
                'attack': 10,
                'release': 100,
                'reason': f'Moderate dynamic range ({self.volume_profile.dynamic_range:.1f} dB)'
            }

    def print_summary(self) -> None:
        """Print formatted analysis summary."""
        print("\n" + "=" * 50)
        print("Pre-Analysis Results")
        print("=" * 50)

        # BGM Intensity
        bgm_pct = self.bgm_intensity * 100
        sep_config = self._get_separation_config()
        print(f"├─ BGM Intensity: {bgm_pct:.1f}%")
        print(f"│  → Ensemble(MDX:{sep_config['mdx_ratio']*100:.0f}%, "
              f"Demucs:{sep_config['demucs_ratio']*100:.0f}%)")

        # Noise Types
        print("├─ Noise Types:")
        noise_config = self._get_noise_reduction_config()
        if self.noise_types.static:
            print(f"│  ├─ Static: HIGH → DeepFilterNet(strength={noise_config['strength']})")
        else:
            print("│  ├─ Static: LOW")

        if self.noise_types.impulse:
            print("│  ├─ Impulse: DETECTED → Apply Gate Filter")
        else:
            print("│  ├─ Impulse: NONE")

        if self.noise_types.howling:
            print("│  └─ Howling: DETECTED → Apply Notch Filter")
        else:
            print("│  └─ Howling: NONE")

        # Voice Bandwidth
        enh_config = self._get_enhancement_config()
        bw_khz = self.voice_bandwidth / 1000
        print(f"├─ Voice Bandwidth: {bw_khz:.1f}kHz")
        print(f"│  → Enhancement(strength={enh_config['strength']})")

        # Volume Profile
        comp_config = self._get_compression_config()
        print("└─ Volume Profile:")
        print(f"   ├─ Avg: {self.volume_profile.avg_db:.1f}dB")
        print(f"   ├─ Peak: {self.volume_profile.peak_db:.1f}dB")
        print(f"   └─ Dynamic Range: {self.volume_profile.dynamic_range:.1f}dB")
        print(f"      → Compression(ratio={comp_config['ratio']:.1f}:1)")
        print("=" * 50 + "\n")


class AnalysisConfig(BaseModel):
    """Configuration for pre-analysis process."""
    sample_rate: int = Field(22050, description="Target sample rate for analysis")
    n_fft: int = Field(2048, description="FFT size for spectral analysis")
    hop_length: int = Field(512, description="Hop length for STFT")
    vad_top_db: float = Field(20.0, description="Voice activity detection threshold (dB)")
    spike_threshold: float = Field(3.0, description="Standard deviations for spike detection")
