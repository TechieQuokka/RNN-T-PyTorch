# Audio Noise Removal System Architecture

## Overview

**Project**: Audio Noise Removal System (Vanish)
**Purpose**: Remove background noise (TV sounds, ambient noise, etc.) from audio recordings while preserving clean speech
**Approach**: Multi-stage pipeline using state-of-the-art audio separation and enhancement models

## System Architecture

### High-Level Pipeline

```
[Input Audio]
    ↓
[Pre-processing & Validation]
    ↓
[Stage 1: Source Separation (Demucs)]
    ↓
[Stage 2: Voice Enhancement (VoiceFixer/Resemble-Enhance)]
    ↓
[Post-processing & Quality Control]
    ↓
[Output: Clean Speech]
```

## Core Components

### 1. Audio Input Module
**Responsibility**: Audio file ingestion and validation

**Supported Formats**:
- WAV (16/24/32-bit PCM)
- MP3 (with automatic conversion)
- FLAC
- M4A/AAC

**Features**:
- Format validation and conversion
- Sample rate normalization (target: 44.1kHz or 48kHz)
- Mono/stereo handling
- Audio quality pre-assessment

**Technical Specifications**:
```python
class AudioInput:
    - validate_format(file_path: str) -> bool
    - convert_to_wav(file_path: str) -> np.ndarray
    - normalize_sample_rate(audio: np.ndarray, target_sr: int) -> np.ndarray
    - analyze_quality(audio: np.ndarray) -> AudioQualityMetrics
```

---

### 2. Source Separation Module (Demucs)
**Responsibility**: Separate speech from background noise

**Model**: Demucs v4 (Hybrid Transformer)
**Alternative**: Demucs v3 with fine-tuned vocals model

**Architecture Details**:
- **Model Type**: Hybrid Transformer (combines convolutional and attention mechanisms)
- **Separation Targets**: vocals (speech) + background (noise/music/ambient)
- **Processing**: Time-domain and frequency-domain dual-path processing
- **Model Size**: ~350MB (v4), ~100MB (v3)

**Configuration**:
```python
demucs_config = {
    "model": "htdemucs",  # Hybrid Transformer Demucs
    "stems": 2,  # vocals + background
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "shifts": 1,  # inference augmentation for quality
    "overlap": 0.25,  # overlap between chunks
    "segment": 10,  # segment length in seconds
}
```

**Technical Specifications**:
```python
class SourceSeparator:
    - load_model(model_name: str, device: str) -> DemucsModel
    - separate(audio: np.ndarray, sr: int) -> SeparatedAudio
    - extract_vocals(separated: SeparatedAudio) -> np.ndarray
    - calculate_separation_quality(vocals, background) -> float
```

---

### 3. Voice Enhancement Module
**Responsibility**: Enhance separated speech quality and remove residual artifacts

**Primary Option**: **Resemble-Enhance** (Recommended)
**Fallback Option**: VoiceFixer

#### 3.1 Resemble-Enhance (Primary)
**Why Better**:
- State-of-the-art (2024) speech enhancement
- Superior artifact removal compared to VoiceFixer
- Better handling of low-quality recordings
- Supports both denoising and super-resolution

**Model Details**:
- **Architecture**: Transformer-based diffusion model
- **Capabilities**:
  - Speech denoising
  - Bandwidth extension (8kHz → 44.1kHz)
  - Artifact removal
  - Prosody preservation
- **Model Size**: ~200MB
- **Processing**: Real-time capable on GPU

**Configuration**:
```python
resemble_config = {
    "model": "resemble-enhance",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "denoiser_run_steps": 30,  # quality vs speed tradeoff
    "enhance_run_steps": 30,
    "solver": "midpoint",  # ODE solver
    "nfe": 64,  # number of function evaluations
}
```

#### 3.2 VoiceFixer (Fallback)
**Use Case**: Lightweight alternative or when Resemble-Enhance unavailable

**Model Details**:
- **Architecture**: U-Net based vocoder
- **Capabilities**:
  - Noise reduction
  - Bandwidth extension
  - Artifact suppression
- **Model Size**: ~50MB
- **Processing**: Faster but lower quality than Resemble-Enhance

**Configuration**:
```python
voicefixer_config = {
    "model": "voicefixer",
    "mode": 2,  # 0: 24kHz, 1: 44.1kHz, 2: auto
    "cuda": torch.cuda.is_available(),
}
```

**Technical Specifications**:
```python
class VoiceEnhancer:
    - load_model(model_type: str, config: dict) -> EnhancerModel
    - enhance_speech(vocals: np.ndarray, sr: int) -> np.ndarray
    - remove_artifacts(audio: np.ndarray) -> np.ndarray
    - upscale_bandwidth(audio: np.ndarray, target_sr: int) -> np.ndarray
```

---

### 4. Post-Processing Module
**Responsibility**: Final quality control and optimization

**Features**:
- **Noise Gate**: Remove silence and low-energy segments
- **Normalization**: Adjust loudness to target level (-16 LUFS for speech)
- **High-pass Filter**: Remove sub-vocal frequencies (<80Hz)
- **De-essing**: Reduce harsh sibilance if needed
- **Quality Metrics**: Calculate SNR, PESQ, STOI scores

**Configuration**:
```python
postprocess_config = {
    "noise_gate_threshold": -40,  # dB
    "target_lufs": -16,  # loudness target
    "highpass_cutoff": 80,  # Hz
    "apply_deessing": True,
    "deess_frequency": 6000,  # Hz
}
```

**Technical Specifications**:
```python
class PostProcessor:
    - apply_noise_gate(audio: np.ndarray, threshold: float) -> np.ndarray
    - normalize_loudness(audio: np.ndarray, target_lufs: float) -> np.ndarray
    - apply_highpass_filter(audio: np.ndarray, cutoff: float) -> np.ndarray
    - calculate_quality_metrics(clean: np.ndarray, noisy: np.ndarray) -> QualityMetrics
```

---

### 5. Quality Assessment Module
**Responsibility**: Evaluate output quality and provide metrics

**Metrics**:
- **SNR (Signal-to-Noise Ratio)**: Target >20dB
- **PESQ (Perceptual Evaluation of Speech Quality)**: Target >3.5
- **STOI (Short-Time Objective Intelligibility)**: Target >0.9
- **Spectral Convergence**: Measure frequency domain accuracy

**Technical Specifications**:
```python
class QualityAssessor:
    - calculate_snr(clean: np.ndarray, noisy: np.ndarray) -> float
    - calculate_pesq(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float
    - calculate_stoi(clean: np.ndarray, noisy: np.ndarray, sr: int) -> float
    - generate_report(metrics: QualityMetrics) -> str
```

---

## Data Flow

### Detailed Processing Pipeline

```
1. Input Audio (noisy.wav)
   ├─ Format: Any supported audio format
   ├─ Sample Rate: Variable (8kHz - 48kHz)
   └─ Channels: Mono or Stereo
        ↓
2. Pre-processing
   ├─ Convert to WAV (16-bit PCM)
   ├─ Resample to 44.1kHz
   ├─ Convert stereo → mono (if needed)
   └─ Normalize amplitude
        ↓
3. Source Separation (Demucs)
   ├─ Input: Preprocessed audio
   ├─ Model: htdemucs (Hybrid Transformer)
   ├─ Output: [vocals.wav, background.wav]
   └─ Separation Quality Score: 0.0 - 1.0
        ↓
4. Voice Enhancement (Resemble-Enhance)
   ├─ Input: vocals.wav (separated speech)
   ├─ Denoising: Remove residual noise
   ├─ Enhancement: Bandwidth extension + artifact removal
   └─ Output: enhanced_vocals.wav
        ↓
5. Post-processing
   ├─ Noise Gate: Remove silence/low-energy
   ├─ Normalization: -16 LUFS target
   ├─ High-pass Filter: Remove <80Hz
   └─ Optional: De-essing
        ↓
6. Quality Assessment
   ├─ Calculate SNR, PESQ, STOI
   ├─ Generate quality report
   └─ Compare: input vs output
        ↓
7. Output (clean_speech.wav)
   ├─ Format: WAV (16-bit or 24-bit PCM)
   ├─ Sample Rate: 44.1kHz
   └─ Quality Metrics: Embedded in metadata
```

---

## Model Comparison

### Recommended Stack

| Component | Primary Choice | Alternative | Rationale |
|-----------|---------------|-------------|-----------|
| **Source Separation** | Demucs v4 (htdemucs) | Demucs v3 | Best vocal/noise separation quality, hybrid architecture |
| **Voice Enhancement** | **Resemble-Enhance** | VoiceFixer | Superior quality, better artifact removal, modern architecture |
| **Post-processing** | librosa + pyloudnorm | pydub | Professional audio processing capabilities |

### Why Resemble-Enhance > VoiceFixer?

| Aspect | Resemble-Enhance | VoiceFixer |
|--------|------------------|------------|
| **Release** | 2024 (state-of-the-art) | 2021 |
| **Architecture** | Transformer + Diffusion | U-Net + Vocoder |
| **Denoising Quality** | Excellent | Good |
| **Artifact Removal** | Superior | Moderate |
| **Bandwidth Extension** | 8kHz → 44.1kHz | Up to 44.1kHz |
| **Prosody Preservation** | Excellent | Good |
| **Processing Speed** | Moderate (GPU recommended) | Fast |
| **Model Size** | ~200MB | ~50MB |
| **Use Case** | High-quality production | Quick processing |

**Recommendation**: Use **Resemble-Enhance** for production quality, VoiceFixer as fallback for speed-critical scenarios.

---

## System Requirements

### Hardware Requirements

**Minimum**:
- CPU: 4-core processor (Intel i5 / AMD Ryzen 5)
- RAM: 8GB
- Storage: 2GB (models + temp files)
- Processing: CPU-only mode (slower)

**Recommended**:
- CPU: 8-core processor (Intel i7 / AMD Ryzen 7)
- GPU: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- RAM: 16GB
- Storage: 5GB (models + cache)
- Processing: GPU-accelerated

**Production**:
- CPU: 16-core processor (Intel Xeon / AMD EPYC)
- GPU: NVIDIA A100 / RTX 4090 (24GB VRAM)
- RAM: 32GB
- Storage: 10GB NVMe SSD
- Processing: Multi-GPU support

### Software Requirements

**Core Dependencies**:
```
Python >= 3.9
PyTorch >= 2.0.0 (with CUDA 11.8+ for GPU)
demucs >= 4.0.0
resemble-enhance >= 0.0.1
voicefixer >= 0.1.2 (fallback)
librosa >= 0.10.0
soundfile >= 0.12.0
pyloudnorm >= 0.1.0
pesq >= 0.0.4
pystoi >= 0.3.3
```

**Optional**:
```
tensorboard  # for monitoring
wandb  # for experiment tracking
gradio  # for web interface
ffmpeg  # for format conversion
```

---

## Performance Characteristics

### Processing Time (Estimates)

**1-minute audio file**:
- **CPU-only**:
  - Demucs: ~60-90 seconds
  - Resemble-Enhance: ~30-45 seconds
  - Total: ~2-3 minutes

- **GPU (RTX 3060)**:
  - Demucs: ~10-15 seconds
  - Resemble-Enhance: ~8-12 seconds
  - Total: ~20-30 seconds

- **GPU (RTX 4090)**:
  - Demucs: ~5-8 seconds
  - Resemble-Enhance: ~4-6 seconds
  - Total: ~10-15 seconds

### Memory Usage

**RAM**:
- Demucs: ~2-4GB
- Resemble-Enhance: ~1-2GB
- Peak usage: ~6GB (with buffers)

**VRAM (GPU)**:
- Demucs: ~4-5GB
- Resemble-Enhance: ~2-3GB
- Peak usage: ~6-8GB

---

## Deployment Architectures

### 1. Local CLI Application

```
User → CLI Interface → Processing Pipeline → Output File
```

**Use Case**: Desktop application, local processing
**Advantages**: Privacy, no network dependency, full control
**Limitations**: Requires local GPU for optimal performance

### 2. Web Service (API)

```
User → REST API → Job Queue (Celery) → Worker Pool → Output Storage → Download Link
```

**Use Case**: Web application, multi-user support
**Advantages**: Centralized GPU resources, scalable, cloud deployment
**Components**:
- FastAPI / Flask for REST API
- Redis for job queue
- Celery for async processing
- S3/MinIO for file storage

### 3. Batch Processing System

```
File Watcher → Job Scheduler → Processing Pool (Multi-GPU) → Archive Storage
```

**Use Case**: Large-scale audio processing, production pipelines
**Advantages**: High throughput, resource optimization
**Components**:
- Apache Airflow for orchestration
- Multi-GPU worker nodes
- Distributed storage

---

## Quality Assurance

### Testing Strategy

**Unit Tests**:
- Audio format validation
- Sample rate conversion
- Each processing module independently

**Integration Tests**:
- Full pipeline with synthetic noisy audio
- Edge cases (very noisy, low quality, multiple speakers)

**Quality Tests**:
- SNR improvement measurement
- PESQ/STOI metrics validation
- Blind A/B testing with human evaluators

### Validation Datasets

**Recommended Datasets**:
- **DNS Challenge Dataset**: Deep Noise Suppression challenge data
- **LibriSpeech + Noise**: Clean speech + synthetic noise
- **VCTK Corpus + Noise**: Multi-speaker validation

**Custom Test Set**:
- Real-world recordings (TV background, ambient noise)
- Various SNR levels (-5dB to 20dB)
- Different languages and accents

---

## Error Handling & Edge Cases

### Error Scenarios

**Input Validation Errors**:
- Unsupported format → Convert or reject with message
- Corrupted file → Validate and report error
- Too short (<1s) → Warn user, process if possible
- Too long (>1 hour) → Chunk processing

**Processing Errors**:
- GPU OOM → Fallback to CPU or reduce batch size
- Model loading failure → Retry or use fallback model
- Separation quality poor → Adjust parameters or warn user

**Output Validation**:
- Silent output → Check input, report failure
- Distorted output → Validate metrics, retry with different params
- Quality below threshold → Warn user, provide original

### Edge Case Handling

**Multiple Speakers**:
- Current: Extract all speech (mixed)
- Future: Speaker diarization + individual extraction

**Music + Speech**:
- Demucs handles well (trained on mixed content)
- Post-processing may need adjustment

**Very Low SNR (<0dB)**:
- May result in artifacts
- Warn user if input quality very poor
- Consider iterative enhancement

---

## Future Enhancements

### Short-term (v1.1)

1. **Real-time Processing**: Streaming audio support
2. **Web Interface**: Gradio/Streamlit UI for easy use
3. **Batch Processing**: Multiple file queue processing
4. **Quality Presets**: Fast/Balanced/High quality modes

### Medium-term (v2.0)

1. **Speaker Diarization**: Separate multiple speakers
2. **Language Detection**: Auto-detect and optimize per language
3. **Cloud Deployment**: AWS/GCP serverless functions
4. **Mobile Support**: iOS/Android SDK

### Long-term (v3.0)

1. **Custom Model Training**: Fine-tune on user-specific data
2. **Real-time Collaboration**: Multi-user audio editing
3. **Video Support**: Extract and enhance audio from video
4. **AI-powered Restoration**: Repair damaged audio segments

---

## References

**Models**:
- Demucs: https://github.com/facebookresearch/demucs
- Resemble-Enhance: https://github.com/resemble-ai/resemble-enhance
- VoiceFixer: https://github.com/haoheliu/voicefixer

**Research Papers**:
- Hybrid Transformers for Music Source Separation (Demucs v4)
- Speech Enhancement with Diffusion Models (Resemble-Enhance)
- VoiceFixer: Speech Restoration with Generative Models

**Quality Metrics**:
- PESQ: ITU-T P.862
- STOI: Short-Time Objective Intelligibility
- DNS Challenge: https://github.com/microsoft/DNS-Challenge
