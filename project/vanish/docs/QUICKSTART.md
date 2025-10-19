# Quick Start Guide

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Using pyenv (recommended for Ubuntu)
pyenv virtualenv 3.11.9 vanish-env
pyenv activate vanish-env

# Or using venv
python3.11 -m venv vanish-env
source vanish-env/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.1 (for RTX 3060)
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install Vanish and dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Install Resemble-Enhance (Optional but Recommended)

```bash
pip install git+https://github.com/resemble-ai/resemble-enhance.git
```

### 4. Verify Installation

```bash
vanish info
```

You should see:
- Python 3.11.9
- PyTorch 2.1.2
- CUDA available: True
- GPU: NVIDIA GeForce RTX 3060
- GPU Memory: 12.0 GB

## Basic Usage

### Command Line

```bash
# Process single file
vanish input.wav -o output.wav

# High quality mode
vanish input.wav -o output.wav --quality high -vv

# Batch processing
vanish batch ./inputs ./outputs
```

### Python API

```python
from vanish import AudioPipeline

# Initialize pipeline
pipeline = AudioPipeline()

# Process audio
result = pipeline.process('input.wav', 'output.wav')

# View metrics
print(result.metrics)
```

## Configuration

### Generate Config File

```bash
vanish create-config config.yaml --preset rtx3060
```

### Use Config File

```bash
vanish input.wav -o output.wav --config config.yaml
```

## Performance Tips for RTX 3060

1. **Use CUDA**: Always use `--device cuda` for 10-15x speedup
2. **Quality Presets**:
   - `fast`: ~15-20 seconds per minute of audio
   - `balanced`: ~20-30 seconds per minute (default)
   - `high`: ~40-60 seconds per minute

3. **Memory Management**:
   - RTX 3060 12GB can handle up to 30-second segments comfortably
   - For longer audio, processing is automatic but may take more time

4. **Batch Processing**:
   - Process multiple files sequentially to maximize GPU utilization
   - Use `vanish batch` for automatic handling

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

```yaml
# Edit config.yaml
demucs:
  segment: 5  # Reduce from 10 to 5 seconds
  batch_size: 1

resemble:
  chunk_size: 220500  # Reduce from 441000 (5 seconds instead of 10)
```

### Slow Processing

1. Verify GPU is being used: `vanish info`
2. Check quality preset: Use `fast` for quicker results
3. Reduce enhancement steps in config

### Missing Dependencies

```bash
# If demucs is missing
pip install demucs==4.0.1

# If quality metrics fail
pip install pesq pystoi

# If ffmpeg is needed
sudo apt-get install ffmpeg  # Ubuntu
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system details
- See [examples/](../examples/) for more usage patterns
- Customize `config.yaml` for your specific needs
- Check [README.md](../README.md) for advanced features
