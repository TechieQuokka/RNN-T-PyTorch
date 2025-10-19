# Installation Guide - Ubuntu + Python 3.11.9 + RTX 3060

Complete installation guide for Vanish Audio Noise Removal System.

## System Requirements

### Verified Configuration
- **OS**: Ubuntu (tested on WSL2)
- **Python**: 3.11.9
- **GPU**: NVIDIA GeForce RTX 3060 12GB
- **CUDA**: 12.1
- **RAM**: 16GB+ recommended

## Step-by-Step Installation

### 1. Verify System Prerequisites

```bash
# Check Python version
python --version
# Should show: Python 3.11.9

# Check GPU
nvidia-smi
# Should show: GeForce RTX 3060, 12GB memory

# Check CUDA
nvcc --version
# Should show: CUDA 12.1 or compatible
```

### 2. Create Virtual Environment

```bash
# Navigate to project directory
cd /home/beethoven/workspace/deeplearning/project/vanish

# Using pyenv (recommended)
pyenv virtualenv 3.11.9 vanish-env
pyenv local vanish-env

# Or using venv
python -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.1.2 with CUDA 12.1
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

### 4. Install Core Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install demucs==4.0.1
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install pyloudnorm==0.1.1
pip install pesq==0.0.4
pip install pystoi==0.3.3
pip install click==8.1.7
pip install pyyaml==6.0.1
```

### 5. Install Vanish Package

```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

### 6. Install Optional Enhancement Models

#### Resemble-Enhance (Recommended - High Quality)

```bash
# Install from GitHub
pip install git+https://github.com/resemble-ai/resemble-enhance.git

# If installation fails, you can use VoiceFixer as fallback
```

#### VoiceFixer (Fallback - Already in requirements.txt)

```bash
# Already installed with requirements.txt
# No additional steps needed
```

### 7. Verify Installation

```bash
# Check Vanish installation
vanish info

# Expected output:
# Python: 3.11.9
# PyTorch: 2.1.2
# CUDA available: True
# CUDA version: 12.1
# GPU: NVIDIA GeForce RTX 3060
# GPU Memory: 12.0 GB
#
# Model availability:
#   ✅ Demucs: 4.0.1
#   ✅ Resemble-Enhance: Installed (or ⚠️ if not installed)
#   ✅ VoiceFixer: Installed
```

### 8. Create Default Configuration

```bash
# Generate optimized config for RTX 3060
vanish create-config config.yaml --preset rtx3060

# Configuration file will be created with optimized settings
```

## Quick Test

### Test with Sample Audio

```bash
# Download or prepare a test audio file
# For testing, you can use any .wav file

# Basic test
vanish test_input.wav -o test_output.wav -vv

# If it works, you should see:
# - Processing stages completing
# - GPU utilization (check with nvidia-smi in another terminal)
# - Output file created
# - Quality metrics displayed
```

## Troubleshooting

### Issue: CUDA not available

**Check 1: NVIDIA drivers**
```bash
nvidia-smi
# If this fails, install drivers:
sudo apt-get update
sudo apt-get install nvidia-driver-535
sudo reboot
```

**Check 2: PyTorch CUDA version**
```bash
python -c "import torch; print(torch.version.cuda)"
# Should match your CUDA version (12.1)

# If mismatch, reinstall PyTorch:
pip uninstall torch torchaudio
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### Issue: ImportError for modules

```bash
# Make sure you're in the virtual environment
pyenv activate vanish-env
# or
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Demucs download errors

```bash
# Demucs downloads models on first use
# Ensure internet connection and sufficient disk space (~2GB)

# Models are cached in ~/.cache/torch/hub/checkpoints/
# If download fails, try again or clear cache:
rm -rf ~/.cache/torch/hub/checkpoints/
```

### Issue: Out of Memory (OOM)

Edit `config.yaml`:
```yaml
demucs:
  segment: 5  # Reduce from 10

resemble:
  chunk_size: 220500  # Reduce from 441000
```

### Issue: FFmpeg not found

```bash
# Install FFmpeg (needed for some audio formats)
sudo apt-get update
sudo apt-get install ffmpeg
```

## Optional: Development Setup

For development and testing:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests
make test

# Format code
make format

# Run linting
make lint
```

## Uninstallation

```bash
# Deactivate virtual environment
pyenv deactivate
# or
deactivate

# Remove virtual environment
pyenv uninstall vanish-env
# or
rm -rf venv

# Remove package
pip uninstall vanish
```

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](docs/QUICKSTART.md) for basic usage
2. See [PERFORMANCE.md](docs/PERFORMANCE.md) for optimization tips
3. Check [examples/](examples/) for code examples
4. Review [config.yaml](config.yaml) for configuration options

## Getting Help

- GitHub Issues: https://github.com/yourusername/vanish/issues
- Documentation: See docs/ directory
- Example Usage: See examples/ directory
