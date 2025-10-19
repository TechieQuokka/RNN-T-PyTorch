"""
Vanish - Audio Noise Removal System

AI-powered audio noise removal using source separation and enhancement.
"""

__version__ = "0.1.0"

from vanish.pipeline import AudioPipeline
from vanish.config import PipelineConfig

__all__ = ["AudioPipeline", "PipelineConfig"]
