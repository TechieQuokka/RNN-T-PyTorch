"""
Example usage of PreAnalyzer module.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.pre_analyzer import PreAnalyzer
from config.analysis_config import AnalysisConfig
import numpy as np
import soundfile as sf


def create_sample_audio(output_path='sample.wav', duration=3.0, sr=22050):
    """Create a sample audio file for testing."""
    t = np.linspace(0, duration, int(sr * duration))

    # Mix of frequencies
    voice = np.sin(2 * np.pi * 500 * t)      # Voice-like (500 Hz)
    bass = np.sin(2 * np.pi * 100 * t) * 0.3 # Bass (100 Hz)
    noise = np.random.randn(len(t)) * 0.05   # Background noise

    # Combine
    audio = voice + bass + noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save
    sf.write(output_path, audio, sr)
    print(f"Sample audio created: {output_path}")
    return output_path


def main():
    """Run example analysis."""
    print("=" * 60)
    print("PreAnalyzer Example")
    print("=" * 60)

    # Create sample audio if needed
    sample_path = 'sample.wav'
    if not os.path.exists(sample_path):
        print("\nCreating sample audio file...")
        create_sample_audio(sample_path)

    # Initialize analyzer
    print("\nInitializing PreAnalyzer...")
    config = AnalysisConfig(sample_rate=22050)
    analyzer = PreAnalyzer(config=config)

    # Run analysis
    print(f"\nAnalyzing: {sample_path}")
    result = analyzer.analyze(sample_path)

    # Print results
    result.print_summary()

    # Get pipeline configuration
    print("\nGenerated Pipeline Configuration:")
    print("-" * 60)
    import json
    pipeline_config = result.to_pipeline_config()
    print(json.dumps(pipeline_config, indent=2))
    print("-" * 60)

    # Save configuration
    config_path = 'pipeline_config.json'
    with open(config_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
