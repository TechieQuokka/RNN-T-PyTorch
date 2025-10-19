"""Basic usage examples for Vanish."""

from vanish import AudioPipeline, PipelineConfig


def example_basic():
    """Basic usage with default settings."""
    print("Example 1: Basic usage with defaults")

    # Initialize pipeline with defaults
    pipeline = AudioPipeline()

    # Process audio file
    result = pipeline.process('input.wav', 'output.wav')

    # Print results
    print(f"Processing complete!")
    print(f"Output: {result.output_path}")
    print(f"Time: {result.processing_time:.2f}s")
    print(f"\n{result.metrics}")


def example_custom_config():
    """Usage with custom configuration."""
    print("\nExample 2: Custom configuration")

    # Create custom configuration
    config = PipelineConfig(
        device='cuda',
        enhancement_mode='resemble',
        save_intermediate=True
    )

    # High quality settings
    config.demucs.shifts = 2
    config.resemble.denoiser_run_steps = 50
    config.resemble.enhance_run_steps = 50

    # Initialize pipeline
    pipeline = AudioPipeline(config)

    # Process
    result = pipeline.process('input.wav', 'output_hq.wav')
    print(f"High quality processing complete: {result.processing_time:.2f}s")


def example_from_yaml():
    """Load configuration from YAML file."""
    print("\nExample 3: Load from YAML config")

    # Load configuration from file
    config = PipelineConfig.from_yaml('config.yaml')

    # Initialize and process
    pipeline = AudioPipeline(config)
    result = pipeline.process('input.wav', 'output.wav')

    print(f"Processing complete with YAML config")


def example_batch_processing():
    """Batch processing multiple files."""
    print("\nExample 4: Batch processing")

    # Initialize pipeline
    pipeline = AudioPipeline()

    # Process multiple files
    input_files = [
        'audio1.wav',
        'audio2.wav',
        'audio3.wav'
    ]

    results = pipeline.process_batch(
        input_files,
        output_dir='outputs',
        prefix='clean_'
    )

    print(f"Processed {len(results)} files")

    # Calculate average metrics
    if results and results[0].metrics:
        avg_snr = sum(r.metrics.snr for r in results if r.metrics.snr) / len(results)
        print(f"Average SNR: {avg_snr:.2f} dB")


def example_quality_presets():
    """Different quality presets."""
    print("\nExample 5: Quality presets")

    # Fast processing (lower quality)
    config_fast = PipelineConfig()
    config_fast.demucs.shifts = 0
    config_fast.resemble.denoiser_run_steps = 15
    config_fast.resemble.enhance_run_steps = 15

    pipeline_fast = AudioPipeline(config_fast)
    result_fast = pipeline_fast.process('input.wav', 'output_fast.wav')
    print(f"Fast processing: {result_fast.processing_time:.2f}s")

    # High quality processing
    config_hq = PipelineConfig()
    config_hq.demucs.shifts = 2
    config_hq.resemble.denoiser_run_steps = 50
    config_hq.resemble.enhance_run_steps = 50

    pipeline_hq = AudioPipeline(config_hq)
    result_hq = pipeline_hq.process('input.wav', 'output_hq.wav')
    print(f"High quality processing: {result_hq.processing_time:.2f}s")


def example_save_intermediate():
    """Save intermediate processing steps."""
    print("\nExample 6: Save intermediate files")

    config = PipelineConfig(save_intermediate=True)
    pipeline = AudioPipeline(config)

    result = pipeline.process('input.wav', 'output.wav')

    print("Intermediate files:")
    for name, path in result.intermediate_files.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    # Run examples (comment out as needed)

    print("="*60)
    print("Vanish - Audio Noise Removal Examples")
    print("="*60)

    # Uncomment to run specific examples:
    # example_basic()
    # example_custom_config()
    # example_from_yaml()
    # example_batch_processing()
    # example_quality_presets()
    # example_save_intermediate()

    print("\nNote: Update input file paths before running examples")
