"""Command-line interface for Vanish."""

import click
import logging
from pathlib import Path
from typing import Optional
import sys

from vanish import AudioPipeline, PipelineConfig
from vanish.utils.logging import setup_logging


@click.group()
@click.version_option()
def cli():
    """Vanish - Audio Noise Removal System

    Remove background noise from audio while preserving clean speech.
    """
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option(
    '-o', '--output',
    type=click.Path(),
    help='Output file path (default: <input>_clean.wav)'
)
@click.option(
    '-c', '--config',
    type=click.Path(exists=True),
    help='Configuration YAML file'
)
@click.option(
    '-d', '--device',
    type=click.Choice(['cuda', 'cpu']),
    help='Processing device (default: auto-detect)'
)
@click.option(
    '-q', '--quality',
    type=click.Choice(['fast', 'balanced', 'high']),
    default='balanced',
    help='Quality preset (default: balanced)'
)
@click.option(
    '--enhancement',
    type=click.Choice(['resemble', 'voicefixer', 'both']),
    default='resemble',
    help='Voice enhancement model (default: resemble)'
)
@click.option(
    '--metrics/--no-metrics',
    default=True,
    help='Calculate and show quality metrics (default: True)'
)
@click.option(
    '--save-intermediate/--no-save-intermediate',
    default=False,
    help='Save intermediate processing files (default: False)'
)
@click.option(
    '-v', '--verbose',
    count=True,
    help='Increase verbosity (-v, -vv, -vvv)'
)
def process(
    input_file: str,
    output: Optional[str],
    config: Optional[str],
    device: Optional[str],
    quality: str,
    enhancement: str,
    metrics: bool,
    save_intermediate: bool,
    verbose: int
):
    """Process a single audio file to remove noise."""

    # Setup logging
    log_level = logging.WARNING
    if verbose == 1:
        log_level = logging.INFO
    elif verbose >= 2:
        log_level = logging.DEBUG

    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    # Determine output path
    if output is None:
        input_path = Path(input_file)
        output = input_path.parent / f"{input_path.stem}_clean.wav"

    # Load configuration
    if config:
        logger.info(f"Loading configuration from {config}")
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    # Apply CLI options
    if device:
        pipeline_config.device = device
        pipeline_config.demucs.device = device
        pipeline_config.resemble.device = device

    pipeline_config.enhancement_mode = enhancement
    pipeline_config.save_intermediate = save_intermediate
    pipeline_config.quality.calculate_snr = metrics
    pipeline_config.quality.calculate_pesq = metrics
    pipeline_config.quality.calculate_stoi = metrics

    # Apply quality preset
    if quality == 'fast':
        pipeline_config.demucs.shifts = 0
        pipeline_config.resemble.denoiser_run_steps = 15
        pipeline_config.resemble.enhance_run_steps = 15
    elif quality == 'high':
        pipeline_config.demucs.shifts = 2
        pipeline_config.resemble.denoiser_run_steps = 50
        pipeline_config.resemble.enhance_run_steps = 50

    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = AudioPipeline(pipeline_config)

        # Process audio
        click.echo(f"Processing: {input_file}")
        result = pipeline.process(input_file, str(output))

        # Display results
        click.echo(f"\n✅ Processing complete!")
        click.echo(f"Output: {result.output_path}")
        click.echo(f"Time: {result.processing_time:.2f}s")

        if metrics and result.metrics:
            click.echo(f"\n{result.metrics}")

        if save_intermediate and result.intermediate_files:
            click.echo("\nIntermediate files:")
            for name, path in result.intermediate_files.items():
                click.echo(f"  {name}: {path}")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option(
    '-c', '--config',
    type=click.Path(exists=True),
    help='Configuration YAML file'
)
@click.option(
    '--pattern',
    default='*.wav',
    help='File pattern to match (default: *.wav)'
)
@click.option(
    '--prefix',
    default='clean_',
    help='Output filename prefix (default: clean_)'
)
@click.option(
    '-v', '--verbose',
    count=True,
    help='Increase verbosity (-v, -vv, -vvv)'
)
def batch(
    input_dir: str,
    output_dir: str,
    config: Optional[str],
    pattern: str,
    prefix: str,
    verbose: int
):
    """Process multiple audio files in batch mode."""

    # Setup logging
    log_level = logging.WARNING
    if verbose == 1:
        log_level = logging.INFO
    elif verbose >= 2:
        log_level = logging.DEBUG

    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    # Find input files
    input_path = Path(input_dir)
    input_files = list(input_path.glob(pattern))

    if not input_files:
        click.echo(f"❌ No files found matching pattern: {pattern}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(input_files)} files to process")

    # Load configuration
    if config:
        logger.info(f"Loading configuration from {config}")
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = AudioPipeline(pipeline_config)

        # Process files
        results = pipeline.process_batch(
            [str(f) for f in input_files],
            output_dir,
            prefix=prefix
        )

        # Display summary
        click.echo(f"\n✅ Batch processing complete!")
        click.echo(f"Successful: {len(results)}/{len(input_files)}")
        click.echo(f"Output directory: {output_dir}")

        # Calculate average metrics
        if results and results[0].metrics:
            avg_snr = sum(r.metrics.snr for r in results if r.metrics.snr) / len(results)
            click.echo(f"\nAverage SNR: {avg_snr:.2f} dB")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('output_file', type=click.Path())
@click.option(
    '--preset',
    type=click.Choice(['default', 'fast', 'high_quality', 'rtx3060']),
    default='rtx3060',
    help='Configuration preset (default: rtx3060)'
)
def create_config(output_file: str, preset: str):
    """Create a configuration file with specified preset."""

    config = PipelineConfig()

    # Apply preset
    if preset == 'fast':
        config.demucs.shifts = 0
        config.resemble.denoiser_run_steps = 15
        config.resemble.enhance_run_steps = 15
    elif preset == 'high_quality':
        config.demucs.shifts = 2
        config.resemble.denoiser_run_steps = 50
        config.resemble.enhance_run_steps = 50
    elif preset == 'rtx3060':
        # Optimized for RTX 3060 12GB
        config.demucs.shifts = 1
        config.demucs.batch_size = 1
        config.demucs.num_workers = 4
        config.resemble.denoiser_run_steps = 30
        config.resemble.enhance_run_steps = 30
        config.resemble.chunk_size = 44100 * 10  # 10 seconds

    # Save configuration
    config.to_yaml(output_file)
    click.echo(f"✅ Configuration saved to: {output_file}")
    click.echo(f"Preset: {preset}")


@cli.command()
def info():
    """Display system and model information."""
    import torch

    click.echo("Vanish - Audio Noise Removal System\n")

    # Python version
    import sys
    click.echo(f"Python: {sys.version.split()[0]}")

    # PyTorch version and CUDA
    click.echo(f"PyTorch: {torch.__version__}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        click.echo(f"CUDA version: {torch.version.cuda}")
        click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        click.echo(f"GPU Memory: {gpu_mem:.1f} GB")

    # Check model availability
    click.echo("\nModel availability:")

    try:
        import demucs  # type: ignore[import-untyped]
        click.echo(f"  ✅ Demucs: {demucs.__version__}")
    except ImportError:
        click.echo("  ❌ Demucs: Not installed")

    try:
        import resemble_enhance  # type: ignore[import-untyped]
        click.echo("  ✅ Resemble-Enhance: Installed")
    except ImportError:
        click.echo("  ⚠️  Resemble-Enhance: Not installed (optional)")

    try:
        import voicefixer  # type: ignore[import-untyped]
        click.echo("  ✅ VoiceFixer: Installed")
    except ImportError:
        click.echo("  ⚠️  VoiceFixer: Not installed (optional)")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
