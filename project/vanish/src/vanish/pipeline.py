"""Main audio processing pipeline orchestration."""

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

from vanish.config import PipelineConfig
from vanish.modules import (
    AudioInput,
    SourceSeparator,
    VoiceEnhancer,
    PostProcessor,
    QualityAssessor
)
from vanish.modules.quality_assessor import QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Container for processing results."""

    output_path: str
    metrics: QualityMetrics
    processing_time: float
    intermediate_files: Optional[dict] = None


class AudioPipeline:
    """
    Complete audio noise removal pipeline.

    Pipeline stages:
    1. Audio input and validation
    2. Source separation (Demucs)
    3. Voice enhancement (Resemble-Enhance/VoiceFixer)
    4. Post-processing
    5. Quality assessment
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize audio pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        if config is None:
            config = PipelineConfig()

        config.validate()
        self.config = config

        # Initialize modules
        self.audio_input = AudioInput(target_sr=config.target_sr)

        self.source_separator = SourceSeparator(
            model_name=config.demucs.model,
            device=config.demucs.device,
            shifts=config.demucs.shifts,
            overlap=config.demucs.overlap,
            segment=config.demucs.segment,
        )

        # Initialize enhancer based on mode
        if config.enhancement_mode in ["resemble", "both"]:
            self.enhancer_primary = VoiceEnhancer(
                model_type="resemble",
                device=config.resemble.device,
                denoiser_run_steps=config.resemble.denoiser_run_steps,
                enhance_run_steps=config.resemble.enhance_run_steps,
                solver=config.resemble.solver,
                nfe=config.resemble.nfe,
            )
        else:
            self.enhancer_primary = None

        if config.enhancement_mode in ["voicefixer", "both"]:
            self.enhancer_fallback = VoiceEnhancer(
                model_type="voicefixer",
                device=config.voicefixer.mode,
                mode=config.voicefixer.mode,
                cuda=config.voicefixer.cuda,
            )
        else:
            self.enhancer_fallback = None

        self.post_processor = PostProcessor(
            noise_gate_threshold=config.postprocess.noise_gate_threshold,
            target_lufs=config.postprocess.target_lufs,
            highpass_cutoff=config.postprocess.highpass_cutoff,
            apply_deessing=config.postprocess.apply_deessing,
            deess_frequency=config.postprocess.deess_frequency,
            deess_threshold=config.postprocess.deess_threshold,
        )

        self.quality_assessor = QualityAssessor(
            calculate_snr=config.quality.calculate_snr,
            calculate_pesq=config.quality.calculate_pesq,
            calculate_stoi=config.quality.calculate_stoi,
        )

        logger.info("Pipeline initialized successfully")

    def process(
        self,
        input_path: str,
        output_path: str,
        save_intermediate: Optional[bool] = None,
    ) -> ProcessingResult:
        """
        Process audio file through complete pipeline.

        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            save_intermediate: Whether to save intermediate files

        Returns:
            ProcessingResult with metrics and paths
        """
        import time

        start_time = time.time()

        logger.info(f"Starting processing: {input_path} -> {output_path}")

        # Determine if we should save intermediate files
        if save_intermediate is None:
            save_intermediate = self.config.save_intermediate

        intermediate_files = {} if save_intermediate else None
        output_dir = Path(output_path).parent

        try:
            # Stage 1: Audio input and preprocessing
            logger.info("Stage 1: Audio input and preprocessing")
            audio, sr, quality_metrics = self.audio_input.preprocess(input_path)

            if save_intermediate:
                preprocessed_path = output_dir / "01_preprocessed.wav"
                sf.write(preprocessed_path, audio, sr)
                intermediate_files['preprocessed'] = str(preprocessed_path)

            # Store original for quality comparison
            original_audio = audio.copy()

            # Stage 2: Source separation
            logger.info("Stage 2: Source separation (Demucs)")
            separated = self.source_separator.separate(audio, sr)

            if save_intermediate:
                vocals_path = output_dir / "02_vocals_separated.wav"
                background_path = output_dir / "02_background.wav"
                sf.write(vocals_path, separated.vocals, sr)
                sf.write(background_path, separated.background, sr)
                intermediate_files['vocals'] = str(vocals_path)
                intermediate_files['background'] = str(background_path)

            # Extract vocals
            vocals = separated.vocals

            # Stage 3: Voice enhancement
            logger.info("Stage 3: Voice enhancement")
            enhanced = self._enhance_vocals(vocals, sr, output_dir, save_intermediate)

            if save_intermediate and enhanced is not None:
                enhanced_path = output_dir / "03_enhanced.wav"
                sf.write(enhanced_path, enhanced, sr)
                intermediate_files['enhanced'] = str(enhanced_path)
            elif enhanced is None:
                # Enhancement failed, use separated vocals
                logger.warning("Enhancement failed, using separated vocals")
                enhanced = vocals

            # Stage 4: Post-processing
            logger.info("Stage 4: Post-processing")
            processed = self.post_processor.process(enhanced, sr)

            if save_intermediate:
                processed_path = output_dir / "04_postprocessed.wav"
                sf.write(processed_path, processed, sr)
                intermediate_files['postprocessed'] = str(processed_path)

            # Stage 5: Save output
            logger.info(f"Stage 5: Saving output to {output_path}")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine bit depth
            subtype = f'PCM_{self.config.output_bitdepth}'

            sf.write(output_path, processed, sr, subtype=subtype)

            # Stage 6: Quality assessment
            logger.info("Stage 6: Quality assessment")
            metrics = self.quality_assessor.assess(processed, original_audio, sr)

            # Processing time
            processing_time = time.time() - start_time

            logger.info(f"Processing complete in {processing_time:.2f}s")
            logger.info(f"\n{metrics}")

            return ProcessingResult(
                output_path=str(output_path),
                metrics=metrics,
                processing_time=processing_time,
                intermediate_files=intermediate_files,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline processing failed: {e}")

    def _enhance_vocals(
        self,
        vocals: np.ndarray,
        sr: int,
        output_dir: Path,
        save_intermediate: bool
    ) -> Optional[np.ndarray]:
        """
        Enhance vocals using configured enhancement mode.

        Args:
            vocals: Vocals waveform
            sr: Sample rate
            output_dir: Directory for intermediate files
            save_intermediate: Whether to save intermediate files

        Returns:
            Enhanced vocals or None if enhancement fails
        """
        try:
            # Try primary enhancer (Resemble-Enhance)
            if self.enhancer_primary is not None:
                logger.info("Enhancing with Resemble-Enhance")
                enhanced = self.enhancer_primary.enhance_speech(vocals, sr)
                return enhanced

            # Try fallback enhancer (VoiceFixer)
            if self.enhancer_fallback is not None:
                logger.info("Enhancing with VoiceFixer")
                enhanced = self.enhancer_fallback.enhance_speech(vocals, sr)
                return enhanced

            # No enhancer available
            logger.warning("No voice enhancer configured")
            return vocals

        except Exception as e:
            logger.error(f"Voice enhancement failed: {e}")

            # Try fallback if primary failed
            if self.enhancer_primary is not None and self.enhancer_fallback is not None:
                try:
                    logger.info("Trying fallback enhancer (VoiceFixer)")
                    enhanced = self.enhancer_fallback.enhance_speech(vocals, sr)
                    return enhanced
                except Exception as e2:
                    logger.error(f"Fallback enhancement also failed: {e2}")

            return None

    def process_batch(
        self,
        input_files: list[str],
        output_dir: str,
        prefix: str = "clean_"
    ) -> list[ProcessingResult]:
        """
        Process multiple audio files in batch.

        Args:
            input_files: List of input file paths
            output_dir: Output directory
            prefix: Prefix for output filenames

        Returns:
            List of ProcessingResult objects
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []

        for i, input_file in enumerate(input_files, 1):
            logger.info(f"Processing file {i}/{len(input_files)}: {input_file}")

            try:
                # Generate output filename
                input_path = Path(input_file)
                output_file = output_path / f"{prefix}{input_path.name}"

                # Process file
                result = self.process(str(input_file), str(output_file))
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                # Continue with next file
                continue

        logger.info(f"Batch processing complete: {len(results)}/{len(input_files)} successful")

        return results
