"""Application factory — wires all components together from config."""

import logging
from typing import Optional

from disappointed.camera.base import CameraSource
from disappointed.commentary.base import CommentaryEngine
from disappointed.config.schema import AppConfig, CameraBackend, DetectorBackend
from disappointed.detection.base import ObjectDetector
from disappointed.lane.detector import LaneDetector
from disappointed.pipeline.coordinator import PipelineCoordinator
from disappointed.triggers.registry import TriggerRegistry

logger = logging.getLogger(__name__)


def _create_camera(config: AppConfig) -> CameraSource:
    """Create the appropriate camera source based on config."""
    match config.camera.backend:
        case CameraBackend.WEBCAM:
            from disappointed.camera.webcam_source import WebcamSource
            return WebcamSource(
                device_index=config.camera.device_index,
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
            )
        case CameraBackend.FILE:
            from disappointed.camera.file_source import FileSource
            if not config.camera.file_path:
                raise ValueError("camera.file_path must be set when using FILE backend")
            return FileSource(file_path=config.camera.file_path, loop=True)
        case CameraBackend.PICAMERA2:
            from disappointed.camera.picamera_source import PicameraSource
            return PicameraSource(
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
            )
        case _:
            raise ValueError(f"Unknown camera backend: {config.camera.backend}")


def _create_detector(config: AppConfig) -> ObjectDetector | None:
    """Create the appropriate object detector based on config."""
    match config.detector.backend:
        case DetectorBackend.ULTRALYTICS:
            from disappointed.detection.ultralytics_detector import UltralyticsDetector
            return UltralyticsDetector(
                model_path=config.detector.model_path,
                confidence_threshold=config.detector.confidence_threshold,
                target_classes=config.detector.target_classes,
                input_size=config.detector.input_size,
            )
        case DetectorBackend.CORAL:
            from disappointed.detection.coral_detector import CoralDetector
            return CoralDetector(
                model_path=config.detector.edgetpu_model_path,
                confidence_threshold=config.detector.confidence_threshold,
                target_classes=config.detector.target_classes,
                input_size=config.detector.input_size,
            )
        case DetectorBackend.OPENCV_DNN:
            raise NotImplementedError("OpenCV DNN detector not yet implemented")
        case _:
            raise ValueError(f"Unknown detector backend: {config.detector.backend}")


def _create_lane_detector(config: AppConfig) -> Optional[LaneDetector]:
    """Create lane detector if enabled."""
    if not config.lane.enabled:
        return None
    return LaneDetector(
        config=config.lane,
        frame_width=config.camera.width,
        frame_height=config.camera.height,
    )


def _create_trigger_registry(config: AppConfig) -> TriggerRegistry:
    """Create and populate the trigger registry based on config."""
    registry = TriggerRegistry()

    if config.triggers.tailgater.enabled:
        from disappointed.triggers.tailgater import TailgaterTrigger
        registry.register(TailgaterTrigger(config.triggers.tailgater))

    if config.triggers.swerver.enabled:
        from disappointed.triggers.swerver import SwerverTrigger
        registry.register(SwerverTrigger(
            config.triggers.swerver,
            lane_confidence_threshold=config.lane.confidence_threshold,
        ))

    if config.triggers.green_light.enabled:
        from disappointed.triggers.green_light import GreenLightTrigger
        registry.register(GreenLightTrigger(config.triggers.green_light))

    if config.triggers.self_critique.enabled:
        from disappointed.triggers.self_critique import SelfCritiqueTrigger
        registry.register(SelfCritiqueTrigger(
            config.triggers.self_critique,
            lane_confidence_threshold=config.lane.confidence_threshold,
        ))

    if config.triggers.hard_brake.enabled:
        from disappointed.triggers.hard_brake import HardBrakeTrigger
        registry.register(HardBrakeTrigger(config.triggers.hard_brake))

    logger.info(f"Triggers registered: {registry.trigger_names}")
    return registry


def _create_prebaked_engine(config: AppConfig) -> Optional[CommentaryEngine]:
    """Create the pre-baked audio commentary engine."""
    from disappointed.commentary.prebaked import PrebakedEngine
    engine = PrebakedEngine(
        audio_dir=config.commentary.prebaked_audio_dir,
        voice_pack=config.commentary.voice_pack,
    )
    return engine


def _create_llm_engine(config: AppConfig) -> Optional[CommentaryEngine]:
    """Create the LLM-based commentary engine if enabled."""
    if not config.commentary.llm_enabled:
        return None

    from disappointed.commentary.llm_engine import LLMEngine
    return LLMEngine(
        voice_pack=config.commentary.voice_pack,
        ollama_model=config.commentary.ollama_model,
        ollama_base_url=config.commentary.ollama_base_url,
        piper_model_path=config.commentary.piper_model_path,
    )


def create_and_run(config: AppConfig, demo_interval: float | None = None) -> None:
    """Create all components from config and run the pipeline."""
    logger.info(f"Camera: {config.camera.backend.value}")
    logger.info(f"Detector: {config.detector.backend.value}")
    logger.info(f"Voice pack: {config.commentary.voice_pack}")
    logger.info(f"Lane detection: {'enabled' if config.lane.enabled else 'disabled'}")
    logger.info(f"LLM commentary: {'enabled' if config.commentary.llm_enabled else 'disabled'}")
    if demo_interval:
        logger.info(f"DEMO MODE: firing random triggers every {demo_interval}s")
    logger.info(f"Debug display: {config.debug_display}")

    camera = _create_camera(config)
    detector = _create_detector(config)
    lane_detector = _create_lane_detector(config)
    trigger_registry = _create_trigger_registry(config)
    prebaked_engine = _create_prebaked_engine(config)
    llm_engine = _create_llm_engine(config)

    coordinator = PipelineCoordinator(
        config=config,
        camera=camera,
        detector=detector,
        lane_detector=lane_detector,
        trigger_registry=trigger_registry,
        prebaked_engine=prebaked_engine,
        llm_engine=llm_engine,
        demo_interval=demo_interval,
    )
    coordinator.run()
