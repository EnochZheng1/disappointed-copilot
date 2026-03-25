"""Pydantic v2 configuration schema for the entire application."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class CameraBackend(str, Enum):
    PICAMERA2 = "picamera2"
    WEBCAM = "webcam"
    FILE = "file"


class DetectorBackend(str, Enum):
    CORAL = "coral"
    ULTRALYTICS = "ultralytics"
    OPENCV_DNN = "opencv_dnn"


class CameraConfig(BaseModel):
    backend: CameraBackend = CameraBackend.WEBCAM
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    file_path: str | None = None  # For FILE backend


class DetectorConfig(BaseModel):
    backend: DetectorBackend = DetectorBackend.ULTRALYTICS
    model_path: str = "models/yolov8n.pt"
    edgetpu_model_path: str = "models/yolov8n_full_integer_quant_edgetpu.tflite"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    input_size: int = 320
    # COCO class IDs: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7, traffic_light=9
    target_classes: list[int] = Field(default=[0, 1, 2, 3, 5, 7, 9])
    detect_every_n_frames: int = 1  # Skip frames to save CPU


class LaneConfig(BaseModel):
    enabled: bool = True
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 50
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 50
    roi_vertices_ratio: list[list[float]] = Field(default=[
        [0.1, 1.0], [0.45, 0.6], [0.55, 0.6], [0.9, 1.0]
    ])
    ema_alpha: float = 0.3  # Smoothing factor for lane line EMA
    confidence_threshold: float = 0.3  # Below this, lane-dependent triggers auto-disable
    lane_every_n_frames: int = 1


class TriggerConfig(BaseModel):
    cooldown_seconds: float = 15.0


class TailgaterConfig(TriggerConfig):
    enabled: bool = True
    bbox_growth_rate_threshold: float = 0.15  # 15% growth per second
    min_bbox_area_ratio: float = 0.02
    consecutive_frames: int = 10


class SwerverConfig(TriggerConfig):
    enabled: bool = True
    lane_cross_ratio: float = 0.3
    consecutive_frames: int = 5


class GreenLightConfig(TriggerConfig):
    enabled: bool = True
    green_detected_seconds: float = 3.0
    speed_threshold: float = 2.0  # Sparse optical flow magnitude
    max_feature_points: int = 20


class SelfCritiqueConfig(TriggerConfig):
    enabled: bool = True
    departure_threshold: float = 0.15


class HardBrakeConfig(TriggerConfig):
    enabled: bool = True
    bbox_growth_rate_threshold: float = 0.25  # All vehicles growing simultaneously
    min_tracked_vehicles: int = 2
    consecutive_frames: int = 5


class TriggersConfig(BaseModel):
    tailgater: TailgaterConfig = TailgaterConfig()
    swerver: SwerverConfig = SwerverConfig()
    green_light: GreenLightConfig = GreenLightConfig()
    self_critique: SelfCritiqueConfig = SelfCritiqueConfig()
    hard_brake: HardBrakeConfig = HardBrakeConfig()


class CommentaryConfig(BaseModel):
    prebaked_audio_dir: Path = Path("assets/audio")
    voice_pack: str = "british_instructor"
    llm_enabled: bool = False
    llm_cpu_threshold: float = 0.7  # Only fire LLM if CPU usage below this
    ollama_model: str = "llama3.2:1b"
    ollama_base_url: str = "http://localhost:11434"
    piper_model_path: str = "models/en_US-lessac-medium.onnx"


class RecordingConfig(BaseModel):
    enabled: bool = True
    output_dir: Path = Path("clips")
    hot_buffer_seconds: int = 60
    pre_trigger_seconds: int = 30
    post_trigger_seconds: int = 15
    output_fps: int = 15
    buffer_resolution: list[int] = Field(default=[640, 360])


class AudioConfig(BaseModel):
    enabled: bool = True
    volume: float = 0.8
    cooldown_seconds: float = 10.0  # Global cooldown between any audio


class AppConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    lane: LaneConfig = LaneConfig()
    triggers: TriggersConfig = TriggersConfig()
    commentary: CommentaryConfig = CommentaryConfig()
    recording: RecordingConfig = RecordingConfig()
    audio: AudioConfig = AudioConfig()
    debug_display: bool = True
    log_level: str = "INFO"
