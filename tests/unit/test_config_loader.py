"""Tests for config loading and validation."""

from pathlib import Path

from disappointed.config.loader import load_config
from disappointed.config.schema import AppConfig, CameraBackend, DetectorBackend


def test_default_config():
    """AppConfig with defaults should be valid."""
    config = AppConfig()
    assert config.camera.backend == CameraBackend.WEBCAM
    assert config.detector.backend == DetectorBackend.ULTRALYTICS
    assert config.debug_display is True


def test_load_yaml_configs():
    """Loading the actual YAML files should produce a valid config."""
    config = load_config(
        Path("config/default.yaml"),
        Path("config/desktop_dev.yaml"),
    )
    assert config.camera.backend == CameraBackend.WEBCAM
    assert config.detector.backend == DetectorBackend.ULTRALYTICS
    assert config.log_level == "DEBUG"  # Overridden by desktop_dev.yaml


def test_later_config_overrides_earlier():
    """Desktop dev config should override default values."""
    config = load_config(
        Path("config/default.yaml"),
        Path("config/desktop_dev.yaml"),
    )
    assert config.recording.enabled is False  # Desktop dev disables recording


def test_missing_config_file_is_ok():
    """A non-existent config file should be silently skipped."""
    config = load_config(Path("config/nonexistent.yaml"))
    assert isinstance(config, AppConfig)
