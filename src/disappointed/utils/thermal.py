"""CPU thermal monitor for Raspberry Pi — throttles pipeline when hot."""

import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

# Pi thermal zone file
_THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


class ThermalMonitor:
    """Monitors CPU temperature and provides throttling recommendations.

    On Raspberry Pi, reads from /sys/class/thermal/thermal_zone0/temp.
    On other platforms, returns a safe default temperature.
    """

    def __init__(
        self,
        throttle_temp: float = 75.0,
        pause_llm_temp: float = 80.0,
        critical_temp: float = 85.0,
    ):
        self.throttle_temp = throttle_temp
        self.pause_llm_temp = pause_llm_temp
        self.critical_temp = critical_temp
        self._is_pi = platform.system() == "Linux" and _THERMAL_PATH.exists()

        if self._is_pi:
            logger.info("Thermal monitor: Raspberry Pi detected, monitoring CPU temp")
        else:
            logger.info("Thermal monitor: not on Pi, temperature monitoring disabled")

    def get_cpu_temp(self) -> float:
        """Get CPU temperature in Celsius."""
        if not self._is_pi:
            return 45.0  # Safe default for non-Pi

        try:
            temp_str = _THERMAL_PATH.read_text().strip()
            return float(temp_str) / 1000.0  # millidegrees to degrees
        except Exception:
            return 45.0

    @property
    def should_throttle_detection(self) -> bool:
        """True if detection should run at reduced frequency."""
        return self.get_cpu_temp() >= self.throttle_temp

    @property
    def should_pause_llm(self) -> bool:
        """True if LLM engine should be paused."""
        return self.get_cpu_temp() >= self.pause_llm_temp

    @property
    def is_critical(self) -> bool:
        """True if temperature is critically high."""
        return self.get_cpu_temp() >= self.critical_temp

    def get_recommended_skip_frames(self) -> int:
        """Return how many frames to skip between detections based on temperature."""
        temp = self.get_cpu_temp()
        if temp < self.throttle_temp:
            return 1  # No skip
        elif temp < self.pause_llm_temp:
            return 2  # Skip every other frame
        elif temp < self.critical_temp:
            return 4  # Run detection every 4th frame
        else:
            return 8  # Minimal detection to prevent shutdown
