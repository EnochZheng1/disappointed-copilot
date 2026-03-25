"""Data models for trigger events."""

from dataclasses import dataclass, field
from enum import Enum


class TriggerSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TriggerEvent:
    """Represents a detected driving behavior event."""

    trigger_name: str  # e.g. "tailgater", "swerver"
    severity: float  # 0.0 - 1.0
    timestamp: float
    frame_index: int
    description: str  # Human-readable, used as LLM prompt context
    metadata: dict = field(default_factory=dict)
