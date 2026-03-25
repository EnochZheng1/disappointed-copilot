"""Simple centroid-based object tracker for consistent IDs across frames."""

from disappointed.utils.math_utils import euclidean_distance
from .models import BoundingBox


class CentroidTracker:
    """Assigns persistent track IDs to detections using centroid distance matching."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 100.0):
        self._next_id = 0
        self._objects: dict[int, tuple[float, float]] = {}  # track_id -> centroid
        self._disappeared: dict[int, int] = {}  # track_id -> frames since last seen
        self._max_disappeared = max_disappeared
        self._max_distance = max_distance

    def update(self, detections: list[BoundingBox]) -> list[BoundingBox]:
        """Match detections to existing tracks and assign track_ids.

        Returns the same detections list with track_id populated.
        """
        if not detections:
            # Mark all existing objects as disappeared
            for track_id in list(self._disappeared.keys()):
                self._disappeared[track_id] += 1
                if self._disappeared[track_id] > self._max_disappeared:
                    del self._objects[track_id]
                    del self._disappeared[track_id]
            return detections

        new_centroids = [det.center for det in detections]

        if not self._objects:
            # No existing tracks — register all detections
            for i, det in enumerate(detections):
                self._register(new_centroids[i])
                det.track_id = self._next_id - 1
            return detections

        # Compute distance matrix between existing tracks and new detections
        existing_ids = list(self._objects.keys())
        existing_centroids = [self._objects[tid] for tid in existing_ids]

        distances: list[list[float]] = []
        for ec in existing_centroids:
            row = [euclidean_distance(ec, nc) for nc in new_centroids]
            distances.append(row)

        # Greedy matching: match closest pairs first
        used_existing: set[int] = set()
        used_new: set[int] = set()
        matches: list[tuple[int, int]] = []  # (existing_idx, new_idx)

        # Flatten and sort all pairs by distance
        pairs = []
        for i in range(len(existing_ids)):
            for j in range(len(new_centroids)):
                pairs.append((distances[i][j], i, j))
        pairs.sort(key=lambda x: x[0])

        for dist, ei, ni in pairs:
            if ei in used_existing or ni in used_new:
                continue
            if dist > self._max_distance:
                break
            matches.append((ei, ni))
            used_existing.add(ei)
            used_new.add(ni)

        # Update matched tracks
        for ei, ni in matches:
            track_id = existing_ids[ei]
            self._objects[track_id] = new_centroids[ni]
            self._disappeared[track_id] = 0
            detections[ni].track_id = track_id

        # Increment disappeared count for unmatched existing tracks
        for i, tid in enumerate(existing_ids):
            if i not in used_existing:
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self._max_disappeared:
                    del self._objects[tid]
                    del self._disappeared[tid]

        # Register new detections that didn't match
        for j in range(len(new_centroids)):
            if j not in used_new:
                self._register(new_centroids[j])
                detections[j].track_id = self._next_id - 1

        return detections

    def _register(self, centroid: tuple[float, float]) -> None:
        self._objects[self._next_id] = centroid
        self._disappeared[self._next_id] = 0
        self._next_id += 1
