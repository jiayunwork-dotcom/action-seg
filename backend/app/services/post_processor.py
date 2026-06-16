import numpy as np
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

from app.core.config import settings


@dataclass
class ActionSegment:
    start_frame: int
    end_frame: int
    action_id: int
    action_name: str
    confidence: float
    start_time: float
    end_time: float


class PostProcessor:
    def __init__(self, action_classes: List[Dict]):
        self.action_classes = action_classes
        self.id_to_name = {c["id"]: c["name"] for c in action_classes}
        self._load_config()

    def _load_config(self):
        with open(settings.MODEL_CONFIG_PATH, "r") as f:
            config = json.load(f)
        self.min_duration_sec = config["postprocessing"]["min_duration_sec"]
        self.merge_gap_sec = config["postprocessing"]["merge_gap_sec"]
        self.nms_iou_threshold = config["postprocessing"]["nms_iou_threshold"]

    def smooth_predictions(self, labels: np.ndarray) -> np.ndarray:
        if len(labels) < 3:
            return labels

        smoothed = labels.copy()
        for i in range(1, len(labels) - 1):
            if labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
                if labels[i - 1] == labels[i + 1]:
                    smoothed[i] = labels[i - 1]

        for i in range(2, len(labels) - 2):
            window = smoothed[i - 2:i + 3]
            counts = np.bincount(window)
            most_common = np.argmax(counts)
            if counts[most_common] >= 4:
                smoothed[i] = most_common

        return smoothed

    def labels_to_segments(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        fps: float,
    ) -> List[ActionSegment]:
        if len(labels) == 0:
            return []

        segments = []
        start = 0
        current_label = labels[0]

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append(self._create_segment(
                    start, i - 1, current_label, labels, probs, fps
                ))
                start = i
                current_label = labels[i]

        segments.append(self._create_segment(
            start, len(labels) - 1, current_label, labels, probs, fps
        ))

        return segments

    def _create_segment(
        self,
        start_frame: int,
        end_frame: int,
        action_id: int,
        labels: np.ndarray,
        probs: np.ndarray,
        fps: float,
    ) -> ActionSegment:
        conf = float(np.mean(probs[start_frame:end_frame + 1, action_id]))
        return ActionSegment(
            start_frame=start_frame,
            end_frame=end_frame,
            action_id=int(action_id),
            action_name=self.id_to_name.get(int(action_id), "未知"),
            confidence=conf,
            start_time=start_frame / fps if fps > 0 else 0,
            end_time=(end_frame + 1) / fps if fps > 0 else 0,
        )

    def filter_short_segments(
        self,
        segments: List[ActionSegment],
        fps: float,
    ) -> List[ActionSegment]:
        if not segments:
            return []

        min_frames = max(1, int(self.min_duration_sec * fps))

        fixed = []
        for i, seg in enumerate(segments):
            duration_frames = seg.end_frame - seg.start_frame + 1
            if duration_frames >= min_frames:
                fixed.append(ActionSegment(
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    action_id=seg.action_id,
                    action_name=seg.action_name,
                    confidence=seg.confidence,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                ))
            else:
                prev_dur = (fixed[-1].end_frame - fixed[-1].start_frame + 1) if fixed else 0
                next_dur = (segments[i + 1].end_frame - segments[i + 1].start_frame + 1) if i + 1 < len(segments) else 0

                if prev_dur >= next_dur and fixed:
                    donor = fixed[-1]
                elif next_dur > 0 and i + 1 < len(segments):
                    donor = segments[i + 1]
                elif fixed:
                    donor = fixed[-1]
                else:
                    fixed.append(ActionSegment(
                        start_frame=seg.start_frame,
                        end_frame=seg.end_frame,
                        action_id=0,
                        action_name=self.id_to_name.get(0, "背景"),
                        confidence=seg.confidence,
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                    ))
                    continue

                fixed.append(ActionSegment(
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    action_id=donor.action_id,
                    action_name=donor.action_name,
                    confidence=seg.confidence,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                ))

        merged = [fixed[0]]
        for seg in fixed[1:]:
            prev = merged[-1]
            if seg.action_id == prev.action_id and seg.start_frame <= prev.end_frame + 1:
                prev = ActionSegment(
                    start_frame=prev.start_frame,
                    end_frame=max(prev.end_frame, seg.end_frame),
                    action_id=prev.action_id,
                    action_name=prev.action_name,
                    confidence=prev.confidence,
                    start_time=prev.start_time,
                    end_time=max(prev.end_time, seg.end_time),
                )
                merged[-1] = prev
            else:
                merged.append(seg)

        return merged

    def merge_adjacent_same_class(
        self,
        segments: List[ActionSegment],
        fps: float,
    ) -> List[ActionSegment]:
        if len(segments) < 2:
            return segments

        max_gap_frames = max(1, int(self.merge_gap_sec * fps))
        merged = [ActionSegment(
            start_frame=segments[0].start_frame,
            end_frame=segments[0].end_frame,
            action_id=segments[0].action_id,
            action_name=segments[0].action_name,
            confidence=segments[0].confidence,
            start_time=segments[0].start_time,
            end_time=segments[0].end_time,
        )]

        for seg in segments[1:]:
            prev = merged[-1]
            if (seg.action_id == prev.action_id and
                seg.start_frame - prev.end_frame - 1 <= max_gap_frames):
                merged[-1] = ActionSegment(
                    start_frame=prev.start_frame,
                    end_frame=seg.end_frame,
                    action_id=prev.action_id,
                    action_name=prev.action_name,
                    confidence=(prev.confidence + seg.confidence) / 2,
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                )
            else:
                merged.append(ActionSegment(
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    action_id=seg.action_id,
                    action_name=seg.action_name,
                    confidence=seg.confidence,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                ))

        return merged

    def temporal_nms(self, segments: List[ActionSegment]) -> List[ActionSegment]:
        if len(segments) < 2:
            return segments

        segments_sorted = sorted(segments, key=lambda s: s.confidence, reverse=True)
        keep = []

        while segments_sorted:
            best = segments_sorted.pop(0)
            keep.append(best)
            segments_sorted = [
                s for s in segments_sorted
                if self._temporal_iou(best, s) <= self.nms_iou_threshold
            ]

        return sorted(keep, key=lambda s: s.start_frame)

    def _temporal_iou(self, seg1: ActionSegment, seg2: ActionSegment) -> float:
        intersection_start = max(seg1.start_frame, seg2.start_frame)
        intersection_end = min(seg1.end_frame, seg2.end_frame)

        if intersection_end < intersection_start:
            return 0.0

        intersection = intersection_end - intersection_start + 1
        union = (seg1.end_frame - seg1.start_frame + 1) + (seg2.end_frame - seg2.start_frame + 1) - intersection

        return intersection / union if union > 0 else 0.0

    def process(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        fps: float,
    ) -> Tuple[np.ndarray, List[Dict]]:
        smoothed_labels = self.smooth_predictions(labels)
        segments = self.labels_to_segments(smoothed_labels, probs, fps)
        segments = self.filter_short_segments(segments, fps)
        segments = self.merge_adjacent_same_class(segments, fps)
        segments = self.temporal_nms(segments)

        segments_dict = [asdict(s) for s in segments]
        return smoothed_labels, segments_dict
