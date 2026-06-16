import copy
from typing import Dict, List, Optional, Tuple
from collections import deque

from app.services.storage_manager import StorageManager


class SegmentEditor:
    _instance = None
    _undo_stacks: Dict[str, deque] = {}
    MAX_UNDO_STEPS = 20

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.storage = StorageManager()

    def _get_undo_stack(self, video_id: str) -> deque:
        if video_id not in self._undo_stacks:
            self._undo_stacks[video_id] = deque(maxlen=self.MAX_UNDO_STEPS)
        return self._undo_stacks[video_id]

    def _push_undo_state(
        self,
        video_id: str,
        model_version: str,
        segments: List[Dict],
        timeline: List[Dict],
        frame_predictions: Dict,
    ) -> None:
        stack = self._get_undo_stack(video_id)
        state = {
            "model_version": model_version,
            "segments": copy.deepcopy(segments),
            "timeline": copy.deepcopy(timeline),
            "frame_predictions": copy.deepcopy(frame_predictions),
        }
        stack.append(state)

    def can_undo(self, video_id: str) -> bool:
        stack = self._get_undo_stack(video_id)
        return len(stack) > 0

    def clear_undo_stack(self, video_id: str) -> None:
        if video_id in self._undo_stacks:
            del self._undo_stacks[video_id]

    def _rebuild_timeline_from_segments(
        self,
        segments: List[Dict],
        fps: float,
        duration: float,
        action_classes: List[Dict],
    ) -> List[Dict]:
        id_to_name = {c["id"]: c["name"] for c in action_classes}
        id_to_color = {c["id"]: c["color"] for c in action_classes}

        total_frames = int(duration * fps) if fps > 0 else 0
        timeline = []

        frame_labels = [0] * total_frames
        for seg in segments:
            start_frame = int(seg["start_frame"])
            end_frame = int(seg["end_frame"])
            for f in range(start_frame, min(end_frame + 1, total_frames)):
                frame_labels[f] = seg["action_id"]

        for i in range(total_frames):
            label_id = frame_labels[i]
            timeline.append({
                "frame": i,
                "time": i / fps if fps > 0 else 0,
                "action_id": label_id,
                "action_name": id_to_name.get(label_id, "未知"),
                "color": id_to_color.get(label_id, "#808080"),
            })

        return timeline

    def _rebuild_frame_predictions(
        self,
        segments: List[Dict],
        total_frames: int,
        num_classes: int,
    ) -> Dict:
        labels = [0] * total_frames
        probabilities = [[0.0] * num_classes for _ in range(total_frames)]

        for seg in segments:
            start_frame = int(seg["start_frame"])
            end_frame = int(seg["end_frame"])
            action_id = seg["action_id"]
            confidence = seg.get("confidence", 0.5)
            for f in range(start_frame, min(end_frame + 1, total_frames)):
                labels[f] = action_id
                probabilities[f][action_id] = confidence
                for c in range(num_classes):
                    if c != action_id:
                        probabilities[f][c] = (1.0 - confidence) / (num_classes - 1)

        return {
            "labels": labels,
            "probabilities": probabilities,
        }

    def update_segment(
        self,
        video_id: str,
        model_version: str,
        segment_index: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action_id: Optional[int] = None,
    ) -> Dict:
        results = self.storage.load_results(video_id, model_version)
        if not results:
            raise ValueError(f"Results not found for video {video_id}")

        segments = results["segments"]
        if segment_index < 0 or segment_index >= len(segments):
            raise ValueError(f"Segment index {segment_index} out of range")

        self._push_undo_state(
            video_id,
            model_version,
            segments,
            results["timeline"],
            results["frame_predictions"],
        )

        seg = segments[segment_index]
        fps = results["video_info"]["target_fps"]

        if start_time is not None:
            seg["start_time"] = start_time
            seg["start_frame"] = int(start_time * fps)

        if end_time is not None:
            seg["end_time"] = end_time
            seg["end_frame"] = int(end_time * fps)

        if action_id is not None:
            action_classes = results["action_classes"]
            action_info = next((c for c in action_classes if c["id"] == action_id), None)
            if action_info:
                seg["action_id"] = action_id
                seg["action_name"] = action_info["name"]

        if seg["start_frame"] > seg["end_frame"]:
            seg["start_frame"], seg["end_frame"] = seg["end_frame"], seg["start_frame"]
            seg["start_time"], seg["end_time"] = seg["end_time"], seg["start_time"]

        results["timeline"] = self._rebuild_timeline_from_segments(
            segments,
            fps,
            results["video_info"]["duration"],
            results["action_classes"],
        )

        num_classes = len(results["action_classes"])
        total_frames = results["video_info"]["sample_frames_count"]
        results["frame_predictions"] = self._rebuild_frame_predictions(
            segments, total_frames, num_classes
        )

        self.storage.save_results(video_id, model_version, results)
        return results

    def split_segment(
        self,
        video_id: str,
        model_version: str,
        segment_index: int,
        split_time: float,
    ) -> Dict:
        results = self.storage.load_results(video_id, model_version)
        if not results:
            raise ValueError(f"Results not found for video {video_id}")

        segments = results["segments"]
        if segment_index < 0 or segment_index >= len(segments):
            raise ValueError(f"Segment index {segment_index} out of range")

        seg = segments[segment_index]
        if split_time <= seg["start_time"] or split_time >= seg["end_time"]:
            raise ValueError(
                f"Split time {split_time}s must be within segment "
                f"[{seg['start_time']}, {seg['end_time']}]s"
            )

        self._push_undo_state(
            video_id,
            model_version,
            segments,
            results["timeline"],
            results["frame_predictions"],
        )

        fps = results["video_info"]["target_fps"]
        split_frame = int(split_time * fps)

        seg1 = copy.deepcopy(seg)
        seg1["end_time"] = split_time
        seg1["end_frame"] = split_frame

        seg2 = copy.deepcopy(seg)
        seg2["start_time"] = split_time
        seg2["start_frame"] = split_frame

        segments.pop(segment_index)
        segments.insert(segment_index, seg2)
        segments.insert(segment_index, seg1)

        segments.sort(key=lambda s: s["start_frame"])

        results["timeline"] = self._rebuild_timeline_from_segments(
            segments,
            fps,
            results["video_info"]["duration"],
            results["action_classes"],
        )

        num_classes = len(results["action_classes"])
        total_frames = results["video_info"]["sample_frames_count"]
        results["frame_predictions"] = self._rebuild_frame_predictions(
            segments, total_frames, num_classes
        )

        self.storage.save_results(video_id, model_version, results)
        return results

    def merge_segments(
        self,
        video_id: str,
        model_version: str,
        index_1: int,
        index_2: int,
    ) -> Dict:
        results = self.storage.load_results(video_id, model_version)
        if not results:
            raise ValueError(f"Results not found for video {video_id}")

        segments = results["segments"]
        for idx in [index_1, index_2]:
            if idx < 0 or idx >= len(segments):
                raise ValueError(f"Segment index {idx} out of range")

        if index_1 == index_2:
            raise ValueError("Cannot merge segment with itself")

        idx_a, idx_b = min(index_1, index_2), max(index_1, index_2)
        if idx_b != idx_a + 1:
            raise ValueError("Can only merge adjacent segments")

        seg_a = segments[idx_a]
        seg_b = segments[idx_b]

        self._push_undo_state(
            video_id,
            model_version,
            segments,
            results["timeline"],
            results["frame_predictions"],
        )

        duration_a = seg_a["end_time"] - seg_a["start_time"]
        duration_b = seg_b["end_time"] - seg_b["start_time"]

        if duration_a >= duration_b:
            merged_action_id = seg_a["action_id"]
            merged_action_name = seg_a["action_name"]
        else:
            merged_action_id = seg_b["action_id"]
            merged_action_name = seg_b["action_name"]

        avg_confidence = (seg_a["confidence"] + seg_b["confidence"]) / 2.0

        merged_seg = {
            "start_frame": seg_a["start_frame"],
            "end_frame": seg_b["end_frame"],
            "start_time": seg_a["start_time"],
            "end_time": seg_b["end_time"],
            "action_id": merged_action_id,
            "action_name": merged_action_name,
            "confidence": avg_confidence,
        }

        segments.pop(idx_b)
        segments.pop(idx_a)
        segments.insert(idx_a, merged_seg)

        fps = results["video_info"]["target_fps"]
        results["timeline"] = self._rebuild_timeline_from_segments(
            segments,
            fps,
            results["video_info"]["duration"],
            results["action_classes"],
        )

        num_classes = len(results["action_classes"])
        total_frames = results["video_info"]["sample_frames_count"]
        results["frame_predictions"] = self._rebuild_frame_predictions(
            segments, total_frames, num_classes
        )

        self.storage.save_results(video_id, model_version, results)
        return results

    def undo(self, video_id: str, model_version: str) -> Optional[Dict]:
        stack = self._get_undo_stack(video_id)
        if not stack:
            return None

        state = stack.pop()

        results = self.storage.load_results(video_id, model_version)
        if not results:
            return None

        results["segments"] = state["segments"]
        results["timeline"] = state["timeline"]
        results["frame_predictions"] = state["frame_predictions"]

        self.storage.save_results(video_id, model_version, results)
        return results
