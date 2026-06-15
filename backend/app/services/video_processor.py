import cv2
import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from app.core.config import settings


@dataclass
class VideoInfo:
    video_id: str
    filename: str
    file_path: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    target_fps: int
    sample_frames_count: int


class VideoProcessor:
    def __init__(self):
        self.target_fps = None
        self.resize_size = None
        self._load_config()

    def _load_config(self):
        with open(settings.MODEL_CONFIG_PATH, "r") as f:
            config = json.load(f)
        self.target_fps = config["preprocessing"]["target_fps"]
        self.resize_size = tuple(config["preprocessing"]["resize_size"])
        self.max_video_minutes = config["preprocessing"]["max_video_minutes"]
        self.segment_minutes = config["preprocessing"]["segment_minutes"]
        self.overlap_frames = config["preprocessing"]["overlap_frames"]

    @staticmethod
    def generate_video_id(file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def get_video_info(self, video_path: str, video_id: str) -> VideoInfo:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        actual_target_fps = min(self.target_fps, fps) if fps > 0 else self.target_fps
        sample_frames_count = int(duration * actual_target_fps)

        cap.release()

        return VideoInfo(
            video_id=video_id,
            filename=Path(video_path).name,
            file_path=video_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            target_fps=actual_target_fps,
            sample_frames_count=sample_frames_count,
        )

    def sample_frames(self, video_path: str, target_fps: Optional[int] = None) -> np.ndarray:
        if target_fps is None:
            target_fps = self.target_fps

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= target_fps:
            frame_indices = list(range(total_frames))
        else:
            step = original_fps / target_fps
            frame_indices = [int(i * step) for i in range(int(total_frames / step))]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resize_size)
            frames.append(frame)

        cap.release()

        frames_array = np.array(frames, dtype=np.float32) / 255.0
        frames_array = frames_array.transpose(0, 3, 1, 2)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        frames_array = (frames_array - mean) / std

        return frames_array

    def split_into_segments(self, frames: np.ndarray, video_info) -> List[Dict]:
        if isinstance(video_info, dict):
            duration = video_info["duration"]
            target_fps = video_info["target_fps"]
        else:
            duration = video_info.duration
            target_fps = video_info.target_fps

        if duration <= self.max_video_minutes * 60:
            return [{
                "frames": frames,
                "start_frame": 0,
                "end_frame": len(frames),
                "segment_idx": 0,
            }]

        segment_frames = int(self.segment_minutes * 60 * target_fps)
        overlap = self.overlap_frames
        segments = []
        total = len(frames)
        start = 0
        seg_idx = 0

        while start < total:
            end = min(start + segment_frames, total)
            seg_frames = frames[start:end]
            segments.append({
                "frames": seg_frames,
                "start_frame": start,
                "end_frame": end,
                "segment_idx": seg_idx,
            })
            seg_idx += 1
            if end >= total:
                break
            start = end - overlap

        return segments

    def merge_segment_predictions(
        self,
        segment_results: List[Tuple[np.ndarray, np.ndarray, int, int]],
        total_frames: int,
        overlap_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(segment_results) == 1:
            return segment_results[0][0], segment_results[0][1]

        num_classes = segment_results[0][1].shape[1]
        final_labels = np.zeros(total_frames, dtype=np.int64)
        final_probs = np.zeros((total_frames, num_classes), dtype=np.float32)
        weight_sum = np.zeros(total_frames, dtype=np.float32)

        for seg_labels, seg_probs, start_frame, end_frame in segment_results:
            seg_len = len(seg_labels)
            weights = np.ones(seg_len, dtype=np.float32)
            if start_frame > 0:
                weights[:overlap_frames] = np.linspace(0.1, 1.0, min(overlap_frames, seg_len))
            if end_frame < total_frames:
                overlap_start = max(0, seg_len - overlap_frames)
                weights[overlap_start:] = np.linspace(1.0, 0.1, seg_len - overlap_start)

            for i in range(seg_len):
                global_idx = start_frame + i
                if global_idx < total_frames:
                    final_probs[global_idx] += seg_probs[i] * weights[i]
                    weight_sum[global_idx] += weights[i]

        valid = weight_sum > 0
        final_probs[valid] /= weight_sum[valid][:, None]
        final_labels[valid] = np.argmax(final_probs[valid], axis=1)

        return final_labels, final_probs
