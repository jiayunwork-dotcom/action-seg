import numpy as np
import torch
from typing import Dict, Optional, Callable

from app.core.config import settings
from app.models.model_manager import ModelManager
from app.services.video_processor import VideoProcessor
from app.services.post_processor import PostProcessor
from app.services.storage_manager import StorageManager


class AnalysisPipeline:
    def __init__(self):
        self.model_manager = ModelManager()
        self.video_processor = VideoProcessor()
        self.storage = StorageManager()
        self.post_processor = PostProcessor(self.model_manager.get_action_classes())

    def run(
        self,
        video_id: str,
        model_version: str = "latest",
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if progress_callback:
            progress_callback(5, "加载视频信息")

        video_info = self.storage.load_video_info(video_id)
        if video_info is None:
            raise ValueError(f"Video {video_id} not found")

        video_path = self.storage.get_video_path(video_id)
        if video_path is None:
            raise ValueError(f"Video file not found for {video_id}")

        if progress_callback:
            progress_callback(10, "检查特征缓存")

        cached_features = self.storage.load_features(video_id, model_version)

        if cached_features is not None:
            if progress_callback:
                progress_callback(40, "特征缓存命中，跳过特征提取")
            features = cached_features
        else:
            if progress_callback:
                progress_callback(15, "视频预处理与抽帧")

            frames = self.video_processor.sample_frames(
                video_path, video_info["target_fps"]
            )

            if progress_callback:
                progress_callback(25, "分割视频片段")

            segments = self.video_processor.split_into_segments(
                frames, video_info
            )

            if progress_callback:
                progress_callback(30, "加载特征提取模型")

            models = self.model_manager.get_models(model_version, device)
            feature_extractor = models["feature_extractor"]

            if progress_callback:
                progress_callback(35, "提取时空特征")

            all_segment_features = []
            for i, seg in enumerate(segments):
                seg_frames = torch.from_numpy(seg["frames"]).unsqueeze(0)
                seg_frames = seg_frames.permute(0, 2, 1, 3, 4).contiguous()
                seg_features = feature_extractor.extract(
                    seg_frames, batch_size=1, device=device
                )
                all_segment_features.append((
                    seg_features.squeeze(0).numpy(),
                    seg["start_frame"],
                    seg["end_frame"],
                ))
                if progress_callback:
                    prog = 35 + int(5 * (i + 1) / len(segments))
                    progress_callback(prog, f"处理片段 {i+1}/{len(segments)}")

            if progress_callback:
                progress_callback(40, "合并片段特征")

            total_sample_frames = video_info["sample_frames_count"]
            feature_dim = all_segment_features[0][0].shape[1]
            features = self._merge_segment_features(
                all_segment_features, total_sample_frames, feature_dim,
                self.video_processor.overlap_frames
            )

            self.storage.save_features(video_id, model_version, features)

        if progress_callback:
            progress_callback(60, "时序行为分割")

        models = self.model_manager.get_models(model_version, device)
        segmentation_model = models["segmentation"]

        features_tensor = torch.from_numpy(features).float()
        pred_labels, pred_probs = segmentation_model.predict(features_tensor, device)

        if progress_callback:
            progress_callback(80, "后处理与片段生成")

        smoothed_labels, segments_dict = self.post_processor.process(
            pred_labels, pred_probs, video_info["target_fps"]
        )

        if progress_callback:
            progress_callback(90, "生成结果")

        timeline = self._build_timeline(
            smoothed_labels, video_info["target_fps"], video_info["duration"]
        )

        results = {
            "video_id": video_id,
            "model_version": model_version,
            "video_info": video_info,
            "frame_predictions": {
                "labels": smoothed_labels.tolist(),
                "probabilities": pred_probs.tolist(),
            },
            "segments": segments_dict,
            "timeline": timeline,
            "action_classes": self.model_manager.get_action_classes(),
        }

        self.storage.save_results(video_id, model_version, results)

        if progress_callback:
            progress_callback(100, "分析完成")

        return results

    def _merge_segment_features(
        self,
        segment_features: list,
        total_frames: int,
        feature_dim: int,
        overlap_frames: int,
    ) -> np.ndarray:
        if len(segment_features) == 1:
            return segment_features[0][0]

        merged = np.zeros((total_frames, feature_dim), dtype=np.float32)
        weight_sum = np.zeros(total_frames, dtype=np.float32)

        for seg_feat, start_frame, end_frame in segment_features:
            seg_len = len(seg_feat)
            weights = np.ones(seg_len, dtype=np.float32)
            if start_frame > 0:
                weights[:overlap_frames] = np.linspace(0.1, 1.0, min(overlap_frames, seg_len))
            if end_frame < total_frames:
                overlap_start = max(0, seg_len - overlap_frames)
                weights[overlap_start:] = np.linspace(1.0, 0.1, seg_len - overlap_start)

            for i in range(seg_len):
                global_idx = start_frame + i
                if global_idx < total_frames:
                    merged[global_idx] += seg_feat[i] * weights[i]
                    weight_sum[global_idx] += weights[i]

        valid = weight_sum > 0
        merged[valid] /= weight_sum[valid][:, None]

        return merged

    def _build_timeline(
        self,
        labels: np.ndarray,
        fps: float,
        duration: float,
    ) -> list:
        id_to_name = {c["id"]: c["name"] for c in self.model_manager.get_action_classes()}
        id_to_color = {c["id"]: c["color"] for c in self.model_manager.get_action_classes()}

        timeline = []
        for i, label in enumerate(labels):
            label_id = int(label)
            timeline.append({
                "frame": i,
                "time": i / fps if fps > 0 else 0,
                "action_id": label_id,
                "action_name": id_to_name.get(label_id, "未知"),
                "color": id_to_color.get(label_id, "#808080"),
            })

        return timeline
