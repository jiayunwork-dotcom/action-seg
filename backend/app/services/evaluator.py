import csv
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from io import StringIO


@dataclass
class EvaluationMetrics:
    frame_accuracy: float
    edit_score: float
    f1_at_10: float
    f1_at_25: float
    f1_at_50: float


class Evaluator:
    def __init__(self, action_classes: List[Dict]):
        self.action_classes = action_classes
        self.name_to_id = {c["name"]: c["id"] for c in action_classes}
        self.id_to_name = {c["id"]: c["name"] for c in action_classes}

    def parse_gt_csv(self, csv_content: str) -> List[Dict]:
        gt_segments = []
        reader = csv.DictReader(StringIO(csv_content))

        for row in reader:
            action_label = row["action_label"].strip()
            action_id = self.name_to_id.get(action_label, 0)
            gt_segments.append({
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "action_id": action_id,
                "action_name": action_label,
            })

        return gt_segments

    def _segments_to_frame_labels(
        self,
        segments: List[Dict],
        total_frames: int,
    ) -> np.ndarray:
        labels = np.zeros(total_frames, dtype=np.int64)
        for seg in segments:
            start = seg["start_frame"]
            end = min(seg["end_frame"], total_frames - 1)
            labels[start:end + 1] = seg["action_id"]
        return labels

    def compute_frame_accuracy(
        self,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
    ) -> float:
        if len(pred_labels) != len(gt_labels):
            min_len = min(len(pred_labels), len(gt_labels))
            pred_labels = pred_labels[:min_len]
            gt_labels = gt_labels[:min_len]

        correct = np.sum(pred_labels == gt_labels)
        return float(correct / len(gt_labels)) if len(gt_labels) > 0 else 0.0

    def compute_edit_score(
        self,
        pred_segments: List[Dict],
        gt_segments: List[Dict],
    ) -> float:
        def segments_to_sequence(segs):
            if not segs:
                return []
            sorted_segs = sorted(segs, key=lambda s: s["start_frame"])
            return [s["action_id"] for s in sorted_segs]

        pred_seq = segments_to_sequence(pred_segments)
        gt_seq = segments_to_sequence(gt_segments)

        if not pred_seq and not gt_seq:
            return 1.0
        if not pred_seq or not gt_seq:
            return 0.0

        n, m = len(pred_seq), len(gt_seq)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if pred_seq[i - 1] == gt_seq[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        edit_distance = dp[n][m]
        max_len = max(n, m)
        return 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0

    def _temporal_iou(self, seg1: Dict, seg2: Dict) -> float:
        intersection_start = max(seg1["start_frame"], seg2["start_frame"])
        intersection_end = min(seg1["end_frame"], seg2["end_frame"])

        if intersection_end < intersection_start:
            return 0.0

        intersection = intersection_end - intersection_start + 1
        union = (seg1["end_frame"] - seg1["start_frame"] + 1) + \
                (seg2["end_frame"] - seg2["start_frame"] + 1) - intersection

        return intersection / union if union > 0 else 0.0

    def compute_f1_at_iou(
        self,
        pred_segments: List[Dict],
        gt_segments: List[Dict],
        iou_threshold: float,
    ) -> float:
        gt_matched = [False] * len(gt_segments)
        tp = 0

        for pred in pred_segments:
            best_iou = 0.0
            best_gt_idx = -1

            for i, gt in enumerate(gt_segments):
                if gt_matched[i]:
                    continue
                if pred["action_id"] != gt["action_id"]:
                    continue
                iou = self._temporal_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                gt_matched[best_gt_idx] = True
                tp += 1

        fp = len(pred_segments) - tp
        fn = len(gt_segments) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def evaluate(
        self,
        pred_labels: np.ndarray,
        pred_segments: List[Dict],
        gt_csv_content: str,
        total_frames: int,
    ) -> Dict:
        gt_segments = self.parse_gt_csv(gt_csv_content)
        gt_labels = self._segments_to_frame_labels(gt_segments, total_frames)

        frame_acc = self.compute_frame_accuracy(pred_labels, gt_labels)
        edit_score = self.compute_edit_score(pred_segments, gt_segments)
        f1_10 = self.compute_f1_at_iou(pred_segments, gt_segments, 0.10)
        f1_25 = self.compute_f1_at_iou(pred_segments, gt_segments, 0.25)
        f1_50 = self.compute_f1_at_iou(pred_segments, gt_segments, 0.50)

        metrics = EvaluationMetrics(
            frame_accuracy=round(frame_acc, 4),
            edit_score=round(edit_score, 4),
            f1_at_10=round(f1_10, 4),
            f1_at_25=round(f1_25, 4),
            f1_at_50=round(f1_50, 4),
        )

        return asdict(metrics)
