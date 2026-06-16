import json
import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from itertools import combinations
from pathlib import Path
from datetime import datetime

from app.core.config import settings
from app.services.storage_manager import StorageManager
from app.services.evaluator import Evaluator
from app.models.model_manager import ModelManager


class ComparisonService:
    _instance = None
    _compare_tasks: Dict[str, Dict] = {}
    _result_cache: Dict[str, Dict] = {}
    CACHE_TTL_SECONDS = 1800

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.storage = StorageManager()
        self.model_manager = ModelManager()
        self._compare_dir = settings.RESULTS_DIR / "comparisons"
        self._compare_dir.mkdir(parents=True, exist_ok=True)
        self._load_tasks_from_disk()

    def _load_tasks_from_disk(self):
        for f in self._compare_dir.glob("task_*.json"):
            try:
                with open(f, "r") as fp:
                    task = json.load(fp)
                self._compare_tasks[task["compare_task_id"]] = task
            except Exception:
                pass

    def _save_task_to_disk(self, compare_task_id: str):
        task = self._compare_tasks.get(compare_task_id)
        if task is None:
            return
        path = self._compare_dir / f"task_{compare_task_id}.json"
        with open(path, "w") as f:
            json.dump(task, f, ensure_ascii=False, indent=2)

    def _save_results_to_disk(self, compare_task_id: str, results: Dict):
        path = self._compare_dir / f"results_{compare_task_id}.json"
        with open(path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _load_results_from_disk(self, compare_task_id: str) -> Optional[Dict]:
        path = self._compare_dir / f"results_{compare_task_id}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def create_comparison(
        self, video_id: str, model_versions: List[str]
    ) -> Dict:
        if not self.storage.video_exists(video_id):
            raise ValueError(f"Video {video_id} not found")

        active_count = sum(
            1 for t in self._compare_tasks.values()
            if t["video_id"] == video_id
            and t["overall_status"] in ("pending", "running")
        )
        if active_count >= 2:
            raise ValueError(
                f"Video {video_id} already has {active_count} active comparison tasks. "
                "Maximum 2 concurrent comparison tasks per video."
            )

        for mv in model_versions:
            results = self.storage.load_results(video_id, mv)
            if results is None:
                raise ValueError(
                    f"No analysis results found for video {video_id} with model {mv}. "
                    "Please analyze the video with this model version first."
                )

        compare_task_id = str(uuid.uuid4())
        sub_tasks = {}
        for mv in model_versions:
            sub_tasks[mv] = {
                "model_version": mv,
                "task_id": None,
                "status": "pending",
                "progress": 0,
                "error": None,
            }

        task = {
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_versions": model_versions,
            "overall_status": "pending",
            "overall_progress": 0,
            "sub_tasks": sub_tasks,
            "failed_models": [],
            "error_details": {},
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }

        self._compare_tasks[compare_task_id] = task
        self._save_task_to_disk(compare_task_id)
        return task

    def get_comparison_status(self, compare_task_id: str) -> Optional[Dict]:
        task = self._compare_tasks.get(compare_task_id)
        if task is None:
            return None
        return task

    def update_sub_task_progress(
        self, compare_task_id: str, model_version: str, progress: int, message: str
    ):
        task = self._compare_tasks.get(compare_task_id)
        if task is None:
            return
        sub = task["sub_tasks"].get(model_version)
        if sub is None:
            return
        sub["progress"] = progress
        sub["status"] = "running"
        self._recalculate_overall(task)
        self._save_task_to_disk(compare_task_id)

    def complete_sub_task(self, compare_task_id: str, model_version: str):
        task = self._compare_tasks.get(compare_task_id)
        if task is None:
            return
        sub = task["sub_tasks"].get(model_version)
        if sub is None:
            return
        sub["status"] = "completed"
        sub["progress"] = 100
        self._recalculate_overall(task)

        all_done = all(
            s["status"] in ("completed", "failed")
            for s in task["sub_tasks"].values()
        )
        if all_done:
            any_failed = any(
                s["status"] == "failed" for s in task["sub_tasks"].values()
            )
            any_success = any(
                s["status"] == "completed" for s in task["sub_tasks"].values()
            )
            if any_failed and any_success:
                task["overall_status"] = "partial"
            elif all(s["status"] == "failed" for s in task["sub_tasks"].values()):
                task["overall_status"] = "failed"
            else:
                task["overall_status"] = "completed"
            task["completed_at"] = datetime.utcnow().isoformat()

        self._save_task_to_disk(compare_task_id)

    def fail_sub_task(
        self, compare_task_id: str, model_version: str, error: str
    ):
        task = self._compare_tasks.get(compare_task_id)
        if task is None:
            return
        sub = task["sub_tasks"].get(model_version)
        if sub is None:
            return
        sub["status"] = "failed"
        sub["progress"] = -1
        sub["error"] = error
        if model_version not in task["failed_models"]:
            task["failed_models"].append(model_version)
        task["error_details"][model_version] = error
        self._recalculate_overall(task)

        all_done = all(
            s["status"] in ("completed", "failed")
            for s in task["sub_tasks"].values()
        )
        if all_done:
            any_success = any(
                s["status"] == "completed" for s in task["sub_tasks"].values()
            )
            if any_success:
                task["overall_status"] = "partial"
            else:
                task["overall_status"] = "failed"
            task["completed_at"] = datetime.utcnow().isoformat()

        self._save_task_to_disk(compare_task_id)

    def _recalculate_overall(self, task: Dict):
        subs = task["sub_tasks"]
        if not subs:
            return
        total = sum(s["progress"] for s in subs.values())
        task["overall_progress"] = int(total / len(subs))
        running = any(s["status"] == "running" for s in subs.values())
        if running or any(s["status"] == "pending" for s in subs.values()):
            task["overall_status"] = "running"

    def compute_comparison_results(self, compare_task_id: str) -> Optional[Dict]:
        task = self._compare_tasks.get(compare_task_id)
        if task is None:
            return None

        cached = self._get_cached_results(compare_task_id)
        if cached is not None:
            return cached

        successful_models = [
            mv for mv in task["model_versions"]
            if task["sub_tasks"][mv]["status"] == "completed"
        ]
        if len(successful_models) < 2:
            return None

        video_id = task["video_id"]
        model_labels: Dict[str, np.ndarray] = {}
        model_segments: Dict[str, list] = {}
        video_info = None
        total_frames = None

        for mv in successful_models:
            results = self.storage.load_results(video_id, mv)
            if results is None:
                continue
            labels = np.array(results["frame_predictions"]["labels"])
            model_labels[mv] = labels
            model_segments[mv] = results["segments"]
            if video_info is None:
                video_info = results["video_info"]
                total_frames = results["video_info"]["sample_frames_count"]

        if len(model_labels) < 2 or video_info is None:
            return None

        min_len = min(len(v) for v in model_labels.values())
        for mv in model_labels:
            model_labels[mv] = model_labels[mv][:min_len]

        diff_matrix, agreement_rates = self._compute_frame_differences(
            model_labels, successful_models, min_len
        )

        disagreement_intervals = self._compute_disagreement_intervals(
            model_labels, successful_models, min_len, video_info["target_fps"]
        )

        results = {
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_versions": successful_models,
            "difference_matrix": diff_matrix,
            "agreement_rates": agreement_rates,
            "disagreement_intervals": disagreement_intervals,
            "metrics_comparison": None,
            "has_ground_truth": False,
            "total_frames": min_len,
            "computed_at": datetime.utcnow().isoformat(),
        }

        self._result_cache[compare_task_id] = {
            "results": results,
            "cached_at": time.time(),
        }
        self._save_results_to_disk(compare_task_id, results)
        return results

    def compute_comparison_results_with_gt(
        self, compare_task_id: str, gt_csv_content: str
    ) -> Optional[Dict]:
        results = self.compute_comparison_results(compare_task_id)
        if results is None:
            return None

        task = self._compare_tasks[compare_task_id]
        video_id = task["video_id"]
        total_frames = results["total_frames"]

        evaluator = Evaluator(self.model_manager.get_action_classes())
        metrics_comparison: Dict[str, Dict[str, float]] = {}

        for mv in results["model_versions"]:
            mv_results = self.storage.load_results(video_id, mv)
            if mv_results is None:
                continue
            pred_labels = np.array(mv_results["frame_predictions"]["labels"][:total_frames])
            pred_segments = mv_results["segments"]
            metrics = evaluator.evaluate(pred_labels, pred_segments, gt_csv_content, total_frames)
            metrics_comparison[mv] = metrics

        results["metrics_comparison"] = metrics_comparison
        results["has_ground_truth"] = True

        self._result_cache[compare_task_id] = {
            "results": results,
            "cached_at": time.time(),
        }
        self._save_results_to_disk(compare_task_id, results)
        return results

    def _compute_frame_differences(
        self,
        model_labels: Dict[str, np.ndarray],
        model_versions: List[str],
        total_frames: int,
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        pairs = list(combinations(model_versions, 2))
        diff_matrix = []
        agreement_rates = {}

        for frame_idx in range(total_frames):
            row = []
            for mv_a, mv_b in pairs:
                agree = 1 if model_labels[mv_a][frame_idx] == model_labels[mv_b][frame_idx] else 0
                row.append(agree)
            diff_matrix.append(row)

        for pair_idx, (mv_a, mv_b) in enumerate(pairs):
            pair_key = f"{mv_a}_vs_{mv_b}"
            agree_count = sum(diff_matrix[f][pair_idx] for f in range(total_frames))
            agreement_rates[pair_key] = round(agree_count / total_frames, 4) if total_frames > 0 else 0.0

        return diff_matrix, agreement_rates

    def _compute_disagreement_intervals(
        self,
        model_labels: Dict[str, np.ndarray],
        model_versions: List[str],
        total_frames: int,
        fps: float,
    ) -> Dict[str, List[Dict]]:
        pairs = list(combinations(model_versions, 2))
        result = {}
        disagreement_threshold = 10

        for mv_a, mv_b in pairs:
            pair_key = f"{mv_a}_vs_{mv_b}"
            disagree_frames = []
            for f in range(total_frames):
                if model_labels[mv_a][f] != model_labels[mv_b][f]:
                    disagree_frames.append(f)

            intervals = []
            if not disagree_frames:
                result[pair_key] = intervals
                continue

            run_start = disagree_frames[0]
            run_end = disagree_frames[0]

            for f in disagree_frames[1:]:
                if f == run_end + 1:
                    run_end = f
                else:
                    if (run_end - run_start + 1) >= disagreement_threshold:
                        intervals.append({
                            "start_frame": int(run_start),
                            "end_frame": int(run_end),
                            "start_time": round(run_start / fps, 4) if fps > 0 else 0.0,
                            "end_time": round(run_end / fps, 4) if fps > 0 else 0.0,
                            "length_frames": int(run_end - run_start + 1),
                        })
                    run_start = f
                    run_end = f

            if (run_end - run_start + 1) >= disagreement_threshold:
                intervals.append({
                    "start_frame": int(run_start),
                    "end_frame": int(run_end),
                    "start_time": round(run_start / fps, 4) if fps > 0 else 0.0,
                    "end_time": round(run_end / fps, 4) if fps > 0 else 0.0,
                    "length_frames": int(run_end - run_start + 1),
                })

            result[pair_key] = intervals

        return result

    def compute_heatmap_data(self, compare_task_id: str) -> Optional[Dict]:
        results = self.compute_comparison_results(compare_task_id)
        if results is None:
            return None

        task = self._compare_tasks[compare_task_id]
        video_id = task["video_id"]
        video_info = None
        for mv in results["model_versions"]:
            mv_results = self.storage.load_results(video_id, mv)
            if mv_results is not None:
                video_info = mv_results["video_info"]
                break

        if video_info is None:
            return None

        fps = video_info["target_fps"]
        total_frames = results["total_frames"]
        diff_matrix = results["difference_matrix"]
        model_versions = results["model_versions"]
        pairs = list(combinations(model_versions, 2))

        is_aggregated = total_frames > 5000
        window_size = 50 if is_aggregated else 1

        heatmap_data: Dict[str, List[Dict]] = {}

        for pair_idx, (mv_a, mv_b) in enumerate(pairs):
            pair_key = f"{mv_a}_vs_{mv_b}"
            data_points = []

            if is_aggregated:
                num_windows = (total_frames + window_size - 1) // window_size
                for w in range(num_windows):
                    f_start = w * window_size
                    f_end = min((w + 1) * window_size, total_frames)
                    disagree_count = 0
                    for f in range(f_start, f_end):
                        if diff_matrix[f][pair_idx] == 0:
                            disagree_count += 1
                    disagree_rate = disagree_count / (f_end - f_start) if (f_end - f_start) > 0 else 0.0
                    data_points.append({
                        "frame_start": f_start,
                        "frame_end": f_end - 1,
                        "time_start": round(f_start / fps, 4) if fps > 0 else 0.0,
                        "time_end": round((f_end - 1) / fps, 4) if fps > 0 else 0.0,
                        "disagreement_rate": round(disagree_rate, 4),
                    })
            else:
                for f in range(total_frames):
                    disagree_rate = 0.0 if diff_matrix[f][pair_idx] == 1 else 1.0
                    data_points.append({
                        "frame_start": f,
                        "frame_end": f,
                        "time_start": round(f / fps, 4) if fps > 0 else 0.0,
                        "time_end": round(f / fps, 4) if fps > 0 else 0.0,
                        "disagreement_rate": disagree_rate,
                    })

            heatmap_data[pair_key] = data_points

        return {
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_pairs": [f"{a}_vs_{b}" for a, b in pairs],
            "is_aggregated": is_aggregated,
            "window_size": window_size if is_aggregated else None,
            "heatmap_data": heatmap_data,
        }

    def _get_cached_results(self, compare_task_id: str) -> Optional[Dict]:
        cache_entry = self._result_cache.get(compare_task_id)
        if cache_entry is None:
            disk_results = self._load_results_from_disk(compare_task_id)
            if disk_results is not None:
                self._result_cache[compare_task_id] = {
                    "results": disk_results,
                    "cached_at": time.time(),
                }
                cache_entry = self._result_cache[compare_task_id]

        if cache_entry is not None:
            elapsed = time.time() - cache_entry["cached_at"]
            if elapsed < self.CACHE_TTL_SECONDS:
                return cache_entry["results"]
            else:
                del self._result_cache[compare_task_id]

        return None

    def invalidate_cache(self, video_id: str, model_version: str):
        to_invalidate = []
        for cid, task in self._compare_tasks.items():
            if task["video_id"] == video_id and model_version in task["model_versions"]:
                to_invalidate.append(cid)

        for cid in to_invalidate:
            if cid in self._result_cache:
                del self._result_cache[cid]
            results_path = self._compare_dir / f"results_{cid}.json"
            if results_path.exists():
                results_path.unlink()

    def get_frame_labels_for_interval(
        self,
        compare_task_id: str,
        model_versions: List[str],
        start_frame: int,
        end_frame: int,
    ) -> Optional[Dict[str, List[int]]]:
        task = self._compare_tasks.get(compare_task_id)
        if task is None:
            return None
        video_id = task["video_id"]
        result = {}
        for mv in model_versions:
            mv_results = self.storage.load_results(video_id, mv)
            if mv_results is None:
                continue
            labels = mv_results["frame_predictions"]["labels"]
            result[mv] = labels[start_frame:end_frame + 1]
        return result if result else None

    def list_comparisons_for_video(self, video_id: str) -> List[Dict]:
        return [
            task for task in self._compare_tasks.values()
            if task["video_id"] == video_id
        ]
