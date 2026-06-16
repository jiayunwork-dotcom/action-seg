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
    _heatmap_cache: Dict[str, Dict] = {}
    CACHE_TTL_SECONDS = 1800
    HEATMAP_FRAME_THRESHOLD = 5000
    HEATMAP_WINDOW_SIZE = 50

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
        slim_results = {k: v for k, v in results.items() if k != "difference_matrix"}
        slim_results["difference_matrix_omitted"] = True
        with open(path, "w") as f:
            json.dump(slim_results, f, ensure_ascii=False, indent=2)

    def _load_results_from_disk(self, compare_task_id: str) -> Optional[Dict]:
        path = self._compare_dir / f"results_{compare_task_id}.json"
        if path.exists():
            with open(path, "r") as f:
                slim_results = json.load(f)
            task = self._compare_tasks.get(compare_task_id)
            if task and slim_results.get("difference_matrix_omitted"):
                diff_matrix = self._rebuild_diff_matrix(
                    task["video_id"],
                    slim_results["model_versions"],
                    slim_results["total_frames"],
                )
                slim_results["difference_matrix"] = diff_matrix
                del slim_results["difference_matrix_omitted"]
            return slim_results
        return None

    def _save_heatmap_to_disk(self, compare_task_id: str, heatmap: Dict):
        path = self._compare_dir / f"heatmap_{compare_task_id}.json"
        with open(path, "w") as f:
            json.dump(heatmap, f, ensure_ascii=False, indent=2)

    def _load_heatmap_from_disk(self, compare_task_id: str) -> Optional[Dict]:
        path = self._compare_dir / f"heatmap_{compare_task_id}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def create_comparison(
        self, video_id: str, model_versions: List[str]
    ) -> Dict:
        if not self.storage.video_exists(video_id):
            raise ValueError(f"Video {video_id} not found")

        self._refresh_tasks_for_video(video_id)

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
        all_have_results = True

        for mv in model_versions:
            has_result = self.storage.load_results(video_id, mv) is not None
            sub_tasks[mv] = {
                "model_version": mv,
                "task_id": None,
                "status": "completed" if has_result else "pending",
                "progress": 100 if has_result else 0,
                "error": None,
            }
            if not has_result:
                all_have_results = False

        task = {
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_versions": model_versions,
            "overall_status": "completed" if all_have_results else "pending",
            "overall_progress": 100 if all_have_results else 0,
            "sub_tasks": sub_tasks,
            "failed_models": [],
            "error_details": {},
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat() if all_have_results else None,
            "all_cached": all_have_results,
        }

        self._compare_tasks[compare_task_id] = task
        self._save_task_to_disk(compare_task_id)

        if all_have_results:
            self.compute_comparison_results(compare_task_id)

        return task

    def _refresh_tasks_for_video(self, video_id: str):
        for f in self._compare_dir.glob("task_*.json"):
            try:
                with open(f, "r") as fp:
                    task = json.load(fp)
                if task["video_id"] == video_id:
                    self._compare_tasks[task["compare_task_id"]] = task
            except Exception:
                pass

    def get_comparison_status(self, compare_task_id: str) -> Optional[Dict]:
        path = self._compare_dir / f"task_{compare_task_id}.json"
        if not path.exists():
            return self._compare_tasks.get(compare_task_id)

        try:
            with open(path, "r") as f:
                task = json.load(f)
            self._compare_tasks[compare_task_id] = task
            return task
        except Exception:
            return self._compare_tasks.get(compare_task_id)

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

            if any_success:
                self.compute_comparison_results(compare_task_id)

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
                self.compute_comparison_results(compare_task_id)
            else:
                task["overall_status"] = "failed"
            task["completed_at"] = datetime.utcnow().isoformat()

        self._save_task_to_disk(compare_task_id)

    def _recalculate_overall(self, task: Dict):
        subs = task["sub_tasks"]
        if not subs:
            return
        progresses = [max(0, s["progress"]) for s in subs.values()]
        task["overall_progress"] = int(sum(progresses) / len(subs))
        running = any(s["status"] == "running" for s in subs.values())
        pending = any(s["status"] == "pending" for s in subs.values())
        if running or pending:
            task["overall_status"] = "running"

    def compute_comparison_results(self, compare_task_id: str) -> Optional[Dict]:
        task = self.get_comparison_status(compare_task_id)
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
        video_info = None

        for mv in successful_models:
            results = self.storage.load_results(video_id, mv)
            if results is None:
                continue
            labels = np.array(results["frame_predictions"]["labels"], dtype=np.int64)
            model_labels[mv] = labels
            if video_info is None:
                video_info = results["video_info"]

        if len(model_labels) < 2 or video_info is None:
            return None

        min_len = min(len(v) for v in model_labels.values())
        for mv in model_labels:
            model_labels[mv] = model_labels[mv][:min_len]

        diff_matrix, agreement_rates = self._compute_frame_differences_fast(
            model_labels, successful_models, min_len
        )

        disagreement_intervals = self._compute_disagreement_intervals_fast(
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
            "fps": video_info["target_fps"],
            "computed_at": datetime.utcnow().isoformat(),
        }

        self._result_cache[compare_task_id] = {
            "results": results,
            "cached_at": time.time(),
        }
        self._save_results_to_disk(compare_task_id, results)

        self._precompute_heatmap(compare_task_id, results)

        return results

    def compute_comparison_results_with_gt(
        self, compare_task_id: str, gt_csv_content: str
    ) -> Optional[Dict]:
        results = self.compute_comparison_results(compare_task_id)
        if results is None:
            return None

        task = self.get_comparison_status(compare_task_id)
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

    def _compute_frame_differences_fast(
        self,
        model_labels: Dict[str, np.ndarray],
        model_versions: List[str],
        total_frames: int,
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        pairs = list(combinations(model_versions, 2))
        num_pairs = len(pairs)

        diff_matrix = [[1] * num_pairs for _ in range(total_frames)]
        agreement_rates = {}

        labels_list = [model_labels[mv] for mv in model_versions]

        for pair_idx, (i, j) in enumerate(combinations(range(len(model_versions)), 2)):
            agree_mask = labels_list[i] == labels_list[j]
            agree_count = int(np.sum(agree_mask))
            pair_key = f"{model_versions[i]}_vs_{model_versions[j]}"
            agreement_rates[pair_key] = round(agree_count / total_frames, 4) if total_frames > 0 else 0.0

            for f in range(total_frames):
                if not agree_mask[f]:
                    diff_matrix[f][pair_idx] = 0

        return diff_matrix, agreement_rates

    def _rebuild_diff_matrix(
        self, video_id: str, model_versions: List[str], total_frames: int
    ) -> List[List[int]]:
        model_labels: Dict[str, np.ndarray] = {}
        for mv in model_versions:
            results = self.storage.load_results(video_id, mv)
            if results is None:
                continue
            labels = np.array(results["frame_predictions"]["labels"], dtype=np.int64)
            model_labels[mv] = labels[:total_frames]

        if len(model_labels) < 2:
            return []

        diff_matrix, _ = self._compute_frame_differences_fast(
            model_labels, model_versions, total_frames
        )
        return diff_matrix

    def _compute_disagreement_intervals_fast(
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
            diff_mask = model_labels[mv_a] != model_labels[mv_b]

            intervals = []
            if not np.any(diff_mask):
                result[pair_key] = intervals
                continue

            diff_indices = np.where(diff_mask)[0]
            if len(diff_indices) == 0:
                result[pair_key] = intervals
                continue

            runs = []
            run_start = diff_indices[0]
            run_end = diff_indices[0]

            for f in diff_indices[1:]:
                if f == run_end + 1:
                    run_end = f
                else:
                    if (run_end - run_start + 1) >= disagreement_threshold:
                        runs.append((run_start, run_end))
                    run_start = f
                    run_end = f

            if (run_end - run_start + 1) >= disagreement_threshold:
                runs.append((run_start, run_end))

            for start_f, end_f in runs:
                intervals.append({
                    "start_frame": int(start_f),
                    "end_frame": int(end_f),
                    "start_time": round(start_f / fps, 4) if fps > 0 else 0.0,
                    "end_time": round(end_f / fps, 4) if fps > 0 else 0.0,
                    "length_frames": int(end_f - start_f + 1),
                })

            result[pair_key] = intervals

        return result

    def _precompute_heatmap(self, compare_task_id: str, results: Dict):
        heatmap = self._compute_heatmap_fast(results)
        if heatmap is not None:
            self._heatmap_cache[compare_task_id] = {
                "heatmap": heatmap,
                "cached_at": time.time(),
            }
            self._save_heatmap_to_disk(compare_task_id, heatmap)

    def compute_heatmap_data(self, compare_task_id: str) -> Optional[Dict]:
        if compare_task_id in self._heatmap_cache:
            cache_entry = self._heatmap_cache[compare_task_id]
            elapsed = time.time() - cache_entry["cached_at"]
            if elapsed < self.CACHE_TTL_SECONDS:
                return cache_entry["heatmap"]

        disk_heatmap = self._load_heatmap_from_disk(compare_task_id)
        if disk_heatmap is not None:
            self._heatmap_cache[compare_task_id] = {
                "heatmap": disk_heatmap,
                "cached_at": time.time(),
            }
            return disk_heatmap

        results = self.compute_comparison_results(compare_task_id)
        if results is None:
            return None

        heatmap = self._compute_heatmap_fast(results)
        if heatmap is not None:
            self._heatmap_cache[compare_task_id] = {
                "heatmap": heatmap,
                "cached_at": time.time(),
            }
            self._save_heatmap_to_disk(compare_task_id, heatmap)

        return heatmap

    def _compute_heatmap_fast(self, results: Dict) -> Optional[Dict]:
        model_versions = results["model_versions"]
        total_frames = results["total_frames"]
        fps = results.get("fps", 1.0)
        video_id = results["video_id"]
        compare_task_id = results["compare_task_id"]

        pairs = list(combinations(model_versions, 2))
        pair_keys = [f"{a}_vs_{b}" for a, b in pairs]

        is_aggregated = total_frames > self.HEATMAP_FRAME_THRESHOLD
        window_size = self.HEATMAP_WINDOW_SIZE if is_aggregated else 1

        model_labels = []
        for mv in model_versions:
            mv_results = self.storage.load_results(video_id, mv)
            if mv_results is None:
                return None
            labels = np.array(mv_results["frame_predictions"]["labels"][:total_frames], dtype=np.int64)
            model_labels.append(labels)

        heatmap_data: Dict[str, List[Dict]] = {}

        for pair_idx, (i, j) in enumerate(combinations(range(len(model_versions)), 2)):
            pair_key = pair_keys[pair_idx]
            diff_mask = model_labels[i] != model_labels[j]
            data_points = []

            if is_aggregated:
                num_windows = (total_frames + window_size - 1) // window_size
                window_starts = np.arange(num_windows) * window_size
                window_ends = np.minimum(window_starts + window_size, total_frames)

                for w in range(num_windows):
                    f_start = window_starts[w]
                    f_end = window_ends[w]
                    window_len = f_end - f_start
                    if window_len == 0:
                        continue
                    disagree_count = int(np.sum(diff_mask[f_start:f_end]))
                    disagree_rate = round(disagree_count / window_len, 4)
                    data_points.append({
                        "s": int(f_start),
                        "e": int(f_end - 1),
                        "ts": round(f_start / fps, 4),
                        "te": round((f_end - 1) / fps, 4),
                        "r": disagree_rate,
                    })
            else:
                for f in range(total_frames):
                    disagree_rate = 1.0 if diff_mask[f] else 0.0
                    data_points.append({
                        "s": f,
                        "e": f,
                        "ts": round(f / fps, 4),
                        "te": round(f / fps, 4),
                        "r": disagree_rate,
                    })

            heatmap_data[pair_key] = data_points

        return {
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_pairs": pair_keys,
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
            if cid in self._heatmap_cache:
                del self._heatmap_cache[cid]
            results_path = self._compare_dir / f"results_{cid}.json"
            if results_path.exists():
                results_path.unlink()
            heatmap_path = self._compare_dir / f"heatmap_{cid}.json"
            if heatmap_path.exists():
                heatmap_path.unlink()

    def get_frame_labels_for_interval(
        self,
        compare_task_id: str,
        model_versions: List[str],
        start_frame: int,
        end_frame: int,
    ) -> Optional[Dict[str, List[int]]]:
        task = self.get_comparison_status(compare_task_id)
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

    def append_model_versions(
        self, compare_task_id: str, new_model_versions: List[str]
    ) -> Optional[Dict]:
        task = self.get_comparison_status(compare_task_id)
        if task is None:
            raise ValueError(f"Comparison task {compare_task_id} not found")

        if task["overall_status"] not in ("completed", "partial"):
            raise ValueError("Can only append to completed or partial comparison tasks")

        video_id = task["video_id"]
        existing_versions = task["model_versions"]

        truly_new = [mv for mv in new_model_versions if mv not in existing_versions]
        if not truly_new:
            raise ValueError("All specified model versions already exist in this comparison task")

        total_after = len(existing_versions) + len(truly_new)
        if total_after > 6:
            raise ValueError(
                f"Total model versions would be {total_after}, maximum is 6"
            )

        for mv in truly_new:
            results = self.storage.load_results(video_id, mv)
            if results is None:
                raise ValueError(
                    f"No analysis results found for video {video_id} with model {mv}"
                )

        for mv in truly_new:
            task["model_versions"].append(mv)
            task["sub_tasks"][mv] = {
                "model_version": mv,
                "task_id": None,
                "status": "completed",
                "progress": 100,
                "error": None,
            }

        task["all_cached"] = all(
            s["status"] == "completed" for s in task["sub_tasks"].values()
        )
        self._save_task_to_disk(compare_task_id)

        self._incremental_compute(compare_task_id, existing_versions, truly_new)

        return task

    def _incremental_compute(
        self,
        compare_task_id: str,
        old_versions: List[str],
        new_versions: List[str],
    ):
        task = self.get_comparison_status(compare_task_id)
        if task is None:
            return

        video_id = task["video_id"]
        all_versions = task["model_versions"]

        successful_models = [
            mv for mv in all_versions
            if task["sub_tasks"][mv]["status"] == "completed"
        ]
        if len(successful_models) < 2:
            return

        model_labels: Dict[str, np.ndarray] = {}
        video_info = None
        for mv in successful_models:
            results = self.storage.load_results(video_id, mv)
            if results is None:
                continue
            labels = np.array(results["frame_predictions"]["labels"], dtype=np.int64)
            model_labels[mv] = labels
            if video_info is None:
                video_info = results["video_info"]

        if len(model_labels) < 2 or video_info is None:
            return

        min_len = min(len(v) for v in model_labels.values())
        for mv in model_labels:
            model_labels[mv] = model_labels[mv][:min_len]

        existing_results = self._get_cached_results(compare_task_id)
        if existing_results is None:
            existing_results = self._load_results_from_disk(compare_task_id)

        existing_agreement_rates = {}
        existing_disagreement_intervals = {}
        existing_diff_matrix = None

        if existing_results is not None:
            existing_agreement_rates = existing_results.get("agreement_rates", {})
            existing_disagreement_intervals = existing_results.get("disagreement_intervals", {})
            existing_diff_matrix = existing_results.get("difference_matrix")

        old_pairs = list(combinations(old_versions, 2))
        old_pair_keys = {f"{a}_vs_{b}" for a, b in old_pairs}

        new_pairs = []
        for nv in new_versions:
            for ov in old_versions:
                pair = tuple(sorted([nv, ov], key=lambda x: all_versions.index(x) if x in all_versions else 999))
                new_pairs.append(pair)
        for pair in combinations(new_versions, 2):
            new_pairs.append(pair)

        new_pair_keys = [f"{a}_vs_{b}" for a, b in new_pairs]

        diff_matrix, new_agreement_rates = self._compute_frame_differences_fast(
            model_labels, all_versions, min_len
        )

        for pk, rate in new_agreement_rates.items():
            if pk not in old_pair_keys:
                existing_agreement_rates[pk] = rate

        new_disagreement_intervals = self._compute_disagreement_intervals_fast(
            model_labels, new_versions + [ov for ov in old_versions if ov not in old_versions],
            min_len, video_info["target_fps"],
        )
        for pk, intervals in new_disagreement_intervals.items():
            if pk not in old_pair_keys:
                existing_disagreement_intervals[pk] = intervals

        results = {
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_versions": successful_models,
            "difference_matrix": diff_matrix,
            "agreement_rates": existing_agreement_rates,
            "disagreement_intervals": existing_disagreement_intervals,
            "metrics_comparison": None,
            "has_ground_truth": False,
            "total_frames": min_len,
            "fps": video_info["target_fps"],
            "computed_at": datetime.utcnow().isoformat(),
        }

        if existing_results and existing_results.get("has_ground_truth"):
            results["has_ground_truth"] = True
            results["metrics_comparison"] = existing_results.get("metrics_comparison")

        self._result_cache[compare_task_id] = {
            "results": results,
            "cached_at": time.time(),
        }
        self._save_results_to_disk(compare_task_id, results)

        self._precompute_heatmap(compare_task_id, results)

    def annotate_disagreement_interval(
        self,
        compare_task_id: str,
        pair_key: str,
        start_frame: int,
        end_frame: int,
        note: Optional[str] = None,
        confirmed: Optional[bool] = None,
    ) -> Optional[Dict]:
        results = self._get_cached_results(compare_task_id)
        if results is None:
            results = self._load_results_from_disk(compare_task_id)
        if results is None:
            return None

        disagreement_intervals = results.get("disagreement_intervals", {})
        if pair_key not in disagreement_intervals:
            return None

        updated_interval = None
        for iv in disagreement_intervals[pair_key]:
            if iv["start_frame"] == start_frame and iv["end_frame"] == end_frame:
                if note is not None:
                    iv["note"] = note
                if confirmed is not None:
                    iv["confirmed"] = confirmed
                updated_interval = iv
                break

        if updated_interval is None:
            return None

        self._result_cache[compare_task_id] = {
            "results": results,
            "cached_at": time.time(),
        }
        self._save_results_to_disk(compare_task_id, results)

        return results

    def export_comparison_report(self, compare_task_id: str) -> Optional[Dict]:
        task = self.get_comparison_status(compare_task_id)
        if task is None:
            return None

        results = self._get_cached_results(compare_task_id)
        if results is None:
            results = self._load_results_from_disk(compare_task_id)
        if results is None:
            return None

        video_id = task["video_id"]
        video_info = self.storage.load_video_info(video_id)
        if video_info is None:
            video_info = {}

        heatmap = self._load_heatmap_from_disk(compare_task_id)
        heatmap_summary = {}
        if heatmap and heatmap.get("heatmap_data"):
            for pair_key, points in heatmap["heatmap_data"].items():
                rates = [p["r"] for p in points]
                heatmap_summary[pair_key] = {
                    "max_disagreement_rate": round(max(rates), 4) if rates else 0.0,
                    "avg_disagreement_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
                    "num_windows": len(points),
                }

        disagreement_summary = []
        for pair_key, intervals in results.get("disagreement_intervals", {}).items():
            for iv in intervals:
                entry = {
                    "pair_key": pair_key,
                    "start_frame": iv["start_frame"],
                    "end_frame": iv["end_frame"],
                    "start_time": iv["start_time"],
                    "end_time": iv["end_time"],
                    "length_frames": iv["length_frames"],
                }
                if iv.get("note"):
                    entry["note"] = iv["note"]
                if iv.get("confirmed") is not None:
                    entry["confirmed"] = iv["confirmed"]
                disagreement_summary.append(entry)

        report = {
            "report_generated_at": datetime.utcnow().isoformat(),
            "compare_task_id": compare_task_id,
            "video_info": {
                "video_id": video_id,
                "filename": video_info.get("filename", ""),
                "duration": video_info.get("duration", 0),
                "fps": video_info.get("target_fps", 0),
                "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
                "total_frames": video_info.get("total_frames", 0),
            },
            "model_versions": results["model_versions"],
            "agreement_rates": results["agreement_rates"],
            "disagreement_intervals_summary": disagreement_summary,
            "heatmap_summary": heatmap_summary,
            "total_frames": results["total_frames"],
            "has_ground_truth": results.get("has_ground_truth", False),
        }

        if results.get("has_ground_truth") and results.get("metrics_comparison"):
            report["metrics_comparison"] = results["metrics_comparison"]

        return report

    def compute_heatmap_data_filtered(
        self, compare_task_id: str, action_class_id: Optional[int] = None
    ) -> Optional[Dict]:
        heatmap = self.compute_heatmap_data(compare_task_id)
        if heatmap is None or action_class_id is None:
            return heatmap

        task = self.get_comparison_status(compare_task_id)
        if task is None:
            return heatmap

        video_id = task["video_id"]
        total_frames = 0
        results_data = None
        for mv in heatmap.get("model_pairs", []):
            parts = mv.split("_vs_")
            mv_name = parts[0]
            mv_results = self.storage.load_results(video_id, mv_name)
            if mv_results is not None:
                results_data = mv_results
                total_frames = len(mv_results["frame_predictions"]["labels"])
                break

        if results_data is None or total_frames == 0:
            return heatmap

        labels = np.array(results_data["frame_predictions"]["labels"][:total_frames], dtype=np.int64)
        action_mask = labels == action_class_id

        filtered_heatmap_data: Dict[str, List[Dict]] = {}
        for pair_key, points in heatmap["heatmap_data"].items():
            filtered_points = []
            for p in points:
                f_start = p["s"] if isinstance(p, dict) and "s" in p else p.frame_start
                f_end = p["e"] if isinstance(p, dict) and "e" in p else p.frame_end

                if isinstance(p, dict):
                    f_start = p["s"]
                    f_end = p["e"]
                else:
                    f_start = p.frame_start
                    f_end = p.frame_end

                if f_end >= total_frames:
                    f_end_safe = total_frames - 1
                else:
                    f_end_safe = f_end
                if f_start >= total_frames:
                    continue

                window_mask = action_mask[f_start:f_end_safe + 1]
                overlap = np.any(window_mask)

                if isinstance(p, dict):
                    new_point = dict(p)
                else:
                    new_point = {
                        "s": p.frame_start,
                        "e": p.frame_end,
                        "ts": p.time_start,
                        "te": p.time_end,
                        "r": p.disagreement_rate,
                    }

                if not overlap:
                    new_point["filtered_out"] = True
                    new_point["r"] = 0.0

                filtered_points.append(new_point)

            filtered_heatmap_data[pair_key] = filtered_points

        return {
            "compare_task_id": heatmap["compare_task_id"],
            "video_id": heatmap["video_id"],
            "model_pairs": heatmap["model_pairs"],
            "is_aggregated": heatmap["is_aggregated"],
            "window_size": heatmap.get("window_size"),
            "heatmap_data": filtered_heatmap_data,
            "action_class_filter": action_class_id,
        }

    def list_comparisons_for_video(self, video_id: str) -> List[Dict]:
        self._refresh_tasks_for_video(video_id)
        return [
            task for task in self._compare_tasks.values()
            if task["video_id"] == video_id
        ]
