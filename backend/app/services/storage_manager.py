import json
import shutil
import time
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from app.core.config import settings


class StorageManager:
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.cache_dir = settings.CACHE_DIR
        self.results_dir = settings.RESULTS_DIR

    def _get_video_dir(self, video_id: str) -> Path:
        video_dir = self.upload_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def _get_cache_dir(self, video_id: str) -> Path:
        cache_dir = self.cache_dir / video_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_results_dir(self, video_id: str) -> Path:
        results_dir = self.results_dir / video_id
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def save_uploaded_file(self, video_id: str, filename: str, content: bytes) -> str:
        video_dir = self._get_video_dir(video_id)
        ext = Path(filename).suffix.lower()
        file_path = video_dir / f"video{ext}"
        with open(file_path, "wb") as f:
            f.write(content)
        return str(file_path)

    def save_video_info(self, video_id: str, info: Dict) -> None:
        video_dir = self._get_video_dir(video_id)
        with open(video_dir / "info.json", "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self._update_access_time(video_id)

    def load_video_info(self, video_id: str) -> Optional[Dict]:
        video_dir = self._get_video_dir(video_id)
        info_path = video_dir / "info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
            self._update_access_time(video_id)
            return info
        return None

    def get_video_path(self, video_id: str) -> Optional[str]:
        video_dir = self._get_video_dir(video_id)
        for ext in [".mp4", ".avi", ".mov"]:
            vp = video_dir / f"video{ext}"
            if vp.exists():
                return str(vp)
        return None

    def save_features(self, video_id: str, model_version: str, features: np.ndarray) -> None:
        cache_dir = self._get_cache_dir(video_id)
        np.save(cache_dir / f"features_{model_version}.npy", features)
        self._update_access_time(video_id)

    def load_features(self, video_id: str, model_version: str) -> Optional[np.ndarray]:
        cache_dir = self._get_cache_dir(video_id)
        feat_path = cache_dir / f"features_{model_version}.npy"
        if feat_path.exists():
            self._update_access_time(video_id)
            return np.load(feat_path)
        return None

    def features_exist(self, video_id: str, model_version: str) -> bool:
        cache_dir = self._get_cache_dir(video_id)
        return (cache_dir / f"features_{model_version}.npy").exists()

    def save_results(self, video_id: str, model_version: str, results: Dict) -> None:
        results_dir = self._get_results_dir(video_id)
        with open(results_dir / f"results_{model_version}.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self._update_access_time(video_id)

    def load_results(self, video_id: str, model_version: str) -> Optional[Dict]:
        results_dir = self._get_results_dir(video_id)
        res_path = results_dir / f"results_{model_version}.json"
        if res_path.exists():
            with open(res_path, "r") as f:
                self._update_access_time(video_id)
                return json.load(f)
        return None

    def save_progress(self, video_id: str, task_id: str, model_version: str, progress: Dict) -> None:
        results_dir = self._get_results_dir(video_id)
        with open(results_dir / f"progress_{task_id}_{model_version}.json", "w") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def load_progress(self, video_id: str, task_id: str, model_version: str) -> Optional[Dict]:
        results_dir = self._get_results_dir(video_id)
        prog_path = results_dir / f"progress_{task_id}_{model_version}.json"
        if prog_path.exists():
            with open(prog_path, "r") as f:
                return json.load(f)
        return None

    def save_checkpoint(self, video_id: str, task_id: str, model_version: str, checkpoint: Dict) -> None:
        results_dir = self._get_results_dir(video_id)
        with open(results_dir / f"checkpoint_{task_id}_{model_version}.json", "w") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, video_id: str, task_id: str, model_version: str) -> Optional[Dict]:
        results_dir = self._get_results_dir(video_id)
        ckpt_path = results_dir / f"checkpoint_{task_id}_{model_version}.json"
        if ckpt_path.exists():
            with open(ckpt_path, "r") as f:
                return json.load(f)
        return None

    def delete_video(self, video_id: str) -> bool:
        deleted = False
        for base_dir in [self.upload_dir, self.cache_dir, self.results_dir]:
            dir_path = base_dir / video_id
            if dir_path.exists():
                shutil.rmtree(dir_path)
                deleted = True
        return deleted

    def video_exists(self, video_id: str) -> bool:
        video_dir = self._get_video_dir(video_id)
        return any(video_dir.glob("video.*"))

    def list_expired_videos(self) -> list:
        expired = []
        current_time = time.time()
        expiry_seconds = settings.DATA_EXPIRY_DAYS * 24 * 3600

        for video_dir in self.upload_dir.iterdir():
            if not video_dir.is_dir():
                continue
            access_path = video_dir / "info.json"
            if access_path.exists():
                mtime = access_path.stat().st_mtime
                if current_time - mtime > expiry_seconds:
                    expired.append({
                        "video_id": video_dir.name,
                        "last_access": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                    })
        return expired

    def _update_access_time(self, video_id: str) -> None:
        video_dir = self._get_video_dir(video_id)
        info_path = video_dir / "info.json"
        if info_path.exists():
            current_time = time.time()
            import os
            os.utime(info_path, (current_time, current_time))
