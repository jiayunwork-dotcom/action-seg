from app.core.celery_app import celery_app
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.storage_manager import StorageManager

_storage = StorageManager()


@celery_app.task(bind=True, name="analyze_video")
def analyze_video_task(self, video_id: str, model_version: str = "latest"):
    pipeline = AnalysisPipeline()

    def progress_callback(percent: int, message: str):
        self.update_state(
            state="PROGRESS",
            meta={
                "current": percent,
                "total": 100,
                "message": message,
                "video_id": video_id,
                "model_version": model_version,
            },
        )
        _storage.save_progress(
            video_id, self.request.id, model_version,
            {"percent": percent, "message": message, "task_id": self.request.id},
        )

    try:
        progress_callback(1, "任务开始")
        results = pipeline.run(video_id, model_version, progress_callback)
        return {
            "status": "success",
            "video_id": video_id,
            "model_version": model_version,
            "segments_count": len(results["segments"]),
        }
    except Exception as e:
        progress_callback(-1, f"错误: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "video_id": video_id,
        }
