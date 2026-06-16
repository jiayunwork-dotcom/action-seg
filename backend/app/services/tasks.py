from app.core.celery_app import celery_app
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.storage_manager import StorageManager
from app.services.comparison_service import ComparisonService

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


@celery_app.task(bind=True, name="compare_sub_task")
def compare_sub_task(self, compare_task_id: str, video_id: str, model_version: str):
    comparison_svc = ComparisonService()
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
        comparison_svc.update_sub_task_progress(
            compare_task_id, model_version, percent, message
        )

    try:
        progress_callback(1, "对比子任务开始")
        results = pipeline.run(video_id, model_version, progress_callback)
        comparison_svc.complete_sub_task(compare_task_id, model_version)
        return {
            "status": "success",
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_version": model_version,
        }
    except Exception as e:
        comparison_svc.fail_sub_task(compare_task_id, model_version, str(e))
        return {
            "status": "failed",
            "error": str(e),
            "compare_task_id": compare_task_id,
            "video_id": video_id,
            "model_version": model_version,
        }


@celery_app.task(name="launch_comparison")
def launch_comparison_task(compare_task_id: str, video_id: str, model_versions: list):
    comparison_svc = ComparisonService()

    for mv in model_versions:
        sub_task = compare_sub_task.delay(compare_task_id, video_id, mv)
        comparison_svc._compare_tasks[compare_task_id]["sub_tasks"][mv]["task_id"] = sub_task.id
        comparison_svc._compare_tasks[compare_task_id]["sub_tasks"][mv]["status"] = "pending"
        comparison_svc._compare_tasks[compare_task_id]["overall_status"] = "running"

    comparison_svc._save_task_to_disk(compare_task_id)
