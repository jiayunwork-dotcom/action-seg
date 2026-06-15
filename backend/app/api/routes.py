from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from pathlib import Path
import httpx
import os

from app.api.schemas import (
    VideoUploadResponse,
    VideoInfoResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    TaskStatusResponse,
    AnalysisResultsResponse,
    TimelineResponse,
    TimelineItem,
    EvaluateResponse,
    DeleteResponse,
    CleanupResponse,
    URLUploadRequest,
)
from app.core.config import settings
from app.services.video_processor import VideoProcessor
from app.services.storage_manager import StorageManager
from app.services.tasks import analyze_video_task
from app.services.evaluator import Evaluator
from app.models.model_manager import ModelManager
from app.core.celery_app import celery_app

router = APIRouter(prefix="", tags=["videos"])

video_processor = VideoProcessor()
storage = StorageManager()
model_manager = ModelManager()


@router.post("/videos/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if file.size and file.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()

    temp_path = Path(settings.UPLOAD_DIR) / f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    video_id = video_processor.generate_video_id(str(temp_path))

    if storage.video_exists(video_id):
        temp_path.unlink(missing_ok=True)
        video_info = storage.load_video_info(video_id)
        return VideoUploadResponse(
            video_id=video_id,
            filename=video_info.get("filename", file.filename),
            file_size=len(content),
            message="Video already exists",
        )

    saved_path = storage.save_uploaded_file(video_id, file.filename, content)
    temp_path.unlink(missing_ok=True)

    info = video_processor.get_video_info(saved_path, video_id)
    storage.save_video_info(video_id, info.__dict__)

    return VideoUploadResponse(
        video_id=video_id,
        filename=file.filename,
        file_size=len(content),
        message="Upload successful",
    )


@router.post("/videos/upload-url", response_model=VideoUploadResponse)
async def upload_video_from_url(request: URLUploadRequest):
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.get(request.url)
            resp.raise_for_status()
            content = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    filename = Path(request.url).name or "video.mp4"
    ext = Path(filename).suffix.lower().lstrip(".")
    if not ext:
        ext = "mp4"
        filename = f"{filename}.mp4"
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    temp_path = Path(settings.UPLOAD_DIR) / f"temp_url_{filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    video_id = video_processor.generate_video_id(str(temp_path))

    if storage.video_exists(video_id):
        temp_path.unlink(missing_ok=True)
        video_info = storage.load_video_info(video_id)
        return VideoUploadResponse(
            video_id=video_id,
            filename=video_info.get("filename", filename),
            file_size=len(content),
            message="Video already exists",
        )

    saved_path = storage.save_uploaded_file(video_id, filename, content)
    temp_path.unlink(missing_ok=True)

    info = video_processor.get_video_info(saved_path, video_id)
    storage.save_video_info(video_id, info.__dict__)

    return VideoUploadResponse(
        video_id=video_id,
        filename=filename,
        file_size=len(content),
        message="Download and save successful",
    )


@router.get("/videos/{video_id}/info", response_model=VideoInfoResponse)
async def get_video_info(video_id: str):
    info = storage.load_video_info(video_id)
    if not info:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoInfoResponse(**info)


@router.post("/videos/{video_id}/analyze", response_model=AnalyzeResponse)
async def analyze_video(video_id: str, request: Optional[AnalyzeRequest] = None):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    model_version = request.model_version if request else "latest"
    available_versions = model_manager.get_available_versions()
    if model_version not in available_versions:
        model_version = "latest"

    task = analyze_video_task.delay(video_id, model_version)

    return AnalyzeResponse(
        task_id=task.id,
        video_id=video_id,
        model_version=model_version,
        status="PENDING",
    )


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)

    status = task.state
    progress = 0
    message = "Waiting"
    result = None
    error = None

    if status == "PROGRESS":
        meta = task.info
        progress = meta.get("current", 0)
        message = meta.get("message", "Processing")
    elif status == "SUCCESS":
        progress = 100
        message = "Completed"
        result = task.result
        if isinstance(result, dict) and result.get("status") == "failed":
            status = "FAILURE"
            error = result.get("error")
    elif status == "FAILURE":
        progress = -1
        message = "Failed"
        error = str(task.info)

    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        progress=progress,
        message=message,
        result=result,
        error=error,
    )


@router.get("/videos/{video_id}/results", response_model=AnalysisResultsResponse)
async def get_video_results(
    video_id: str,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    results = storage.load_results(video_id, model_version)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Results not found for video {video_id} with model {model_version}",
        )

    return AnalysisResultsResponse(
        video_id=results["video_id"],
        model_version=results["model_version"],
        video_info=results["video_info"],
        segments=results["segments"],
        action_classes=results["action_classes"],
    )


@router.get("/videos/{video_id}/results/timeline", response_model=TimelineResponse)
async def get_video_timeline(
    video_id: str,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    results = storage.load_results(video_id, model_version)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Results not found for video {video_id} with model {model_version}",
        )

    timeline_items = [TimelineItem(**t) for t in results["timeline"]]

    return TimelineResponse(
        video_id=video_id,
        duration=results["video_info"]["duration"],
        fps=results["video_info"]["target_fps"],
        timeline=timeline_items,
    )


@router.post("/videos/{video_id}/evaluate", response_model=EvaluateResponse)
async def evaluate_video(
    video_id: str,
    gt_file: UploadFile = File(...),
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    results = storage.load_results(video_id, model_version)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Results not found. Please analyze the video first.",
        )

    gt_content = await gt_file.read()
    gt_text = gt_content.decode("utf-8")

    evaluator = Evaluator(model_manager.get_action_classes())
    total_frames = results["video_info"]["sample_frames_count"]

    import numpy as np
    pred_labels = np.array(results["frame_predictions"]["labels"])
    pred_segments = results["segments"]

    metrics = evaluator.evaluate(pred_labels, pred_segments, gt_text, total_frames)

    return EvaluateResponse(
        video_id=video_id,
        metrics=metrics,
    )


@router.get("/videos/{video_id}/features")
async def download_features(
    video_id: str,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    if not storage.features_exist(video_id, model_version):
        raise HTTPException(
            status_code=404,
            detail=f"Features not found for video {video_id}",
        )

    cache_dir = settings.CACHE_DIR / video_id
    feat_path = cache_dir / f"features_{model_version}.npy"

    return FileResponse(
        path=str(feat_path),
        filename=f"{video_id}_features_{model_version}.npy",
        media_type="application/octet-stream",
    )


@router.delete("/videos/{video_id}", response_model=DeleteResponse)
async def delete_video(video_id: str):
    deleted = storage.delete_video(video_id)
    return DeleteResponse(
        video_id=video_id,
        deleted=deleted,
        message="Video deleted successfully" if deleted else "Video not found",
    )


@router.get("/storage/expired", response_model=CleanupResponse)
async def get_expired_videos():
    expired = storage.list_expired_videos()
    return CleanupResponse(
        expired_videos=expired,
        message=f"Found {len(expired)} expired videos",
    )


@router.post("/storage/cleanup", response_model=CleanupResponse)
async def cleanup_expired_videos():
    expired = storage.list_expired_videos()
    deleted_count = 0
    for item in expired:
        if storage.delete_video(item["video_id"]):
            deleted_count += 1

    return CleanupResponse(
        expired_videos=expired,
        message=f"Cleaned up {deleted_count} expired videos",
    )


@router.get("/models/versions")
async def get_model_versions():
    return {
        "versions": model_manager.get_available_versions(),
        "default": "latest",
        "action_classes": model_manager.get_action_classes(),
    }
