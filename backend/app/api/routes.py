from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse, Response
from typing import Optional
from pathlib import Path
import httpx
import os
from io import BytesIO

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
    SegmentUpdateRequest,
    SplitSegmentRequest,
    MergeSegmentsRequest,
    UndoResponse,
    CompareCreateRequest,
    CompareStatusResponse,
    CompareSubTaskInfo,
    CompareResultsResponse,
    DisagreementInterval,
    HeatmapDataPoint,
    HeatmapResponse,
    CompareAppendRequest,
    IntervalAnnotationRequest,
    CompareExportResponse,
)
from app.services.segment_editor import SegmentEditor
from app.services.export_service import ExportService
from app.services.comparison_service import ComparisonService
from app.core.config import settings
from app.services.video_processor import VideoProcessor
from app.services.storage_manager import StorageManager
from app.services.tasks import analyze_video_task, launch_comparison_task
from app.services.evaluator import Evaluator
from app.models.model_manager import ModelManager
from app.core.celery_app import celery_app

router = APIRouter(prefix="", tags=["videos"])

video_processor = VideoProcessor()
storage = StorageManager()
model_manager = ModelManager()
segment_editor = SegmentEditor()
export_service = ExportService()
comparison_service = ComparisonService()


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


@router.put("/videos/{video_id}/segments/{seg_index}", response_model=AnalysisResultsResponse)
async def update_segment(
    video_id: str,
    seg_index: int,
    request: SegmentUpdateRequest,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        results = segment_editor.update_segment(
            video_id,
            model_version,
            seg_index,
            start_time=request.start_time,
            end_time=request.end_time,
            action_id=request.action_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AnalysisResultsResponse(
        video_id=results["video_id"],
        model_version=results["model_version"],
        video_info=results["video_info"],
        segments=results["segments"],
        action_classes=results["action_classes"],
    )


@router.post("/videos/{video_id}/segments/{seg_index}/split", response_model=AnalysisResultsResponse)
async def split_segment(
    video_id: str,
    seg_index: int,
    request: SplitSegmentRequest,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        results = segment_editor.split_segment(
            video_id,
            model_version,
            seg_index,
            request.split_time,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AnalysisResultsResponse(
        video_id=results["video_id"],
        model_version=results["model_version"],
        video_info=results["video_info"],
        segments=results["segments"],
        action_classes=results["action_classes"],
    )


@router.post("/videos/{video_id}/segments/merge", response_model=AnalysisResultsResponse)
async def merge_segments(
    video_id: str,
    request: MergeSegmentsRequest,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        results = segment_editor.merge_segments(
            video_id,
            model_version,
            request.segment_index_1,
            request.segment_index_2,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AnalysisResultsResponse(
        video_id=results["video_id"],
        model_version=results["model_version"],
        video_info=results["video_info"],
        segments=results["segments"],
        action_classes=results["action_classes"],
    )


@router.post("/videos/{video_id}/segments/undo", response_model=UndoResponse)
async def undo_segment_edit(
    video_id: str,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    results = segment_editor.undo(video_id, model_version)
    if results is None:
        return UndoResponse(
            success=False,
            message="No undo history available",
            can_undo=False,
            segments_count=0,
        )

    return UndoResponse(
        success=True,
        message="Undo successful",
        can_undo=segment_editor.can_undo(video_id),
        segments_count=len(results["segments"]),
    )


@router.get("/videos/{video_id}/segments/can-undo")
async def can_undo(video_id: str):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "can_undo": segment_editor.can_undo(video_id),
    }


@router.get("/videos/{video_id}/export/{format}")
async def export_segments(
    video_id: str,
    format: str,
    model_version: str = Query("latest"),
):
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    format_lower = format.lower()
    try:
        if format_lower == "json":
            filename, media_type, content = export_service.export_json(
                video_id, model_version
            )
        elif format_lower == "srt":
            filename, media_type, content = export_service.export_srt(
                video_id, model_version
            )
        elif format_lower == "csv":
            filename, media_type, content = export_service.export_csv(
                video_id, model_version
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Use json, srt, or csv.",
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.delete("/videos/{video_id}", response_model=DeleteResponse)
async def delete_video(video_id: str):
    segment_editor.clear_undo_stack(video_id)
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


@router.post("/compare/create")
async def create_comparison(request: CompareCreateRequest):
    try:
        task = comparison_service.create_comparison(
            request.video_id, request.model_versions
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    pending_models = [
        mv for mv, sub in task["sub_tasks"].items()
        if sub["status"] == "pending"
    ]
    if pending_models:
        launch_comparison_task.delay(
            task["compare_task_id"], request.video_id, request.model_versions
        )

    return {
        "compare_task_id": task["compare_task_id"],
        "video_id": task["video_id"],
        "model_versions": task["model_versions"],
        "status": task["overall_status"],
    }


@router.get("/compare/{task_id}/status", response_model=CompareStatusResponse)
async def get_comparison_status(task_id: str):
    status = comparison_service.get_comparison_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Comparison task not found")

    sub_tasks = []
    for mv, sub in status["sub_tasks"].items():
        sub_tasks.append(CompareSubTaskInfo(
            model_version=mv,
            task_id=sub.get("task_id"),
            status=sub["status"],
            progress=sub["progress"],
            error=sub.get("error"),
        ))

    return CompareStatusResponse(
        compare_task_id=status["compare_task_id"],
        video_id=status["video_id"],
        model_versions=status["model_versions"],
        overall_status=status["overall_status"],
        overall_progress=status["overall_progress"],
        sub_tasks=sub_tasks,
        failed_models=status.get("failed_models", []),
        error_details=status.get("error_details", {}),
    )


@router.get("/compare/{task_id}/results")
async def get_comparison_results(task_id: str):
    results = comparison_service.compute_comparison_results(task_id)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="Comparison results not available. Task may not be completed yet.",
        )

    disagreement_intervals = {}
    for pair_key, intervals in results["disagreement_intervals"].items():
        disagreement_intervals[pair_key] = [
            DisagreementInterval(**iv) for iv in intervals
        ]

    return CompareResultsResponse(
        compare_task_id=results["compare_task_id"],
        video_id=results["video_id"],
        model_versions=results["model_versions"],
        difference_matrix=results["difference_matrix"],
        agreement_rates=results["agreement_rates"],
        disagreement_intervals=disagreement_intervals,
        metrics_comparison=results.get("metrics_comparison"),
        has_ground_truth=results.get("has_ground_truth", False),
        total_frames=results["total_frames"],
        computed_at=results["computed_at"],
    )


@router.get("/compare/{task_id}/heatmap")
async def get_comparison_heatmap(
    task_id: str,
    action_class: Optional[int] = Query(None, description="Filter by action class ID"),
):
    heatmap = comparison_service.compute_heatmap_data_filtered(
        task_id, action_class_id=action_class
    )
    if heatmap is None:
        raise HTTPException(
            status_code=404,
            detail="Heatmap data not available. Task may not be completed yet.",
        )

    heatmap_data = {}
    for pair_key, points in heatmap["heatmap_data"].items():
        converted = []
        for p in points:
            if isinstance(p, dict):
                converted.append(HeatmapDataPoint(
                    frame_start=p["s"],
                    frame_end=p["e"],
                    time_start=p["ts"],
                    time_end=p["te"],
                    disagreement_rate=p["r"],
                    filtered_out=p.get("filtered_out"),
                ))
            else:
                converted.append(p)
        heatmap_data[pair_key] = converted

    return HeatmapResponse(
        compare_task_id=heatmap["compare_task_id"],
        video_id=heatmap["video_id"],
        model_pairs=heatmap["model_pairs"],
        is_aggregated=heatmap["is_aggregated"],
        window_size=heatmap.get("window_size"),
        heatmap_data=heatmap_data,
        action_class_filter=heatmap.get("action_class_filter"),
    )


@router.post("/compare/{task_id}/evaluate")
async def evaluate_comparison(task_id: str, gt_file: UploadFile = File(...)):
    status = comparison_service.get_comparison_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Comparison task not found")

    if status["overall_status"] not in ("completed", "partial"):
        raise HTTPException(
            status_code=400,
            detail="Comparison task is not completed yet",
        )

    gt_content = await gt_file.read()
    gt_text = gt_content.decode("utf-8")

    results = comparison_service.compute_comparison_results_with_gt(
        task_id, gt_text
    )
    if results is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to compute comparison results with Ground Truth",
        )

    disagreement_intervals = {}
    for pair_key, intervals in results["disagreement_intervals"].items():
        disagreement_intervals[pair_key] = [
            DisagreementInterval(**iv) for iv in intervals
        ]

    return CompareResultsResponse(
        compare_task_id=results["compare_task_id"],
        video_id=results["video_id"],
        model_versions=results["model_versions"],
        difference_matrix=results["difference_matrix"],
        agreement_rates=results["agreement_rates"],
        disagreement_intervals=disagreement_intervals,
        metrics_comparison=results.get("metrics_comparison"),
        has_ground_truth=True,
        total_frames=results["total_frames"],
        computed_at=results["computed_at"],
    )


@router.get("/compare/{task_id}/frame-labels")
async def get_frame_labels(
    task_id: str,
    start_frame: int = Query(...),
    end_frame: int = Query(...),
    model_versions: str = Query(...),
):
    status = comparison_service.get_comparison_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Comparison task not found")

    mv_list = [v.strip() for v in model_versions.split(",")]
    labels = comparison_service.get_frame_labels_for_interval(
        task_id, mv_list, start_frame, end_frame
    )
    if labels is None:
        raise HTTPException(status_code=404, detail="Frame labels not available")

    return {
        "compare_task_id": task_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_labels": labels,
    }


@router.post("/compare/{task_id}/append")
async def append_model_versions(task_id: str, request: CompareAppendRequest):
    try:
        task = comparison_service.append_model_versions(
            task_id, request.model_versions
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "compare_task_id": task["compare_task_id"],
        "video_id": task["video_id"],
        "model_versions": task["model_versions"],
        "status": task["overall_status"],
        "message": "Model versions appended successfully, comparison results recalculated",
    }


@router.patch("/compare/{task_id}/intervals/annotate")
async def annotate_interval(
    task_id: str,
    pair_key: str = Query(...),
    start_frame: int = Query(...),
    end_frame: int = Query(...),
    request: IntervalAnnotationRequest = None,
):
    status = comparison_service.get_comparison_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Comparison task not found")

    results = comparison_service.annotate_disagreement_interval(
        task_id,
        pair_key,
        start_frame,
        end_frame,
        note=request.note if request else None,
        confirmed=request.confirmed if request else None,
    )
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="Disagreement interval not found for the given pair_key and frame range",
        )

    disagreement_intervals = {}
    for pk, intervals in results["disagreement_intervals"].items():
        disagreement_intervals[pk] = [
            DisagreementInterval(**iv) for iv in intervals
        ]

    return CompareResultsResponse(
        compare_task_id=results["compare_task_id"],
        video_id=results["video_id"],
        model_versions=results["model_versions"],
        difference_matrix=results["difference_matrix"],
        agreement_rates=results["agreement_rates"],
        disagreement_intervals=disagreement_intervals,
        metrics_comparison=results.get("metrics_comparison"),
        has_ground_truth=results.get("has_ground_truth", False),
        total_frames=results["total_frames"],
        computed_at=results["computed_at"],
    )


@router.post("/compare/{task_id}/export", response_model=CompareExportResponse)
async def export_comparison_report(task_id: str):
    report = comparison_service.export_comparison_report(task_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Comparison report not available. Task may not be completed yet.",
        )

    return CompareExportResponse(**report)
