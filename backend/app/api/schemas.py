from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    file_size: int
    message: str


class VideoInfoResponse(BaseModel):
    video_id: str
    filename: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    target_fps: int
    sample_frames_count: int


class AnalyzeRequest(BaseModel):
    model_version: str = "latest"


class AnalyzeResponse(BaseModel):
    task_id: str
    video_id: str
    model_version: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ActionSegmentData(BaseModel):
    start_frame: int
    end_frame: int
    action_id: int
    action_name: str
    confidence: float
    start_time: float
    end_time: float


class AnalysisResultsResponse(BaseModel):
    video_id: str
    model_version: str
    video_info: Dict[str, Any]
    segments: List[Dict[str, Any]]
    action_classes: List[Dict[str, Any]]


class TimelineItem(BaseModel):
    frame: int
    time: float
    action_id: int
    action_name: str
    color: str


class TimelineResponse(BaseModel):
    video_id: str
    duration: float
    fps: float
    timeline: List[TimelineItem]


class EvaluateResponse(BaseModel):
    video_id: str
    metrics: Dict[str, float]


class DeleteResponse(BaseModel):
    video_id: str
    deleted: bool
    message: str


class ExpiredVideoItem(BaseModel):
    video_id: str
    last_access: str


class CleanupResponse(BaseModel):
    expired_videos: List[ExpiredVideoItem]
    message: str


class URLUploadRequest(BaseModel):
    url: str = Field(..., description="视频URL地址")
