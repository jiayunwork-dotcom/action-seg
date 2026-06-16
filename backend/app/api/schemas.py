from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    file_size: int
    message: str


class VideoInfoResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


class SegmentUpdateRequest(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    action_id: Optional[int] = None


class SplitSegmentRequest(BaseModel):
    split_time: float = Field(..., description="拆分时间点（秒）")


class MergeSegmentsRequest(BaseModel):
    segment_index_1: int = Field(..., description="第一个片段索引")
    segment_index_2: int = Field(..., description="第二个片段索引")


class UndoResponse(BaseModel):
    success: bool
    message: str
    can_undo: bool
    segments_count: int


class CompareCreateRequest(BaseModel):
    video_id: str
    model_versions: List[str] = Field(..., min_length=2, max_length=4)


class CompareSubTaskInfo(BaseModel):
    model_version: str
    task_id: str
    status: str
    progress: int
    error: Optional[str] = None


class CompareStatusResponse(BaseModel):
    compare_task_id: str
    video_id: str
    model_versions: List[str]
    overall_status: str
    overall_progress: int
    sub_tasks: List[CompareSubTaskInfo]
    failed_models: List[str] = []
    error_details: Dict[str, str] = {}


class DisagreementInterval(BaseModel):
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    length_frames: int


class CompareResultsResponse(BaseModel):
    compare_task_id: str
    video_id: str
    model_versions: List[str]
    difference_matrix: List[List[int]]
    agreement_rates: Dict[str, float]
    disagreement_intervals: Dict[str, List[DisagreementInterval]]
    metrics_comparison: Optional[Dict[str, Dict[str, float]]] = None
    has_ground_truth: bool = False
    total_frames: int
    computed_at: str


class HeatmapDataPoint(BaseModel):
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    disagreement_rate: float


class HeatmapResponse(BaseModel):
    compare_task_id: str
    video_id: str
    model_pairs: List[str]
    is_aggregated: bool
    window_size: Optional[int] = None
    heatmap_data: Dict[str, List[HeatmapDataPoint]]
