import csv
import io
import json
from datetime import datetime
from typing import Dict, List, Tuple

from app.services.storage_manager import StorageManager


class ExportService:
    def __init__(self):
        self.storage = StorageManager()

    def _format_timecode(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _generate_filename(self, video_id: str, fmt: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{video_id}_{timestamp}.{fmt}"

    def export_json(self, video_id: str, model_version: str) -> Tuple[str, str, bytes]:
        results = self.storage.load_results(video_id, model_version)
        if not results:
            raise ValueError(f"Results not found for video {video_id}")

        filename = self._generate_filename(video_id, "json")
        content = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
        return filename, "application/json", content

    def export_srt(self, video_id: str, model_version: str) -> Tuple[str, str, bytes]:
        results = self.storage.load_results(video_id, model_version)
        if not results:
            raise ValueError(f"Results not found for video {video_id}")

        segments = results["segments"]
        srt_lines = []

        for i, seg in enumerate(segments, 1):
            start_tc = self._format_timecode(seg["start_time"])
            end_tc = self._format_timecode(seg["end_time"])
            srt_lines.append(str(i))
            srt_lines.append(f"{start_tc} --> {end_tc}")
            srt_lines.append(seg["action_name"])
            srt_lines.append("")

        filename = self._generate_filename(video_id, "srt")
        content = "\n".join(srt_lines).encode("utf-8")
        return filename, "application/x-subrip", content

    def export_csv(self, video_id: str, model_version: str) -> Tuple[str, str, bytes]:
        results = self.storage.load_results(video_id, model_version)
        if not results:
            raise ValueError(f"Results not found for video {video_id}")

        segments = results["segments"]
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "片段序号",
            "动作类别",
            "起始时间码",
            "结束时间码",
            "持续秒数",
            "置信度",
        ])

        for i, seg in enumerate(segments, 1):
            duration = seg["end_time"] - seg["start_time"]
            writer.writerow([
                i,
                seg["action_name"],
                self._format_timecode(seg["start_time"]),
                self._format_timecode(seg["end_time"]),
                f"{duration:.3f}",
                f"{seg['confidence']:.4f}",
            ])

        filename = self._generate_filename(video_id, "csv")
        content = output.getvalue().encode("utf-8-sig")
        return filename, "text/csv", content
