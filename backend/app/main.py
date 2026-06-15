from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router as api_router

app = FastAPI(
    title=settings.APP_NAME,
    description="视频动作识别与时序行为分割后端推理服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload_video": "POST /videos/upload",
            "upload_video_url": "POST /videos/upload-url",
            "get_video_info": "GET /videos/{id}/info",
            "analyze_video": "POST /videos/{id}/analyze",
            "get_task_status": "GET /tasks/{id}/status",
            "get_results": "GET /videos/{id}/results",
            "get_timeline": "GET /videos/{id}/results/timeline",
            "evaluate": "POST /videos/{id}/evaluate",
            "download_features": "GET /videos/{id}/features",
            "delete_video": "DELETE /videos/{id}",
            "model_versions": "GET /models/versions",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
