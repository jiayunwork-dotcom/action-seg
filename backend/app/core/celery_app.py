from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "action_seg_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
    task_queues={
        "default": {
            "exchange": "default",
            "routing_key": "default",
        }
    },
    task_default_queue="default",
)

celery_app.autodiscover_tasks(["app.services"])
