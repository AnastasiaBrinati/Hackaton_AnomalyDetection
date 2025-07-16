from celery import Celery

REDIS_URL = "redis://redis:6379/0"

# Inizializzazione dell'app Celery
celery_app = Celery(
    "task_in_OD",
    broker=REDIS_URL,
    backend=REDIS_URL,
) 