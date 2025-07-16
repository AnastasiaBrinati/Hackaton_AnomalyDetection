FROM python:3.13-slim

WORKDIR /usr/src/app

# Installo le librerie di sistema necessarie per OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Questa linea serve per installare le dipendenze complesse in modo più veloce
ENV PIP_PREFER_BINARY=1 

COPY ml_worker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY ml_worker/ .
COPY shared ./shared

CMD ["celery", "-A", "shared.celery_app.celery_app", "worker", "--loglevel=info", "--concurrency=1", "--include=shared.celery_config"]