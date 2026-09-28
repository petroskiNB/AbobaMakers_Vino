FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    WINE_PREVIEW=0

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

# RapidOCR pulls desktop OpenCV alongside our headless package. Both own cv2;
# remove both before restoring the single server-only implementation.
RUN python -m pip uninstall -y opencv-python opencv-python-headless \
    && python -m pip install --no-cache-dir opencv-python-headless==4.13.0.92 \
    && python -c "import cv2; from rapidocr import RapidOCR"

RUN useradd --create-home --uid 10001 wine
COPY wine_ml/ ./wine_ml/
COPY web/ ./web/
USER wine
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4)); assert r['ready']"
CMD ["python", "-m", "uvicorn", "wine_ml.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
