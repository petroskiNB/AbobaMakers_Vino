"""Local evaluation API; product rejection/uncertainty calibration is pending."""
import io
import os
from pathlib import Path
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from wine_ml.sommelier import recommend, PairingUnavailable
from PIL import Image, UnidentifiedImageError

recognizer = None
lock = Lock()


@asynccontextmanager
async def lifespan(app):
    global recognizer
    recognizer = None
    if os.environ.get('WINE_PREVIEW') == '1':
        yield
        return
    required = ['artifacts/base_model/config.json', 'artifacts/experiment/index.pt',
                'artifacts/experiment/adapter_best.pt', 'artifacts/experiment/manifest.json']
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError('Missing model artifacts: ' + ', '.join(missing)
                           + '. Restore trained artifacts or use scripts/run.ps1 -Preview for UI testing.')
    from wine_ml.predict import Recognizer
    from wine_ml.hybrid import HybridRecognizer
    recognizer = HybridRecognizer() if Path('artifacts/experiment/ocr_index.json').exists() else Recognizer()
    recognizer.predict(Image.new('RGB', (384, 384), 'white'))
    yield


app = FastAPI(title='Wine retrieval experimental baseline', lifespan=lifespan)
app.mount('/static', StaticFiles(directory='web'), name='static')


class PairingRequest(BaseModel):
    slug: str
    dish: str = Field(min_length=1, max_length=300)


@app.get('/')
def home():
    return FileResponse('web/index.html', headers={'Cache-Control': 'no-cache'})


@app.get('/health')
def health():
    return {'ready': recognizer is not None, 'catalog_size': len(recognizer.slugs) if recognizer else 0,
            'pipeline': type(recognizer).__name__ if recognizer else None}


@app.get('/manifest.webmanifest')
def manifest():
    return FileResponse('web/manifest.webmanifest', media_type='application/manifest+json')


@app.get('/sw.js')
def service_worker():
    return FileResponse('web/sw.js', media_type='application/javascript',
                        headers={'Cache-Control': 'no-cache'})


def run(image):
    require_recognizer()
    from wine_ml.core import load_image
    data = image.file.read(20*1024*1024+1)
    if len(data) > 20*1024*1024:
        raise HTTPException(413, 'Image exceeds 20 MB')
    try:
        im = load_image(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, 'Invalid image')
    with lock:
        return recognizer.predict(im)


def require_recognizer():
    if recognizer is None:
        raise HTTPException(503, 'Режим просмотра: модель не загружена. Для распознавания нужны веса модели и поисковый индекс.')


@app.post('/v1/eval/predict')
def evaluation_predict(image: UploadFile = File(...)):
    return {'slug': run(image)['slug']}


@app.post('/v1/predict')
def product_predict(image: UploadFile = File(...)):
    result = run(image)
    result['card'] = recognizer.cards[result['slug']]
    result['candidates'] = [
        {**candidate, 'card': recognizer.cards[candidate['slug']]}
        for candidate in result['candidates'][:5]
    ]
    return result


@app.post('/v1/pairing')
def pairing(request: PairingRequest):
    require_recognizer()
    card = recognizer.cards.get(request.slug)
    if not card:
        raise HTTPException(404, 'Wine not found in active index')
    dish = request.dish.strip()
    if not dish:
        raise HTTPException(422, 'Укажите блюдо')
    try:
        return {'slug': request.slug, **recommend(card, dish)}
    except PairingUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
