"""Local evaluation API; product rejection/uncertainty calibration is pending."""
import io
import os
from pathlib import Path
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
    dish: str


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
    return result


@app.post('/v1/pairing')
def pairing(request: PairingRequest):
    require_recognizer()
    card = recognizer.cards.get(request.slug)
    if not card:
        raise HTTPException(404, 'Wine not found in active index')
    category = str(card.get('Категория', '')).lower()
    grape = str(card.get('Сорт винограда', '')).lower()
    dish = request.dish.strip()
    if not dish:
        raise HTTPException(422, 'Dish is required')
    if 'игрист' in category or 'брют' in category:
        advice = 'Игристый стиль хорошо работает с лёгкими закусками, мягкими сырами и блюдами с хрустящей текстурой.'
    elif 'крас' in category or any(x in grape for x in ['каберне','саперави','мерло']):
        advice = 'Попробуйте подать к насыщенному мясному блюду, грибам или выдержанному сыру.'
    elif 'роз' in category:
        advice = 'Подойдёт к птице, лёгким мясным блюдам, овощам и неострым закускам.'
    else:
        advice = 'Подойдёт к рыбе, морепродуктам, птице или свежим сырам.'
    return {'slug': request.slug, 'dish': dish,
            'explanation': f'Для блюда «{dish}»: {advice} Это общая гастрономическая рекомендация по данным карточки.'}
