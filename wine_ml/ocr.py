"""Offline OCR index and conservative text retrieval for wine labels."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from rapidocr import RapidOCR

from wine_ml.core import load_image

CYRILLIC = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
LATIN = ['a','b','v','g','d','e','e','zh','z','i','j','k','l','m','n','o','p','r','s','t','u','f','h','cz','ch','sh','shh','','y','','e','yu','ya']
TRANSLIT = dict(zip(CYRILLIC, LATIN))


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKD', text.lower().replace('ё', 'е'))
    text = ''.join(TRANSLIT.get(char, char) for char in text)
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def ngrams(text: str, size=3) -> Counter:
    grams = Counter()
    for token in normalize(text).split():
        padded = f'^{token}$'
        if len(padded) < size:
            grams[padded] += 1
        else:
            grams.update(padded[i:i+size] for i in range(len(padded)-size+1))
    return grams


def cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    denominator = math.sqrt(sum(x*x for x in left.values()) * sum(x*x for x in right.values()))
    return numerator / denominator if denominator else 0.0


class OCR:
    def __init__(self):
        self.engine = RapidOCR()

    def read(self, image) -> dict:
        result = self.engine(image)
        texts = list(result.txts or ())
        scores = list(result.scores or ())
        accepted = [text for text, score in zip(texts, scores) if score >= .55 and len(normalize(text)) >= 2]
        return {'text': ' '.join(accepted), 'lines': texts, 'scores': scores,
                'latency_ms': float(result.elapse * 1000)}


def card_text(row: dict) -> str:
    card = row['card']
    return ' '.join(str(card.get(field, '')) for field in
                    ['Название вина', 'Винодельня', 'Сорт винограда', 'Категория', 'Цвет'])


def build(root=Path('artifacts/experiment')):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    reader = OCR()
    records = []
    for i, row in enumerate(manifest):
        result = reader.read(load_image(row['candidates'][0]))
        records.append({'slug': row['slug'], 'ocr_text': result['text'],
                        'card_text': card_text(row), 'ocr_lines': result['lines'],
                        'ocr_scores': result['scores']})
        if (i+1) % 50 == 0:
            print(f'OCR {i+1}/{len(manifest)}', flush=True)
    (root/'ocr_index.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')


class OCRIndex:
    def __init__(self, root=Path('artifacts/experiment')):
        records = json.loads((root/'ocr_index.json').read_text(encoding='utf-8'))
        self.records = records
        self.grams = [ngrams(row['ocr_text'] + ' ' + row['card_text']) for row in records]
        self.reader = OCR()

    def search(self, image, top_k=30):
        # The product asks the user to center one bottle. Cropping suppresses
        # neighbouring labels that otherwise inject contradictory words.
        w, h = image.size
        result = self.reader.read(image.crop((int(w*.18), int(h*.08), int(w*.82), int(h*.96))))
        query = ngrams(result['text'])
        ranked = sorted(((cosine(query, grams), row['slug']) for row, grams in zip(self.records, self.grams)),
                        reverse=True)[:top_k]
        return {'text': result['text'], 'lines': result['lines'], 'ocr_latency_ms': result['latency_ms'],
                'candidates': [{'slug': slug, 'text_similarity': score} for score, slug in ranked]}


if __name__ == '__main__':
    build()
