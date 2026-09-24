"""Hybrid visual retrieval plus OCR reciprocal-rank fusion."""
import json
from pathlib import Path

from wine_ml.ocr import OCRIndex
from wine_ml.predict import Recognizer


class HybridRecognizer:
    def __init__(self, artifacts='artifacts/experiment'):
        self.visual = Recognizer(artifacts)
        self.ocr = OCRIndex(Path(artifacts))
        self.cards = self.visual.cards
        self.slugs = self.visual.slugs
        metrics_path = Path(artifacts)/'hybrid_metrics.json'
        self.text_weight = json.loads(metrics_path.read_text())['selected_alpha_on_validation'] if metrics_path.exists() else .45

    def predict(self, image, top_k=5):
        visual = self.visual.predict(image, top_k=60)
        text = self.ocr.search(image, top_k=60)
        visual_rank = {row['slug']: rank for rank, row in enumerate(visual['candidates'], 1)}
        text_rank = {row['slug']: rank for rank, row in enumerate(text['candidates'], 1)}
        visual_score = {row['slug']: row['similarity'] for row in visual['candidates']}
        text_score = {row['slug']: row['text_similarity'] for row in text['candidates']}
        union = set(visual_rank) | set(text_rank)
        # OCR weight is intentionally moderate. The validation script can replace
        # this fixed value once real labeled field photos exist.
        text_weight = self.text_weight
        combined = []
        for slug in union:
            score = ((1-text_weight)/(60+visual_rank[slug]) if slug in visual_rank else 0.0)
            score += (text_weight/(60+text_rank[slug]) if slug in text_rank else 0.0)
            combined.append({'slug': slug, 'fusion_score': score,
                             'visual_similarity': visual_score.get(slug),
                             'text_similarity': text_score.get(slug),
                             'visual_rank': visual_rank.get(slug), 'text_rank': text_rank.get(slug)})
        combined.sort(key=lambda row: row['fusion_score'], reverse=True)
        best = combined[0]
        visual_margin = visual['margin']
        # Refuse a product answer when modalities strongly disagree or the visual
        # margin fails the known-wine validation threshold.
        agreement = best['visual_rank'] is not None and best['text_rank'] is not None
        status = 'found_candidate' if agreement and best['visual_rank'] <= 10 and best['text_rank'] <= 10 else 'low_confidence_reshoot'
        return {'slug': best['slug'], 'status': status, 'candidates': combined[:top_k],
                'ocr_text': text['text'], 'ocr_lines': text['lines'],
                'visual_margin': visual_margin,
                'latency_ms': visual['latency_ms'] + text['ocr_latency_ms'],
                'warning': 'Fusion weights are experimental; no labeled field validation is available.'}
