import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from wine_ml.core import Adapter, Encoder, load_image, query_views


class Recognizer:
    def __init__(self, artifacts='artifacts/experiment'):
        root = Path(artifacts)
        self.encoder = Encoder()
        index = torch.load(root/'index.pt', map_location='cpu', weights_only=True)
        if hashlib.sha256((root/'adapter_best.pt').read_bytes()).hexdigest() != index['adapter_sha256']:
            raise ValueError('Adapter and index do not match; rebuild the index')
        if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() != index['manifest_sha256']:
            raise ValueError('Manifest and index do not match; rebuild the index')
        if json.loads(Path('artifacts/base_model/source.json').read_text()) != index['source_revision']:
            raise ValueError('Encoder and index do not match; rebuild the features and index')
        checkpoint = torch.load(root/'adapter_best.pt', map_location='cpu', weights_only=True)
        self.adapter = Adapter(index['dimension']).to(self.encoder.device).eval()
        self.adapter.load_state_dict(checkpoint['state_dict'])
        self.gallery = index['gallery'].to(self.encoder.device)
        self.slugs = index['slugs']
        self.cards = {row['slug']: row['card'] for row in json.loads((root/'manifest.json').read_text(encoding='utf-8'))}
        calibration_path = root/'calibration.json'
        self.calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None

    @torch.inference_mode()
    def predict(self, image, top_k=5):
        started = time.perf_counter()
        representations = query_views(image)
        query = self.adapter(self.encoder.encode(representations).to(self.encoder.device))
        similarities = (query @ self.gallery.T).reshape(len(query), len(self.slugs), 2)
        scores = similarities.amax(dim=(0, 2))
        values, indices = scores.topk(min(top_k, len(scores)))
        candidates = [{'slug': self.slugs[i], 'similarity': float(v)} for v,i in zip(values.cpu(), indices.cpu())]
        margin = candidates[0]['similarity']-candidates[1]['similarity'] if len(candidates)>1 else None
        status = 'uncalibrated_candidate'
        if self.calibration and margin is not None:
            threshold = self.calibration['margin']['threshold']
            status = 'found_known_wine' if margin >= threshold else 'low_confidence_reshoot'
        return {'slug': candidates[0]['slug'], 'candidates': candidates,
                'margin': margin,
                'latency_ms': (time.perf_counter()-started)*1000,
                'status': status,
                'warning': 'Partial catalog; threshold is calibrated only on synthetic known wines. Similarity is not probability.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--artifacts', default='artifacts/experiment')
    args = parser.parse_args()
    print(json.dumps(Recognizer(args.artifacts).predict(load_image(args.image)), ensure_ascii=False, indent=2))
