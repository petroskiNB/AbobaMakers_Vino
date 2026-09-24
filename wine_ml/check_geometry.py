"""Diagnostic ablation on public, unlabeled development images. Not an accuracy test."""
import json
import time
from pathlib import Path
from wine_ml.core import load_image
from wine_ml.predict import Recognizer
from wine_ml.local_features import GeometricVerifier


if __name__ == '__main__':
    recognizer = Recognizer()
    verifier = GeometricVerifier()
    results = []
    for path in sorted(Path('data/eval/queries').iterdir()):
        image = load_image(path)
        started = time.perf_counter()
        visual = recognizer.predict(image, top_k=40)
        reranked = verifier.verify(image, visual['candidates'])
        row = {'image': path.name, 'baseline_top5': visual['candidates'][:5],
               'geometry_top5': reranked[:5], 'latency_ms': (time.perf_counter()-started)*1000,
               'ground_truth_available': False}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    Path('artifacts/geometry_ablation.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
