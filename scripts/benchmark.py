"""Sequential latency benchmark for the local evaluation endpoint."""
import json
import statistics
import time
from pathlib import Path

import requests

endpoint = 'http://127.0.0.1:8080/v1/eval/predict'
images = sorted(Path('data/eval/queries').iterdir())
measurements = []
for _ in range(2):
    for path in images:
        started = time.perf_counter()
        with path.open('rb') as stream:
            response = requests.post(endpoint, files={'image': (path.name, stream)}, timeout=(5, 10))
        response.raise_for_status()
        measurements.append({'image': path.name, 'latency_ms': (time.perf_counter()-started)*1000,
                             'slug': response.json()['slug']})
values = sorted(row['latency_ms'] for row in measurements)
summary = {'requests': len(values), 'warm_sequential': True,
           'mean_ms': statistics.mean(values), 'median_p50_ms': statistics.median(values),
           'p95_nearest_rank_ms': values[max(0, int(.95*len(values)+.999999)-1)],
           'min_ms': min(values), 'max_ms': max(values), 'measurements': measurements,
           'note': 'Three public images repeated twice; not a load or accuracy test.'}
Path('artifacts/api_benchmark.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
