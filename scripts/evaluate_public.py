"""Sequential Windows-compatible HTTP smoke test matching organizer JSONL fields.

No accuracy computation: organizer labels are not public.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--endpoint', default='http://127.0.0.1:8080/v1/eval/predict')
parser.add_argument('--manifest', type=Path, default=Path('data/eval/queries.tsv'))
parser.add_argument('--images-dir', type=Path, default=Path('data/eval/queries'))
parser.add_argument('--output', type=Path, default=Path('artifacts/public_predictions.jsonl'))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x', encoding='utf-8') as output:
    for row in csv.DictReader(args.manifest.open(encoding='utf-8-sig'), delimiter='\t'):
        path = (args.images_dir/row['image_path']).resolve()
        if not path.is_relative_to(args.images_dir.resolve()):
            raise ValueError('Unsafe image path')
        data = path.read_bytes()
        started = time.perf_counter()
        slug = None
        try:
            response = requests.post(args.endpoint, files={'image': (path.name, data)}, timeout=(5, 10))
            if response.status_code in (200,201):
                answer = response.json()
                if isinstance(answer, list) and answer:
                    answer = answer[0]
                candidate = answer.get('slug') if isinstance(answer, dict) else None
                slug = candidate if isinstance(candidate, str) and candidate else None
        except (requests.RequestException, ValueError) as exc:
            print(f"{row['query_id']}: {exc}", flush=True)
        result = {'query_id': row['query_id'], 'image_path': row['image_path'],
                  'image_sha256': hashlib.sha256(data).hexdigest(), 'predicted_slug': slug,
                  'latency_ms': round((time.perf_counter()-started)*1000)}
        output.write(json.dumps(result, ensure_ascii=False)+'\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
