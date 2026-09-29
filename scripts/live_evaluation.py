"""Sequential live HTTP evaluation with optional independent ground truth."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import requests


def summarize(rows, labels=None):
    times = sorted(r['latency_ms'] for r in rows)
    result = {'queries': len(rows), 'successful_responses': sum(r['ok'] for r in rows),
              'latency_ms': {'mean': statistics.mean(times), 'p50': statistics.median(times),
                             'p95_nearest_rank': times[math.ceil(.95*len(times))-1], 'max': max(times)},
              'successful_under_3s': sum(r['ok'] and r['latency_ms'] < 3000 for r in rows),
              'quality': None}
    if labels is None:
        result['quality_note'] = 'Ground truth unavailable; HTTP success is not recognition accuracy.'
        return result
    tp, fp, fn = Counter(), Counter(), Counter()
    hits = correct = retrieved = predicted_count = 0
    for row in rows:
        truth = labels[row['query_id']]
        predicted = row['predicted_slug']
        predicted_count += predicted is not None
        if predicted == truth:
            correct += 1
            tp[truth] += 1
        else:
            fn[truth] += 1
            if predicted is not None:
                fp[predicted] += 1
        candidates = row['top5']
        hits += truth in candidates
        retrieved += len(candidates)
    classes = set(tp) | set(fp) | set(fn)
    macro = statistics.mean(2*tp[c]/(2*tp[c]+fp[c]+fn[c]) for c in classes)
    precision5 = hits/retrieved if retrieved else 0.0
    recall5 = hits/len(rows)
    result['quality'] = {'accuracy_top1': correct/len(rows), 'macro_f1_top1': macro,
                         'micro_set_f1_at_1': 2*correct/(predicted_count+len(rows)),
                         'recall_at_5': recall5, 'micro_precision_at_5': precision5,
                         'micro_set_f1_at_5': 2*hits/(retrieved+len(rows)),
                         'classes_in_macro_average': len(classes)}
    result['quality_note'] = ('Top1 macro-F1 uses union of true and predicted classes. '
                              'Top5 is a set: micro F1=2*hits/(returned candidates+queries). '
                              'One relevant slug per query; this is not an organizer-approved F1@5 definition.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080/v1/predict')
    parser.add_argument('--manifest', type=Path, default=Path('data/queries.tsv'))
    parser.add_argument('--images-dir', type=Path, default=Path('data/queries'))
    parser.add_argument('--labels', type=Path, help='Independent TSV: query_id, expected_slug')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwritten')
    parser.add_argument('--timeout', type=float, default=10)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8-sig') as stream:
        queries = list(csv.DictReader(stream, delimiter='\t'))
    ids = [r['query_id'] for r in queries]
    if not ids or len(set(ids)) != len(ids):
        parser.error('Manifest must contain unique, nonempty queries')
    labels = None
    if args.labels:
        with args.labels.open(encoding='utf-8-sig') as stream:
            labeled = list(csv.DictReader(stream, delimiter='\t'))
        labels = {r['query_id']: r['expected_slug'].strip() for r in labeled}
        if len(labels) != len(labeled) or set(labels) != set(ids) or not all(labels.values()):
            parser.error('Labels must match every query exactly once and contain nonempty expected_slug')
    paths = []
    for query in queries:
        path = (args.images_dir / query['image_path']).resolve()
        if not path.is_relative_to(args.images_dir.resolve()) or not path.is_file():
            parser.error(f'Invalid image path: {query["image_path"]}')
        paths.append(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with (args.output_dir/'predictions.jsonl').open('x', encoding='utf-8') as out:
        for query, path in zip(queries, paths):
            content = path.read_bytes()
            row = {**query, 'image_sha256': hashlib.sha256(content).hexdigest(),
                   'predicted_slug': None, 'top5': [], 'ok': False, 'http_status': None}
            started = time.perf_counter()
            try:
                response = requests.post(args.endpoint, files={'image': (path.name, content)},
                                         timeout=(5, args.timeout))
                row['http_status'] = response.status_code
                response.raise_for_status()
                answer = response.json()
                slug = answer.get('slug')
                candidates = answer.get('candidates')
                if not isinstance(slug, str) or not slug or not isinstance(candidates, list):
                    raise ValueError('Product endpoint must return slug and candidates')
                top5 = list(dict.fromkeys(c['slug'] for c in candidates
                                         if isinstance(c, dict) and isinstance(c.get('slug'), str) and c['slug']))[:5]
                if not top5 or top5[0] != slug:
                    raise ValueError('Candidate ranking is missing or inconsistent with top1')
                row.update(predicted_slug=slug, top5=top5, ok=True)
            except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
                row['error'] = type(exc).__name__
            row['latency_ms'] = round((time.perf_counter()-started)*1000, 3)
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False)+'\n')
            out.flush()
            print(f'{len(rows)}/{len(queries)} {query["query_id"]}: {row["latency_ms"]:.0f} ms '
                  f'top1={row["predicted_slug"]} top5={row["top5"]}', flush=True)
            print(json.dumps(summarize(rows, labels), ensure_ascii=False), flush=True)
    summary = summarize(rows, labels)
    summary.update(endpoint=args.endpoint, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                   labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest() if args.labels else None,
                   protocol='Sequential HTTP, no retries, no excluded warmup; includes failed requests')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
