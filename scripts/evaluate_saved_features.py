"""Recompute top1/top5 set F1 on the saved synthetic test (not field photos)."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wine_ml.core import Adapter
from scripts.live_evaluation import summarize


def main():
    root = ROOT/'artifacts/experiment'
    data = torch.load(root/'features.pt', map_location='cpu', weights_only=True)
    checkpoint = torch.load(root/'adapter_best.pt', map_location='cpu', weights_only=True)
    model = Adapter(checkpoint['dimension']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    features = data['features']
    ids = [i for i, split in enumerate(data['splits']) if split == 'test']
    with torch.inference_mode():
        gallery = model(features[:, :2].reshape(-1, features.shape[-1]))
        query = model(features[ids, 2:].reshape(-1, features.shape[-1]))
        scores = (query @ gallery.T).reshape(len(query), len(features), 2).amax(-1)
        orders = scores.topk(5).indices.tolist()
    targets = [str(i) for i in ids for _ in range(features.shape[1]-2)]
    rows = [dict(query_id=str(i), predicted_slug=str(order[0]), top5=list(map(str, order)),
                 ok=True, latency_ms=0) for i, order in enumerate(orders)]
    result = summarize(rows, {str(i): t for i, t in enumerate(targets)})
    for name in ('latency_ms', 'successful_under_3s'):
        result.pop(name)
    result.update(dataset='synthetic_saved_test', label_source='provisional_catalog_links',
                  adapter_sha256=hashlib.sha256((root/'adapter_best.pt').read_bytes()).hexdigest(),
                  note='Recomputed from saved synthetic features; not API latency or public dataset quality.')
    path = ROOT/'web/synthetic-evaluation.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
