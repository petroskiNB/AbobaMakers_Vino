"""Calibrate a known-wine acceptance threshold on validation only."""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wine_ml.core import Adapter


def per_class_macro_f1(predictions, targets, classes):
    scores = []
    for cls in classes.tolist():
        tp = int(((predictions == cls) & (targets == cls)).sum())
        fp = int(((predictions == cls) & (targets != cls)).sum())
        fn = int(((predictions != cls) & (targets == cls)).sum())
        denominator = 2*tp + fp + fn
        scores.append((2*tp/denominator) if denominator else 0.0)
    return sum(scores)/len(scores)


@torch.no_grad()
def collect(model, features, ids):
    gallery = model(features[:, :2].reshape(-1, features.shape[-1]))
    query = model(features[ids, 2:].reshape(-1, features.shape[-1]))
    scores = (query @ gallery.T).reshape(len(query), len(features), 2).amax(-1)
    values, indices = scores.topk(min(5, len(features)), dim=-1)
    targets = ids.repeat_interleave(2)
    correct = indices[:, 0].eq(targets)
    margins = values[:, 0] - values[:, 1]
    reciprocal = torch.where((indices == targets[:, None]).any(-1),
                             1.0 / ((indices == targets[:, None]).float().argmax(-1)+1), 0.0)
    return {'scores': values[:, 0], 'margins': margins, 'correct': correct,
            'predictions': indices[:, 0], 'targets': targets,
            'recall5': (indices == targets[:, None]).any(-1).float().mean().item(),
            'mrr5': reciprocal.float().mean().item()}


def threshold_for_precision(values, correct, target=.95):
    candidates = sorted(set(float(x) for x in values))
    best = None
    for threshold in candidates:
        accepted = values >= threshold
        count = int(accepted.sum())
        if not count:
            continue
        precision = float(correct[accepted].float().mean())
        coverage = count/len(values)
        if precision >= target and (best is None or coverage > best['coverage']):
            best = {'threshold': threshold, 'precision': precision, 'coverage': coverage, 'accepted': count}
    return best


def summarize(result, ids):
    correct = result['correct']
    return {'queries': len(correct), 'accuracy_at_1': float(correct.float().mean()),
            'recall_at_5': result['recall5'], 'mrr_at_5': result['mrr5'],
            'macro_f1': per_class_macro_f1(result['predictions'], result['targets'], ids),
            'mean_top1_similarity': float(result['scores'].mean()),
            'mean_margin': float(result['margins'].mean())}


def apply_threshold(result, threshold, field):
    accepted = result[field] >= threshold
    count = int(accepted.sum())
    return {'accepted': count, 'coverage': count/len(accepted),
            'precision_among_accepted': float(result['correct'][accepted].float().mean()) if count else None}


if __name__ == '__main__':
    root = Path('artifacts/experiment')
    data = torch.load(root/'features.pt', weights_only=True)
    features = data['features']
    checkpoint = torch.load(root/'adapter_best.pt', weights_only=True)
    model = Adapter(checkpoint['dimension']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    baseline = Adapter(checkpoint['dimension']).eval()
    validation = torch.tensor([i for i,s in enumerate(data['splits']) if s == 'validation'])
    test = torch.tensor([i for i,s in enumerate(data['splits']) if s == 'test'])
    val_result = collect(model, features, validation)
    test_result = collect(model, features, test)
    similarity_threshold = threshold_for_precision(val_result['scores'], val_result['correct'])
    margin_threshold = threshold_for_precision(val_result['margins'], val_result['correct'])
    calibration = {'scope': 'known_wines_synthetic_validation_only',
                   'cannot_calibrate_unknown_wine_rejection': True,
                   'target_precision': .95,
                   'top1_similarity': similarity_threshold,
                   'margin': margin_threshold,
                   'test_at_validation_margin_threshold': apply_threshold(test_result, margin_threshold['threshold'], 'margins'),
                   'selected_validation': summarize(val_result, validation),
                   'selected_test': summarize(test_result, test),
                   'baseline_test': summarize(collect(baseline, features, test), test)}
    (root/'calibration.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    print(json.dumps(calibration, indent=2))
