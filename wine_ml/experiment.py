"""Train a residual visual adapter on provisional catalog labels and synthetic views.

No field accuracy is claimed. Splits hold out source images/slugs from training.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import ImageEnhance, ImageFilter
from torchvision.transforms import functional as TF

from wine_ml.core import Adapter, Encoder, foreground, load_image, views


def augment(image, seed):
    rng = random.Random(seed)
    image = image.copy()
    image.thumbnail((640, 640))
    w, h = image.size
    dx, dy = max(1, int(w*.12)), max(1, int(h*.06))
    start = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
    end = [[rng.randint(0, dx), rng.randint(0, dy)],
           [w-1-rng.randint(0, dx), rng.randint(0, dy)],
           [w-1-rng.randint(0, dx), h-1-rng.randint(0, dy)],
           [rng.randint(0, dx), h-1-rng.randint(0, dy)]]
    image = TF.perspective(image, start, end, fill=255)
    image = image.rotate(rng.uniform(-9, 9), expand=True, fillcolor='white')
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.60, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.75, 1.2))
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.1)))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def extract(args):
    torch.manual_seed(42)
    links = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    selected, excluded = [], []
    for row_number, row in enumerate(links):
        if row_number and row_number % 250 == 0:
            print(f'Preparing {row_number}/{len(links)} catalog rows', flush=True)
        if row['status'] != 'candidate' or row['cross_slug_content_conflict']:
            excluded.append({'slug': row['slug'], 'reason': 'ambiguous_missing_or_content_conflict'})
            continue
        try:
            image = foreground(load_image(row['candidates'][0]))
            # Conservative, explicitly provisional bottle-shaped subset.
            if image.width / image.height > .85 or min(image.size) < 40:
                excluded.append({'slug': row['slug'], 'reason': 'geometry_filter'})
                continue
        except Exception as exc:
            excluded.append({'slug': row['slug'], 'reason': str(exc)})
            continue
        fraction = int(hashlib.sha256(row['sha256'][0].encode()).hexdigest()[:8], 16) / 2**32
        row['split'] = 'train' if fraction < .70 else ('validation' if fraction < .85 else 'test')
        selected.append(row)
    if args.limit:
        selected = selected[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', selected)
    write_json(out/'excluded.json', excluded)
    encoder = Encoder()
    all_features, pending, started = [], [], time.perf_counter()
    def flush():
        if pending:
            all_features.append(encoder.encode(pending))
            pending.clear()
    for i, row in enumerate(selected):
        image_views = views(load_image(row['candidates'][0]))
        # Two canonical and two augmented views per source. Augmented views
        # from validation/test sources never enter the optimizer.
        seed = int(row['sha256'][0][:8], 16)
        for image in image_views + [augment(image_views[0], seed), augment(image_views[1], seed+1)]:
            pending.append(image)
            if len(pending) >= args.batch_size:
                flush()
        if (i+1) % 50 == 0:
            print(f'Encoded {i+1}/{len(selected)} sources, {time.perf_counter()-started:.1f}s', flush=True)
    flush()
    features = torch.cat(all_features).reshape(len(selected), 4, -1)
    torch.save({'features': features, 'slugs': [x['slug'] for x in selected],
                'splits': [x['split'] for x in selected],
                'source_revision': json.loads(Path('artifacts/base_model/source.json').read_text()),
                'manifest_sha256': hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()}, out/'features.pt')
    print(f'Saved features: {tuple(features.shape)}', flush=True)


@torch.no_grad()
def evaluate(adapter, features, ids, batch_size=128):
    gallery = adapter(features[:, :2].reshape(-1, features.shape[-1]))
    correct1 = correct5 = total = 0
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start+batch_size]
        queries = adapter(features[batch_ids, 2:].reshape(-1, features.shape[-1]))
        scores = (queries @ gallery.T).reshape(len(queries), len(features), 2).amax(dim=-1)
        top = scores.topk(min(5, len(features)), dim=-1).indices
        targets = batch_ids.repeat_interleave(2)
        correct1 += (top[:, 0] == targets).sum().item()
        correct5 += (top == targets[:, None]).any(dim=-1).sum().item()
        total += len(queries)
    return {'queries': total, 'sources': len(ids), 'accuracy_at_1': correct1/total, 'recall_at_5': correct5/total}


def train(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    out = Path(args.out)
    data = torch.load(out/'features.pt', weights_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = data['features'].to(device)
    ids = {name: torch.tensor([i for i, s in enumerate(data['splits']) if s == name], device=device)
           for name in ['train', 'validation', 'test']}
    if min(map(len, ids.values())) == 0:
        raise ValueError('Each source-disjoint split must be nonempty')
    model = Adapter(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    baseline_val = evaluate(model, x, ids['validation'])
    best_score = baseline_val['accuracy_at_1']
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        order = ids['train'][torch.randperm(len(ids['train']), device=device)]
        losses = []
        for group in order.split(128):
            # Distinct nonconflicting slugs; augmented views match either
            # canonical view of the same source. All other batch slugs are negatives.
            gallery = model(x[group, :2].reshape(-1, x.shape[-1]))
            query = model(x[group, 2:].reshape(-1, x.shape[-1]))
            logits = (query @ gallery.T).reshape(len(query), len(group), 2).amax(-1) / .07
            targets = torch.arange(len(group), device=device).repeat_interleave(2)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val = evaluate(model, x, ids['validation'])
        result = {'epoch': epoch, 'loss': float(np.mean(losses)), 'validation': val}
        history.append(result)
        print(json.dumps(result), flush=True)
        if val['accuracy_at_1'] > best_score:
            best_score, best_epoch = val['accuracy_at_1'], epoch
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # Always retain the actually trained weights, even if validation selects identity.
    torch.save({'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                'dimension': x.shape[-1], 'epochs': args.epochs}, out/'adapter_last.pt')
    model.load_state_dict(best_state)
    torch.save({'state_dict': best_state, 'dimension': x.shape[-1], 'epoch': best_epoch}, out/'adapter_best.pt')
    baseline = Adapter(x.shape[-1]).to(device).eval()
    metrics = {'evaluation_type': 'synthetic_source_disjoint_not_field_accuracy',
               'provisional_filename_labels': True, 'seed': 42,
               'split_counts': {k:len(v) for k,v in ids.items()},
               'baseline_validation': baseline_val, 'best_epoch': best_epoch,
               'baseline_test': evaluate(baseline, x, ids['test']),
               'selected_test': evaluate(model, x, ids['test']), 'history': history,
               'encoder_frozen': True, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
               'torch': torch.__version__, 'device': device,
               'source_revision': data['source_revision'], 'manifest_sha256': data['manifest_sha256']}
    write_json(out/'metrics.json', metrics)
    with torch.no_grad():
        gallery = model(x[:, :2].reshape(-1, x.shape[-1])).cpu()
    torch.save({'gallery': gallery, 'slugs': data['slugs'], 'dimension': x.shape[-1],
                'source_revision': data['source_revision'], 'manifest_sha256': data['manifest_sha256'],
                'adapter_sha256': hashlib.sha256((out/'adapter_best.pt').read_bytes()).hexdigest()}, out/'index.pt')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'history'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['extract', 'train'])
    parser.add_argument('--manifest', default='artifacts/data_audit/candidate_links.json')
    parser.add_argument('--out', default='artifacts/experiment')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    (extract if args.command == 'extract' else train)(args)
