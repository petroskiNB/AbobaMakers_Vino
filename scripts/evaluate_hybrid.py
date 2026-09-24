"""Select visual/OCR fusion weight on validation, report once on synthetic test."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wine_ml.core import Adapter, load_image, views
from wine_ml.experiment import augment
from wine_ml.ocr import OCRIndex


def macro_f1(predictions, targets, classes):
    values = []
    for cls in classes:
        tp = sum(p == cls and t == cls for p,t in zip(predictions,targets))
        fp = sum(p == cls and t != cls for p,t in zip(predictions,targets))
        fn = sum(p != cls and t == cls for p,t in zip(predictions,targets))
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return sum(values)/len(values)


def fuse(visual_order, text_order, alpha):
    visual_rank = {slug: rank for rank,slug in enumerate(visual_order,1)}
    text_rank = {slug: rank for rank,slug in enumerate(text_order,1)}
    scores = {}
    for slug, rank in visual_rank.items():
        scores[slug] = scores.get(slug,0) + (1-alpha)/(60+rank)
    for slug, rank in text_rank.items():
        scores[slug] = scores.get(slug,0) + alpha/(60+rank)
    return max(scores, key=scores.get)


def evaluate(rows, visual_orders, text_orders, alpha):
    targets, predictions = [], []
    for row, visual, text in zip(rows, visual_orders, text_orders):
        for query_index in range(2):
            targets.append(row['slug'])
            predictions.append(fuse(visual[query_index], text[query_index], alpha))
    return {'queries': len(targets),
            'accuracy_at_1': sum(p==t for p,t in zip(predictions,targets))/len(targets),
            'macro_f1': macro_f1(predictions, targets, [row['slug'] for row in rows])}


if __name__ == '__main__':
    root = Path('artifacts/experiment')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    stored = torch.load(root/'features.pt', weights_only=True)
    checkpoint = torch.load(root/'adapter_best.pt', weights_only=True)
    adapter = Adapter(checkpoint['dimension']).eval()
    adapter.load_state_dict(checkpoint['state_dict'])
    features = stored['features']
    gallery = adapter(features[:,:2].reshape(-1,features.shape[-1]))
    ocr = OCRIndex(root)
    sets = {}
    started = time.perf_counter()
    for split in ['validation','test']:
        ids = [i for i,name in enumerate(stored['splits']) if name == split]
        rows = [manifest[i] for i in ids]
        visual_orders, text_orders = [], []
        for position,(source_id,row) in enumerate(zip(ids,rows),1):
            query_features = adapter(features[source_id,2:])
            score = (query_features @ gallery.T).reshape(2,len(manifest),2).amax(-1)
            visual_orders.append([[manifest[i]['slug'] for i in order.tolist()] for order in score.argsort(-1,descending=True)])
            source_views = views(load_image(row['candidates'][0]))
            seed = int(row['sha256'][0][:8],16)
            query_images = [augment(source_views[0],seed),augment(source_views[1],seed+1)]
            text_orders.append([[x['slug'] for x in ocr.search(image,top_k=len(manifest))['candidates']]
                                for image in query_images])
            if position % 50 == 0:
                print(f'{split}: OCR {position}/{len(rows)}',flush=True)
        sets[split] = (rows,visual_orders,text_orders)
    alphas = [0,.2,.35,.45,.5,.55,.65,.8,1]
    validation = {str(alpha): evaluate(*sets['validation'],alpha) for alpha in alphas}
    best_alpha = max(alphas,key=lambda alpha:(validation[str(alpha)]['accuracy_at_1'],-alpha))
    output = {'evaluation_type':'synthetic_source_disjoint_not_field_accuracy',
              'fusion_method':'weighted_reciprocal_rank_fusion_k60',
              'selected_alpha_on_validation':best_alpha,
              'validation_grid':validation,
              'test_visual_only':evaluate(*sets['test'],0),
              'test_ocr_only':evaluate(*sets['test'],1),
              'test_selected_hybrid':evaluate(*sets['test'],best_alpha),
              'ocr_evaluation_seconds':time.perf_counter()-started}
    (root/'hybrid_metrics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
