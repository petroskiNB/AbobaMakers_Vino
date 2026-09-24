"""Optional geometric verification of visual candidates; no learned confidence."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageOps
from wine_ml.core import load_image, foreground

cv2.setNumThreads(4)


def describe(image, maximum=720):
    image = ImageOps.contain(image, (maximum, maximum))
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=700).detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    return np.asarray([x.pt for x in keypoints], np.float32), descriptors


def build(root=Path('artifacts/experiment')):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    points, descriptors, offsets = [], [], [0]
    for i, row in enumerate(manifest):
        p, d = describe(foreground(load_image(row['candidates'][0])))
        points.append(p)
        descriptors.append(d)
        offsets.append(offsets[-1]+len(d))
        if (i+1) % 100 == 0:
            print(f'Local descriptors {i+1}/{len(manifest)}', flush=True)
    np.savez_compressed(root/'local_features.npz', points=np.concatenate(points),
                        descriptors=np.concatenate(descriptors), offsets=np.asarray(offsets),
                        slugs=np.asarray([x['slug'] for x in manifest]))


class GeometricVerifier:
    def __init__(self, root=Path('artifacts/experiment')):
        with np.load(root/'local_features.npz', allow_pickle=False) as data:
            self.points, self.descriptors = data['points'], data['descriptors']
            self.offsets = data['offsets']
            self.ids = {slug: i for i, slug in enumerate(data['slugs'].tolist())}
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def verify(self, image, candidates):
        # Suppress shelf neighbours using a generic central query region.
        w, h = image.size
        central = image.crop((int(w*.18), 0, int(w*.82), h))
        query_points, query_descriptors = describe(central, 960)
        results = []
        for candidate in candidates:
            row = dict(candidate)
            i = self.ids[row['slug']]
            start, end = self.offsets[i:i+2]
            reference = self.descriptors[start:end]
            inliers = 0
            if len(reference) >= 4 and len(query_descriptors) >= 4:
                pairs = self.matcher.knnMatch(reference, query_descriptors, k=2)
                good = [m for m,n in pairs if m.distance < .7*n.distance]
                # A single query feature cannot provide independent evidence twice.
                unique = {}
                for match in sorted(good, key=lambda x: x.distance):
                    unique.setdefault(match.trainIdx, match)
                good = list(unique.values())
                if len(good) >= 8:
                    source = np.float32([self.points[start+m.queryIdx] for m in good])
                    target = np.float32([query_points[m.trainIdx] for m in good])
                    cv2.setRNGSeed(42)
                    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
                    if matrix is not None and mask is not None:
                        inliers = int(mask.sum())
            row['geometric_inliers'] = inliers
            results.append(row)
        # Uncalibrated experimental tie-breaker, exposed separately from embeddings.
        strong = [x for x in results if x['geometric_inliers'] >= 8]
        return sorted(results, key=lambda x: (x['geometric_inliers'], x['similarity']), reverse=True) if strong else results


if __name__ == '__main__':
    build()
