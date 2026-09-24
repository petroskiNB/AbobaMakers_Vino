"""Audit local assets and propose CSV-to-image links; never invent slug labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.jfif', '.heic'}
SIZE_PREFIX = re.compile(r'^(thumbnail|small|medium|large)_', re.I)
TRANSLIT = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
    ['a','b','v','g','d','e','e','zh','z','i','j','k','l','m','n','o','p','r','s','t','u','f','h','cz','ch','sh','shh','','y','','e','yu','ya']))


def normalize(value):
    value = ''.join(TRANSLIT.get(c, c) for c in value.lower())
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', value))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('data'))
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--out', type=Path, default=Path('artifacts/data_audit'))
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error(f'Image directory missing: {args.images}')
    if args.catalog and not args.catalog.is_file():
        parser.error(f'Catalog missing: {args.catalog}')
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.images.rglob('*') if p.is_file())
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    originals = [p for p in images if not SIZE_PREFIX.match(p.name)]
    by_name, by_normalized = defaultdict(list), defaultdict(list)
    for p in originals:
        by_name[p.name].append(p)
        stem = re.sub(r'_[0-9a-f]{10}$', '', p.stem, flags=re.I)
        by_normalized[normalize(stem)].append(p)
    with (args.out / 'inventory.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'bytes', 'derived_size', 'normalized_source_name'])
        for p in images:
            stem = SIZE_PREFIX.sub('', p.stem)
            stem = re.sub(r'_[0-9a-f]{10}$', '', stem, flags=re.I)
            writer.writerow([p.as_posix(), p.stat().st_size, bool(SIZE_PREFIX.match(p.name)), normalize(stem)])
    report = {'files': len(files), 'extensions': dict(sorted(Counter(p.suffix.lower() for p in files).items())),
              'raster_images': len(images), 'original_image_files': len(originals),
              'size_variants': len(images) - len(originals),
              'catalog_provided': args.catalog is not None,
              'training_status': 'not_started',
              'note': 'Filename matches are candidates, not verified wine labels.'}
    if args.catalog:
        with args.catalog.open(encoding='utf-8-sig', newline='') as stream:
            sample = stream.readline()
            stream.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(stream, dialect=dialect, doublequote=True)
            if not {'Slug', 'Название фото'} <= set(reader.fieldnames or []):
                parser.error('CSV requires Slug and Название фото columns')
            rows = list(reader)
            if any(None in row or any(v is None for v in row.values()) for row in rows):
                parser.error('Malformed CSV: inconsistent number of columns')
        unique = {json.dumps(row, sort_keys=True, ensure_ascii=False): row for row in rows}.values()
        links, hashes = [], defaultdict(set)
        hash_cache = {}
        for row in unique:
            name = row['Название фото']
            candidates = by_name.get(name, []) or by_normalized.get(normalize(Path(name).stem), [])
            status = 'missing' if not candidates else ('candidate' if len(candidates) == 1 else 'ambiguous')
            for p in candidates:
                if p not in hash_cache:
                    hash_cache[p] = sha256(p)
                hashes[hash_cache[p]].add(row['Slug'])
            links.append({'slug': row['Slug'], 'photo_name': name, 'status': status,
                          'verified': False, 'candidates': [p.as_posix() for p in candidates],
                          'sha256': [hash_cache[p] for p in candidates], 'card': row})
        conflicts = {h: sorted(slugs) for h, slugs in hashes.items() if len(slugs) > 1}
        for link in links:
            link['cross_slug_content_conflict'] = any(h in conflicts for h in link['sha256'])
        (args.out / 'candidate_links.json').write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.out / 'content_conflicts.json').write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding='utf-8')
        report.update(csv_rows=len(rows), unique_rows=len(links), unique_slugs=len({x['slug'] for x in links}),
                      link_statuses=dict(Counter(x['status'] for x in links)), content_conflict_groups=len(conflicts))
    else:
        report['blocker'] = 'Missing CSV/JSON catalog connecting image names to real wine slugs.'
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
