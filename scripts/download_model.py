"""Download a fixed snapshot to the project; record its immutable revision."""
import json
import hashlib
import time
import urllib.request
from pathlib import Path

model_id = 'google/siglip2-base-patch16-384'
with urllib.request.urlopen(f'https://huggingface.co/api/models/{model_id}?blobs=true', timeout=60) as response:
    info = json.load(response)
    revision = info['sha']
expected_hash = next((x.get('lfs', {}).get('sha256') for x in info.get('siblings', [])
                      if x['rfilename'] == 'model.safetensors'), None)
target = Path('artifacts/base_model')
target.mkdir(parents=True, exist_ok=True)
for name in ['config.json', 'preprocessor_config.json', 'model.safetensors']:
    destination = target / name
    if destination.exists() and (target/'source.json').exists():
        previous = json.loads((target/'source.json').read_text())
        if previous['revision'] == revision:
            continue
    partial = target / (name+'.part')
    url = f'https://huggingface.co/{model_id}/resolve/{revision}/{name}'
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                mode = 'ab' if response.status == 206 and offset else 'wb'
                with partial.open(mode) as stream:
                    while chunk := response.read(1024*1024):
                        stream.write(chunk)
            partial.replace(destination)
            print(f'Downloaded {name}: {destination.stat().st_size} bytes', flush=True)
            break
        except Exception as exc:
            print(f'{name}: attempt {attempt+1} failed: {type(exc).__name__}', flush=True)
            if attempt == 4:
                raise
            time.sleep(2)
with (target/'model.safetensors').open('rb') as stream:
    actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
if expected_hash and actual_hash != expected_hash:
    raise ValueError('Downloaded weights failed SHA256 verification')
(target / 'source.json').write_text(json.dumps({'model_id': model_id, 'revision': revision,
                                              'weights_sha256': actual_hash}, indent=2))
print(f'Downloaded {model_id}@{revision}', flush=True)
