"""Verify delivered bytes, checkpoint identities, and framework import location."""
import hashlib
from importlib import metadata, util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = (ROOT/'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    for line in lines:
        expected, name = line.split('  ', 1)
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT.resolve()), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Changed or damaged file: {name}'
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for artifact in manifest['weights'].values():
        assert hashlib.sha256((ROOT/artifact['path']).read_bytes()).hexdigest() == artifact['sha256']
    framework = Path(util.find_spec('mjlab').origin).resolve()
    assert framework.is_relative_to(ROOT/'vendor/mjlab/src'), f'Wrong mjlab imported: {framework}'
    versions = {n: metadata.version(n) for n in ('torch','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','numpy','tensordict')}
    print(json.dumps({'verified_files': len(lines), 'description': manifest['description'],
                     'framework': str(framework), 'versions': versions}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
