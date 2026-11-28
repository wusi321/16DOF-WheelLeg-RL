"""Reuse proven straight-only trials after changing only the turn network.

Switching trials must always come from the new combined controller. This checks
the actual inference/force source contracts, checkpoint hashes, and matrix keys.
It records the originating result file for every reused row.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def signature(folder):
    tree = ast.parse((folder/'evaluation_snapshot.py').read_text(encoding='utf-8'))
    names = {'bounded_spec', 'bounded_servo', 'residual_actions', 'HybridController', 'HorizontalPushes', 'command_for'}
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--straight-results', type=Path, required=True)
    ap.add_argument('--turn-results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT.resolve())
    straight, turn = read(args.straight_results), read(args.turn_results)
    assert Path(straight['straight']).resolve() == Path(turn['straight']).resolve(), 'Straight checkpoint changed'
    assert signature(args.straight_results.parent) == signature(args.turn_results.parent), 'Inference/force source contract changed'
    for key in ('terrain', 'dt', 'checkpoint_sha256', 'force_duration_s', 'force_frame', 'application_point', 'limits'):
        assert straight[key] == turn[key], f'Environment/evaluation contract changed: {key}'
    for mode in ('straight', 'turn'):
        checkpoint = Path(turn[mode])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert turn['evaluated_weight_sha256'][mode] == digest
        # Prior straight-only trials may predate per-evaluation SHA metadata.
        # The original training manifest binds the immutable final checkpoint.
        if mode == 'straight':
            manifest = read(checkpoint.parent/'manifest.json')
            assert manifest['final_sha256'] == digest
            if straight.get('evaluated_weight_sha256', {}).get('straight'):
                assert straight['evaluated_weight_sha256']['straight'] == digest
            assert checkpoint.stat().st_mtime < min(p.stat().st_mtime for p in args.straight_results.parent.glob('forward*_seed*.npz'))
    output.mkdir(parents=True, exist_ok=False)
    rows, seen = [], set()
    for source, data, scenarios in (
        (args.straight_results, straight, {'stop', 'forward04', 'forward08'}),
        (args.turn_results, turn, {'left', 'right', 'switch'})):
        for row in data['rows']:
            if row['scenario'] not in scenarios: continue
            key = (row['scenario'], row['seed'], row['force_n'])
            assert key not in seen
            seen.add(key)
            rows.append({**row, 'origin_results': str(source.resolve())})
        for scenario in scenarios:
            for file in source.parent.glob(scenario+'_seed*'):
                shutil.copy2(file, output/file.name)
        shutil.copy2(source.parent/'evaluation_snapshot.py', output/(source.parent.name+'_snapshot.py'))
    assert len(seen) == 72, f'Incomplete matrix: {len(seen)}/72 rows'
    merged = {**turn, 'rows': rows, 'compositional_evaluation': {
        'straight_trials': str(args.straight_results.resolve()),
        'turn_and_switch_trials': str(args.turn_results.resolve()),
        'reason': 'Only the turn network changed; pure straight/stand never invoke it. All mode-switch trials were rerun with both final networks.',
        'checks': 'same straight checkpoint with training SHA256; same inference and force AST; same terrain, dt, forces, original checkpoint; unique complete matrix'}}
    (output/'results.json').write_text(json.dumps(merged, indent=2), encoding='utf-8')
    shutil.copy2(args.turn_results.parent/'evaluation_snapshot.py', output/'evaluation_snapshot.py')
    shutil.copy2(__file__, output/'merge_snapshot.py')
    print(f'Merged {len(rows)} trial rows into {output}', flush=True)


if __name__ == '__main__':
    main()
