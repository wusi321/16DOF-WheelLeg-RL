"""Summarize paired robustness trials, enforce gates, and draw saved curves."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'logs/mxz_robust'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def aggregate(rows):
    result = []
    for scenario in dict.fromkeys(r['scenario'] for r in rows):
        for force in sorted({r['force_n'] for r in rows}):
            selected = [r for r in rows if r['scenario'] == scenario and r['force_n'] == force]
            if not selected: continue
            n = sum(r['envs'] for r in selected)
            ne = sum(r['push_events'] for r in selected)
            result.append({'scenario': scenario, 'force_n': force, 'episodes': n,
                'episode_success_rate': sum(r['episode_success_rate']*r['envs'] for r in selected)/n,
                'push_events': ne,
                'recovery_rate': sum((r['recovery_rate'] or 0)*r['push_events'] for r in selected)/ne if ne else None,
                'vx_mae': float(np.mean([r['vx_mae'] for r in selected if r['vx_mae'] is not None])),
                'yaw_mae': float(np.mean([r['yaw_mae'] for r in selected if r['yaw_mae'] is not None])),
                'nonfinite_frames': sum(r.get('nonfinite_frames', 0) + r.get('termination_counts', {}).get('nan_detection', 0) for r in selected),
                'max_roll_pitch_deg': np.max([r['roll_pitch_abs_max_deg'] for r in selected], axis=0).tolist(),
                'max_calf_penetration_rad': max(r['max_calf_penetration_rad'] for r in selected)})
    return result


def gates(rows, baseline_rows, force):
    results = []
    for row in rows:
        if row['force_n'] != force: continue
        failures = []
        if any(r['nonfinite_frames'] for r in rows): failures.append('nonfinite state observed')
        if row['episode_success_rate'] < .95: failures.append('episode_success < 95%')
        if row['recovery_rate'] is None or row['recovery_rate'] < .95: failures.append('push_recovery < 95%')
        own_nominal = next((r for r in rows if r['force_n'] == 0 and r['scenario'] == row['scenario']), None)
        ref = next((r for r in baseline_rows if r['force_n'] == 0 and r['scenario'] == row['scenario']), None)
        if own_nominal is None or ref is None:
            failures.append('missing no-push regression data')
        else:
            if own_nominal['episode_success_rate'] < .95: failures.append('nominal episode_success < 95%')
            if own_nominal['vx_mae'] > ref['vx_mae']+.02: failures.append('nominal vx regression > .02m/s')
            if own_nominal['yaw_mae'] > ref['yaw_mae']+.05: failures.append('nominal yaw regression > .05rad/s')
        results.append({'scenario': row['scenario'], 'force_n': force, 'pass': not failures, 'failures': failures})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=OUTPUT/'baseline_plane/results.json')
    ap.add_argument('--baseline', type=Path, default=OUTPUT/'baseline_plane/results.json')
    ap.add_argument('--gate-force', type=float, default=40.)
    ap.add_argument('--select', action='store_true', help='select this controller only after complete acceptance')
    args = ap.parse_args()
    data = read(args.results)
    baseline_data = read(args.baseline)
    rows, baseline = aggregate(data['rows']), aggregate(baseline_data['rows'])
    gate = gates(rows, baseline, args.gate_force)
    if data.get('terrain') != baseline_data.get('terrain'):
        for g in gate:
            g['pass'] = False
            g['failures'].append('candidate and baseline terrain do not match')
    expected = {'stop', 'forward04', 'forward08', 'left', 'right', 'switch'}
    complete = all(len({r['seed'] for r in data['rows'] if r['scenario'] == s and r['force_n'] == f
        and r['envs'] >= 32 and r['steps'] >= 1500}) >= 3 for s in expected for f in (0, 20, 40, 60))
    result = {'complete_matrix': complete, 'aggregate': rows, 'gate_force_n': args.gate_force,
              'gate': gate, 'passed': bool(complete and len(gate) == 6 and all(g['pass'] for g in gate))}
    matrix_key = lambda r: (r['scenario'], r['seed'], r['force_n'], r['envs'], r['steps'])
    result['paired_matrix'] = {matrix_key(r) for r in data['rows']} == {matrix_key(r) for r in baseline_data['rows']}
    result['passed'] &= result['paired_matrix']
    light_gate = gates(rows, baseline, 20.)
    result['light_20N_gate'] = light_gate
    result['passed'] &= len(light_gate) == 6 and all(g['pass'] for g in light_gate)
    dest = args.results.parent
    if not dest.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Reports must stay inside the F-drive project')
    (dest/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if args.select:
        if not result['passed']:
            raise SystemExit('Selection refused: acceptance matrix incomplete or failed. Original remains default.')
        bundle = ROOT/'checkpoints/mxz_robust'/dest.name
        bundle.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        original_path = ROOT/'vendor/mxz/checkpoints/model_1795.pt'
        for mode, source in [('straight', data['straight']), ('turn', data['turn']), ('original_turn', original_path)]:
            if source is None: continue
            source = Path(source)
            target = bundle/(mode+'.pt')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if mode != 'original_turn' and data.get('evaluated_weight_sha256', {}).get(mode) != digest:
                raise ValueError(f'Evaluated weight identity missing or changed: {mode}')
            if mode == 'original_turn' and digest != data['checkpoint_sha256']:
                raise ValueError('Original comparison weight changed')
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Refusing to overwrite a different accepted weight: {target}')
            if not target.exists(): shutil.copy2(source, target)
            artifacts[mode] = {'path': str(target.resolve()), 'sha256': digest, 'source': str(source.resolve())}
        shutil.copy2(ROOT/'assets/16dof/training_zero.json', bundle/'training_zero.json')
        shutil.copy2(ROOT/'tools/mxz_robust.py', bundle/'mxz_robust_snapshot.py')
        shutil.copy2(ROOT/'docs/MXZ_RANDOM_PUSH.md', bundle/'README.md')
        (bundle/'manifest.json').write_text(json.dumps({'artifacts': artifacts,
            'evaluation': str(args.results.resolve()), 'baseline': str(args.baseline.resolve()),
            'source_commit': '7f97d153240c22f796c80421e77bf0a86dc0e6f2',
            'status': '20N/40N simulation acceptance passed; 60N stress-only',
            'stand': 'original zero-action PD', 'straight': 'bounded residual on deterministic wheel feed-forward',
            'turn': 'fine-tuned original actor, all wheel targets zero',
            'requires': 'existing project environment and original MXZ source/assets; these checkpoints alone are not a standalone robot simulator'}, indent=2), encoding='utf-8')
        selection = {'straight': artifacts['straight']['path'] if data['straight'] else None,
                     'turn': artifacts['turn']['path'] if data['turn'] else None,
                     'evaluation': str(args.results.resolve()), 'bundle': str(bundle.resolve()),
                     'artifacts': artifacts,
                     'status': 'passed simulated 40N/0.2s acceptance; 60N is stress-only',
                     'checkpoint_sha256': data['checkpoint_sha256']}
        (OUTPUT/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for scenario in dict.fromkeys(r['scenario'] for r in rows):
        selected = [r for r in rows if r['scenario'] == scenario]
        x = [r['force_n'] for r in selected]
        axes[0, 0].plot(x, [r['episode_success_rate']*100 for r in selected], 'o-', label=scenario)
        pushed = [r for r in selected if r['recovery_rate'] is not None]
        axes[0, 1].plot([r['force_n'] for r in pushed], [r['recovery_rate']*100 for r in pushed], 'o-', label=scenario)
        axes[1, 0].plot(x, [r['vx_mae'] for r in selected], 'o-', label=scenario)
        axes[1, 1].plot(x, [r['yaw_mae'] for r in selected], 'o-', label=scenario)
    for ax, title, unit in zip(axes.flat,
        ['Episode success (falls stay failed)', 'Push recovery: <=2s, hold 0.5s', 'Forward speed error', 'Yaw speed error'],
        ['Success (%)', 'Recovered (%)', 'MAE (m/s)', 'MAE (rad/s)']):
        ax.set(title=title, xlabel='Horizontal push (N), duration 0.2s', ylabel=unit)
        ax.grid(alpha=.3)
    for ax in axes[0]:
        ax.axhline(95, color='red', ls='--', lw=1)
        ax.set_ylim(0, 105)
    axes[0, 0].legend(ncol=2)
    fig.savefig(dest/'robustness_curves.png', dpi=160)
    plt.close(fig)

    order = ['stop', 'forward04', 'forward08', 'left', 'right', 'switch']
    have = [s for s in order if any(r['scenario'] == s and r['force_n'] == 40 for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(have))
    for values, shift, label, color in ((baseline, -.18, 'Original', '#9a9fa5'), (rows, .18, 'Candidate', '#2677c5')):
        for ax, force, field, multiplier, title in (
            (axes[0, 0], 40, 'recovery_rate', 100, '40N push recovery (2s + 0.5s hold)'),
            (axes[0, 1], 40, 'episode_success_rate', 100, '40N episodes without falls'),
            (axes[1, 0], 0, 'vx_mae', 1, 'No-push forward speed MAE (m/s)'),
            (axes[1, 1], 0, 'yaw_mae', 1, 'No-push yaw speed MAE (rad/s)')):
            y = [next((r[field]*multiplier for r in values if r['scenario'] == s and r['force_n'] == force), np.nan) for s in have]
            ax.bar(x+shift, y, .34, label=label, color=color)
            ax.set(title=title, xticks=x, xticklabels=have)
            ax.grid(axis='y', alpha=.25)
    for ax in axes[0]:
        ax.axhline(95, color='red', ls='--', lw=1)
        ax.set_ylim(0, 105)
    axes[0, 0].legend()
    fig.savefig(dest/'before_after.png', dpi=160)
    plt.close(fig)

    labels = {'stop': '停止', 'forward04': '直行 0.4m/s', 'forward08': '直行 0.8m/s',
              'left': '左转 +0.8rad/s', 'right': '右转 −0.8rad/s', 'switch': '模式切换'}
    lines = ['# 随机推力验收结果', '',
        f"完整矩阵：{'是' if complete else '尚未完成'}；20/40N 正式验收：{'通过' if result['passed'] else '未通过'}。",
        f"地面：{data.get('terrain', '旧版本未记录')}。每场景每档 32 环境 × 3 种子 × 30 秒。", '',
        '原始模型与候选使用相同的推力时序和方向。恢复要求：推力结束后 2 秒内达到姿态/速度标准并保持 0.5 秒。', '',
        '| 场景 | 原模型 40N 恢复率 | 候选 40N 恢复率 | 候选无摔倒率 | 候选无推力 vx误差 | 候选无推力 yaw误差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s in have:
        b = next(r for r in baseline if r['scenario'] == s and r['force_n'] == 40)
        r = next(r for r in rows if r['scenario'] == s and r['force_n'] == 40)
        nominal = next(r for r in rows if r['scenario'] == s and r['force_n'] == 0)
        lines.append(f"| {labels[s]} | {b['recovery_rate']:.1%} | {r['recovery_rate']:.1%} | {r['episode_success_rate']:.1%} | {nominal['vx_mae']:.4f}m/s | {nominal['yaw_mae']:.4f}rad/s |")
    lines += ['', f"总仿真过程数：{sum(r['envs'] for r in data['rows'])}；总推力事件数：{sum(r['push_events'] for r in data['rows'])}。",
        f"实际小腿限位最大穿透：{max(r['max_calf_penetration_rad'] for r in rows):.6f}rad。",
        '滤波后的动作目标每步检查物理范围；NaN/终止统计保存在 results.json，首次摔倒后该环境持续计失败。', '',
        f"直行残差：`{data['straight']}`；转向策略：`{data['turn'] or '原 model_1795.pt'}`。", '',
        '原模型右转的低恢复率表示未连续满足本报告严格的速度/姿态门槛，不表示它无法转向，也不等于摔倒率。', '',
        '60N 是压力测试；单次脉冲为 0.2 秒，间隔 5～8 秒，作用点在机身质心上方 8cm。',
        '本结论只覆盖这些仿真工况，不扩展到任意扰动、未知地形或实机。', '',
        '对比曲线：before_after.png；各力档曲线：robustness_curves.png；模式切换时序：switch_trace_40N.png。',
        '原有限方块场地结果留作诊断；验收必须使用与候选相同的无限平面对照。']
    if data.get('compositional_evaluation'):
        lines += ['', '组合验收的数据来源：直行/停车权重及控制契约未变，复用已完成的直行/停车测试；',
            '左右转向和模式切换全部由最终两份权重重新测试。合并前核对直行权重 SHA256、推力/推理代码 AST、限位、地面及全部矩阵键。',
            f"直行/停车来源：`{data['compositional_evaluation']['straight_trials']}`。",
            f"转向/切换来源：`{data['compositional_evaluation']['turn_and_switch_trials']}`。"]
    (dest/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    sample = dest/'switch_seed42.npz'
    if sample.exists():
        v = np.load(sample)['trace']
        i = v.shape[1]//2  # first robot in the 40N group of default four groups
        t = np.arange(len(v))*.02
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        ax[0].plot(t, v[:, i, 0], label='actual vx')
        ax[0].plot(t, v[:, i, 6], '--', label='command vx')
        ax[0].set_ylabel('m/s')
        ax[1].plot(t, v[:, i, 2], label='actual yaw')
        ax[1].plot(t, v[:, i, 7], '--', label='command yaw')
        ax[1].set_ylabel('rad/s')
        ax[2].plot(t, np.rad2deg(v[:, i, 4]), label='roll')
        ax[2].plot(t, np.rad2deg(v[:, i, 5]), label='pitch')
        ax[2].set(xlabel='Simulation time (s)', ylabel='degrees')
        for ev in read(dest/'switch_seed42_pushes.json'):
            if ev['env'] == i:
                for a in ax: a.axvspan(ev['step']*.02, ev['end_step']*.02, color='red', alpha=.2)
        for a in ax:
            a.grid(alpha=.3)
            a.legend(loc='upper right')
        fig.savefig(dest/'switch_trace_40N.png', dpi=160)
        plt.close(fig)

    # Training rewards use different weights between stages; keep each stage
    # separate and label them. They are learning diagnostics, not acceptance.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    runs = sorted(p for p in OUTPUT.glob('*stage*') if p.is_dir() and list(p.glob('events.out.tfevents.*')))
    if runs:
        fig, axes = plt.subplots(len(runs), 3, figsize=(13, 3.5*len(runs)), squeeze=False, constrained_layout=True)
        training_rows = {}
        for i, run in enumerate(runs):
            acc = EventAccumulator(str(run), size_guidance={'scalars': 0})
            acc.Reload()
            for j, tag in enumerate(('Train/mean_reward', 'Metrics/twist/error_vel_xy', 'Train/mean_episode_length')):
                if tag not in acc.Tags()['scalars']: continue
                series = acc.Scalars(tag)
                axes[i, j].plot([s.step for s in series], [s.value for s in series])
                axes[i, j].set(title=run.name+'\n'+tag, xlabel='Local iteration')
                axes[i, j].grid(alpha=.3)
                training_rows.setdefault(run.name, {})[tag] = [{'iteration': s.step, 'value': s.value} for s in series]
        fig.savefig(OUTPUT/'training_curves.png', dpi=150)
        (OUTPUT/'training_curves.json').write_text(json.dumps(training_rows), encoding='utf-8')
        plt.close(fig)


if __name__ == '__main__':
    main()
