"""Reproducible horizontal-push evaluation and conservative MXZ adaptation.

The external source/checkpoint is immutable. Imports are intentionally isolated
from this project's different `robot` package. All artifacts stay in this project.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/mxz_7f97d153/source/站立行走转弯/mxz'
CHECKPOINT = SOURCE / 'checkpoints/model_1795.pt'
OUTPUT = ROOT / 'logs/mxz_robust'
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'tools')]

import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml
import robot
from robot import naming
from eval_policy import load_policy, pin_command

REFERENCE = json.loads((ROOT / 'assets/16dof/training_zero.json').read_text(encoding='utf-8'))
LIMITS = REFERENCE['calf_limits_rad']
EXPECTED_HASH = 'e4bdaa2653842d700a9b03594ba30fd7e01736254071cf19d4339fb7d2f2c9c5'
SCENARIOS = {'stop': (0., 0., 0.), 'forward04': (.4, 0., 0.),
             'forward08': (.8, 0., 0.), 'left': (0., 0., .8),
             'right': (0., 0., -.8), 'switch': (.4, 0., 0.)}
SWITCH = [(.4, 0., 0.), (0., 0., .8), (.8, 0., 0.), (0., 0., -.8), (0., 0., 0.)]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def check_source():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_HASH, 'Source checkpoint changed'
    assert Path(robot.__file__).resolve().is_relative_to(SOURCE.resolve()), robot.__file__


def bounded_spec(original):
    def build():
        spec = original()
        for name, bounds in LIMITS.items():
            spec.joint(name).range = bounds
            spec.joint(name).limited = mujoco.mjtLimited.mjLIMITED_TRUE
        return spec
    return build


def bounded_servo(self, spec, names):
    BuiltinPositionActuator.edit_spec(self, spec, names)
    for act in spec.actuators:
        if act.target in names and spec.joint(act.target).limited:
            act.inheritrange = 0.
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = spec.joint(act.target).range


class BoundedPositionCfg(BuiltinPositionActuatorCfg):
    def build(self, entity, target_ids, target_names):
        instance = super().build(entity, target_ids, target_names)
        # Preserve builtin type: mjlab dispatches its fused implementation by type.
        instance.edit_spec = MethodType(bounded_servo, instance)
        return instance


def configure(num_envs, seed, training=False, mode='straight'):
    task = 'Robot-WalkTurn-Locked-v0' if mode == 'turn' else 'Robot-WalkTurn-Straight-v0'
    cfg = load_env_cfg(task, play=not training)
    # Hybrid inference always uses the original generic wheel scale 10.
    cfg.actions['wheel_joint_vel'].scale = 10.
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = 20. if training else 100000.
    cfg.auto_reset = training
    cfg.events.pop('push_robot', None)
    if training and mode == 'straight':
        cfg.rewards['joint_pos_penalty'].weight = -2.
        cfg.rewards['flat_orientation'].weight = -.8
        cfg.rewards['base_height_l2'].weight = -40.
    asset = cfg.scene.entities['robot']
    groups = tuple(BoundedPositionCfg(**vars(a)) if isinstance(a, BuiltinPositionActuatorCfg) else a
                   for a in asset.articulation.actuators)
    cfg.scene.entities['robot'] = replace(asset, spec_fn=bounded_spec(asset.spec_fn),
        articulation=replace(asset.articulation, actuators=groups))
    spec = cfg.scene.entities['robot'].spec_fn()
    cfg.actions['leg_joint_pos'].clip = {j.name: tuple(float(v) for v in j.range) for j in spec.joints
        if j.name.endswith('_joint') and 'wheel' not in j.name}
    cfg.actions['wheel_joint_vel'].clip = {'.*': (-20., 20.)}
    # Training keeps source noise, 0--2 step delays, and 0.8--1.2 wheel friction.
    cfg.sim.nan_guard.enabled = True
    cfg.sim.nan_guard.output_dir = str(OUTPUT / 'nan_dumps')
    return cfg


def verify(env):
    model = env.sim.mj_model
    assert model.nu == 16
    records = []
    leg = env.action_manager._terms['leg_joint_pos']
    assert len(leg.target_names) == 12
    assert [n.split('_')[0] for n in leg.target_names[::3]] == ['FR', 'FL', 'RL', 'RR']
    for name, bounds in LIMITS.items():
        jid = model.joint('robot/' + name).id
        np.testing.assert_allclose(model.jnt_range[jid], bounds)
        aids = np.flatnonzero(model.actuator_trnid[:, 0] == jid)
        assert len(aids) == 1
        aid = int(aids[0])
        assert model.actuator_ctrllimited[aid]
        np.testing.assert_allclose(model.actuator_ctrlrange[aid], bounds)
        records.append({'joint': name, 'physical_range': list(bounds), 'target_range': list(bounds)})
    for value in (-100., 100.):
        for _ in range(5):
            env.action_manager.process_action(torch.full((env.num_envs, 16), value, device=env.device))
        check_targets(env)
    env.reset(seed=42)
    assert abs(env.step_dt - .02) < 1e-9, 'Push schedule assumes 50 Hz'
    # Verify a full 8-direction pulse at every force level, without simulating.
    probe = HorizontalPushes(env, 40., seed=100)
    probe.wait[:] = 100000
    for direction in range(8):
        angle = direction * math.pi / 4
        f = torch.tensor([40*math.cos(angle), 40*math.sin(angle), 0.], device=env.device)
        probe.remaining[:] = 10
        probe.force[:] = f
        for _ in range(10):
            probe.step()
            torch.testing.assert_close(probe.force[:, 0], f.expand(env.num_envs, 3))
            assert (probe.force[:, :, 2] == 0).all()
        probe.step()
        assert not probe.force.any(), 'A force persisted past its 0.2s duration'
    probe.force[:] = 40
    probe.reset()
    probe.step()
    assert not probe.force.any(), 'Reset must clear outstanding forces'
    # The zero residual must reproduce baseline, and saturation must be bounded.
    from tensordict import TensorDict
    values = torch.zeros(env.num_envs, 53, device=env.device)
    values[:, 6] = .4
    obs = TensorDict({'actor': values}, batch_size=[env.num_envs])
    base = residual_actions(obs, torch.zeros(env.num_envs, 16, device=env.device))
    assert not base[:, :12].any()
    for sign in (-1, 1):
        delta = residual_actions(obs, torch.full_like(base, sign*100.))-base
        torch.testing.assert_close(delta[:, :12].abs(), torch.tensor([.64, .48, .48]*4, device=env.device).expand(env.num_envs, -1))
        torch.testing.assert_close(delta[:, 12:].abs(), torch.full_like(delta[:, 12:], .2))
    return records


def check_targets(env):
    term = env.action_manager._terms['leg_joint_pos']
    target = term._processed_actions
    assert torch.isfinite(target).all()
    assert (target >= term._clip[:, :, 0] - 1e-6).all()
    assert (target <= term._clip[:, :, 1] + 1e-6).all()


class HorizontalPushes:
    """World-horizontal 8-direction forces at COM+(body z)*8cm.

    Before-step scheduling gives exactly round(duration/dt) forced steps.
    Separate RNG is unaffected by policy/reset random number consumption.
    This owns only the base body's external wrench; GUI mouse qfrc is separate.
    """
    def __init__(self, env, magnitudes, seed=42, duration=.2, training=False):
        self.env, self.asset = env, env.scene['robot']
        self.body_ids = self.asset.find_bodies(('base',), preserve_order=True)[0]
        self.magnitudes = np.broadcast_to(np.asarray(magnitudes, dtype=float), (env.num_envs,)).copy()
        self.rng = np.random.default_rng(seed)
        self.duration = max(1, round(duration / env.step_dt))
        self.training = training
        self.enabled = True
        self.force = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self.offset = torch.tensor([0., 0., .08], device=env.device)
        self.remaining = np.zeros(env.num_envs, dtype=int)
        self.wait = np.full(env.num_envs, round(5. / env.step_dt), dtype=int)
        self.episode_enabled = np.ones(env.num_envs, dtype=bool)
        if training:
            self.episode_enabled = self.rng.random(env.num_envs) >= .25
        self.starts = np.zeros(env.num_envs, dtype=bool)
        self.direction = np.zeros(env.num_envs, dtype=int)

    def reset(self, ids=None):
        ids = np.arange(self.env.num_envs) if ids is None else np.asarray(ids, dtype=int)
        self.remaining[ids] = 0
        self.wait[ids] = round(5. / self.env.step_dt)
        self.force[torch.as_tensor(ids, device=self.env.device)] = 0
        if self.training:
            self.episode_enabled[ids] = self.rng.random(len(ids)) >= .25

    def step(self):
        self.starts[:] = False
        self.force[self.remaining == 0] = 0
        eligible = (self.remaining == 0) & (self.wait <= 0)
        if self.enabled:
            ids = np.flatnonzero(eligible)
            # Draw even for zero-force controls, preserving paired schedules.
            for i in ids:
                direction = int(self.rng.integers(8))
                self.direction[i] = direction
                angle = direction * math.pi / 4
                mag = self.magnitudes[i] if self.episode_enabled[i] else 0.
                self.force[i, 0] = torch.tensor([mag * math.cos(angle), mag * math.sin(angle), 0.], device=self.env.device)
                self.remaining[i] = self.duration
                self.wait[i] = self.duration + int(self.rng.integers(round(5/self.env.step_dt), round(8/self.env.step_dt)+1))
                self.starts[i] = mag > 0
        if not self.enabled:
            self.force.zero_()
            self.remaining[:] = 0
        quat = self.asset.data.body_com_quat_w[:, self.body_ids].reshape(-1, 4)
        offset_w = quat_apply(quat, self.offset.expand(self.env.num_envs, 3)).reshape(-1, 1, 3)
        torque = torch.cross(offset_w, self.force, dim=-1)
        self.asset.write_external_wrench_to_sim(self.force, torque, body_ids=self.body_ids)
        self.remaining = np.maximum(0, self.remaining - 1)
        self.wait -= 1

    def debug_vis(self, visualizer):
        for i in visualizer.get_env_indices(self.env.num_envs):
            f = self.force[i, 0].cpu().numpy()
            if np.linalg.norm(f) < 1:
                continue
            pos = self.asset.data.body_com_pos_w[i, self.body_ids[0]]
            q = self.asset.data.body_com_quat_w[i, self.body_ids[0]]
            start = (pos + quat_apply(q, self.offset)).cpu().numpy()
            visualizer.add_arrow(start=start, end=start+f*.005,
                                 color=(1., .15, .05, 1.), width=.015)


def residual_actions(obs, residual):
    """53D actor input; actual leg bound .08/.12rad, wheel bound 2rad/s."""
    values = obs['actor']
    out = torch.tanh(residual).clone()
    # Source scale .125 hip and .25 thigh/calf, in FR/FL/RL/RR order.
    out[:, :12] *= torch.tensor([.08/.125, .12/.25, .12/.25]*4, device=out.device)
    out[:, 12:16] *= .2
    out[:, 12:16] += values[:, 6:7] / naming.WHEEL_RADIUS * 1.14 / 10.
    return out


class HybridController:
    def __init__(self, wrapped, straight=None, turn=None):
        self.original = load_policy('Robot-WalkTurn-Locked-v0', wrapped, str(wrapped.device), CHECKPOINT)
        self.straight = load_policy('Robot-WalkTurn-Straight-v0', wrapped, str(wrapped.device), Path(straight)) if straight else None
        self.turn = load_policy('Robot-WalkTurn-Locked-v0', wrapped, str(wrapped.device), Path(turn)) if turn else self.original

    @torch.inference_mode()
    def __call__(self, obs, cmd):
        if abs(cmd[2]) < .1:
            if self.straight:
                return residual_actions(obs, self.straight(obs))
            actions = torch.zeros(obs.shape[0], 16, device=obs.device)
            actions[:, 12:16] = cmd[0] / naming.WHEEL_RADIUS * 1.14 / 10.
            return actions
        actions = self.turn(obs).clone()
        actions[:, 12:16] = 0.
        return actions


def command_for(scenario, step):
    return SWITCH[(step // 300) % len(SWITCH)] if scenario == 'switch' else SCENARIOS[scenario]


def set_command(env, cmd):
    pin_command(env, cmd)
    # Replace the observation's previous command before computing this action.
    return env.observation_manager.compute()


def recovery_summary(trace, failed, events, dt):
    # trace: steps, envs, [vx,vy,wz,z,roll,pitch,cmdvx,cmdwz]
    ok = ((np.abs(trace[:, :, 4:6]) < np.deg2rad(5)).all(-1) &
          (np.linalg.norm(trace[:, :, :2] - np.stack((trace[:, :, 6], np.zeros_like(trace[:, :, 6])), -1), axis=-1) < .08) &
          (np.abs(trace[:, :, 2] - trace[:, :, 7]) < .15) & ~failed)
    hold, deadline = round(.5/dt), round(2./dt)
    records = []
    for ev in events:
        i, start, end = ev['env'], ev['step'], ev['end_step']
        last = min(len(ok), end + deadline + hold)
        hit = None
        # Recovery must begin within 2s after the force ends, then last .5s.
        for s in range(end, min(end+deadline+1, last-hold+1)):
            if ok[s:s+hold, i].all():
                hit = (s-end)*dt
                break
        # Episode failure during the recovery window invalidates the event.
        if failed[start:min(last, end+deadline+hold), i].any():
            hit = None
        records.append({**ev, 'recovered': hit is not None, 'recovery_s': hit})
    return records


@torch.inference_mode()
def evaluate(args):
    from tensordict import TensorDict
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    forces = [float(x) for x in args.forces.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    scenarios = args.scenarios.split(',')
    n = args.num_envs * len(forces)
    cfg = configure(n, seeds[0])
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    rows = []
    try:
        verification = verify(env)
        policy = HybridController(wrapped, args.straight, args.turn)
        asset = env.scene['robot']
        leg = env.action_manager._terms['leg_joint_pos']
        calf_ids = asset.find_joints(list(LIMITS), preserve_order=True)[0]
        lo = torch.tensor([v[0] for v in LIMITS.values()], device=env.device)
        hi = torch.tensor([v[1] for v in LIMITS.values()], device=env.device)
        for scenario in scenarios:
            for seed in seeds:
                env.reset(seed=seed)
                pushes = HorizontalPushes(env, np.repeat(forces, args.num_envs), seed)
                # Pair the same push sequence across force groups by construction.
                schedule_rng = np.random.default_rng(seed)
                schedule = []
                for i in range(args.num_envs):
                    t = 250
                    while t + 10 + 125 <= args.steps:
                        schedule.append((t, i, int(schedule_rng.integers(8))))
                        t += 10 + int(schedule_rng.integers(250, 401))
                by_step = {}
                events = []
                for t, i, direction in schedule:
                    by_step.setdefault(t, []).append((i, direction))
                    for group, force in enumerate(forces):
                        if force > 0:
                            events.append({'step': t, 'end_step': t+10, 'env': group*args.num_envs+i,
                                           'force_n': force, 'direction': direction})
                pushes.wait[:] = args.steps + 100
                trace, failures, targets, actual = [], [], [], []
                ever_failed = torch.zeros(n, dtype=torch.bool, device=env.device)
                term_counts = {k: [0]*len(forces) for k in env.termination_manager.active_terms}
                max_penetration = torch.zeros(n, device=env.device)
                for step in range(args.steps):
                    cmd = command_for(scenario, step)
                    obs = TensorDict(set_command(env, cmd), batch_size=[n])
                    for i, direction in by_step.get(step, []):
                        angle = direction * math.pi / 4
                        for group, mag in enumerate(forces):
                            idx = group*args.num_envs + i
                            pushes.force[idx, 0] = torch.tensor([mag*math.cos(angle), mag*math.sin(angle), 0.], device=env.device)
                            pushes.remaining[idx] = pushes.duration
                    pushes.step()
                    action = policy(obs, cmd)
                    assert torch.isfinite(action).all()
                    _, _, done, extra = wrapped.step(action)
                    check_targets(env)
                    finite = torch.isfinite(env.sim.data.qpos).all(dim=-1) & torch.isfinite(env.sim.data.qvel).all(dim=-1)
                    ever_failed |= done.bool() | ~finite
                    for key in term_counts:
                        count = env.termination_manager.get_term(key).reshape(len(forces), -1).sum(1).cpu().tolist()
                        term_counts[key] = [a+b for a,b in zip(term_counts[key], count)]
                    g = asset.data.projected_gravity_b
                    state = torch.stack((asset.data.root_link_lin_vel_b[:, 0], asset.data.root_link_lin_vel_b[:, 1],
                        asset.data.root_link_ang_vel_b[:, 2], asset.data.root_link_pos_w[:, 2],
                        torch.atan2(-g[:, 1], -g[:, 2]), torch.asin(g[:, 0].clamp(-1, 1)),
                        torch.full((n,), cmd[0], device=env.device), torch.full((n,), cmd[2], device=env.device)), -1)
                    trace.append(state.cpu().numpy())
                    failures.append(ever_failed.cpu().numpy().copy())
                    q = asset.data.joint_pos[:, calf_ids]
                    max_penetration = torch.maximum(max_penetration, torch.maximum((lo-q).clamp_min(0), (q-hi).clamp_min(0)).max(1).values)
                    targets.append(leg._processed_actions[:, [leg.target_names.index(k) for k in LIMITS]].cpu().numpy())
                    actual.append(q.cpu().numpy())
                    # Capture terminal state/metrics first; a reset cannot erase failure.
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    if len(ids):
                        env.reset(env_ids=ids)
                        pushes.reset(ids.cpu().numpy())
                    if step and step % 500 == 0:
                        print(f'EVAL {scenario} seed={seed} step={step}/{args.steps}', flush=True)
                values, failed = np.asarray(trace), np.asarray(failures)
                event_records = recovery_summary(values, failed, events, env.step_dt)
                for group, force in enumerate(forces):
                    sl = slice(group*args.num_envs, (group+1)*args.num_envs)
                    v = values[:, sl]
                    valid = ~failed[:, sl]
                    valid[:250] = False
                    if scenario == 'switch':
                        for t in range(0, args.steps, 300):
                            valid[t:t+50] = False
                    err_v = np.abs(v[:, :, 0]-v[:, :, 6])
                    err_w = np.abs(v[:, :, 2]-v[:, :, 7])
                    evs = [e for e in event_records if sl.start <= e['env'] < sl.stop]
                    row = {'scenario': scenario, 'seed': seed, 'force_n': force, 'envs': args.num_envs,
                        'steps': args.steps, 'episode_success_rate': float((~failed[-1, sl]).mean()),
                        'recovery_rate': float(np.mean([e['recovered'] for e in evs])) if evs else None,
                        'push_events': len(evs), 'vx_mae': float(err_v[valid].mean()) if valid.any() else None,
                        'yaw_mae': float(err_w[valid].mean()) if valid.any() else None,
                        'roll_pitch_abs_max_deg': np.rad2deg(np.abs(v[:, :, 4:6])).max((0, 1)).tolist(),
                        'max_calf_penetration_rad': float(max_penetration[sl].max()),
                        'calf_actual_range': [np.asarray(actual)[:, sl].min().item(), np.asarray(actual)[:, sl].max().item()],
                        'calf_target_range': [np.asarray(targets)[:, sl].min().item(), np.asarray(targets)[:, sl].max().item()],
                        'termination_counts': {k: v[group] for k,v in term_counts.items()}}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                np.savez_compressed(out/f'{scenario}_seed{seed}.npz', trace=values, failed=failed,
                                    calf_target=np.asarray(targets), calf_actual=np.asarray(actual))
                write_json(out/f'{scenario}_seed{seed}_pushes.json', event_records)
                write_json(out/'results.json', {'rows': rows, 'limits': verification,
                    'checkpoint_sha256': EXPECTED_HASH, 'straight': args.straight, 'turn': args.turn,
                    'dt': env.step_dt, 'force_duration_s': .2, 'force_frame': 'world horizontal',
                    'application_point': 'base COM + body z * 0.08m', 'failed_episodes_remain_failed': True})
    finally:
        wrapped.close()
    return rows


class TrainingWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, mode, force, seed):
        self.mode = mode
        super().__init__(env)
        self.pushes = HorizontalPushes(env, force, seed, training=True)

    def step(self, actions):
        self.pushes.step()
        if self.mode == 'straight':
            actions = residual_actions(self.get_observations(), actions)
        else:
            actions = actions.clone()
            actions[:, 12:16] = 0.
        obs, reward, done, extra = super().step(actions)
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            self.pushes.reset(ids.cpu().numpy())
        return obs, reward, done, extra


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configure(args.num_envs, args.seed, training=True, mode=args.mode)
    task = 'Robot-WalkTurn-Locked-v0' if args.mode == 'turn' else 'Robot-WalkTurn-Straight-v0'
    agent = load_rl_cfg(task)
    agent.algorithm.learning_rate = 1e-4
    agent.algorithm.schedule = 'fixed'
    agent.save_interval = 50
    agent.max_iterations = args.iterations
    agent.upload_model = False
    if args.mode == 'straight':
        agent.actor.distribution_cfg['init_std'] = .2
    dump_yaml(out/'params/env.yaml', asdict(cfg))
    dump_yaml(out/'params/agent.yaml', asdict(agent))
    shutil.copy2(__file__, out/'mxz_robust_snapshot.py')
    shutil.copy2(ROOT/'assets/16dof/training_zero.json', out/'training_zero.json')
    shutil.copytree(SOURCE/'src', out/'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    write_json(out/'manifest.json', {'mode': args.mode, 'force_n': args.force, 'duration_s': .2,
        'iterations': args.iterations, 'seed': args.seed, 'num_envs': args.num_envs,
        'source_commit': '7f97d153240c22f796c80421e77bf0a86dc0e6f2',
        'original_checkpoint': str(CHECKPOINT), 'original_sha256': EXPECTED_HASH,
        'resume': args.checkpoint, 'optimizer_reset': True, 'no_push_episode_fraction': .25,
        'motor_limits_nm': {'hip_thigh': 40, 'calf': 60, 'wheel': 15},
        'straight_residual_limits': {'hip_rad': .08, 'thigh_calf_rad': .12, 'wheel_rad_s': 2.},
        'actor_observation': 'source 53D proprioception; no true linear velocity or force',
        'notes': 'straight: train residual on deterministic base; turn: original actor+critic, wheels zero'})
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = TrainingWrapper(env, args.mode, args.force, args.seed)
    try:
        verify(env)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), str(out), device=args.device)
        if args.checkpoint or args.mode == 'turn':
            runner.load(str(args.checkpoint or CHECKPOINT), load_cfg={'actor': True, 'critic': True}, map_location=args.device)
        else:
            # Zero residual mean starts at exactly the known deterministic base.
            last = [m for m in runner.alg.get_policy().mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()
        runner.current_learning_iteration = 0
        print(f'TRAIN mode={args.mode} force={args.force}N out={out}', flush=True)
        runner.learn(args.iterations, init_at_random_ep_len=False)
        runner.save(str(out/'final.pt'))
    finally:
        wrapped.close()


def view(args):
    """Live simulation/video, using the same force and controller code as tests."""
    from collections import deque
    from tensordict import TensorDict
    from mjlab.viewer import NativeMujocoViewer, ViewerConfig
    selection = OUTPUT/'selection.json'
    if selection.exists() and not args.original:
        chosen = json.loads(selection.read_text(encoding='utf-8'))
        args.straight = args.straight or chosen.get('straight')
        args.turn = args.turn or chosen.get('turn')
    cfg = configure(1, args.seed)
    cfg.auto_reset = True
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = 'robot'
    cfg.viewer.lookat = (0., 0., .24)
    cfg.viewer.distance = 2.4
    cfg.viewer.elevation = -18.
    cfg.viewer.azimuth = 135.
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.width, cfg.viewer.height = 960, 640
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode='rgb_array' if args.video else None)
    wrapped = RslRlVecEnvWrapper(env)
    verify(env)
    controller = HybridController(wrapped, args.straight, args.turn)
    pushes = HorizontalPushes(env, args.force, args.seed)
    original_visuals = env.update_visualizers
    def visuals(visualizer):
        original_visuals(visualizer)
        pushes.debug_vis(visualizer)
    env.update_visualizers = visuals

    class Demo:
        def __init__(self):
            self.keys = deque()
            self.auto = not args.manual
            self.cmd = (0., 0., 0.)
            self.steps = self.falls = 0
            self.label = ''

        def reset(self):
            self.steps = 0
            pushes.reset()

        def on_key(self, key):
            # Viewer thread queues commands; the sim thread alone mutates tensors.
            self.keys.append(key)

        def __call__(self, obs):
            while self.keys:
                key = self.keys.popleft()
                vx, _, wz = self.cmd
                if key in (87, 83, 65, 68, 88):
                    self.auto = False
                if key == 87: self.cmd = (min(.8, vx+.1), 0., 0.)
                elif key == 83: self.cmd = (max(0., vx-.1), 0., 0.)
                elif key == 65: self.cmd = (0., 0., min(.8, wz+.2))
                elif key == 68: self.cmd = (0., 0., max(-.8, wz-.2))
                elif key == 88: self.cmd = (0., 0., 0.)
                elif key == 84: self.auto = True
                elif key == 66: pushes.enabled = not pushes.enabled
                elif key == 70: pushes.magnitudes[:] = {20.: 40., 40.: 60., 60.: 20.}.get(float(pushes.magnitudes[0]), 20.)
                elif key == 85: pushes.duration = {10: 15, 15: 5, 5: 10}.get(pushes.duration, 10)
                elif key == 71: pushes.wait[:] = 0
            if self.auto:
                self.cmd = command_for('switch', self.steps)
            obs = TensorDict(set_command(env, self.cmd), batch_size=[1])
            pushes.step()
            self.steps += 1
            mag = float(pushes.force.norm().item())
            self.label = (f"{'STRAIGHT' if abs(self.cmd[2]) < .1 else 'TURN'}  "
                f"vx={self.cmd[0]:.1f} wz={self.cmd[2]:+.1f} | "
                f"push {'ON' if pushes.enabled else 'OFF'} {pushes.magnitudes[0]:.0f}N "
                f"{pushes.duration*env.step_dt:.2f}s active={mag:.0f}N | falls={self.falls}")
            if pushes.starts.any():
                print(self.label, flush=True)
            return controller(obs, self.cmd)

    demo = Demo()
    class DemoWrapper(RslRlVecEnvWrapper):
        def step(self, action):
            result = super().step(action)
            if result[2].any():
                demo.falls += int(result[2].sum())
                pushes.reset()
                print(f'FALL/TERMINATION count={demo.falls}; auto-reset for demo only', flush=True)
            return result
    # Reuse the existing wrapper without an extra environment reset.
    wrapped.__class__ = DemoWrapper
    class PushViewer(NativeMujocoViewer):
        def _set_status_overlay(self, handle):
            s = self.get_status()
            handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                f"{'PAUSED' if s.paused else 'RUNNING'} | {s.actual_realtime:.2f}x\n{demo.label}\n"
                'W/S speed | A/D turn | X stop | T auto\nB push on/off | F force | U duration | G push now\nEnter reset | Space pause | drag mouse to rotate', ''))
    print('LIVE: W/S speed, A/D turn, X stop, T auto; B pushes, F 20/40/60N, U duration, G push now.', flush=True)
    try:
        if args.headless or args.video:
            writer = None
            if args.video:
                import imageio.v2 as imageio
                Path(args.video).parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(args.video, fps=25, codec='libx264', quality=7)
            try:
                for step in range(args.steps or 750):
                    wrapped.step(demo(wrapped.get_observations()))
                    if writer and step % 2 == 0:
                        from PIL import Image, ImageDraw
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 960, 32), fill=(0, 0, 0))
                        draw.text((8, 8), demo.label, fill=(255, 255, 255))
                        writer.append_data(np.asarray(frame))
                print(demo.label, flush=True)
            finally:
                if writer: writer.close()
        else:
            PushViewer(wrapped, demo, key_callback=demo.on_key).run(num_steps=args.steps or None)
    finally:
        wrapped.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='operation', required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--num-envs', type=int, default=32, help='envs per force group')
    ev.add_argument('--steps', type=int, default=1500)
    ev.add_argument('--seeds', default='42,43,44')
    ev.add_argument('--forces', default='0,20,40,60')
    ev.add_argument('--scenarios', default=','.join(SCENARIOS))
    ev.add_argument('--straight')
    ev.add_argument('--turn')
    ev.add_argument('--output', default=str(OUTPUT/'baseline'))
    tr = sub.add_parser('train')
    tr.add_argument('--mode', choices=['straight', 'turn'], required=True)
    tr.add_argument('--force', type=float, default=20)
    tr.add_argument('--num-envs', type=int, default=512)
    tr.add_argument('--iterations', type=int, default=200)
    tr.add_argument('--seed', type=int, default=42)
    tr.add_argument('--checkpoint')
    tr.add_argument('--output', required=True)
    vw = sub.add_parser('view')
    vw.add_argument('--straight')
    vw.add_argument('--turn')
    vw.add_argument('--original', action='store_true')
    vw.add_argument('--manual', action='store_true')
    vw.add_argument('--headless', action='store_true')
    vw.add_argument('--video')
    vw.add_argument('--force', type=float, default=40.)
    vw.add_argument('--seed', type=int, default=42)
    vw.add_argument('--steps', type=int, default=0)
    for parser in (ev, tr, vw):
        parser.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    check_source()
    configure_torch_backends()
    if args.operation == 'evaluate':
        evaluate(args)
    elif args.operation == 'train':
        train(args)
    else:
        view(args)


if __name__ == '__main__':
    main()
