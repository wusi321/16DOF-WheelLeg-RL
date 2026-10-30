"""组合赛道（沟壑 + 绕桩并排）的奖励/指标分发。

地图里两条赛道**并排**（mjlab 的 curriculum 模式：每个 sub-terrain 占一列），
每个环境按 `terrain_types` 落到某一列上。于是同一个奖励项必须只对"属于它那条赛道"
的环境生效，否则沟壑的环境会被绕桩的必经点奖励白送分（反之亦然）。

`course_mask(env, "gap" | "slalom")` 就是干这个的：读 `terrain_types`，
再按 `sub_terrains` 的键序（=列序）反查每个环境在哪条赛道上。
"""

from __future__ import annotations

import torch


def _keys(env) -> list[str]:
    gen = env.cfg.scene.terrain.terrain_generator
    return list(gen.sub_terrains.keys())


def column_of(env, keyword: str) -> int:
    for i, key in enumerate(_keys(env)):
        if keyword in key:
            return i
    raise KeyError(f"no sub-terrain matching {keyword!r} in {_keys(env)}")


def course_mask(env, keyword: str) -> torch.Tensor:
    """``(B,)`` bool：哪些环境落在名字含 `keyword` 的那条赛道上。"""
    col = column_of(env, keyword)
    types = getattr(env.scene.terrain, "terrain_types", None)
    if types is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return (types.to(torch.long) % len(_keys(env))) == col


def leap_bonus_on_gap(env, **kwargs) -> torch.Tensor:
    from .gap_course import leap_bonus

    return leap_bonus(env, **kwargs) * course_mask(env, "gap").to(torch.float32)


def gaps_crossed_on_gap(env, **kwargs) -> torch.Tensor:
    from .gap_course import gaps_crossed

    return gaps_crossed(env, **kwargs) * course_mask(env, "gap").to(torch.float32)


def waypoint_reward_on_slalom(env, **kwargs) -> torch.Tensor:
    from .slalom import waypoint_reward

    return waypoint_reward(env, **kwargs) * course_mask(env, "slalom").to(torch.float32)


def all_waypoints_on_slalom(env, **kwargs) -> torch.Tensor:
    from .slalom import all_waypoints

    return all_waypoints(env, **kwargs) * course_mask(env, "slalom").to(torch.float32)


def skip_penalty_on_slalom(env, **kwargs) -> torch.Tensor:
    from .slalom import skip_penalty

    return skip_penalty(env, **kwargs) * course_mask(env, "slalom").to(torch.float32)


def pillar_hit_on_slalom(env, **kwargs) -> torch.Tensor:
    from .slalom import pillar_hit_penalty

    return pillar_hit_penalty(env, **kwargs) * course_mask(env, "slalom").to(torch.float32)


def slalom_progress_on_slalom(env, **kwargs) -> torch.Tensor:
    from .slalom import slalom_progress

    return slalom_progress(env, **kwargs) * course_mask(env, "slalom").to(torch.float32)
