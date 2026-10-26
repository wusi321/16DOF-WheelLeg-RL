"""HIMLoco-style "only positive step rewards" clamping, opt-in per environment.

Mechanics:

* PPO advantage is ``reward + gamma * V(s') - V(s)``. Early in training the penalties
  can dwarf the positive terms, so every step yields a negative advantage and the
  policy learns to *end the episode as fast as possible* to stop the bleeding.
* Clamping the **total** step reward at zero keeps penalties from creating that
  "terminate early" gradient while still letting them cancel the positive terms.
  Per-term episode sums are untouched, so ``Episode_Reward/*`` stays interpretable.

Why per-environment instead of a global switch: ``RewardManager.compute`` is a class
method, and the original implementation toggled it with a module-level patch. That
made the behaviour depend on *which task config was built last in the process* — a
Silent state leak between tasks (build Rough, then StepStone, and StepStone inherits
the clamp). Here the patch is installed once and reads a flag from the environment
config, so the behaviour is a property of the task, not of import order.

Enable it for a task with::

    cfg.only_positive_rewards = True

``_make_base_env_cfg()`` resets the flag to ``False`` for every task.
"""

from __future__ import annotations

import torch
from mjlab.managers.reward_manager import RewardManager

_original_compute = RewardManager.compute


def _compute_maybe_clamped(self: RewardManager, dt: float) -> torch.Tensor:
    """Compute step rewards, clamping the total to >= 0 when the env opted in."""
    reward = _original_compute(self, dt)
    if getattr(self._env.cfg, "only_positive_rewards", False):
        return torch.clamp(reward, min=0.0)
    return reward


def install_only_positive_rewards() -> None:
    """Install the (idempotent) patched ``compute`` that honours the per-env flag."""
    if RewardManager.compute is not _compute_maybe_clamped:
        RewardManager.compute = _compute_maybe_clamped


def uninstall_only_positive_rewards() -> None:
    """Restore the stock ``RewardManager.compute`` (tests / debugging only)."""
    if RewardManager.compute is _compute_maybe_clamped:
        RewardManager.compute = _original_compute
