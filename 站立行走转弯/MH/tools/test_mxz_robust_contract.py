"""Regression tests for recovery measurements and physical-force contracts."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mxz_robust import HorizontalPushes, recovery_summary, residual_actions
from mxz_robust_report import gates


class RecoveryTests(unittest.TestCase):
    def trial(self):
        trace = np.zeros((400, 1, 8))
        trace[:, :, 0] = trace[:, :, 6] = .4
        failed = np.zeros((400, 1), dtype=bool)
        events = [{'step': 0, 'end_step': 10, 'env': 0, 'force_n': 40, 'direction': 0}]
        return trace, failed, events

    def test_recovery_deadline_includes_exactly_two_seconds(self):
        v, f, e = self.trial()
        v[10:110, :, 0] = 0
        result = recovery_summary(v, f, e, .02)[0]
        self.assertTrue(result['recovered'])
        self.assertAlmostEqual(result['recovery_s'], 2.)
        v[110, :, 0] = 0
        self.assertFalse(recovery_summary(v, f, e, .02)[0]['recovered'])

    def test_must_hold_good_state_for_half_second(self):
        v, f, e = self.trial()
        # Recurring spikes prevent any contiguous .5s stable period.
        v[10:140:20, :, 4] = np.deg2rad(6)
        self.assertFalse(recovery_summary(v, f, e, .02)[0]['recovered'])

    def test_lateral_velocity_counts_even_when_forward_matches(self):
        v, f, e = self.trial()
        v[:, :, 1] = .09
        self.assertFalse(recovery_summary(v, f, e, .02)[0]['recovered'])

    def test_reset_cannot_erase_a_previous_fall(self):
        v, f, e = self.trial()
        f[5:] = True
        self.assertFalse(recovery_summary(v, f, e, .02)[0]['recovered'])


class ForceTests(unittest.TestCase):
    def test_force_is_horizontal_and_has_correct_duration_and_offset_torque(self):
        class Asset:
            data = SimpleNamespace(body_com_quat_w=torch.tensor([[[1., 0., 0., 0.]]]))
            def find_bodies(self, *args, **kwargs): return [0], ['base']
            def write_external_wrench_to_sim(self, force, torque, **kwargs):
                self.force, self.torque = force.clone(), torque.clone()
        asset = Asset()
        env = SimpleNamespace(num_envs=1, device='cpu', step_dt=.02, scene={'robot': asset})
        push = HorizontalPushes(env, 40)
        push.wait[:] = 10000
        push.remaining[:] = 10
        push.force[0, 0] = torch.tensor([40., 0., 0.])
        for _ in range(10):
            push.step()
            torch.testing.assert_close(asset.force[0, 0], torch.tensor([40., 0., 0.]))
            torch.testing.assert_close(asset.torque[0, 0], torch.tensor([0., 3.2, 0.]))
        push.step()
        self.assertFalse(asset.force.any())
        self.assertFalse(asset.torque.any())
        push.remaining[:] = 10
        push.force[:] = 40
        push.reset()
        push.step()
        self.assertFalse(asset.force.any())
        self.assertFalse(asset.torque.any())

    def test_stand_uses_proven_original_pd_and_moving_residual_is_bounded(self):
        obs = {'actor': torch.zeros(2, 53)}
        obs['actor'][1, 6] = .4
        extreme = torch.full((2, 16), 100.)
        output = residual_actions(obs, extreme)
        self.assertFalse(output[0].any())
        torch.testing.assert_close(output[1, :12], torch.tensor([.64, .48, .48]*4))
        base_wheel = .4/.0839*1.14/10
        torch.testing.assert_close(output[1, 12:], torch.full((4,), base_wheel+.2))


class SelectionTests(unittest.TestCase):
    def row(self, force=0, vx=.01, success=1.):
        return {'scenario': 'forward04', 'force_n': force, 'vx_mae': vx,
                'yaw_mae': .01, 'episode_success_rate': success,
                'nonfinite_frames': 0,
                'recovery_rate': 1. if force else None}

    def test_cannot_select_a_model_that_moves_well_only_with_pushes(self):
        baseline = [self.row()]
        candidate = [self.row(success=.9), self.row(force=40)]
        self.assertFalse(gates(candidate, baseline, 40)[0]['pass'])

    def test_nominal_regression_limit_is_enforced(self):
        baseline = [self.row()]
        self.assertTrue(gates([self.row(vx=.029), self.row(force=40)], baseline, 40)[0]['pass'])
        self.assertFalse(gates([self.row(vx=.031), self.row(force=40)], baseline, 40)[0]['pass'])

    def test_one_nonfinite_frame_rejects_the_whole_candidate(self):
        nominal, forced = self.row(), self.row(force=40)
        forced['nonfinite_frames'] = 1
        self.assertFalse(gates([nominal, forced], [self.row()], 40)[0]['pass'])


if __name__ == '__main__':
    unittest.main()
