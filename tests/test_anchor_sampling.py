"""PTX候选分组回归：真实原生入口/扩散，微型网络及替代trunk、置信度头。"""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from ml_collections import ConfigDict

from fixture_inference import _example, _tiny_config
from oridynamic.inference_runtime import predict_anchor
from oridynamic.inference_timing import TimingLog
from protenix.model.generator import InferenceNoiseScheduler
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.protenix import Protenix


class AnchorSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        timer = TimingLog(self.temp.name)
        timer.configure_cuda(lambda: None, device='cpu_fixture')
        self.args = SimpleNamespace(anchor_seed=101, anchor_candidates=32,
                                    anchor_samples_per_seed=8, timing=timer)
        root = Path(__file__).resolve().parents[1]
        cfg = ConfigDict(json.loads((root / 'configs/inference_defaults.json').read_text()))
        cfg.sample_diffusion.N_step = 2  # 仅缩短CPU测试；正式入口仍校验SDE200。
        example = _example()
        self.features = dict(example['input_feature_dict'], residue_index=torch.arange(3),
                             ref_space_uid=torch.zeros(6, dtype=torch.long),
                             asym_id=torch.zeros(3, dtype=torch.long), has_frame=torch.ones(3),
                             is_ligand=torch.zeros(6), msa=torch.zeros(1, 3, dtype=torch.long))
        self.native = Protenix.__new__(Protenix)
        nn.Module.__init__(self.native)
        self.native.configs = cfg
        self.native.N_cycle, self.native.N_model_seed = 10, 1
        self.native.enable_diffusion_shared_vars_cache = True
        self.native.enable_efficient_fusion = False
        self.native.relative_position_encoding = SimpleNamespace(generate_relp=lambda f: f)
        self.native.inference_noise_scheduler = InferenceNoiseScheduler(**cfg.inference_noise_scheduler)
        self.native.diffusion_module = DiffusionModule(**_tiny_config())
        self.native.distogram_head = nn.Linear(4, 64)
        self.native.get_pairformer_output = Mock(return_value=tuple(
            example[k] for k in ('s_inputs', 's_trunk', 'z_trunk')))
        self.groups = []

        def confidence(**kwargs):
            self.groups.append(kwargs['x_pred_coords'].detach().clone())
            return tuple(torch.zeros(1) for _ in range(4))

        self.native.run_confidence_head = Mock(side_effect=confidence)
        self.native.eval().requires_grad_(False)

    def summary(self, **kwargs):
        offset = sum(len(x) for x in self.groups[:-1])
        return ([{'ranking_score': float(offset + i)} for i in range(len(kwargs['atom_coordinate']))], [])

    def predict(self):
        with redirect_stdout(io.StringIO()), patch(
                'protenix.model.protenix.sample_confidence.compute_full_data_and_summary',
                side_effect=self.summary), torch.no_grad():
            return predict_anchor(self.native, self.features, self.args, 6)

    def test_four_trunks_generate_32_real_diffusion_samples_and_rank_all(self):
        before = copy.deepcopy(self.features)
        anchor, selection = self.predict()
        self.assertEqual(self.native.get_pairformer_output.call_count, 4)
        self.assertEqual([len(g) for g in self.groups], [8, 8, 8, 8])
        self.assertEqual(selection['seeds'], [101, 102, 103, 104])
        self.assertEqual(selection['scores'], list(map(float, range(32))))
        self.assertEqual(selection['selected'], 31)
        coordinates = torch.cat(self.groups)
        self.assertTrue(torch.isfinite(coordinates).all())
        self.assertEqual(torch.unique(coordinates.reshape(32, -1), dim=0).shape[0], 32)
        torch.testing.assert_close(anchor, coordinates[31])
        self.assertEqual(set(self.features), set(before))
        for k, value in before.items():
            if torch.is_tensor(value): torch.testing.assert_close(self.features[k], value)
        stages = self.args.timing.data['stages']
        self.assertEqual([(s['seed'], s['candidates']) for s in stages],
                         [(101, 8), (102, 8), (103, 8), (104, 8)])
        for call in self.native.get_pairformer_output.call_args_list:
            self.assertEqual(call.kwargs['N_cycle'], 10)

    def test_partial_group_keeps_exact_budget_and_restores_config(self):
        self.args.anchor_candidates = 10
        anchor, selection = self.predict()
        self.assertEqual([len(g) for g in self.groups], [8, 2])
        self.assertEqual(self.native.get_pairformer_output.call_count, 2)
        self.assertEqual(self.native.configs.sample_diffusion.N_sample, 8)
        self.assertEqual(selection['total_candidates'], 10)
        self.assertEqual(selection['selected'], 9)
        torch.testing.assert_close(anchor, self.groups[-1][-1])

    def test_single_candidate_and_repeatability(self):
        self.args.anchor_candidates = 1
        first, first_selection = self.predict()
        self.assertEqual(self.native.get_pairformer_output.call_count, 1)
        self.groups.clear()
        second, second_selection = self.predict()
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(first_selection, second_selection)

    def test_count_mismatch_fails_and_restores_config(self):
        self.args.anchor_candidates = 10
        def bad_summary(**kwargs):
            if len(kwargs['atom_coordinate']) == 8:
                return self.summary(**kwargs)
            return ([{'ranking_score': 1.}], [])
        with redirect_stdout(io.StringIO()), patch(
                'protenix.model.protenix.sample_confidence.compute_full_data_and_summary',
                side_effect=bad_summary), torch.no_grad(), self.assertRaisesRegex(ValueError, 'counts differ'):
            predict_anchor(self.native, self.features, self.args, 6)
        self.assertEqual(self.native.configs.sample_diffusion.N_sample, 8)


if __name__ == '__main__':
    unittest.main()
