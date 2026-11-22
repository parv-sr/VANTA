"""VANTA-RNX contract tests: associative aggregation, exact RN identity, score bounds."""
import math
import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'VANTA_RNX'))
from model import NonlinearFeatureAttentionRegressor, ReferenceRN


class TestRNX(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(147)
        self.kwargs = dict(num_features=4, numeric_positions=[0, 1],
            categorical_positions=[2, 3], categorical_cardinalities=[4, 5],
            latent_dim=7, gamma_init=.1)
        self.n = torch.tensor([[.5,-2.], [12.,.3], [-4.,3.]])
        self.c = torch.tensor([[1,2], [0,3], [2,4]], dtype=torch.long)

    def build(self, kernel, value):
        return NonlinearFeatureAttentionRegressor(
            **self.kwargs, kernel_map=kernel, value_map=value)

    def test_identity_equivalence(self):
        base = ReferenceRN(**self.kwargs)
        ext = self.build('identity', 'identity')
        ext.load_state_dict(base.state_dict(), strict=True)
        torch.testing.assert_close(base(self.n,self.c), ext(self.n,self.c), atol=0, rtol=0)
        torch.testing.assert_close(
            base.attention_matrix(self.n,self.c),
            ext.attention_matrix(self.n,self.c), atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in base.parameters()),
                         sum(p.numel() for p in ext.parameters()))

    def test_learned_maps_initialize_as_identity(self):
        base = ReferenceRN(**self.kwargs)
        for kernel,value in [('cubic','identity'),
                             ('identity','learned_asinh'),
                             ('identity','gelu2'), ('cubic','gelu2')]:
            with self.subTest(kernel=kernel,value=value):
                model = self.build(kernel,value)
                model.load_state_dict(base.state_dict(), strict=False)
                torch.testing.assert_close(base(self.n,self.c),
                                           model(self.n,self.c), atol=1e-6,rtol=1e-6)

    def test_fast_equals_explicit_all_combinations(self):
        for kernel in NonlinearFeatureAttentionRegressor.KERNEL_MAPS:
            for value in NonlinearFeatureAttentionRegressor.VALUE_MAPS:
                with self.subTest(kernel=kernel,value=value):
                    model = self.build(kernel,value)
                    q,k,u,v = model.mapped_components(self.n,self.c)
                    A = model.attention_matrix(self.n,self.c)
                    explicit = model.readout(v + model.gamma*(A @ u.unsqueeze(-1)).squeeze(-1)).squeeze(-1)
                    actual, returned = model(self.n,self.c,return_attention=True)
                    torch.testing.assert_close(actual,explicit,rtol=1e-5,atol=1e-5)
                    torch.testing.assert_close(A,returned)
                    self.assertEqual(q.shape, (3,4,7))
                    self.assertEqual(k.shape, (3,4,7))
                    self.assertLessEqual(A.abs().max().item(),1/math.sqrt(4)+1e-6)

    def test_score_norms_not_increased(self):
        for kernel in NonlinearFeatureAttentionRegressor.KERNEL_MAPS:
            model = self.build(kernel,'identity')
            q,k,_,_ = model.mapped_components(self.n,self.c)
            rq,rk,_ = ReferenceRN.project(model,self.n,self.c)
            torch.testing.assert_close(q.norm(dim=-1),rq.norm(dim=-1),rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(k.norm(dim=-1),rk.norm(dim=-1),rtol=1e-5,atol=1e-6)

    def test_value_mapping_only_numeric_attention_values(self):
        base = self.build('identity','identity')
        changed = self.build('identity','asinh')
        changed.load_state_dict(base.state_dict(),strict=True)
        q1,k1,u1,v1 = base.mapped_components(self.n,self.c)
        q2,k2,u2,v2 = changed.mapped_components(self.n,self.c)
        for a,b in [(q1,q2),(k1,k2),(v1,v2),(u1[:,2:],u2[:,2:])]:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertGreater((u1[:,:2]-u2[:,:2]).abs().max().item(),.5)
        base.gamma.data.zero_()
        changed.gamma.data.zero_()
        torch.testing.assert_close(base(self.n,self.c),changed(self.n,self.c))

    def test_finite_gradients(self):
        for kernel in NonlinearFeatureAttentionRegressor.KERNEL_MAPS:
            for value in NonlinearFeatureAttentionRegressor.VALUE_MAPS:
                model = self.build(kernel,value)
                model(self.n,self.c).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_learned_maps_receive_gradient(self):
        model = self.build('cubic','gelu2')
        model(self.n,self.c).square().mean().backward()
        self.assertTrue(torch.isfinite(model.query_cubic.grad).all())
        self.assertTrue(torch.isfinite(model.key_cubic.grad).all())
        self.assertGreater(model.value_out.grad.abs().sum().item(),0)


if __name__=='__main__':
    unittest.main()
