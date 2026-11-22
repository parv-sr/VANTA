"""Algebraic contract tests: python -m unittest discover -s tests -v."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = load_module("vanta_r_test", ROOT / "VANTA_R" / "model.py")
RN = load_module("vanta_rn_test", ROOT / "VANTA_RN" / "model.py")


class TestAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.numeric = torch.tensor([[0.1, -2.0], [3.0, 0.0], [-1.3, 0.2]])
        self.categorical = torch.tensor([[1, 2], [3, 1], [0, 0]], dtype=torch.long)
        self.kwargs = dict(num_features=4, numeric_positions=[0,1],
            categorical_positions=[2,3], categorical_cardinalities=[4,3],
            latent_dim=7, gamma_init=.1)

    def assert_associative(self, model, divisor):
        model.eval()
        with torch.no_grad():
            q,k,v = model.project(self.numeric, self.categorical)
            a = model.attention_matrix(self.numeric, self.categorical)
            explicit = (a @ v.unsqueeze(-1)).squeeze(-1)
            assoc = (q * (k*v.unsqueeze(-1)).sum(dim=1).unsqueeze(1)).sum(dim=-1)/divisor
            torch.testing.assert_close(assoc, explicit, rtol=1e-5, atol=1e-6)
            pred = model(self.numeric, self.categorical)
            torch.testing.assert_close(pred, model.readout(v+model.gamma*explicit).squeeze(-1), rtol=1e-5, atol=1e-6)
            pred2,a2 = model(self.numeric, self.categorical, return_attention=True)
            torch.testing.assert_close(pred2,pred)
            torch.testing.assert_close(a2,a)

    def test_r_associativity(self):
        self.assert_associative(R.ResidualFeatureAttentionRegressor(**self.kwargs), math.sqrt(7))

    def test_rn_associativity(self):
        self.assert_associative(RN.NormalizedFeatureAttentionRegressor(**self.kwargs), math.sqrt(4))

    def test_rn_norms_and_scores_bounded(self):
        model = RN.NormalizedFeatureAttentionRegressor(**self.kwargs)
        x=self.numeric.clone(); x[0,0]=1e5
        with torch.no_grad():
            q,k,_=model.project(x,self.categorical)
            self.assertLessEqual(q.norm(dim=-1).max().item(),1+1e-6)
            self.assertLessEqual(k.norm(dim=-1).max().item(),1+1e-6)
            self.assertLessEqual(model.attention_matrix(x,self.categorical).abs().max().item(),.5+1e-6)

    def test_rn_same_parameter_schema(self):
        a=R.ResidualFeatureAttentionRegressor(**self.kwargs)
        b=RN.NormalizedFeatureAttentionRegressor(**self.kwargs)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))

    def test_finite_gradients(self):
        for cls in [R.ResidualFeatureAttentionRegressor,RN.NormalizedFeatureAttentionRegressor]:
            with self.subTest(cls=cls.__name__):
                m=cls(**self.kwargs);m(self.numeric,self.categorical).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_unknown_categorical_value_zero(self):
        m=RN.NormalizedFeatureAttentionRegressor(**self.kwargs)
        with torch.no_grad():
            _,v=m.tokenize(self.numeric,torch.zeros_like(self.categorical))
        torch.testing.assert_close(v[:,2:],torch.zeros_like(v[:,2:]))


if __name__=="__main__":
    unittest.main()
