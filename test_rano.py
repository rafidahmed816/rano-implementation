"""Unit tests for Rano modules — forward/inverse correctness and loss computation."""

import pytest
import torch

from acg import AnonymizationConditionGenerator
from anonymizer import Anonymizer
from blocks import CINNBlock, INNBlock
from loss import RanoLoss
from metrics import compute_pitch_correlation, compute_wer
from model import Rano
from speaker_encoder import SpeakerEncoder


B, F, T, D = 2, 80, 64, 256


class TestINNBlock:
    def test_forward_shape(self):
        block = INNBlock(D)
        x = torch.randn(B, D)
        y, ld = block(x)
        assert y.shape == (B, D)
        assert ld.shape == (B,)

    def test_invertibility(self):
        block = INNBlock(D)
        x = torch.randn(B, D)
        y, _ = block(x)
        x_rec = block.inverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5)


class TestCINNBlock:
    def test_forward_shape(self):
        block = CINNBlock(F, D)
        x = torch.randn(B, F, 16, T)
        c = torch.randn(B, D)
        y = block(x, c)
        assert y.shape == x.shape

    def test_invertibility(self):
        block = CINNBlock(F, D)
        x = torch.randn(B, F, 16, T)
        c = torch.randn(B, D)
        y = block(x, c)
        x_rec = block.inverse(y, c)
        assert torch.allclose(x, x_rec, atol=1e-4)


class TestACG:
    def test_generate_shape(self):
        acg = AnonymizationConditionGenerator(D, num_blocks=2)
        key = torch.randn(B, D)
        sa = acg.generate(key)
        assert sa.shape == (B, D)

    def test_nll_loss_positive(self):
        acg = AnonymizationConditionGenerator(D, num_blocks=2)
        s = torch.randn(B, D)
        loss = acg.loss(s)
        assert loss.item() > 0


class TestAnonymizer:
    def test_forward_shape(self):
        anon = Anonymizer(F, D, num_blocks=2)
        x = torch.randn(B, 1, F, T)
        c = torch.randn(B, D, 1, 1)
        xa = anon(x, c)
        assert xa.shape == x.shape

    def test_invertibility(self):
        anon = Anonymizer(F, D, num_blocks=2)
        x = torch.randn(B, 1, F, T)
        c = torch.randn(B, D, 1, 1)
        xa = anon(x, c)
        xr = anon.inverse(xa, c)
        assert torch.allclose(x, xr, atol=1e-4)


class TestSpeakerEncoder:
    def test_output_normalised(self):
        enc = SpeakerEncoder(F, D)
        mel = torch.randn(B, 1, F, T)
        emb = enc(mel)
        norms = torch.norm(emb, dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)


class TestRanoLoss:
    def test_loss_keys(self):
        loss_fn = RanoLoss()
        x = torch.randn(B, 1, F, T)
        x_hat = torch.randn(B, 1, F, T)
        anchor = torch.randn(B, D)
        pos = torch.randn(B, D)
        neg = torch.randn(B, D)
        losses = loss_fn(x, x_hat, anchor, pos, neg)
        assert set(losses.keys()) == {"total", "consistency", "triplet"}


class TestMetrics:
    def test_pitch_correlation_perfect(self):
        f0 = torch.randn(100).abs().numpy()
        assert abs(compute_pitch_correlation(f0, f0) - 1.0) < 1e-5

    def test_wer_identical(self):
        assert compute_wer("hello world", "hello world") < 0.01

    def test_wer_all_wrong(self):
        wer = compute_wer("hello world", "foo bar baz")
        assert wer > 50


class TestRanoModel:
    def test_anonymize_restore_cycle(self):
        model = Rano(F, D, num_cinn_blocks=2, num_acg_blocks=2)
        model.eval()
        x = torch.randn(1, 1, F, T)
        key = torch.randn(1, D)
        with torch.no_grad():
            xa, _ = model.anonymize(x, key)
            xr = model.restore(xa, key)
        assert torch.allclose(x, xr, atol=1e-3), "Restoration must be lossless"

    def test_different_keys_give_different_anon(self):
        model = Rano(F, D, num_cinn_blocks=2, num_acg_blocks=2)
        model.eval()
        x = torch.randn(1, 1, F, T)
        k1, k2 = torch.randn(1, D), torch.randn(1, D)
        with torch.no_grad():
            xa1, _ = model.anonymize(x, k1)
            xa2, _ = model.anonymize(x, k2)
        assert not torch.allclose(xa1, xa2), "Different keys must produce different anonymized mel"
