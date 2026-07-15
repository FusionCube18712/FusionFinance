from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.verifier.market_head import HORIZONS, MarketVerifier


ROOT = Path(__file__).resolve().parents[1]


def test_rocm_runtime_is_strict_without_hardcoding_an_unproven_sku(
    monkeypatch,
) -> None:
    from alpha.gpu_ml import assert_rocm_stage

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "",
            current_device=lambda: 0,
        ),
        version=SimpleNamespace(hip="7.2"),
        is_tensor=lambda _value: False,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert_rocm_stage("receipt-backed-stage")
    with pytest.raises(RuntimeError, match="unexpected device"):
        assert_rocm_stage("explicit-device-check", expect_device_substr="named-device")

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "alpha").rglob("*.py")
    )
    assert "W7900" not in source


def test_market_verifier_cpu_fallback_exposes_probabilistic_heads() -> None:
    rng = np.random.default_rng(7)
    names = [f"T{i:02d}" for i in range(24)]
    panels = []
    for date_id in range(3):
        features = pd.DataFrame(
            rng.normal(size=(len(names), 9)), index=names,
            columns=("mom21", "mom63", "mom126", "mom252", "rev5", "vol21",
                     "dvol", "dist_high", "vsurge"),
        )
        labels = {
            horizon: pd.Series(
                0.01 * features["mom21"] + rng.normal(scale=0.005, size=len(names)),
                index=names,
            )
            for horizon in HORIZONS
        }
        panels.append((pd.Timestamp("2020-01-01") + pd.Timedelta(days=date_id), features, labels))

    verifier = MarketVerifier(backend="cpu-gbm", ensemble=2)
    verifier.fit(panels)
    forecast = verifier.forecast(panels[-1][1].iloc[0])

    assert set(forecast) == {
        "expected_residual_bps",
        "p_adverse",
        "epistemic_var",
        "aleatoric_var",
        "epistemic_mi",
        "epistemic_mi_basis",
        "out_of_distribution_score",
        "calibration_hash",
    }
    assert set(forecast["p_adverse"]) == {f"{horizon}d" for horizon in HORIZONS}
    assert all(0.0 <= value <= 1.0 for value in forecast["p_adverse"].values())
    assert all(len(model.classifiers) == 2 for model in verifier.models.values())
    assert any(value > 0.0 for value in forecast["epistemic_mi"].values())

    class _TestCalibrator:
        artifact_hash = "a" * 64
        fitted_horizons = tuple(f"{horizon}d" for horizon in HORIZONS)

        @staticmethod
        def transform(probabilities, horizon):
            values = np.asarray(probabilities)
            return np.square(values) / (
                np.square(values) + np.square(1.0 - values)
            )

    verifier.calibrator = _TestCalibrator()
    calibrated = verifier.forecast(panels[-1][1].iloc[0])
    assert calibrated["epistemic_mi_basis"] == "calibrated_member_probabilities"
    assert any(
        calibrated["epistemic_mi"][key] != pytest.approx(value)
        for key, value in forecast["epistemic_mi"].items()
    )


def test_tensor_policy_is_on_device_int8_and_uncertainty_first() -> None:
    torch = pytest.importorskip("torch")
    from alpha.verifier.gpu_policy import (
        ABSTAIN,
        APPROVE,
        REJECT,
        TensorPolicyThresholds,
        tensor_policy,
    )

    result = tensor_policy(
        torch.tensor([0.10, 0.90, 0.10, 0.50]),
        torch.tensor([0.01, 0.01, 0.50, 0.01]),
        torch.tensor([0.01, 0.01, 0.01, 0.90]),
        thresholds=TensorPolicyThresholds(
            approve_max_adverse=0.35,
            reject_min_adverse=0.65,
            max_epistemic_mi=0.20,
            max_novelty=0.50,
        ),
    )

    assert result.decisions.dtype == torch.int8
    assert result.decisions.device.type == "cpu"
    assert result.decisions.tolist() == [APPROVE, REJECT, ABSTAIN, ABSTAIN]
    assert len(result.decision_hash) == 64
    with pytest.raises(ValueError, match="reserved"):
        tensor_policy(
            torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([0.1]),
            context={"p_adverse": "spoofed"},
        )


def test_deep_ensemble_three_head_shapes_and_date_block_weights() -> None:
    torch = pytest.importorskip("torch")
    from alpha.gpu_ml import DeepEnsemble, EnsembleConfig

    rng = np.random.default_rng(11)
    X = rng.normal(size=(24, 4)).astype(np.float32)
    y = rng.normal(scale=0.01, size=24).astype(np.float32)
    groups = np.repeat(np.arange(6), 4)
    model = DeepEnsemble(EnsembleConfig(
        members=3, hidden=(8,), epochs=2, dropout=0.0, block_len=2,
        device="cpu", strict=False,
    ))
    receipt = model.fit(X, y, groups)
    prediction = model.predict(X)
    counts = model._block_counts(6).cpu()

    assert receipt["dates"] == 6
    assert set(prediction) == {
        "p_adverse", "mu", "epistemic_var", "aleatoric_var", "epistemic_mi",
    }
    assert all(value.shape == (24,) for value in prediction.values())
    assert np.all((prediction["p_adverse"] >= 0.0) & (prediction["p_adverse"] <= 1.0))
    assert counts.shape == (3, 6)
    assert torch.all(counts.sum(dim=1) == 6)


def test_inverse_free_ood_is_bounded_and_rocm_assertion_fails_closed(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from alpha.gpu_ml import TorchOOD, assert_rocm_stage

    rng = np.random.default_rng(19)
    train = rng.normal(size=(40, 5)).astype(np.float32)
    scores = TorchOOD(train, device="cpu").score(
        np.vstack([train[0], np.full(5, 1e6, dtype=np.float32)])
    )
    assert np.all((scores >= 0.0) & (scores < 1.0))
    assert scores[1] > scores[0]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        assert_rocm_stage("negative-test")


def test_expanded_market_features_are_pit_and_keep_asof_universe() -> None:
    from alpha.features_market import expanded_market_features

    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2019-01-01", periods=320)
    names = ["A", "B", "C", "D", "E", "SPY"]
    returns = rng.normal(0.0003, 0.01, size=(len(dates), len(names)))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=names)
    volume = pd.DataFrame(rng.integers(100_000, 2_000_000, size=prices.shape),
                          index=dates, columns=names)
    asof = dates[280]
    prices.loc[dates[290]:, "E"] = np.nan
    before = expanded_market_features(prices, volume, asof, names[:-1])

    future_prices = prices.copy()
    future_volume = volume.copy()
    future_prices.loc[future_prices.index > asof] = 1e9
    future_volume.loc[future_volume.index > asof] = 1
    after = expanded_market_features(future_prices, future_volume, asof, names[:-1])

    pd.testing.assert_frame_equal(before, after)
    assert before.shape[1] >= 30
    assert list(before.index) == names[:-1]
    assert np.isfinite(before.to_numpy()).all()


@pytest.mark.parametrize("model_name", ["deep_sets", "set_transformer"])
def test_set_models_are_permutation_equivariant_and_center_mu(model_name: str) -> None:
    torch = pytest.importorskip("torch")
    from alpha.gpu_setmodel import (
        DeepSetsVerifier, SetModelConfig, SetTransformerVerifier,
    )

    torch.manual_seed(29)
    cfg = SetModelConfig(n_features=6, horizons=3, d_model=16, dropout=0.0)
    model = (DeepSetsVerifier(cfg) if model_name == "deep_sets"
             else SetTransformerVerifier(cfg)).eval()
    features = torch.randn(2, 7, 6)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    inverse = torch.argsort(permutation)

    original = model(features, mask)
    permuted = model(features[:, permutation], mask[:, permutation])

    for key in ("sign_logit", "p_adverse", "mu", "scale"):
        assert torch.allclose(original[key], permuted[key][:, inverse], atol=1e-5)
    valid_mu_sum = (original["mu"] * mask.unsqueeze(-1)).sum(dim=1)
    assert torch.allclose(valid_mu_sum, torch.zeros_like(valid_mu_sum), atol=1e-5)


def test_proper_scores_reward_better_forecasts_and_average_dates() -> None:
    from alpha.metrics_proper import (
        date_averaged_brier,
        date_averaged_log_loss,
        deflated_sharpe_ratio,
        equal_mass_reliability,
        gaussian_crps,
        gaussian_nll,
        pit_histogram,
    )

    outcomes = np.array([0, 1, 0, 1], dtype=float)
    dates = np.array([0, 0, 1, 1])
    good = np.array([0.05, 0.95, 0.10, 0.90])
    bad = 1.0 - good
    assert date_averaged_log_loss(outcomes, good, dates) < date_averaged_log_loss(outcomes, bad, dates)
    assert date_averaged_brier(outcomes, good, dates) < date_averaged_brier(outcomes, bad, dates)

    values = np.array([-1.0, 0.0, 1.0])
    assert gaussian_nll(values, values, np.ones(3)) < gaussian_nll(values, values + 2.0, np.ones(3))
    assert gaussian_crps(values, values, np.ones(3)) < gaussian_crps(values, values + 2.0, np.ones(3))
    histogram = pit_histogram(values, values, np.ones(3), bins=5)
    assert int(histogram["counts"].sum()) == 3
    assert sum(item["count"] for item in equal_mass_reliability(outcomes, good, bins=3)) == 4

    rng = np.random.default_rng(31)
    returns = rng.normal(0.001, 0.01, size=300)
    trial_sharpes = np.linspace(-0.5, 0.5, 20)
    assert deflated_sharpe_ratio(
        returns, trial_sharpes=trial_sharpes
    ) <= deflated_sharpe_ratio(returns)


def test_purged_folds_have_strict_label_boundaries() -> None:
    from alpha.verifier.calibration import (
        purged_walk_forward_folds,
    )

    all_dates = pd.bdate_range("2020-01-01", periods=105)
    dates = all_dates[:100]
    label_ends = pd.DatetimeIndex([
        all_dates[index + 5] for index in range(len(dates))
    ])
    folds = purged_walk_forward_folds(
        dates, label_ends, min_train_size=30, validation_size=10,
        calibration_size=10, test_size=10, purge=10,
    )
    assert folds
    for fold in folds:
        assert max(label_ends[list(fold.train)]) < dates[min(fold.validation)]
        assert max(label_ends[list(fold.validation)]) < dates[min(fold.calibration)]
        assert max(label_ends[list(fold.calibration)]) < dates[min(fold.test)]
        consumed = fold.train + fold.validation + fold.calibration + fold.test
        assert len(set(consumed)) == len(consumed)


def test_affine_calibrator_starts_as_identity() -> None:
    torch = pytest.importorskip("torch")
    from alpha.verifier.calibration import AffineLogitCalibrator

    calibrator = AffineLogitCalibrator.identity(("10d",), device="cpu")
    probabilities = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(calibrator.transform(probabilities, "10d"), probabilities, atol=1e-6)


def test_gpu_manifest_is_canonical_and_hash_chained() -> None:
    from alpha.gpu_receipts import build_run_manifest, canonical_hash

    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    digest = "a" * 64
    commit = "b" * 40
    first = build_run_manifest(
        config={"members": 4}, input_hashes={"prices": digest},
        feature_hash=digest, universe_hash=digest, fold_hash=digest,
        seed=7, autocast_dtype="bfloat16", stages={"fit": {"peak_allocated_bytes": 0}},
        git_commit=commit, dirty_hash=digest, allow_test_git_state=True,
    )
    second = build_run_manifest(
        config={"members": 8}, input_hashes={"prices": digest},
        feature_hash=digest, universe_hash=digest, fold_hash=digest,
        seed=7, autocast_dtype="bfloat16", stages={},
        previous_receipt_hash=first["receipt_hash"], git_commit=commit,
        dirty_hash=digest, allow_test_git_state=True,
    )
    assert second["previous_receipt_hash"] == first["receipt_hash"]
    assert second["receipt_hash"] != first["receipt_hash"]


def test_m_ablation_selects_the_smallest_member_count_tied_with_best() -> None:
    from alpha.gpu_ablation import MEMBER_GRID, select_smallest_tied

    rows = [
        {"members": members, "log_loss": score}
        for members, score in zip(MEMBER_GRID, (0.55, 0.51, 0.505, 0.504, 0.503, 0.502))
    ]
    assert select_smallest_tied(rows, tolerance=0.01) == 8


def test_thesis_commit_is_immutable_and_verifier_values_fail_closed() -> None:
    from pydantic import ValidationError

    from alpha.verifier.contract import ExpectedOutcome, ThesisContract, VerifierOutput

    draft = ThesisContract(
        ticker="AMD", as_of="2026-01-01", claim_type="near_term_catalyst",
        direction="positive", expected_outcome=ExpectedOutcome(direction="positive"),
    )
    committed = draft.commit(model_version="test", prompt_hash="prompt")
    assert not draft.thesis_hash
    committed.verify_commit()
    with pytest.raises(ValidationError):
        committed.ticker = "NVDA"
    with pytest.raises(ValidationError):
        VerifierOutput(
            thesis_hash=committed.thesis_hash,
            expected_residual_bps={"10d": float("nan")},
        )
    output = VerifierOutput(thesis_hash=committed.thesis_hash, p_adverse={"10d": 0.2})
    with pytest.raises(TypeError):
        output.p_adverse["10d"] = 0.9
