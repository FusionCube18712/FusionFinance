"""Integration tests for the namespaced Filing Alpha transforms.

The core cases are adapted from the MIT-licensed ``filing-alpha-cleanroom``
0.6.0 test suite.  Additional assertions cover the local immutability and
missing-data contracts required by FusionFinance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from alpha.filing_alpha import (
    FilingAvailabilityConfig,
    add_filing_decay_features,
    annualized_return,
    build_longitudinal_text_features,
    build_xbrl_features,
    equity_curve,
    extract_text_statistics,
    fit_fusion_model,
    information_coefficient,
    map_filings_to_available_sessions,
    point_in_time_filing_join,
    robust_cross_sectional_standardize,
    summarize_daily_returns,
    validate_filing_features,
    validate_market_data,
    validate_predictions,
)


def _market_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"] * 2),
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "open": [10, 11, 12, 20, 20, 21],
            "high": [11, 12, 13, 21, 21, 22],
            "low": [9, 10, 11, 19, 19, 20],
            "close": [10, 11, 12, 20, 20, 21],
            "volume": [1_000] * 6,
        }
    )


def test_market_validation_normalizes_without_mutating_input() -> None:
    source = _market_fixture().sample(frac=1.0, random_state=7).reset_index(drop=True)
    original = source.copy(deep=True)

    result = validate_market_data(source)

    pdt.assert_frame_equal(source, original)
    assert list(result.columns) == list(source.columns)
    assert result["ticker"].tolist() == sorted(result["ticker"].tolist())


def test_market_validation_rejects_impossible_bar() -> None:
    source = _market_fixture().iloc[[0]].copy()
    source.loc[:, "high"] = 8.0

    with pytest.raises(ValueError, match="Invalid OHLC"):
        validate_market_data(source)


def test_market_validation_rejects_non_finite_volume() -> None:
    source = _market_fixture().iloc[[0]].astype({"volume": float}).copy()
    source.loc[:, "volume"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        validate_market_data(source)


def test_filing_and_prediction_schemas_normalize_and_fail_closed() -> None:
    filings = pd.DataFrame(
        {
            "accepted_at": ["2026-01-09T20:30:00Z"],
            "ticker": [" aaa "],
            "form": ["10-q"],
            "score": ["1.5"],
        }
    )
    predictions = pd.DataFrame(
        {
            "date": ["2026-01-09"],
            "ticker": [" aaa "],
            "pred_h5": ["0.1"],
        }
    )

    clean_filings = validate_filing_features(filings)
    clean_predictions = validate_predictions(predictions, [5])

    assert clean_filings.loc[0, "ticker"] == "AAA"
    assert clean_filings.loc[0, "form"] == "10-Q"
    assert clean_predictions.loc[0, "pred_h5"] == 0.1
    bad_predictions = predictions.assign(pred_h5=np.inf)
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_predictions(bad_predictions, [5])


def test_xbrl_ratios_comparable_changes_and_missing_debt() -> None:
    rows: list[dict[str, object]] = []
    for quarter, revenue in enumerate([100.0, 110.0, 120.0, 130.0, 150.0]):
        accepted = pd.Timestamp("2023-05-01", tz="UTC") + pd.DateOffset(months=3 * quarter)
        facts = {
            "Revenues": revenue,
            "GrossProfit": revenue * 0.4,
            "OperatingIncomeLoss": revenue * 0.1,
            "NetIncomeLoss": revenue * 0.07,
            "NetCashProvidedByUsedInOperatingActivities": revenue * 0.09,
            "PaymentsToAcquirePropertyPlantAndEquipment": revenue * 0.02,
            "Assets": 500.0 + quarter * 20,
            "AssetsCurrent": 200.0,
            "LiabilitiesCurrent": 100.0,
            "CashAndCashEquivalentsAtCarryingValue": 50.0,
        }
        for concept, value in facts.items():
            rows.append(
                {
                    "ticker": "AAA",
                    "accepted_at": accepted.isoformat(),
                    "form": "10-Q",
                    "period_end": accepted.date().isoformat(),
                    "concept": concept,
                    "value": value,
                }
            )
    source = pd.DataFrame(rows)
    original = source.copy(deep=True)

    result = build_xbrl_features(source)
    latest = result.sort_values("accepted_at").iloc[-1]

    pdt.assert_frame_equal(source, original)
    assert np.isclose(latest["gross_margin"], 0.4)
    assert np.isclose(latest["free_cash_flow_margin"], 0.07)
    assert np.isclose(latest["comparable_pct__revenue"], 0.5)
    assert np.isnan(latest["total_debt"]), "unknown debt must not be fabricated as zero"


def test_longitudinal_text_features_handle_deterioration_and_missing_text() -> None:
    source = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "accepted_at": [
                "2023-01-10T20:00:00Z",
                "2024-01-10T20:00:00Z",
                "2024-01-10T20:00:00Z",
            ],
            "form": ["10-K", "10-K", "10-K"],
            "section": ["Risk Factors", "Risk Factors", "Risk Factors"],
            "text": [
                "Demand remains strong and liquidity is adequate.",
                "Demand may decline. We face liquidity uncertainty and a material weakness.",
                pd.NA,
            ],
        }
    )
    original = source.copy(deep=True)

    result = build_longitudinal_text_features(source)
    latest = result.loc[result["ticker"] == "AAA"].sort_values("accepted_at").iloc[-1]
    missing_stats = extract_text_statistics(pd.NA)

    pdt.assert_frame_equal(source, original)
    assert latest["risk_factors__delta_adverse_rate"] > 0
    assert latest["risk_factors__material_weakness_flag"] == 1.0
    assert latest["risk_factors__text_novelty"] > 0
    assert missing_stats["word_count"] == 0.0
    assert missing_stats["adverse_rate"] == 0.0


def test_filing_cutoff_weekend_join_decay_and_standardization() -> None:
    market = _market_fixture()
    filings = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "accepted_at": ["2026-01-09T20:30:00Z", "2026-01-09T22:30:00Z"],
            "score": [1.0, 2.0],
        }
    )
    market_original = market.copy(deep=True)
    filings_original = filings.copy(deep=True)

    mapped = map_filings_to_available_sessions(market, filings)
    joined = point_in_time_filing_join(market, filings, ["score"])
    aaa = joined.loc[joined["ticker"] == "AAA"].sort_values("date")
    decayed = add_filing_decay_features(
        pd.DataFrame({"filing_age_sessions": [0, 10, 20], "score": [2.0, 2.0, 2.0]}),
        ["score"],
        half_life_sessions=10,
    )
    standardized = robust_cross_sectional_standardize(
        pd.DataFrame({"date": ["2026-01-01"] * 3, "x": [1.0, 2.0, 100.0]}),
        ["x"],
    )

    pdt.assert_frame_equal(market, market_original)
    pdt.assert_frame_equal(filings, filings_original)
    assert list(mapped["available_date"].dt.strftime("%Y-%m-%d")) == [
        "2026-01-09",
        "2026-01-12",
    ]
    assert aaa["score"].tolist() == [1.0, 2.0, 2.0]
    assert np.isclose(decayed.loc[0, "score_decayed"], 2.0)
    assert np.isclose(decayed.loc[1, "score_decayed"], 1.0)
    assert standardized["x_robust_z"].abs().max() <= 5.0
    assert standardized.loc[1, "x_robust_z"] == 0.0


def test_filing_mapping_rejects_naive_acceptance_timestamps() -> None:
    filings = pd.DataFrame(
        {"ticker": ["AAA"], "accepted_at": ["2026-01-09 15:30:00"], "score": [1.0]}
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        map_filings_to_available_sessions(_market_fixture(), filings)


def test_fusion_ridge_baseline_and_configuration_are_explicit() -> None:
    train = pd.DataFrame(
        {
            "market": [0.0, 1.0, 2.0],
            "filing": [1.0, 1.0, 2.0],
            "target": [0.0, 1.0, 2.0],
        }
    )

    fitted = fit_fusion_model(train, "target", ["market"], ["filing"], ridge_alpha=1.0)
    predictions = fitted.predict(train)

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
    with pytest.raises(ValueError, match="cutoff_hour"):
        FilingAvailabilityConfig(cutoff_hour_eastern=24)
    with pytest.raises(ValueError, match="at least one finite target"):
        fit_fusion_model(
            train.assign(target=np.nan),
            "target",
            ["market"],
            ["filing"],
        )


def test_metrics_summary_has_demo_ready_risk_and_trade_fields() -> None:
    returns = pd.Series([0.10, -0.05, 0.02], dtype=float)
    trades = pd.DataFrame(
        {"net_return": [0.10, -0.05], "holding_sessions": [2, 3]}
    )

    result = summarize_daily_returns(returns, trades)

    assert result["total_return"] == pytest.approx((1.10 * 0.95 * 1.02) - 1.0)
    assert result["max_drawdown"] == pytest.approx(0.05)
    assert result["trade_count"] == 2
    assert result["win_rate"] == 0.5
    assert equity_curve([0.1, -0.1], compound=False).tolist() == pytest.approx([1.1, 1.0])
    assert information_coefficient(
        pd.DataFrame({"prediction": [1, 2, 3], "target": [10, 20, 30]}),
        "prediction",
        "target",
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="periods_per_year"):
        annualized_return(returns, periods_per_year=0)
