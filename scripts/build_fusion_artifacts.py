#!/usr/bin/env python3
"""Deterministically rebuild the public FusionFinance v1 replay artifacts.

The input is a checked-in, sealed retrospective replay. No ignored cache,
network request, API credential, or wall clock is allowed into this build.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PERIODS = 252
SOURCE_RELATIVE_PATH = Path("evidence/replay/v1_source.json")
RECEIPT_PATHS = {
    "environment": Path("evidence/amd/environment.json"),
    "hardware": Path("evidence/amd/hardware.json"),
    "training": Path("evidence/amd/training.json"),
}
ARM_LABELS = {
    "pure_ml": "Pure ML",
    "pure_llm": "Pure LLM",
    "fusion": "FusionFinance",
    "benchmark": "SPY benchmark",
}


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required public evidence is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON evidence at {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _positive_number(value: Any, *, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return converted


def _parsed_amd_claims(
    probe: Any, hardware: Any, training: Any
) -> dict[str, Any]:
    """Derive publication claims from three mutually consistent receipts."""
    probe = _required_mapping(probe, field="AMD environment receipt")
    hardware = _required_mapping(hardware, field="AMD hardware receipt")
    training = _required_mapping(training, field="AMD training receipt")
    if probe.get("status") != "complete" or hardware.get("status") != "complete":
        raise ValueError("AMD environment and hardware receipts must be complete")

    hardware_text = str(hardware.get("smi_short", ""))
    vendor_match = re.search(r"Card Vendor:\s*([^\n]+)", hardware_text)
    vendor = vendor_match.group(1).strip() if vendor_match else ""
    if "Advanced Micro Devices" not in vendor or "AMD/ATI" not in vendor:
        raise ValueError("AMD hardware receipt does not identify an AMD/ATI vendor")
    gfx_match = re.search(r"GFX Version:\s*(gfx[0-9a-z]+)\b", hardware_text)
    if not gfx_match:
        raise ValueError("AMD hardware receipt has no valid gfx architecture")
    gfx_version = gfx_match.group(1)

    vram_match = re.search(
        r"VRAM Total Memory \(B\):\s*([0-9]+)", str(hardware.get("smi_vram", ""))
    )
    if not vram_match:
        raise ValueError("AMD hardware receipt has no VRAM total")
    vram_bytes = int(vram_match.group(1))
    if vram_bytes < 1_000_000_000:
        raise ValueError("AMD hardware receipt reports implausible VRAM")

    runtime = str(probe.get("rocm_hip", "")).strip()
    torch_version = str(probe.get("torch", "")).strip()
    device_count = int(probe.get("device_count", 0))
    if not runtime or not torch_version or device_count < 1:
        raise ValueError("AMD environment receipt lacks ROCm, PyTorch, or a device")

    training_device = _required_mapping(training.get("device"), field="training.device")
    device_probe = _required_mapping(
        training.get("device_probe"), field="training.device_probe"
    )
    if training_device.get("rocm_hip") != runtime:
        raise ValueError("AMD environment and training runtime mismatch")
    if str(training_device.get("torch", "")).strip() != torch_version:
        raise ValueError("AMD environment and training PyTorch mismatch")
    if int(training_device.get("device_count", 0)) != device_count:
        raise ValueError("AMD environment and training device-count mismatch")
    if not bool(device_probe.get("cuda_available")):
        raise ValueError("AMD training receipt does not show accelerator availability")

    reported_mem_gb = _positive_number(device_probe.get("mem_gb"), field="GPU memory")
    if not math.isclose(reported_mem_gb, vram_bytes / 1_000_000_000, abs_tol=0.2):
        raise ValueError("AMD hardware and training VRAM mismatch")
    training_seconds = _positive_number(
        training.get("gpu_train_seconds"), field="GPU training seconds"
    )
    policies = _required_mapping(training.get("policies"), field="training.policies")
    expanding = _required_mapping(
        policies.get("gpu_mlp_expanding"), field="training.policies.gpu_mlp_expanding"
    )
    requests_processed = int(expanding.get("n_retrains", 0))
    if requests_processed < 1:
        raise ValueError("AMD training receipt has no walk-forward retrains")

    return {
        "vendor": vendor,
        "gfx_version": gfx_version,
        "runtime": f"ROCm/HIP {runtime}",
        "torch": torch_version,
        "device_count": device_count,
        "vram_bytes": vram_bytes,
        "training_seconds": training_seconds,
        "requests_processed": requests_processed,
        "raw_device_name": str(probe.get("gpu", "")).strip() or None,
        "hardware_excerpt": hardware_text[:1000],
    }


def _finite_numbers(values: Any, *, field: str, size: int) -> list[float]:
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"{field} must contain exactly {size} observations")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{field} contains a non-finite observation")
    return converted


def _validated_source(root: Path) -> tuple[Mapping[str, Any], Path]:
    source_path = root / SOURCE_RELATIVE_PATH
    source = _read(source_path)
    if not isinstance(source, Mapping):
        raise ValueError("public replay source must be a JSON object")
    if source.get("schema_version") != 1:
        raise ValueError("public replay source schema_version must equal 1")
    if source.get("claim_status") != "provisional_uncontrolled_legacy_race":
        raise ValueError("public replay claim boundary is missing or changed")

    dates = source.get("dates")
    if not isinstance(dates, list) or not dates or not all(
        isinstance(value, str) for value in dates
    ):
        raise ValueError("public replay dates must be a non-empty string list")
    if dates != sorted(set(dates)):
        raise ValueError("public replay dates must be unique and increasing")
    if source.get("window") != [dates[0], dates[-1]]:
        raise ValueError("public replay window must match its first and last date")

    arms = source.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARM_LABELS):
        raise ValueError("public replay must contain exactly four expected arms")
    for key in ARM_LABELS:
        arm = arms[key]
        if not isinstance(arm, Mapping):
            raise ValueError(f"arms.{key} must be an object")
        _finite_numbers(
            arm.get("daily_returns"), field=f"arms.{key}.daily_returns", size=len(dates)
        )
        _finite_numbers(arm.get("equity"), field=f"arms.{key}.equity", size=len(dates))

    decisions = source.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("public replay must retain its decision timeline")
    required_decision_fields = {
        "timestamp",
        "asset",
        "proposed_action",
        "final_action",
        "verification",
        "rationale",
    }
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping) or not required_decision_fields <= decision.keys():
            raise ValueError(f"decisions[{index}] is missing required fields")
    return source, source_path


def _metrics(returns: list[float]) -> dict[str, float | int | None]:
    if not returns:
        raise ValueError("cannot summarize an empty return stream")
    wealth: list[float] = []
    value = 1.0
    for item in returns:
        value *= 1.0 + item
        wealth.append(value)
    peak = 1.0
    drawdowns: list[float] = []
    for item in wealth:
        peak = max(peak, item)
        drawdowns.append(item / peak - 1.0)
    sigma = stdev(returns) if len(returns) > 1 else 0.0
    downside = [min(item, 0.0) for item in returns]
    downside_sigma = math.sqrt(fmean(item * item for item in downside))
    annual_return = value ** (PERIODS / len(returns)) - 1.0 if value > 0 else -1.0
    annual_vol = sigma * math.sqrt(PERIODS)
    sharpe = fmean(returns) / sigma * math.sqrt(PERIODS) if sigma > 0 else 0.0
    sortino = (
        fmean(returns) / downside_sigma * math.sqrt(PERIODS)
        if downside_sigma > 0
        else 0.0
    )
    max_drawdown = min(drawdowns, default=0.0)
    ordered = sorted(returns)
    tail_index = max(0, math.ceil(0.05 * len(ordered)) - 1)
    return {
        "cumulative_return": value - 1.0,
        "annualized_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annual_vol,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        "tail_loss_95": ordered[tail_index],
        "positive_day_rate": sum(item > 0 for item in returns) / len(returns),
        "observations": len(returns),
        "turnover": None,
        "transaction_costs": None,
        "trade_count": None,
        "win_rate": None,
        "current_exposure": None,
        "cash_balance": None,
    }


def _amd(root: Path) -> dict[str, Any]:
    paths = {name: root / path for name, path in RECEIPT_PATHS.items()}
    probe = _read(paths["environment"])
    hardware = _read(paths["hardware"])
    training = _read(paths["training"])
    claims = _parsed_amd_claims(probe, hardware, training)
    return {
        "status": "verified_receipt_with_device_name_unavailable",
        "vendor": claims["vendor"],
        "device_claim": (
            f"AMD {claims['gfx_version']} accelerator; "
            "commercial SKU not proven by receipt"
        ),
        "gfx_version": claims["gfx_version"],
        "runtime": claims["runtime"],
        "torch": claims["torch"],
        "device_count": claims["device_count"],
        "vram_bytes": claims["vram_bytes"],
        "workload": "walk-forward quantitative model training",
        "training_seconds": claims["training_seconds"],
        "model_loaded": "expanding walk-forward MLP ranker",
        "requests_processed": claims["requests_processed"],
        "utilization": None,
        "raw_device_name": claims["raw_device_name"],
        "receipts": {
            name: {
                "path": relative_path.as_posix(),
                "sha256": _sha256(paths[name]),
            }
            for name, relative_path in RECEIPT_PATHS.items()
        },
        "hardware_excerpt": claims["hardware_excerpt"],
    }


def build_payload(root: Path = ROOT) -> dict[str, Any]:
    source, source_path = _validated_source(root)
    dates = list(source["dates"])
    arms = {
        key: {
            "label": label,
            "daily_returns": _finite_numbers(
                source["arms"][key]["daily_returns"],
                field=f"arms.{key}.daily_returns",
                size=len(dates),
            ),
            "equity": _finite_numbers(
                source["arms"][key]["equity"],
                field=f"arms.{key}.equity",
                size=len(dates),
            ),
            "metrics": _metrics(
                _finite_numbers(
                    source["arms"][key]["daily_returns"],
                    field=f"arms.{key}.daily_returns",
                    size=len(dates),
                )
            ),
            **(
                {
                    "source": (
                        "sealed public replay; original provenance: "
                        "data/cache/pead/ohlcv.parquet:Adj Close:SPY"
                    )
                }
                if key == "benchmark"
                else {}
            ),
        }
        for key, label in ARM_LABELS.items()
    }
    return {
        "schema_version": 1,
        "generated_at": str(source["generated_at"]),
        "title": "FusionFinance",
        "mode": "retrospective_point_in_time_replay",
        "claim_status": source["claim_status"],
        "window": list(source["window"]),
        "dates": dates,
        "starting_capital": float(source["starting_capital"]),
        "arms": arms,
        "decisions": [dict(item) for item in source["decisions"]],
        "architecture": list(source["architecture"]),
        "assumptions": dict(source["assumptions"]),
        "limitations": list(source["limitations"]),
        "amd_compute": _amd(root),
        "source_artifact": {
            "path": SOURCE_RELATIVE_PATH.as_posix(),
            "sha256": _sha256(source_path),
        },
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def render_artifacts(payload: Mapping[str, Any]) -> dict[str, bytes]:
    replay = _json_bytes(payload)
    arms = payload["arms"]
    metrics = _json_bytes({key: value["metrics"] for key, value in arms.items()})
    amd = _json_bytes(payload["amd_compute"])
    decisions = "".join(
        json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
        for item in payload["decisions"]
    ).encode("utf-8")
    return {
        "results/demo_run.json": replay,
        "demo/replay.json": replay,
        "results/metrics.json": metrics,
        "results/amd_compute.json": amd,
        "results/decisions.jsonl": decisions,
    }


def write_artifacts(root: Path, artifacts: Mapping[str, bytes]) -> None:
    for relative_path, payload in artifacts.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def main() -> int:
    payload = build_payload(ROOT)
    artifacts = render_artifacts(payload)
    write_artifacts(ROOT, artifacts)
    print(
        f"wrote deterministic public replay with {len(payload['decisions'])} "
        "decision events"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
