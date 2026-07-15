#!/usr/bin/env python3
"""Verify the checked-in AMD compute receipt chain without probing the host."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path, *, name: str, errors: list[str]) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{name} receipt is unreadable: {exc}")
        return None
    if not isinstance(value, Mapping):
        errors.append(f"{name} receipt must be a JSON object")
        return None
    return value


def _semantic_receipt_errors(
    manifest: Mapping[str, Any], receipts: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    errors: list[str] = []
    environment = receipts["environment"]
    hardware = receipts["hardware"]
    training = receipts["training"]
    hardware_text = str(hardware.get("smi_short", ""))

    vendor_match = re.search(r"Card Vendor:\s*([^\n]+)", hardware_text)
    vendor = vendor_match.group(1).strip() if vendor_match else ""
    if "Advanced Micro Devices" not in vendor or "AMD/ATI" not in vendor:
        errors.append("hardware receipt does not identify AMD/ATI as the vendor")
    if manifest.get("vendor") != vendor:
        errors.append("manifest vendor does not match the hardware receipt")

    gfx_match = re.search(r"GFX Version:\s*(gfx[0-9a-z]+)\b", hardware_text)
    gfx_version = gfx_match.group(1) if gfx_match else None
    if not gfx_version:
        errors.append("hardware receipt has no valid gfx architecture")
    if manifest.get("gfx_version") != gfx_version:
        errors.append("manifest gfx architecture does not match the hardware receipt")

    vram_match = re.search(
        r"VRAM Total Memory \(B\):\s*([0-9]+)", str(hardware.get("smi_vram", ""))
    )
    vram_bytes = int(vram_match.group(1)) if vram_match else 0
    if vram_bytes < 1_000_000_000:
        errors.append("hardware receipt has no plausible VRAM total")
    if manifest.get("vram_bytes") != vram_bytes:
        errors.append("manifest VRAM does not match the hardware receipt")

    runtime = str(environment.get("rocm_hip", "")).strip()
    expected_runtime = f"ROCm/HIP {runtime}" if runtime else ""
    if not runtime or manifest.get("runtime") != expected_runtime:
        errors.append("manifest ROCm/HIP runtime does not match the environment receipt")
    training_device = training.get("device")
    if not isinstance(training_device, Mapping):
        errors.append("training device receipt is missing")
    else:
        if training_device.get("rocm_hip") != runtime:
            errors.append("environment and training runtime mismatch")
        if training_device.get("torch") != environment.get("torch"):
            errors.append("environment and training PyTorch mismatch")
        if training_device.get("device_count") != environment.get("device_count"):
            errors.append("environment and training device-count mismatch")

    try:
        seconds = float(training.get("gpu_train_seconds"))
    except (TypeError, ValueError):
        seconds = 0.0
    if not math.isfinite(seconds) or seconds <= 0:
        errors.append("training receipt has no valid elapsed time")
    if manifest.get("training_seconds") != seconds:
        errors.append("manifest training time does not match the training receipt")

    policies = training.get("policies")
    expanding = policies.get("gpu_mlp_expanding") if isinstance(policies, Mapping) else None
    retrains = expanding.get("n_retrains") if isinstance(expanding, Mapping) else None
    if not isinstance(retrains, int) or retrains < 1:
        errors.append("training receipt has no walk-forward retrain count")
    if manifest.get("requests_processed") != retrains:
        errors.append("manifest retrain count does not match the training receipt")
    return errors


def validate_manifest(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Return every publication error; an empty list means the chain is valid."""
    errors: list[str] = []
    if "AMD" not in str(manifest.get("vendor", "")):
        errors.append("receipt does not identify AMD as the device vendor")
    if manifest.get("gfx_version") != "gfx1100":
        errors.append("expected the recorded gfx1100 architecture")
    if not str(manifest.get("runtime", "")).startswith("ROCm/HIP "):
        errors.append("ROCm/HIP runtime is missing")
    if not manifest.get("workload") or not manifest.get("training_seconds"):
        errors.append("meaningful workload or timing evidence is missing")

    receipts = manifest.get("receipts", {})
    if not isinstance(receipts, Mapping) or not receipts:
        return [*errors, "receipt map is empty"]
    expected_names = {"environment", "hardware", "training"}
    if set(receipts) != expected_names:
        errors.append("receipt map must contain environment, hardware, and training")

    resolved_root = root.resolve()
    parsed_receipts: dict[str, Mapping[str, Any]] = {}
    for name, record in receipts.items():
        if not isinstance(record, Mapping):
            errors.append(f"{name} receipt record is invalid")
            continue
        path = (resolved_root / str(record.get("path", ""))).resolve()
        if resolved_root != path and resolved_root not in path.parents:
            errors.append(f"{name} receipt path is unsafe or missing")
            continue
        if not path.is_file():
            errors.append(f"{name} receipt path is unsafe or missing")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            errors.append(f"{name} receipt hash mismatch")
        parsed = _load_json(path, name=name, errors=errors)
        if parsed is not None:
            parsed_receipts[name] = parsed
    if set(parsed_receipts) == expected_names:
        errors.extend(_semantic_receipt_errors(manifest, parsed_receipts))
    return errors


def main() -> int:
    manifest_path = ROOT / "results/amd_compute.json"
    if not manifest_path.is_file():
        print("FAIL AMD manifest missing; run scripts/build_fusion_artifacts.py")
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL AMD manifest is invalid JSON: {exc}")
        return 1

    errors = validate_manifest(ROOT, manifest)
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS AMD receipt chain: vendor=AMD gfx=gfx1100 runtime=" + manifest["runtime"])
    print(
        f"PASS workload={manifest['workload']} "
        f"training_seconds={manifest['training_seconds']}"
    )
    print("NOTE commercial SKU name is not asserted because the source receipt left it blank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
