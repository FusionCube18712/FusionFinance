from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_amd.py"
EXPECTED_RECEIPTS = {
    "environment": (
        "evidence/amd/environment.json",
        "cf7382dc842323579a87f031c09a9384db555c58f63a65dcf64bd23a36b10fcf",
    ),
    "hardware": (
        "evidence/amd/hardware.json",
        "b1dab964b344b00e80780b19097652492a020b65539f8ee175f658a046db2ab8",
    ),
    "training": (
        "evidence/amd/training.json",
        "4a9c246d0345b3a32f1660ad77a75809098f160f82d285de3d5a0e0969f85894",
    ),
}


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_amd", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_amd_receipts_preserve_the_recorded_bytes() -> None:
    for relative_path, expected_digest in EXPECTED_RECEIPTS.values():
        path = ROOT / relative_path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_digest


def test_amd_verifier_fails_closed_on_tampering(tmp_path: Path) -> None:
    verifier = _load_verifier()
    manifest = json.loads((ROOT / "results/amd_compute.json").read_text())

    clean_errors = verifier.validate_manifest(ROOT, manifest)
    assert clean_errors == []

    receipt_root = tmp_path / "candidate"
    for relative_path, _ in EXPECTED_RECEIPTS.values():
        source = ROOT / relative_path
        destination = receipt_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    manifest_path = receipt_root / "results/amd_compute.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    training = receipt_root / EXPECTED_RECEIPTS["training"][0]
    training.write_bytes(training.read_bytes() + b"\n")
    errors = verifier.validate_manifest(receipt_root, manifest)

    assert any("training receipt hash mismatch" in error for error in errors)


def test_amd_verifier_rejects_rehashed_semantic_tampering(tmp_path: Path) -> None:
    verifier = _load_verifier()
    candidate = tmp_path / "candidate"
    shutil.copytree(ROOT / "evidence", candidate / "evidence")
    manifest = json.loads((ROOT / "results/amd_compute.json").read_text())

    hardware_path = candidate / "evidence/amd/hardware.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware["smi_short"] = "Card Vendor: Not AMD\nGFX Version: not-gfx\n"
    hardware["smi_vram"] = "VRAM Total Memory (B): 1\n"
    hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
    manifest["receipts"]["hardware"]["sha256"] = hashlib.sha256(
        hardware_path.read_bytes()
    ).hexdigest()

    errors = verifier.validate_manifest(candidate, manifest)

    assert any("AMD" in error for error in errors)
    assert any("gfx" in error for error in errors)
    assert any("VRAM" in error for error in errors)
