from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_fusion_artifacts.py"
PUBLIC_SOURCE = ROOT / "evidence/replay/v1_source.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_fusion_artifacts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_artifacts_rebuild_byte_identically_from_checked_in_evidence() -> None:
    assert PUBLIC_SOURCE.is_file()
    builder = _load_builder()

    first = builder.render_artifacts(builder.build_payload(ROOT))
    second = builder.render_artifacts(builder.build_payload(ROOT))

    assert first == second
    for relative_path, expected_bytes in first.items():
        assert (ROOT / relative_path).read_bytes() == expected_bytes


def test_public_source_preserves_the_v1_claim_boundary() -> None:
    source = json.loads(PUBLIC_SOURCE.read_text(encoding="utf-8"))

    assert source["claim_status"] == "provisional_uncontrolled_legacy_race"
    assert len(source["dates"]) == 109
    assert len(source["decisions"]) == 12
    assert set(source["arms"]) == {"pure_ml", "pure_llm", "fusion", "benchmark"}
    assert all(arm["daily_returns"] for arm in source["arms"].values())


def _copy_public_evidence(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    shutil.copytree(ROOT / "evidence", candidate / "evidence")
    return candidate


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("smi_short", "Card Vendor: Not AMD\nGFX Version: gfx1100\n", "AMD"),
        (
            "smi_short",
            "Card Vendor: Advanced Micro Devices, Inc. [AMD/ATI]\n"
            "GFX Version: not-gfx\n",
            "gfx architecture",
        ),
        ("smi_vram", "VRAM Total Memory (B): 1\n", "VRAM"),
    ],
)
def test_public_builder_rejects_semantically_tampered_hardware(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    builder = _load_builder()
    candidate = _copy_public_evidence(tmp_path)
    hardware_path = candidate / "evidence/amd/hardware.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware[field] = replacement
    hardware_path.write_text(json.dumps(hardware), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        builder.build_payload(candidate)


def test_public_builder_rejects_cross_receipt_runtime_mismatch(tmp_path: Path) -> None:
    builder = _load_builder()
    candidate = _copy_public_evidence(tmp_path)
    training_path = candidate / "evidence/amd/training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["device"]["rocm_hip"] = "tampered-runtime"
    training_path.write_text(json.dumps(training), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime mismatch"):
        builder.build_payload(candidate)
