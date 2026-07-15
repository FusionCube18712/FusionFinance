from __future__ import annotations

import json
import importlib.util
import re
import tomllib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_archive_builder():
    script = ROOT / "scripts/build_submission_archive.py"
    spec = importlib.util.spec_from_file_location("build_submission_archive", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_demo_is_the_only_public_demo_surface() -> None:
    replay = json.loads((ROOT / "demo/replay.json").read_text(encoding="utf-8"))
    index = (ROOT / "demo/index.html").read_text(encoding="utf-8")

    assert replay["claim_status"] == "provisional_uncontrolled_legacy_race"
    assert len(replay["decisions"]) == 12
    assert "Content-Security-Policy" in index
    assert "trade ledgers are shown as an em dash" in index
    assert "not causal proof" in index
    assert not (ROOT / "demo/race.html").exists()
    assert not (ROOT / "demo/slides.html").exists()
    assert not (ROOT / "demo/FusionFinance_Slides.pdf").exists()


def test_package_metadata_has_one_gpl_source_of_truth() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert project["license"] == "GPL-3.0-only"
    assert not any("License ::" in item for item in project["classifiers"])
    assert "alpha/filing_alpha/NOTICE" in project["license-files"]


def test_public_markdown_links_do_not_point_to_removed_local_files() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    failures: list[str] = []
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

    for document in markdown_files:
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path_text = target.split("#", 1)[0]
            candidate = (document.parent / path_text).resolve()
            if ROOT != candidate and ROOT not in candidate.parents:
                failures.append(f"{document.relative_to(ROOT)}: unsafe link {target}")
            elif not candidate.exists():
                failures.append(f"{document.relative_to(ROOT)}: missing link {target}")

    assert failures == []


def test_submission_archive_is_deterministic_and_allowlisted(tmp_path: Path) -> None:
    builder = _load_archive_builder()
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_count, first_digest = builder.build_archive(first)
    second_count, second_digest = builder.build_archive(second)

    assert first_count == second_count
    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
    assert "fusionfinance/README.md" in names
    assert "fusionfinance/evidence/replay/v1_source.json" in names
    assert "fusionfinance/presentation/FusionFinance_Demo.mp4" in names
    assert "fusionfinance/presentation/FusionFinance_Submission.pdf" in names
    assert "fusionfinance/SHA256SUMS.txt" in names
    assert not any(
        name.startswith(
            (
                "fusionfinance/TradingAgents/",
                "fusionfinance/books/",
                "fusionfinance/evalh/",
                "fusionfinance/receipts/",
            )
        )
        for name in names
    )
    assert not any(name.lower().endswith((".wav", ".aiff", ".m4a")) for name in names)


def test_archive_binary_allowlist_excludes_private_media_shapes() -> None:
    builder = _load_archive_builder()

    assert builder.PUBLIC_BINARY_PATHS == {
        "docs/assets/dashboard.png",
        "presentation/FusionFinance_Demo.mp4",
        "presentation/FusionFinance_Submission.pdf",
    }
    assert builder._max_file_bytes(Path("presentation/FusionFinance_Demo.mp4")) >= (
        100 * 1024 * 1024
    )
    assert builder._max_file_bytes(Path("presentation/avatar.mp4")) == 0
    assert builder._max_file_bytes(Path("presentation/presenter.jpg")) == 0
    collected_binary_paths = {
        path.relative_to(ROOT).as_posix()
        for path in builder.collect_files()
        if not builder._is_text_path(path.relative_to(ROOT))
    }
    assert collected_binary_paths == builder.PUBLIC_BINARY_PATHS


def test_archive_default_denies_unrecognized_text_and_binary_shapes(
    tmp_path: Path,
) -> None:
    builder = _load_archive_builder()

    forbidden = (
        "docs/narration.flac",
        "docs/narration.ogg",
        "docs/voice.opus",
        "docs/audio.aac",
        "docs/presenter.heic",
        "docs/presenter.tiff",
        "docs/presenter.avif",
        "docs/diagram.svg",
        "docs/private.sqlite",
        "docs/model.bin",
        "docs/unscanned.cfg",
        "docs/extensionless-binary",
    )

    assert all(builder._max_file_bytes(Path(path)) == 0 for path in forbidden)
    assert builder._max_file_bytes(Path("README.md")) > 0
    assert builder._max_file_bytes(Path("LICENSE")) > 0
    assert builder._max_file_bytes(Path("alpha/filing_alpha/NOTICE")) > 0

    disguised_binary = tmp_path / "voice.txt"
    disguised_binary.write_bytes(b"RIFF\x00private-audio")
    assert builder._is_utf8_text(disguised_binary) is False


def test_deck_copy_matches_the_public_evidence_boundary() -> None:
    deck = (ROOT / "presentation/FusionFinance_Submission.html").read_text(
        encoding="utf-8"
    )

    assert "0.62" not in deck
    assert "−280 bps" not in deck
    assert "PASS</span> ticker and horizon" not in deck
    assert "quantitative verification workloads" not in deck
    assert "Public GitHub URL must be attached" not in deck
    assert "confidence: 0.60" in deck
    assert "Expected outcome: not recorded" in deck
    assert "sparse five-session marks" in deck
    assert "not independent attestation" in deck
    assert "https://github.com/FusionCube18712/FusionFinance" in deck
    assert "https://fusionfinance2.vercel.app" in deck
