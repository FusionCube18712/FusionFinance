from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORYBOARD = ROOT / "presentation/video-storyboard.json"
VIDEO_SCRIPT = ROOT / "presentation/video-script.md"


def _load_storyboard() -> dict:
    return json.loads(STORYBOARD.read_text(encoding="utf-8"))


def test_video_storyboard_has_a_complete_hackathon_arc() -> None:
    data = _load_storyboard()
    scenes = data["scenes"]

    assert data["schema_version"] == 1
    assert data["resolution"] == [1920, 1080]
    assert data["fps"] == 30
    assert 225 <= data["target_duration_seconds"] <= 270
    assert [scene["id"] for scene in scenes] == [
        "hook",
        "fair-race",
        "pure-ml",
        "pure-llm",
        "fusion-decision",
        "results-boundary",
        "amd-receipts",
        "close",
    ]
    assert sum(scene["duration_seconds"] for scene in scenes) == data[
        "target_duration_seconds"
    ]


def test_video_storyboard_preserves_the_claim_boundary_and_submission_facts() -> None:
    data = _load_storyboard()
    narration = " ".join(scene["narration"] for scene in data["scenes"])
    on_screen = " ".join(
        text for scene in data["scenes"] for text in scene["on_screen"]
    )

    assert "+10.1%" in on_screen
    assert "2.05 Sharpe" in on_screen
    assert "−3.9% max drawdown" in on_screen
    assert "gfx1100" in narration
    assert "231.29 seconds" in narration
    assert "provisional" in narration.lower()
    assert "not causal proof" in narration.lower()
    assert "research" in narration.lower()
    assert "not financial advice" in on_screen.lower()


def test_avatar_is_used_as_a_punctuating_device_not_a_constant_talking_head() -> None:
    data = _load_storyboard()
    avatar_seconds = sum(
        scene["duration_seconds"] * scene["avatar_fraction"]
        for scene in data["scenes"]
    )

    assert 0.08 <= avatar_seconds / data["target_duration_seconds"] <= 0.25
    assert data["scenes"][0]["avatar_fraction"] == 1.0
    assert data["scenes"][-1]["avatar_fraction"] == 1.0


def test_caption_beats_are_short_and_readable() -> None:
    data = _load_storyboard()
    captions = [
        caption
        for scene in data["scenes"]
        for caption in scene["caption_beats"]
    ]

    assert captions
    assert all(2 <= len(caption.split()) <= 7 for caption in captions)
    assert all("\n" not in caption for caption in captions)


def test_public_video_script_matches_the_storyboard_narration() -> None:
    data = _load_storyboard()
    script = VIDEO_SCRIPT.read_text(encoding="utf-8")

    assert "Target length: 4:10" in script
    for scene in data["scenes"]:
        assert scene["narration"] in script


def test_video_walkthrough_uses_only_fields_in_the_sealed_mu_sell_record() -> None:
    data = _load_storyboard()
    replay = json.loads((ROOT / "results/demo_run.json").read_text(encoding="utf-8"))
    mu_sell = next(
        item
        for item in replay["decisions"]
        if item["asset"] == "MU" and item["proposed_action"] == "sell"
    )
    scene = next(item for item in data["scenes"] if item["id"] == "fusion-decision")
    copy = " ".join([scene["narration"], *scene["on_screen"]])

    assert mu_sell["llm_confidence"] == 0.6
    assert "0.60" in copy or "60 percent" in copy
    assert "−280" not in copy
    assert "expected move −" not in copy.lower()
    assert "expected move of" not in copy.lower()
    assert "schema passed" not in copy.lower()
    assert "missing horizon" in copy.lower()


def test_video_never_presents_the_legacy_replay_as_the_controlled_race() -> None:
    data = _load_storyboard()
    fair_scene = next(item for item in data["scenes"] if item["id"] == "fair-race")
    result_scene = next(
        item for item in data["scenes"] if item["id"] == "results-boundary"
    )
    fair_copy = " ".join([fair_scene["narration"], *fair_scene["on_screen"]]).lower()
    result_copy = " ".join(
        [result_scene["narration"], *result_scene["on_screen"]]
    ).lower()

    assert "one fair race" not in fair_copy
    assert "different legacy kernels" in fair_copy
    assert "sparse five-session marks" in result_copy
    assert "contains the ml champion" in result_copy


def test_video_describes_receipt_hashes_as_integrity_not_attestation() -> None:
    data = _load_storyboard()
    scene = next(item for item in data["scenes"] if item["id"] == "amd-receipts")
    copy = " ".join([scene["narration"], *scene["on_screen"]]).lower()

    assert "integrity" in copy
    assert "not independent attestation" in copy
