from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    script = ROOT / "scripts/verify_media.py"
    spec = importlib.util.spec_from_file_location("verify_media", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_probe(*, duration: str = "250.048") -> dict[str, object]:
    return {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": duration,
            "tags": {"major_brand": "isom"},
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
                "duration": "250.000",
                "disposition": {"default": 1},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": duration,
                "disposition": {"default": 1},
            },
        ],
    }


def test_release_media_contract_accepts_the_final_master_shape(tmp_path: Path) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")

    report = verifier.verify_media(video, probe=lambda _: _valid_probe())

    assert (
        report.width,
        report.height,
        report.video_codec,
        report.audio_codec,
        report.duration_seconds,
    ) == (1920, 1080, "h264", "aac", 250.048)
    success = verifier._success_message(report)
    assert "30.000 fps" in success
    assert "yuv420p" in success


def test_checked_in_release_video_passes_a_real_ffprobe() -> None:
    verifier = _load_verifier()

    report = verifier.verify_media(verifier.DEFAULT_VIDEO)

    assert report.path == verifier.DEFAULT_VIDEO.resolve()
    assert report.frame_rate == 30.0
    assert report.pixel_format == "yuv420p"


def test_release_media_contract_rejects_a_renamed_non_mp4_container(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")
    probe_data = _valid_probe()
    probe_data["format"] = {
        "format_name": "matroska,webm",
        "duration": "250.048",
    }

    with pytest.raises(verifier.MediaVerificationError, match="MP4 container"):
        verifier.verify_media(video, probe=lambda _: probe_data)


def test_release_media_contract_rejects_a_quicktime_mov_renamed_mp4(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")
    probe_data = _valid_probe()
    probe_data["format"] = {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "250.048",
        "tags": {"major_brand": "qt  "},
    }

    with pytest.raises(verifier.MediaVerificationError, match="ISO MP4 brand"):
        verifier.verify_media(video, probe=lambda _: probe_data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("avg_frame_rate", "24/1", "30 fps"),
        ("pix_fmt", "yuv444p", "yuv420p"),
    ],
)
def test_release_media_contract_requires_judge_compatible_video_encoding(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")
    probe_data = _valid_probe()
    video_stream = probe_data["streams"][0]
    assert isinstance(video_stream, dict)
    video_stream[field] = value

    with pytest.raises(verifier.MediaVerificationError, match=message):
        verifier.verify_media(video, probe=lambda _: probe_data)


def test_release_media_contract_validates_the_default_streams(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")
    probe_data = _valid_probe()
    probe_data["streams"] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "duration": "250.000",
            "disposition": {"default": 0},
        },
        {
            "index": 1,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 640,
            "height": 360,
            "duration": "250.000",
            "disposition": {"default": 1},
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "aac",
            "duration": "250.048",
            "disposition": {"default": 1},
        },
    ]

    with pytest.raises(verifier.MediaVerificationError, match="1920x1080"):
        verifier.verify_media(video, probe=lambda _: probe_data)


def test_release_media_contract_rejects_short_primary_streams_in_a_padded_file(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")
    probe_data = _valid_probe()
    probe_data["streams"] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30/1",
            "pix_fmt": "yuv420p",
            "duration": "1.000",
            "disposition": {"default": 1},
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "duration": "1.000",
            "disposition": {"default": 1},
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "aac",
            "duration": "250.048",
            "disposition": {"default": 0},
        },
    ]

    with pytest.raises(verifier.MediaVerificationError, match="video stream duration"):
        verifier.verify_media(video, probe=lambda _: probe_data)


@pytest.mark.parametrize(
    ("probe_patch", "message"),
    [
        ({"streams": [{"codec_type": "audio", "codec_name": "aac"}]}, "video"),
        (
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "hevc",
                        "width": 1920,
                        "height": 1080,
                    },
                    {"codec_type": "audio", "codec_name": "aac"},
                ]
            },
            "H.264",
        ),
        (
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1280,
                        "height": 720,
                    },
                    {"codec_type": "audio", "codec_name": "aac"},
                ]
            },
            "1920x1080",
        ),
        (
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                    }
                ]
            },
            "audio",
        ),
        (
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                    },
                    {"codec_type": "audio", "codec_name": "mp3"},
                ]
            },
            "AAC",
        ),
        (
            {
                "format": {
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                    "duration": "251.0",
                    "tags": {"major_brand": "isom"},
                }
            },
            "duration",
        ),
    ],
)
def test_release_media_contract_rejects_nonconforming_probe_data(
    tmp_path: Path,
    probe_patch: dict[str, object],
    message: str,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")
    probe_data = _valid_probe()
    probe_data.update(probe_patch)

    with pytest.raises(verifier.MediaVerificationError, match=message):
        verifier.verify_media(video, probe=lambda _: probe_data)


def test_release_media_contract_rejects_the_github_size_boundary(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    with video.open("wb") as fixture:
        fixture.truncate(verifier.GITHUB_MAX_FILE_BYTES)

    with pytest.raises(verifier.MediaVerificationError, match="GitHub"):
        verifier.verify_media(video, probe=lambda _: _valid_probe())


def test_ffprobe_is_invoked_without_a_shell_and_returns_json(tmp_path: Path) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")
    observed: dict[str, object] = {}

    def runner(command, **kwargs):
        observed.update({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(_valid_probe()),
            stderr="",
        )

    payload = verifier.probe_media(video, runner=runner)

    assert payload["format"] == {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "250.048",
        "tags": {"major_brand": "isom"},
    }
    assert observed["shell"] is False
    assert observed["command"][-1] == str(video)


def test_ffprobe_failure_is_reported_without_leaking_a_traceback(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"synthetic fixture")

    def runner(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="invalid media",
        )

    with pytest.raises(verifier.MediaVerificationError, match="ffprobe failed"):
        verifier.probe_media(video, runner=runner)


def test_release_documentation_exposes_the_media_verification_command() -> None:
    submission = (ROOT / "docs/SUBMISSION_PACKAGE.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "make verify-media" in submission
    assert "verify-media:" in makefile
