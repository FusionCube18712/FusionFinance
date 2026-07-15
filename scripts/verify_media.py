#!/usr/bin/env python3
"""Verify the public demo video against the judge-facing release contract."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "presentation/FusionFinance_Demo.mp4"
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
EXPECTED_FRAME_RATE = 30.0
EXPECTED_PIXEL_FORMAT = "yuv420p"
EXPECTED_DURATION_SECONDS = 250.0
DURATION_TOLERANCE_SECONDS = 0.25
GITHUB_MAX_FILE_BYTES = 100 * 1024 * 1024
ISO_MP4_MAJOR_BRANDS = frozenset(
    {"avc1", "dash", "iso2", "iso4", "iso5", "iso6", "isom", "mp41", "mp42"}
)

Runner = Callable[..., subprocess.CompletedProcess[str]]
Probe = Callable[[Path], dict[str, Any]]


class MediaVerificationError(ValueError):
    """Raised when the public video cannot satisfy the release contract."""


@dataclass(frozen=True)
class MediaReport:
    """Immutable summary of a conforming release video."""

    path: Path
    size_bytes: int
    width: int
    height: int
    video_codec: str
    audio_codec: str
    duration_seconds: float
    video_duration_seconds: float
    audio_duration_seconds: float
    container_format: str
    frame_rate: float
    pixel_format: str
    major_brand: str


def probe_media(
    path: Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Return the minimal ffprobe JSON needed by the release contract."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "format=format_name,duration:"
            "stream=index,codec_type,codec_name,width,height,duration,"
            "avg_frame_rate,pix_fmt:"
            "stream_disposition=default:format_tags=major_brand"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise MediaVerificationError(
            "ffprobe is required; install FFmpeg before verifying release media"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaVerificationError("ffprobe timed out after 30 seconds") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        raise MediaVerificationError(f"ffprobe failed: {detail[:500]}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaVerificationError("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MediaVerificationError("ffprobe JSON must be an object")
    return payload


def _stream(payload: dict[str, Any], codec_type: str) -> dict[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaVerificationError("ffprobe did not return a stream list")
    matches = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type
    ]
    if not matches:
        raise MediaVerificationError(f"release video is missing a {codec_type} stream")
    defaults = [
        stream
        for stream in matches
        if isinstance(stream.get("disposition"), dict)
        and stream["disposition"].get("default") == 1
    ]
    if len(defaults) == 1:
        return defaults[0]
    if len(defaults) > 1:
        raise MediaVerificationError(
            f"release video has multiple default {codec_type} streams"
        )
    if len(matches) == 1:
        return matches[0]
    raise MediaVerificationError(
        f"release video must identify one default {codec_type} stream"
    )


def _finite_duration(raw_duration: Any, *, label: str) -> float:
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise MediaVerificationError(f"{label} is unavailable") from exc
    if not math.isfinite(duration):
        raise MediaVerificationError(f"{label} must be finite")
    return duration


def _format_contract(payload: dict[str, Any]) -> tuple[str, str, float]:
    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        raise MediaVerificationError("ffprobe did not return container metadata")
    container_format = str(format_data.get("format_name", "")).lower()
    if "mp4" not in {name.strip() for name in container_format.split(",")}:
        raise MediaVerificationError(
            "release video must use an MP4 container, "
            f"found {container_format or 'unknown'}"
        )
    tags = format_data.get("tags")
    major_brand = (
        str(tags.get("major_brand", "")).strip().lower()
        if isinstance(tags, dict)
        else ""
    )
    if major_brand not in ISO_MP4_MAJOR_BRANDS:
        raise MediaVerificationError(
            "release video must use an ISO MP4 brand, "
            f"found {major_brand or 'unknown'}"
        )
    duration = _finite_duration(
        format_data.get("duration"), label="release video duration"
    )
    return container_format, major_brand, duration


def _frame_rate(raw_frame_rate: Any) -> float:
    try:
        frame_rate = float(Fraction(str(raw_frame_rate)))
    except (ValueError, ZeroDivisionError) as exc:
        raise MediaVerificationError("video frame rate is unavailable") from exc
    if not math.isfinite(frame_rate):
        raise MediaVerificationError("video frame rate must be finite")
    return frame_rate


def verify_media(path: Path, *, probe: Probe | None = None) -> MediaReport:
    """Validate one MP4 against the public FusionFinance release requirements."""
    video = path.resolve()
    if not video.is_file():
        raise MediaVerificationError(f"release video does not exist: {video}")

    size_bytes = video.stat().st_size
    if size_bytes <= 0:
        raise MediaVerificationError("release video is empty")
    if size_bytes >= GITHUB_MAX_FILE_BYTES:
        size_mib = size_bytes / (1024 * 1024)
        raise MediaVerificationError(
            f"release video is {size_mib:.2f} MiB; GitHub requires files under 100 MiB"
        )

    payload = (probe or probe_media)(video)
    video_stream = _stream(payload, "video")
    audio_stream = _stream(payload, "audio")
    container_format, major_brand, duration_seconds = _format_contract(payload)
    video_codec = str(video_stream.get("codec_name", "")).lower()
    audio_codec = str(audio_stream.get("codec_name", "")).lower()
    width = video_stream.get("width")
    height = video_stream.get("height")
    raw_frame_rate = video_stream.get("avg_frame_rate")
    pixel_format = str(video_stream.get("pix_fmt", "")).lower()

    if video_codec != "h264":
        raise MediaVerificationError(
            f"release video must use H.264, found {video_codec or 'unknown'}"
        )
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise MediaVerificationError(
            "release video must be 1920x1080, "
            f"found {width or 'unknown'}x{height or 'unknown'}"
        )
    if audio_codec != "aac":
        raise MediaVerificationError(
            f"release audio must use AAC, found {audio_codec or 'unknown'}"
        )
    frame_rate = _frame_rate(raw_frame_rate)
    if abs(frame_rate - EXPECTED_FRAME_RATE) > 0.001:
        raise MediaVerificationError(
            f"release video must use 30 fps, found {frame_rate:.3f} fps"
        )
    if pixel_format != EXPECTED_PIXEL_FORMAT:
        raise MediaVerificationError(
            "release video must use yuv420p for judge-compatible playback, "
            f"found {pixel_format or 'unknown'}"
        )
    video_duration_seconds = _finite_duration(
        video_stream.get("duration"), label="video stream duration"
    )
    audio_duration_seconds = _finite_duration(
        audio_stream.get("duration"), label="audio stream duration"
    )
    if abs(duration_seconds - EXPECTED_DURATION_SECONDS) > DURATION_TOLERANCE_SECONDS:
        raise MediaVerificationError(
            "release duration must be "
            f"{EXPECTED_DURATION_SECONDS:.2f}±{DURATION_TOLERANCE_SECONDS:.2f} seconds, "
            f"found {duration_seconds:.3f}"
        )
    for label, stream_duration in (
        ("video stream duration", video_duration_seconds),
        ("audio stream duration", audio_duration_seconds),
    ):
        if abs(stream_duration - EXPECTED_DURATION_SECONDS) > DURATION_TOLERANCE_SECONDS:
            raise MediaVerificationError(
                f"{label} must be "
                f"{EXPECTED_DURATION_SECONDS:.2f}±{DURATION_TOLERANCE_SECONDS:.2f} "
                f"seconds, found {stream_duration:.3f}"
            )

    return MediaReport(
        path=video,
        size_bytes=size_bytes,
        width=int(width),
        height=int(height),
        video_codec=video_codec,
        audio_codec=audio_codec,
        duration_seconds=duration_seconds,
        video_duration_seconds=video_duration_seconds,
        audio_duration_seconds=audio_duration_seconds,
        container_format=container_format,
        frame_rate=frame_rate,
        pixel_format=pixel_format,
        major_brand=major_brand,
    )


def _success_message(report: MediaReport) -> str:
    size_mib = report.size_bytes / (1024 * 1024)
    return (
        "PASS release media: "
        f"MP4/{report.major_brand}, {report.width}x{report.height} "
        f"at {report.frame_rate:.3f} fps, "
        f"{report.video_codec}/{report.pixel_format} + {report.audio_codec}, "
        f"{report.duration_seconds:.3f}s, {size_mib:.2f} MiB"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help="MP4 to verify (default: presentation/FusionFinance_Demo.mp4)",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_media(args.video)
    except MediaVerificationError as exc:
        print(f"FAIL release media: {exc}", file=sys.stderr)
        return 1
    print(_success_message(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
