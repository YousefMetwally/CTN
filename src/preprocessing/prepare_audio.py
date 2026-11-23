#!/usr/bin/env python3
"""
Prepare media files for speech-recognition experiments.

Features:
- Recursively or non-recursively scans an input directory.
- Inspects media streams with ffprobe.
- Extracts audio as:
    - mono
    - 16 kHz
    - signed 16-bit little-endian PCM
    - WAV
- Detects embedded subtitle streams.
- Attempts to extract every subtitle stream as SRT.
- Optionally normalizes audio loudness.
- Verifies generated WAV files.
- Produces a detailed JSON report.

Only Python's standard library is required.

External requirements:
- ffmpeg
- ffprobe
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
    ".mts",
    ".m2ts",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".flac",
    ".wav",
    ".wma",
}

# These subtitle codecs are normally text-based and can usually be converted
# to SRT by FFmpeg.
TEXT_SUBTITLE_CODECS = {
    "subrip",
    "srt",
    "mov_text",
    "ass",
    "ssa",
    "webvtt",
    "text",
    "microdvd",
    "jacosub",
    "sami",
    "realtext",
    "subviewer",
    "subviewer1",
    "mpl2",
    "vplayer",
    "pjs",
}


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None = None
    codec_long_name: str | None = None
    language: str | None = None
    title: str | None = None
    disposition_default: bool = False
    disposition_forced: bool = False
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class SubtitleExtraction:
    stream_index: int
    subtitle_number: int
    codec_name: str | None
    language: str | None
    title: str | None
    output_file: str | None
    status: str
    error: str | None = None


@dataclass
class FileReport:
    input_file: str
    relative_path: str
    status: str

    format_name: str | None = None
    format_long_name: str | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    bit_rate: int | None = None

    video_stream_count: int = 0
    audio_stream_count: int = 0
    subtitle_stream_count: int = 0

    streams: list[StreamInfo] = field(default_factory=list)

    audio_output: str | None = None
    audio_status: str | None = None
    audio_verification: dict[str, Any] | None = None

    subtitles: list[SubtitleExtraction] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect media files, extract Whisper-ready audio, "
            "and extract embedded subtitles."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing source media files.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where processed audio, subtitles, and reports are saved.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for media files inside subdirectories.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing audio and subtitle files.",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply FFmpeg loudness normalization to extracted audio.",
    )

    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Inspect subtitle streams but do not extract them.",
    )

    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable or path. Default: ffmpeg",
    )

    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="FFprobe executable or path. Default: ffprobe",
    )

    parser.add_argument(
        "--report-name",
        default="conversion_report.json",
        help="JSON report filename. Default: conversion_report.json",
    )

    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an external command and capture its output."""
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def ensure_executable_exists(executable: str) -> None:
    """
    Verify that an executable is available.

    This supports both commands discoverable through PATH and explicit paths.
    """
    candidate = Path(executable).expanduser()

    if candidate.parent != Path(".") or "/" in executable:
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Executable not found: {candidate}"
            )
        return

    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Executable '{executable}' was not found in PATH."
        )


def discover_media_files(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:
    """Find supported media files."""
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")

    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def probe_media(
    input_file: Path,
    ffprobe: str,
) -> dict[str, Any]:
    """Return complete FFprobe stream and format metadata."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_file),
    ]

    result = run_command(command)

    if result.returncode != 0:
        error = result.stderr.strip() or "Unknown ffprobe error."
        raise RuntimeError(error)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"FFprobe returned invalid JSON: {exc}"
        ) from exc


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stream_from_probe(raw_stream: dict[str, Any]) -> StreamInfo:
    tags = raw_stream.get("tags") or {}
    disposition = raw_stream.get("disposition") or {}

    return StreamInfo(
        index=optional_int(raw_stream.get("index")) or 0,
        codec_type=str(raw_stream.get("codec_type", "unknown")),
        codec_name=raw_stream.get("codec_name"),
        codec_long_name=raw_stream.get("codec_long_name"),
        language=tags.get("language"),
        title=tags.get("title"),
        disposition_default=bool(disposition.get("default", 0)),
        disposition_forced=bool(disposition.get("forced", 0)),
        sample_rate=optional_int(raw_stream.get("sample_rate")),
        channels=optional_int(raw_stream.get("channels")),
        channel_layout=raw_stream.get("channel_layout"),
        width=optional_int(raw_stream.get("width")),
        height=optional_int(raw_stream.get("height")),
    )


def safe_filename_component(
    value: str | None,
    fallback: str,
) -> str:
    """
    Convert metadata such as language or title into a safe filename component.
    """
    if not value:
        return fallback

    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_-]+", "", cleaned)

    return cleaned or fallback


def create_output_stem(
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Preserve source subdirectories in the output directory.

    Example:
        downloads/interviews/example.mp4
        output/interviews/example
    """
    relative_path = input_file.relative_to(input_dir)
    target_parent = output_dir / relative_path.parent
    target_parent.mkdir(parents=True, exist_ok=True)

    return target_parent / input_file.stem


def extract_audio(
    input_file: Path,
    output_file: Path,
    ffmpeg: str,
    overwrite: bool,
    normalize: bool,
) -> tuple[str, str | None]:
    """
    Extract the first audio stream as mono 16 kHz PCM WAV.

    Returns:
        (status, error)
    """
    if output_file.exists() and not overwrite:
        return "skipped_existing", None

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_file),

        # Select the first audio stream.
        "-map",
        "0:a:0",

        # Do not include video or subtitles in the WAV.
        "-vn",
        "-sn",

        # Whisper-oriented output format.
        "-ac",
        "1",
        "-ar",
        "16000",
    ]

    if normalize:
        command.extend([
            "-af",
            "loudnorm=I=-16:LRA=11:TP=-1.5",
        ])

    command.extend([
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ])

    result = run_command(command)

    if result.returncode != 0:
        output_file.unlink(missing_ok=True)
        return "failed", result.stderr.strip() or "Unknown FFmpeg error."

    if not output_file.exists() or output_file.stat().st_size == 0:
        return "failed", "FFmpeg did not create a valid output file."

    return "converted", None


def verify_wav(
    wav_file: Path,
    ffprobe: str,
) -> dict[str, Any]:
    """Verify that the generated WAV has the expected properties."""
    probe_data = probe_media(wav_file, ffprobe)
    raw_streams = probe_data.get("streams", [])

    audio_stream = next(
        (
            stream
            for stream in raw_streams
            if stream.get("codec_type") == "audio"
        ),
        None,
    )

    if audio_stream is None:
        return {
            "valid": False,
            "reason": "No audio stream was found.",
        }

    codec_name = audio_stream.get("codec_name")
    sample_rate = optional_int(audio_stream.get("sample_rate"))
    channels = optional_int(audio_stream.get("channels"))
    sample_format = audio_stream.get("sample_fmt")

    valid = (
        codec_name == "pcm_s16le"
        and sample_rate == 16000
        and channels == 1
    )

    return {
        "valid": valid,
        "codec_name": codec_name,
        "sample_format": sample_format,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": optional_float(
            audio_stream.get("duration")
            or (probe_data.get("format") or {}).get("duration")
        ),
        "size_bytes": optional_int(
            (probe_data.get("format") or {}).get("size")
        ),
    }


def subtitle_output_path(
    output_stem: Path,
    subtitle_number: int,
    stream: StreamInfo,
) -> Path:
    language = safe_filename_component(
        stream.language,
        "und",
    )

    title = safe_filename_component(
        stream.title,
        "",
    )

    name_parts = [
        output_stem.name,
        f"sub{subtitle_number}",
        language,
    ]

    if title:
        name_parts.append(title)

    filename = ".".join(name_parts) + ".srt"
    return output_stem.parent / filename


def extract_subtitle_stream(
    input_file: Path,
    output_file: Path,
    subtitle_number: int,
    stream: StreamInfo,
    ffmpeg: str,
    overwrite: bool,
) -> SubtitleExtraction:
    """
    Attempt to convert one embedded subtitle stream to SRT.

    subtitle_number refers to the subtitle-relative index used by FFmpeg:
        0:s:0
        0:s:1
        0:s:2

    stream.index is the absolute stream index displayed by FFprobe.
    """
    base_result = SubtitleExtraction(
        stream_index=stream.index,
        subtitle_number=subtitle_number,
        codec_name=stream.codec_name,
        language=stream.language,
        title=stream.title,
        output_file=str(output_file),
        status="pending",
    )

    if output_file.exists() and not overwrite:
        base_result.status = "skipped_existing"
        return base_result

    codec_name = (stream.codec_name or "").lower()

    if codec_name and codec_name not in TEXT_SUBTITLE_CODECS:
        base_result.status = "unsupported_bitmap_or_unknown_codec"
        base_result.output_file = None
        base_result.error = (
            f"Subtitle codec '{codec_name}' may be image-based and cannot "
            "normally be converted directly to SRT."
        )
        return base_result

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_file),

        # Select one subtitle stream by its subtitle-relative number.
        "-map",
        f"0:s:{subtitle_number}",

        # Convert supported text subtitles to SubRip.
        "-c:s",
        "srt",
        str(output_file),
    ]

    result = run_command(command)

    if result.returncode != 0:
        output_file.unlink(missing_ok=True)
        base_result.status = "failed"
        base_result.output_file = None
        base_result.error = (
            result.stderr.strip()
            or "Unknown subtitle extraction error."
        )
        return base_result

    if not output_file.exists() or output_file.stat().st_size == 0:
        output_file.unlink(missing_ok=True)
        base_result.status = "failed"
        base_result.output_file = None
        base_result.error = "The extracted subtitle file was empty."
        return base_result

    base_result.status = "extracted"
    return base_result


def process_file(
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    overwrite: bool,
    normalize: bool,
    extract_subtitles: bool,
) -> FileReport:
    relative_path = input_file.relative_to(input_dir)

    report = FileReport(
        input_file=str(input_file),
        relative_path=str(relative_path),
        status="processing",
    )

    try:
        probe_data = probe_media(input_file, ffprobe)
    except RuntimeError as exc:
        report.status = "probe_failed"
        report.errors.append(str(exc))
        return report

    raw_format = probe_data.get("format") or {}
    raw_streams = probe_data.get("streams") or []

    report.format_name = raw_format.get("format_name")
    report.format_long_name = raw_format.get("format_long_name")
    report.duration_seconds = optional_float(raw_format.get("duration"))
    report.size_bytes = optional_int(raw_format.get("size"))
    report.bit_rate = optional_int(raw_format.get("bit_rate"))

    report.streams = [
        stream_from_probe(stream)
        for stream in raw_streams
    ]

    video_streams = [
        stream
        for stream in report.streams
        if stream.codec_type == "video"
    ]

    audio_streams = [
        stream
        for stream in report.streams
        if stream.codec_type == "audio"
    ]

    subtitle_streams = [
        stream
        for stream in report.streams
        if stream.codec_type == "subtitle"
    ]

    report.video_stream_count = len(video_streams)
    report.audio_stream_count = len(audio_streams)
    report.subtitle_stream_count = len(subtitle_streams)

    output_stem = create_output_stem(
        input_file,
        input_dir,
        output_dir,
    )

    # Audio extraction
    if audio_streams:
        audio_output = output_stem.with_suffix(".wav")
        report.audio_output = str(audio_output)

        audio_status, audio_error = extract_audio(
            input_file=input_file,
            output_file=audio_output,
            ffmpeg=ffmpeg,
            overwrite=overwrite,
            normalize=normalize,
        )

        report.audio_status = audio_status

        if audio_error:
            report.errors.append(f"Audio extraction: {audio_error}")
        elif audio_output.exists():
            try:
                report.audio_verification = verify_wav(
                    audio_output,
                    ffprobe,
                )
            except RuntimeError as exc:
                report.errors.append(
                    f"Audio verification: {exc}"
                )
    else:
        report.audio_status = "no_audio_stream"
        report.errors.append("The file contains no audio stream.")

    # Subtitle extraction
    if extract_subtitles:
        for subtitle_number, subtitle_stream in enumerate(
            subtitle_streams
        ):
            output_file = subtitle_output_path(
                output_stem=output_stem,
                subtitle_number=subtitle_number,
                stream=subtitle_stream,
            )

            subtitle_result = extract_subtitle_stream(
                input_file=input_file,
                output_file=output_file,
                subtitle_number=subtitle_number,
                stream=subtitle_stream,
                ffmpeg=ffmpeg,
                overwrite=overwrite,
            )

            report.subtitles.append(subtitle_result)
    else:
        for subtitle_number, subtitle_stream in enumerate(
            subtitle_streams
        ):
            report.subtitles.append(
                SubtitleExtraction(
                    stream_index=subtitle_stream.index,
                    subtitle_number=subtitle_number,
                    codec_name=subtitle_stream.codec_name,
                    language=subtitle_stream.language,
                    title=subtitle_stream.title,
                    output_file=None,
                    status="inspection_only",
                )
            )

    if report.audio_status == "failed":
        report.status = "completed_with_errors"
    elif report.errors:
        report.status = "completed_with_warnings"
    else:
        report.status = "completed"

    return report


def print_file_summary(report: FileReport) -> None:
    print(f"\n{report.relative_path}")
    print(f"  Status:    {report.status}")
    print(f"  Video:     {report.video_stream_count} stream(s)")
    print(f"  Audio:     {report.audio_stream_count} stream(s)")
    print(f"  Subtitles: {report.subtitle_stream_count} stream(s)")

    if report.audio_status:
        print(f"  WAV:       {report.audio_status}")

    for subtitle in report.subtitles:
        language = subtitle.language or "und"
        codec = subtitle.codec_name or "unknown"

        print(
            "  Subtitle:  "
            f"#{subtitle.subtitle_number} "
            f"language={language} "
            f"codec={codec} "
            f"status={subtitle.status}"
        )

    for error in report.errors:
        print(f"  Warning:   {error}")


def build_summary(reports: list[FileReport]) -> dict[str, int]:
    return {
        "files_total": len(reports),
        "files_completed": sum(
            report.status == "completed"
            for report in reports
        ),
        "files_with_warnings": sum(
            report.status == "completed_with_warnings"
            for report in reports
        ),
        "files_with_errors": sum(
            report.status in {
                "completed_with_errors",
                "probe_failed",
            }
            for report in reports
        ),
        "audio_converted": sum(
            report.audio_status == "converted"
            for report in reports
        ),
        "audio_skipped": sum(
            report.audio_status == "skipped_existing"
            for report in reports
        ),
        "files_with_subtitles": sum(
            report.subtitle_stream_count > 0
            for report in reports
        ),
        "subtitle_streams_found": sum(
            report.subtitle_stream_count
            for report in reports
        ),
        "subtitle_streams_extracted": sum(
            subtitle.status == "extracted"
            for report in reports
            for subtitle in report.subtitles
        ),
        "subtitle_streams_skipped": sum(
            subtitle.status == "skipped_existing"
            for report in reports
            for subtitle in report.subtitles
        ),
        "subtitle_streams_unsupported": sum(
            subtitle.status == "unsupported_bitmap_or_unknown_codec"
            for report in reports
            for subtitle in report.subtitles
        ),
        "subtitle_streams_failed": sum(
            subtitle.status == "failed"
            for report in reports
            for subtitle in report.subtitles
        ),
    }


def write_report(
    report_path: Path,
    reports: list[FileReport],
    arguments: argparse.Namespace,
) -> None:
    summary = build_summary(reports)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(arguments.input_dir),
        "output_directory": str(arguments.output_dir),
        "settings": {
            "recursive": arguments.recursive,
            "overwrite": arguments.overwrite,
            "normalize": arguments.normalize,
            "extract_subtitles": not arguments.no_subtitles,
            "ffmpeg": arguments.ffmpeg,
            "ffprobe": arguments.ffprobe,
        },
        "summary": summary,
        "files": [
            asdict(report)
            for report in reports
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> int:
    args = parse_arguments()

    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.input_dir.is_dir():
        print(
            f"Error: input directory does not exist: {args.input_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        ensure_executable_exists(args.ffmpeg)
        ensure_executable_exists(args.ffprobe)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    media_files = discover_media_files(
        input_dir=args.input_dir,
        recursive=args.recursive,
    )

    if not media_files:
        print(
            f"No supported media files found in {args.input_dir}.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(media_files)} media file(s).")

    reports: list[FileReport] = []

    for position, input_file in enumerate(media_files, start=1):
        print(
            f"\n[{position}/{len(media_files)}] "
            f"Processing {input_file.name}"
        )

        report = process_file(
            input_file=input_file,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            overwrite=args.overwrite,
            normalize=args.normalize,
            extract_subtitles=not args.no_subtitles,
        )

        reports.append(report)
        print_file_summary(report)

    report_path = args.output_dir / args.report_name

    write_report(
        report_path=report_path,
        reports=reports,
        arguments=args,
    )

    summary = build_summary(reports)

    print("\nProcessing complete")
    print(f"  Files:               {summary['files_total']}")
    print(f"  Audio converted:     {summary['audio_converted']}")
    print(f"  Files with subtitles:{summary['files_with_subtitles']}")
    print(f"  Subtitle streams:    {summary['subtitle_streams_found']}")
    print(f"  Subtitles extracted: {summary['subtitle_streams_extracted']}")
    print(f"  Report:               {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())