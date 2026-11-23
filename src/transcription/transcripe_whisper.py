#!/usr/bin/env python3
"""
Transcribe prepared WAV files using faster-whisper.

Outputs:
- Plain-text transcript
- Detailed JSON transcript
- SRT subtitles

External dependency:
    pip install faster-whisper
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
}


@dataclass
class WordResult:
    start: float | None
    end: float | None
    word: str
    probability: float | None


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None
    no_speech_probability: float | None
    words: list[WordResult]


@dataclass
class TranscriptionReport:
    input_file: str
    status: str
    language: str | None = None
    language_probability: float | None = None
    duration_seconds: float | None = None
    duration_after_vad_seconds: float | None = None
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None
    transcript_text: str | None = None
    text_output: str | None = None
    json_output: str | None = None
    srt_output: str | None = None
    segments: list[SegmentResult] | None = None
    error: str | None = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using faster-whisper."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing prepared audio files.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for transcript files.",
    )

    parser.add_argument(
        "--model",
        default="large-v3",
        help=(
            "Whisper model name or local model path. "
            "Default: large-v3"
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. Default: auto",
    )

    parser.add_argument(
        "--compute-type",
        default=None,
        help=(
            "CTranslate2 compute type, such as float16, int8, "
            "int8_float16, or float32."
        ),
    )

    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Force a language code such as ar. "
            "When omitted, Whisper detects the language."
        ),
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task. Default: transcribe",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for audio files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing transcript files.",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam-search size. Default: 5",
    )

    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable voice activity detection.",
    )

    parser.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable word-level timestamps.",
    )

    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import ctranslate2

        return (
            "cuda"
            if ctranslate2.get_cuda_device_count() > 0
            else "cpu"
        )
    except Exception:
        return "cpu"


def resolve_compute_type(
    device: str,
    requested_compute_type: str | None,
) -> str:
    if requested_compute_type:
        return requested_compute_type

    if device == "cuda":
        return "float16"

    return "int8"


def discover_audio_files(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")

    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def format_srt_timestamp(seconds: float) -> str:
    milliseconds_total = max(0, round(seconds * 1000))

    hours, remainder = divmod(milliseconds_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)

    return (
        f"{hours:02d}:{minutes:02d}:{secs:02d},"
        f"{milliseconds:03d}"
    )


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_srt_file(
    path: Path,
    segments: list[SegmentResult],
) -> None:
    entries: list[str] = []

    for number, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(segment.start)
        end = format_srt_timestamp(segment.end)
        text = segment.text.strip()

        entries.append(
            f"{number}\n"
            f"{start} --> {end}\n"
            f"{text}\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries), encoding="utf-8")


def write_json_file(
    path: Path,
    report: TranscriptionReport,
) -> None:
    payload = asdict(report)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def create_output_stem(
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    relative_path = input_file.relative_to(input_dir)
    target_parent = output_dir / relative_path.parent / input_file.stem
    target_parent.mkdir(parents=True, exist_ok=True)

    return target_parent / input_file.stem


def convert_word(word: Any) -> WordResult:
    probability = getattr(word, "probability", None)

    return WordResult(
        start=getattr(word, "start", None),
        end=getattr(word, "end", None),
        word=str(getattr(word, "word", "")),
        probability=(
            float(probability)
            if probability is not None
            else None
        ),
    )


def transcribe_file(
    model: WhisperModel,
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    task: str,
    beam_size: int,
    vad_filter: bool,
    word_timestamps: bool,
    overwrite: bool,
) -> TranscriptionReport:
    output_stem = create_output_stem(
        input_file,
        input_dir,
        output_dir,
    )

    text_output = output_stem.with_suffix(".txt")
    json_output = output_stem.with_suffix(".json")
    srt_output = output_stem.with_suffix(".srt")

    report = TranscriptionReport(
        input_file=str(input_file),
        status="processing",
        model=model_name,
        device=device,
        compute_type=compute_type,
        text_output=str(text_output),
        json_output=str(json_output),
        srt_output=str(srt_output),
    )

    outputs_exist = all(
        path.exists()
        for path in [text_output, json_output, srt_output]
    )

    if outputs_exist and not overwrite:
        report.status = "skipped_existing"
        return report

    try:
        segments_generator, info = model.transcribe(
            str(input_file),
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            condition_on_previous_text=True,
        )

        segment_results: list[SegmentResult] = []

        for segment in segments_generator:
            words = [
                convert_word(word)
                for word in (segment.words or [])
            ]

            segment_results.append(
                SegmentResult(
                    id=int(segment.id),
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    avg_logprob=(
                        float(segment.avg_logprob)
                        if segment.avg_logprob is not None
                        else None
                    ),
                    no_speech_probability=(
                        float(segment.no_speech_prob)
                        if segment.no_speech_prob is not None
                        else None
                    ),
                    words=words,
                )
            )

        transcript_text = "\n".join(
            segment.text
            for segment in segment_results
            if segment.text
        ).strip()

        report.language = info.language
        report.language_probability = float(
            info.language_probability
        )
        report.duration_seconds = float(info.duration)
        report.duration_after_vad_seconds = float(
            info.duration_after_vad
        )
        report.transcript_text = transcript_text
        report.segments = segment_results
        report.status = "completed"

        write_text_file(text_output, transcript_text)
        write_srt_file(srt_output, segment_results)
        write_json_file(json_output, report)

        return report

    except Exception as exc:
        report.status = "failed"
        report.error = str(exc)

        try:
            write_json_file(json_output, report)
        except OSError:
            pass

        return report


def write_run_report(
    output_dir: Path,
    reports: list[TranscriptionReport],
    arguments: argparse.Namespace,
    device: str,
    compute_type: str,
) -> Path:
    report_path = output_dir / "transcription_report.json"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "model": arguments.model,
            "device": device,
            "compute_type": compute_type,
            "language": arguments.language,
            "task": arguments.task,
            "beam_size": arguments.beam_size,
            "vad_filter": arguments.vad_filter,
            "word_timestamps": arguments.word_timestamps,
        },
        "summary": {
            "total": len(reports),
            "completed": sum(
                report.status == "completed"
                for report in reports
            ),
            "skipped": sum(
                report.status == "skipped_existing"
                for report in reports
            ),
            "failed": sum(
                report.status == "failed"
                for report in reports
            ),
        },
        "files": [
            asdict(report)
            for report in reports
        ],
    }

    report_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report_path


def main() -> int:
    args = parse_arguments()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(
            f"Error: input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = discover_audio_files(
        input_dir=input_dir,
        recursive=args.recursive,
    )

    if not audio_files:
        print(
            f"No supported audio files found in {input_dir}.",
            file=sys.stderr,
        )
        return 1

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(
        device,
        args.compute_type,
    )

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")

    try:
        model = WhisperModel(
            args.model,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        print(
            f"Could not load Whisper model: {exc}",
            file=sys.stderr,
        )
        return 1

    reports: list[TranscriptionReport] = []

    for number, input_file in enumerate(audio_files, start=1):
        print(
            f"\n[{number}/{len(audio_files)}] "
            f"Transcribing {input_file.name}"
        )

        report = transcribe_file(
            model=model,
            input_file=input_file,
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=args.model,
            device=device,
            compute_type=compute_type,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            word_timestamps=args.word_timestamps,
            overwrite=args.overwrite,
        )

        reports.append(report)

        print(f"Status: {report.status}")

        if report.language:
            print(
                f"Language: {report.language} "
                f"({report.language_probability:.2%})"
            )

        if report.error:
            print(f"Error: {report.error}")

    run_report = write_run_report(
        output_dir=output_dir,
        reports=reports,
        arguments=args,
        device=device,
        compute_type=compute_type,
    )

    completed = sum(
        report.status == "completed"
        for report in reports
    )
    failed = sum(
        report.status == "failed"
        for report in reports
    )

    print("\nTranscription complete")
    print(f"  Files found: {len(reports)}")
    print(f"  Completed:   {completed}")
    print(f"  Failed:      {failed}")
    print(f"  Report:      {run_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())