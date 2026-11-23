#!/usr/bin/env python3
"""
assemble_redistributed.py

Reuse already-generated Chatterbox WAV clips and a redistributed dubbing JSON
to create a new timed master voice-over WAV.

NO TTS GENERATION IS PERFORMED.

Expected workflow:
    dubbing.json
        -> Chatterbox generation
        -> output/clips/*.wav + timing_report.json
        -> redistribute_timing.py
        -> dubbing_redistributed.json
        -> THIS SCRIPT
        -> output/english_voiceover_redistributed.wav

Requirements:
    pip install pydub
    ffmpeg must be installed and available on PATH.

The script:
- reads redistributed start/end timestamps
- finds existing WAV clip for each segment
- measures the real clip duration
- speeds up clips only when needed
- does NOT slow clips down
- trims tiny rounding overflow if necessary
- overlays each fitted clip at its redistributed start time
- exports a master WAV
- writes a fresh assembly report JSON

Recommended clip filenames:
    001_RANDA.wav
    002_RANDA.wav
    003_interviewer.wav

But the finder also accepts filenames beginning with the zero-padded segment:
    001_*.wav
    002_*.wav
etc.
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment


EPS = 0.010  # 10 ms tolerance


def ts_to_ms(ts: str) -> int:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm to milliseconds."""
    ts = ts.replace(",", ".")
    hh, mm, ss = ts.split(":")
    total = int(hh) * 3600 + int(mm) * 60 + float(ss)
    return int(round(total * 1000))


def ms_to_ts(ms: int) -> str:
    if ms < 0:
        ms = 0
    hh, rem = divmod(ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, msec = divmod(rem, 1_000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{msec:03d}"


def safe_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return value or "speaker"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "FFmpeg was not found on PATH.\n"
            "Install FFmpeg and confirm that `ffmpeg -version` works."
        )
    return exe


def find_clip(clips_dir: Path, segment: int, speaker: str) -> Path:
    """
    Try several naming conventions, then fall back to any WAV beginning
    with the zero-padded segment number.
    """
    n3 = f"{segment:03d}"
    n4 = f"{segment:04d}"
    sp = safe_name(speaker)

    exact_candidates = [
        clips_dir / f"{n3}_{sp}.wav",
        clips_dir / f"{n4}_{sp}.wav",
        clips_dir / f"{n3}.wav",
        clips_dir / f"{n4}.wav",
        clips_dir / f"segment_{n3}.wav",
        clips_dir / f"segment_{n4}.wav",
    ]

    for p in exact_candidates:
        if p.exists():
            return p

    candidates = sorted(
        set(clips_dir.glob(f"{n3}_*.wav"))
        | set(clips_dir.glob(f"{n4}_*.wav"))
        | set(clips_dir.glob(f"*_{n3}_*.wav"))
        | set(clips_dir.glob(f"*_{n4}_*.wav"))
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No WAV clip found for segment {segment} in {clips_dir}"
        )

    raise RuntimeError(
        f"Multiple possible WAV clips found for segment {segment}:\n"
        + "\n".join(f"  {p.name}" for p in candidates)
    )


def atempo_chain(ratio: float) -> str:
    """
    FFmpeg atempo supports 0.5 to 100 in modern FFmpeg, but chaining smaller
    factors is safer and more portable.

    ratio > 1 means speed up.
    """
    if ratio <= 0:
        raise ValueError("Speed ratio must be positive.")

    factors = []
    remaining = ratio

    # Keep individual factors <= 2.0
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    # We never intentionally slow down, so remaining should be >= 1.
    if remaining < 1.0:
        remaining = 1.0

    factors.append(remaining)
    return ",".join(f"atempo={x:.10f}" for x in factors)


def ffmpeg_speed_fit(
    ffmpeg: str,
    source: Path,
    destination: Path,
    ratio: float,
):
    destination.parent.mkdir(parents=True, exist_ok=True)

    filt = atempo_chain(ratio)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-filter:a", filt,
        "-c:a", "pcm_s16le",
        str(destination),
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble redistributed dubbing using existing WAV clips."
    )
    parser.add_argument(
        "redistributed_json",
        help="Path to dubbing_redistributed.json"
    )
    parser.add_argument(
        "--clips-dir",
        default="output/clips",
        help="Directory containing original generated segment WAV files"
    )
    parser.add_argument(
        "--output",
        default="output/english_voiceover_redistributed.wav",
        help="Output master WAV"
    )
    parser.add_argument(
        "--fitted-dir",
        default="output/redistributed_fitted",
        help="Where timing-fitted WAV clips are written"
    )
    parser.add_argument(
        "--report",
        default="output/redistributed_assembly_report.json",
        help="Output assembly report JSON"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Master output sample rate"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Master WAV channels (1 mono, 2 stereo)"
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=2.0,
        help="Safety limit: abort if any required speed ratio exceeds this"
    )

    args = parser.parse_args()

    redistributed_json = Path(args.redistributed_json)
    clips_dir = Path(args.clips_dir)
    output_path = Path(args.output)
    fitted_dir = Path(args.fitted_dir)
    report_path = Path(args.report)

    if not redistributed_json.exists():
        raise FileNotFoundError(redistributed_json)
    if not clips_dir.exists():
        raise FileNotFoundError(clips_dir)

    ffmpeg = find_ffmpeg()
    items = load_json(redistributed_json)

    if not isinstance(items, list) or not items:
        raise ValueError("Redistributed JSON must be a non-empty JSON array.")

    # Chronological check
    starts = [ts_to_ms(x["start"]) for x in items]
    if starts != sorted(starts):
        raise ValueError("Segments in redistributed JSON are not chronological.")

    fitted_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timeline_start_ms = min(ts_to_ms(x["start"]) for x in items)
    timeline_end_ms = max(ts_to_ms(x["end"]) for x in items)
    total_ms = timeline_end_ms - timeline_start_ms

    # Create silent master.
    master = (
        AudioSegment.silent(duration=total_ms, frame_rate=args.sample_rate)
        .set_channels(args.channels)
        .set_sample_width(2)
    )

    report = []
    previous_end = None

    print("=" * 80)
    print("ASSEMBLING REDISTRIBUTED DUB")
    print("=" * 80)

    for idx, item in enumerate(items, start=1):
        segment = int(item["segment"])
        speaker = str(item.get("speaker", "speaker"))

        start_abs_ms = ts_to_ms(item["start"])
        end_abs_ms = ts_to_ms(item["end"])
        available_ms = end_abs_ms - start_abs_ms

        if available_ms <= 0:
            raise ValueError(f"Segment {segment}: non-positive allocated duration.")

        if previous_end is not None and start_abs_ms < previous_end:
            raise ValueError(
                f"Segment {segment} starts before previous segment ends. "
                "Redistributed JSON contains overlapping windows."
            )
        previous_end = end_abs_ms

        clip_path = find_clip(clips_dir, segment, speaker)
        original_audio = AudioSegment.from_wav(clip_path)
        actual_ms = len(original_audio)

        if actual_ms <= 0:
            raise ValueError(f"Segment {segment}: WAV is empty.")

        # Exact ratio from the actual WAV, not from stale report metadata.
        required_ratio = actual_ms / available_ms

        if required_ratio > args.max_speed:
            raise RuntimeError(
                f"Segment {segment} would require {required_ratio:.3f}x speed, "
                f"above --max-speed {args.max_speed:.3f}x."
            )

        fitted_name = f"{segment:03d}_{safe_name(speaker)}.wav"
        fitted_path = fitted_dir / fitted_name

        if actual_ms > available_ms + int(EPS * 1000):
            # Speed up just enough to fit.
            ffmpeg_speed_fit(
                ffmpeg=ffmpeg,
                source=clip_path,
                destination=fitted_path,
                ratio=required_ratio,
            )
            fitted_audio = AudioSegment.from_wav(fitted_path)
            status = "SPED_UP"
        else:
            # No slowdown. Reuse unchanged audio.
            fitted_audio = original_audio
            status = "UNCHANGED"

            # Write a normalized copy so fitted_dir is complete.
            fitted_audio.export(
                fitted_path,
                format="wav",
                parameters=["-acodec", "pcm_s16le"]
            )

        fitted_ms_before_trim = len(fitted_audio)

        # FFmpeg atempo can differ by a few milliseconds because of codec/frame
        # boundaries. Trim only tiny overflow so no clip crosses its window.
        if len(fitted_audio) > available_ms:
            fitted_audio = fitted_audio[:available_ms]
            trim_ms = fitted_ms_before_trim - available_ms
        else:
            trim_ms = 0

        # If a clip is slightly shorter due to atempo rounding, DO NOT add
        # semantic silence here. The next segment still begins at its planned
        # redistributed timestamp. Usually this is only a few milliseconds.
        residual_gap_ms = max(0, available_ms - len(fitted_audio))

        # Normalize master format.
        fitted_audio = (
            fitted_audio
            .set_frame_rate(args.sample_rate)
            .set_channels(args.channels)
            .set_sample_width(2)
        )

        position_ms = start_abs_ms - timeline_start_ms
        master = master.overlay(fitted_audio, position=position_ms)

        final_ratio = actual_ms / max(len(fitted_audio), 1)

        report.append({
            "segment": segment,
            "speaker": speaker,
            "source_clip": str(clip_path),
            "fitted_clip": str(fitted_path),
            "start": item["start"],
            "end": item["end"],
            "allocated_duration_seconds": round(available_ms / 1000, 3),
            "actual_source_duration_seconds": round(actual_ms / 1000, 3),
            "fitted_duration_seconds": round(len(fitted_audio) / 1000, 3),
            "required_speed_ratio": round(required_ratio, 6),
            "effective_speed_ratio": round(final_ratio, 6),
            "effective_speed_percent": round((final_ratio - 1) * 100, 3),
            "residual_gap_ms": residual_gap_ms,
            "trimmed_ms": trim_ms,
            "status": status,
            "text": item.get("text", "")
        })

        print(
            f"{segment:03d}  {speaker:<14} "
            f"window={available_ms/1000:7.3f}s  "
            f"clip={actual_ms/1000:7.3f}s  "
            f"speed={required_ratio:6.3f}x  "
            f"{status}"
        )

    # Export master.
    master.export(
        output_path,
        format="wav",
        parameters=[
            "-acodec", "pcm_s16le",
            "-ar", str(args.sample_rate),
            "-ac", str(args.channels)
        ]
    )

    # Summary
    max_ratio = max(x["required_speed_ratio"] for x in report)
    avg_ratio = sum(x["required_speed_ratio"] for x in report) / len(report)
    total_residual_ms = sum(x["residual_gap_ms"] for x in report)
    sped_up = sum(x["status"] == "SPED_UP" for x in report)

    result = {
        "summary": {
            "segments": len(report),
            "timeline_start": ms_to_ts(timeline_start_ms),
            "timeline_end": ms_to_ts(timeline_end_ms),
            "master_duration_seconds": round(total_ms / 1000, 3),
            "sped_up_segments": sped_up,
            "unchanged_segments": len(report) - sped_up,
            "maximum_required_speed_ratio": round(max_ratio, 6),
            "maximum_required_speed_percent": round((max_ratio - 1) * 100, 3),
            "mean_required_speed_ratio": round(avg_ratio, 6),
            "total_residual_gap_ms": total_residual_ms,
            "output_wav": str(output_path),
        },
        "segments": report,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Master WAV:       {output_path.resolve()}")
    print(f"Fitted clips:     {fitted_dir.resolve()}")
    print(f"Assembly report:  {report_path.resolve()}")
    print(f"Max speed-up:     {max_ratio:.4f}x ({(max_ratio - 1)*100:.2f}%)")
    print(f"Residual gap:     {total_residual_ms} ms total")
    print("=" * 80)


if __name__ == "__main__":
    main()
