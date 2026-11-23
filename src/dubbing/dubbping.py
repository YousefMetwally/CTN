import argparse
import json
import subprocess
from pathlib import Path

import torch
import torchaudio as ta
from pydub import AudioSegment

from chatterbox.tts import ChatterboxTTS


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MAX_SPEEDUP = 5
TAIL_SECONDS = 1.0

# Conservative defaults for interview/documentary speech.
EXAGGERATION = 0.45
CFG_WEIGHT = 0.50


# ---------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------

def timestamp_to_ms(timestamp: str) -> int:
    """
    Convert HH:MM:SS.mmm into milliseconds.
    """
    hh, mm, rest = timestamp.split(":")
    ss, msec = rest.split(".")

    total_ms = (
        int(hh) * 3600000
        + int(mm) * 60000
        + int(ss) * 1000
        + int(msec.ljust(3, "0")[:3])
    )

    return total_ms


def audio_duration_seconds(path: Path) -> float:
    audio = AudioSegment.from_wav(path)
    return len(audio) / 1000.0


# ---------------------------------------------------------
# FFMPEG DURATION FITTING
# ---------------------------------------------------------

def speed_up_audio(input_path: Path, output_path: Path, factor: float):
    """
    Speed audio up without changing pitch using FFmpeg atempo.
    factor > 1 = faster
    """

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-filter:a",
        f"atempo={factor:.6f}",
        str(output_path),
    ]

    subprocess.run(command, check=True)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        default="dubbing.json",
        help="Chatterbox-ready dubbing JSON"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Output directory"
    )

    parser.add_argument(
        "--voices",
        default="voices",
        help="Directory containing SPEAKER.wav voice references"
    )

    parser.add_argument(
        "--start-segment",
        type=int,
        default=1,
        help="Start segment number"
    )

    args = parser.parse_args()

    json_path = Path(args.json)
    output_dir = Path(args.output)
    voices_dir = Path(args.voices)
    start_segment = args.start_segment

    clips_dir = output_dir / "clips"
    fitted_dir = output_dir / "fitted"

    clips_dir.mkdir(parents=True, exist_ok=True)
    fitted_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load dubbing data
    # -----------------------------------------------------

    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        raise RuntimeError("The dubbing JSON contains no segments.")

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print()
    print("=" * 60)
    print(f"DEVICE: {device}")
    print("=" * 60)

    # -----------------------------------------------------
    # Load Chatterbox
    # -----------------------------------------------------

    print("\nLoading Chatterbox...")

    model = ChatterboxTTS.from_pretrained(device=device)

    print(f"Sample rate: {model.sr}")
    print("Model loaded.\n")

    generated_segments = []

    # -----------------------------------------------------
    # Generate each segment
    # -----------------------------------------------------

    for item in segments:
        
        edited = bool(item.get("edited", True))
        number = int(item["segment"])
        speaker = str(item["speaker"]).strip()
        text = str(item["text"]).strip()

        start = item["start"]
        end = item["end"]

        start_ms = timestamp_to_ms(start)
        end_ms = timestamp_to_ms(end)

        target_duration = (end_ms - start_ms) / 1000.0

        filename = f"{number:03d}_{speaker}.wav"
        raw_path = clips_dir / filename

        print()
        print("-" * 60)
        print(f"SEGMENT {number:03d}")
        print(f"Speaker : {speaker}")
        print(f"Window  : {start} -> {end}")
        print(f"Target  : {target_duration:.2f}s")
        print(f"Text    : {text}")

        # ---------------------------------------------
        # Speaker reference
        # ---------------------------------------------

        voice_reference = voices_dir / f"{speaker}.wav"

        generate_kwargs = {
            "exaggeration": EXAGGERATION,
            "cfg_weight": CFG_WEIGHT,
        }

        if voice_reference.exists():
            print(f"Voice   : {voice_reference}")
            generate_kwargs["audio_prompt_path"] = str(voice_reference)
        else:
            print("Voice   : Chatterbox default")

        # ---------------------------------------------
        # TTS
        # ---------------------------------------------
        if (raw_path.exists() and not edited) or (start_segment > 1 and number < start_segment):
            print(f"Skipping generation segment {number:03d}")
            
        else:
            print(f"Generating segment {number:03d}...")
            wav = model.generate(
                text,
                **generate_kwargs
            )

            ta.save(
            str(raw_path),
            wav.cpu(),
            model.sr
            )

        raw_duration = audio_duration_seconds(raw_path)

        print(f"Generated: {raw_duration:.2f}s")

        selected_path = raw_path
        final_duration = raw_duration
        timing_status = "OK"

        # ---------------------------------------------
        # Clip too long
        # ---------------------------------------------

        if raw_duration > target_duration:

            required_speed = raw_duration / target_duration

            if required_speed <= MAX_SPEEDUP:

                fitted_path = fitted_dir / filename

                print(
                    f"Fitting : {required_speed:.3f}x "
                    f"speed ({(required_speed - 1) * 100:.1f}% faster)"
                )

                speed_up_audio(
                    raw_path,
                    fitted_path,
                    required_speed
                )

                selected_path = fitted_path
                final_duration = audio_duration_seconds(fitted_path)

                timing_status = "FITTED"

            else:

                timing_status = "WARNING_TOO_LONG"

                print()
                print("!!! TIMING WARNING !!!")
                print(
                    f"Would require {required_speed:.2f}x speed."
                )
                print(
                    "English text should be shortened rather than "
                    "compressing the voice this much."
                )

        # ---------------------------------------------
        # Clip shorter than window
        # ---------------------------------------------

        else:

            spare = target_duration - raw_duration

            print(
                f"Fits naturally. "
                f"{spare:.2f}s silence remains in window."
            )

        generated_segments.append(
            {
                "segment": number,
                "speaker": speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "target_duration": target_duration,
                "generated_duration": raw_duration,
                "final_duration": final_duration,
                "status": timing_status,
                "audio_path": selected_path,
                "text": text,
                "spare" : spare if raw_duration <= target_duration else 0.0,
                "required_speed": required_speed if raw_duration > target_duration else 1.0

            }
        )

    # -----------------------------------------------------
    # Construct master timeline
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ASSEMBLING MASTER VOICE-OVER")
    print("=" * 60)

    final_end_ms = max(x["end_ms"] for x in generated_segments)

    master_length_ms = final_end_ms + int(TAIL_SECONDS * 1000)

    master = AudioSegment.silent(
        duration=master_length_ms,
        frame_rate=model.sr
    )

    master = master.set_channels(1)
    master = master.set_sample_width(2)

    # -----------------------------------------------------
    # Place every clip at its timestamp
    # -----------------------------------------------------

    for item in generated_segments:

        clip = AudioSegment.from_wav(item["audio_path"])

        clip = clip.set_frame_rate(model.sr)
        clip = clip.set_channels(1)
        clip = clip.set_sample_width(2)

        master = master.overlay(
            clip,
            position=item["start_ms"]
        )

        print(
            f"Placed segment {item['segment']:03d} "
            f"at {item['start_ms'] / 1000:.3f}s"
        )

    # -----------------------------------------------------
    # Export master
    # -----------------------------------------------------

    master_path = output_dir / "english_voiceover.wav"

    master.export(
        master_path,
        format="wav"
    )

    # -----------------------------------------------------
    # Timing report
    # -----------------------------------------------------

    report_path = output_dir / "timing_report.json"

    report = []

    for item in generated_segments:

        report.append(
            {
                "segment": item["segment"],
                "speaker": item["speaker"],
                "target_duration_seconds": round(
                    item["target_duration"], 3
                ),
                "generated_duration_seconds": round(
                    item["generated_duration"], 3
                ),
                "final_duration_seconds": round(
                    item["final_duration"], 3
                ),
                "status": item["status"],
                "text": item["text"],
            }
        )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    warnings = [
        x for x in generated_segments
        if x["status"] == "WARNING_TOO_LONG"
    ]

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"\nMaster voice-over:")
    print(master_path.resolve())

    print(f"\nTiming report:")
    print(report_path.resolve())

    print(f"\nIndividual clips:")
    print(clips_dir.resolve())

    if warnings:

        print()
        print("=" * 60)
        print("SEGMENTS THAT NEED SHORTER ENGLISH")
        print("=" * 60)

        for item in warnings:
            print(
                f"Segment {item['segment']:03d}: "
                f"{item['generated_duration']:.2f}s generated / "
                f"{item['target_duration']:.2f}s available"
            )

    else:

        print("\nAll segments fit within the configured timing limit.")


if __name__ == "__main__":
    main()