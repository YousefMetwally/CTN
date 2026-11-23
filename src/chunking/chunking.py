from pathlib import Path
import re
import argparse

parser = argparse.ArgumentParser(description="Chunk SRT files")
parser.add_argument("input", type=Path, help="Input SRT file")
parser.add_argument("output", type=Path, help="Output directory")
parser.add_argument("--chunk-size", type=int, default=400, help="Chunk size")
parser.add_argument("--overlap", type=int, default=20, help="Overlap size")
args = parser.parse_args()

INPUT_SRT = args.input
OUTPUT_DIR = args.output
CHUNK_SIZE = args.chunk_size
OVERLAP = args.overlap


def parse_srt(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()

    blocks = re.split(r"\n\s*\n", text)

    entries = []

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        timestamp = lines[1].strip()
        subtitle_text = "\n".join(lines[2:]).strip()

        entries.append({
            "index": index,
            "timestamp": timestamp,
            "text": subtitle_text,
        })

    return entries


def write_chunk(entries, chunk_number, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    first_id = entries[0]["index"]
    last_id = entries[-1]["index"]

    filename = (
        output_dir
        / f"chunk_{chunk_number:03d}_subs_{first_id}-{last_id}.srt"
    )

    parts = []

    for entry in entries:
        parts.append(
            f"{entry['index']}\n"
            f"{entry['timestamp']}\n"
            f"{entry['text']}"
        )

    filename.write_text(
        "\n\n".join(parts) + "\n",
        encoding="utf-8",
    )

    return filename


def main():
    entries = parse_srt(INPUT_SRT)

    print(f"Total subtitle entries: {len(entries)}")

    step = CHUNK_SIZE - OVERLAP

    chunk_number = 1

    for start in range(0, len(entries), step):
        chunk = entries[start:start + CHUNK_SIZE]

        if not chunk:
            break

        output_file = write_chunk(
            chunk,
            chunk_number,
            OUTPUT_DIR,
        )

        print(
            f"Created: {output_file.name} "
            f"({len(chunk)} subtitles)"
        )

        chunk_number += 1


if __name__ == "__main__":
    main()