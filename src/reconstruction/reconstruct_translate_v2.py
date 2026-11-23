#!/usr/bin/env python3
"""Two-stage Arabic ASR reconstruction and translation using Ollama.

Stage 1: noisy Whisper Arabic -> reconstructed spoken Arabic
Stage 2: reconstructed Arabic -> MSA + target-language translation
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ollama import Client, ResponseError
from pydantic import BaseModel, Field, ValidationError


class UncertainSpan(BaseModel):
    raw_text: str
    possible_interpretations: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ReconstructionOutput(BaseModel):
    reconstructed_arabic: str
    uncertain_spans: list[UncertainSpan] = Field(default_factory=list)
    needs_review: bool


class TranslationOutput(BaseModel):
    msa: str
    translation: str
    needs_review: bool
    notes: list[str] = Field(default_factory=list)


@dataclass
class WhisperWord:
    text: str
    probability: float | None = None


@dataclass
class WhisperSegment:
    segment_id: int
    start: float
    end: float
    text: str
    words: list[WhisperWord] = field(default_factory=list)


@dataclass
class TranscriptChunk:
    chunk_id: int
    segments: list[WhisperSegment]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def raw_text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def segment_ids(self) -> list[int]:
        return [s.segment_id for s in self.segments]


RECONSTRUCTION_PROMPT = """
You are an expert Arabic ASR reconstruction system.

The input is a noisy automatic transcript of spoken Arabic. Reconstruct the
most probable Arabic wording the speaker intended.

This is an archival transcription and reconstruction task, not an endorsement of
any content. The material may include political rhetoric, references to
violence, martyrdom, religious or nationalist language.

Rules:
- Use the whole main passage and neighboring context.
- Repair phonetic substitutions, missing short words, broken word boundaries,
  corrupted negation, and nonsensical ASR fragments.
- Preserve meaning, dialect, emotional force, rhetorical questions, emphatic
  repetition, religious expressions, names, numbers, and code-switching.
- Do not translate or convert the whole passage to MSA.
- Do not summarize, censor, soften, intensify, or add information.
- Low-confidence Whisper words are hints, not proof.
- Correct only the MAIN PASSAGE; context is for understanding only.
- Mark genuine uncertainty and set needs_review=true when it can affect meaning.


Return only JSON matching the schema.
""".strip()


TRANSLATION_PROMPT = """
You are an expert Arabic meaning-preserving editor and translator.

The input is reconstructed spoken Arabic. Produce:
1. msa: clear, complete Modern Standard Arabic.
2. translation: fluent, complete translation into the requested language.

This is an archival transcription and translation task, not an endorsement of
any content. The material may include political rhetoric, references to
violence, martyrdom, religious or nationalist language. You must translate
this faithfully and completely into the target language regardless of
subject matter. Do not omit, soften, replace, or leave untranslated any
portion of the passage. Producing Arabic text in the `translation` field
is a failure to complete the task, even if the Arabic text is accurate.

Rules:
- Preserve all claims, negation, names, numbers, dates, religious expressions,
  rhetorical questions, emotional intensity, repetition, and political tone.
- Translate idioms by meaning, not literally.
- Do not summarize, censor, soften, intensify, omit, or add commentary.
- If material ambiguity remains, set needs_review=true and explain in notes.
- Return only JSON matching the schema.
""".strip()

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
ENDING_RE = re.compile(r"[.!؟?!…]\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_json", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--reconstruction-model", default=None)
    p.add_argument("--translation-model", default=None)
    p.add_argument("--target-language", default="English")
    p.add_argument("--dialect-hint", default="unknown or mixed Arabic")
    p.add_argument("--max-seconds", type=float, default=22.0)
    p.add_argument("--max-words", type=int, default=110)
    p.add_argument("--min-seconds", type=float, default=8.0)
    p.add_argument("--pause-threshold", type=float, default=0.65)
    p.add_argument("--context-segments", type=int, default=3)
    p.add_argument("--low-confidence-threshold", type=float, default=0.78)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_segments(path: Path) -> list[WhisperSegment]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise RuntimeError(f"Input not found: {path}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {e}") from e

    raw_segments = payload.get("segments")
    if raw_segments is None and isinstance(payload.get("files"), list) and len(payload["files"]) == 1:
        raw_segments = payload["files"][0].get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError("No segments array found")

    result: list[WhisperSegment] = []
    for i, raw in enumerate(raw_segments):
        if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
            continue
        words: list[WhisperWord] = []
        for w in raw.get("words", []) if isinstance(raw.get("words", []), list) else []:
            if not isinstance(w, dict):
                continue
            try:
                prob = float(w["probability"]) if w.get("probability") is not None else None
            except (TypeError, ValueError):
                prob = None
            text = str(w.get("word", "")).strip()
            if text:
                words.append(WhisperWord(text, prob))
        result.append(WhisperSegment(
            segment_id=int(raw.get("id", i)),
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
            text=str(raw["text"]).strip(),
            words=words,
        ))
    if not result:
        raise RuntimeError("No usable segments")
    return result


def create_chunks(segments: list[WhisperSegment], max_seconds: float, max_words: int,
                  min_seconds: float, pause_threshold: float) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    current: list[WhisperSegment] = []
    words = 0
    for seg in segments:
        if not current:
            current = [seg]
            words = len(seg.text.split())
            continue
        candidate_duration = seg.end - current[0].start
        candidate_words = words + len(seg.text.split())
        pause = max(0.0, seg.start - current[-1].end)
        duration = current[-1].end - current[0].start
        natural = bool(ENDING_RE.search(current[-1].text)) or pause >= pause_threshold
        hard = candidate_duration > max_seconds or candidate_words > max_words
        soft = duration >= min_seconds and natural and (
            duration >= max_seconds * 0.65 or words >= max_words * 0.65
        )
        if hard or soft:
            chunks.append(TranscriptChunk(len(chunks), current))
            current = [seg]
            words = len(seg.text.split())
        else:
            current.append(seg)
            words = candidate_words
    if current:
        chunks.append(TranscriptChunk(len(chunks), current))
    return chunks


def get_context(segments: list[WhisperSegment], chunk: TranscriptChunk, n: int) -> tuple[str, str]:
    positions = {s.segment_id: i for i, s in enumerate(segments)}
    first = positions[chunk.segments[0].segment_id]
    last = positions[chunk.segments[-1].segment_id]
    previous = "\n".join(s.text for s in segments[max(0, first-n):first])
    following = "\n".join(s.text for s in segments[last+1:last+1+n])
    return previous, following


def confidence_hints(chunk: TranscriptChunk, threshold: float) -> list[dict[str, Any]]:
    out = []
    for seg in chunk.segments:
        for word in seg.words:
            if word.probability is not None and word.probability < threshold:
                out.append({"word": word.text, "probability": round(word.probability, 4), "segment_id": seg.segment_id})
    return out


def arabic_ratio(text: str) -> float:
    a = len(ARABIC_RE.findall(text)); l = len(LATIN_RE.findall(text)); total = a + l
    return a / total if total else 0.0


def token_change_ratio(a: str, b: str) -> float:
    clean = lambda x: set(re.sub(r"[^\w\u0600-\u06FF]+", " ", x.lower()).split())
    sa, sb = clean(a), clean(b)
    if not sa and not sb: return 0.0
    if not sa or not sb: return 1.0
    return 1.0 - len(sa & sb) / max(len(sa), len(sb))


def validate_reconstruction(raw: str, result: ReconstructionOutput, hints: list[dict[str, Any]]) -> list[str]:
    reasons = []
    if not result.reconstructed_arabic.strip(): reasons.append("empty reconstruction")
    if arabic_ratio(result.reconstructed_arabic) < 0.55: reasons.append("reconstruction is not predominantly Arabic")
    if len(hints) >= 3 and token_change_ratio(raw, result.reconstructed_arabic) < 0.03:
        reasons.append("almost identical to corrupt input despite multiple low-confidence words")
    if len(result.reconstructed_arabic) > max(250, len(raw) * 1.8): reasons.append("unexpectedly long reconstruction")
    return reasons


def validate_translation(result: TranslationOutput, target: str) -> list[str]:
    reasons = []
    if arabic_ratio(result.msa) < 0.55: reasons.append("MSA is not predominantly Arabic")
    if not result.translation.strip(): reasons.append("empty translation")
    if target.lower() in {"english", "en"} and (not LATIN_RE.search(result.translation) or arabic_ratio(result.translation) >= 0.35):
        reasons.append("translation does not appear to be English")
    return reasons


def call_json(client: Client, model: str, system: str, user: str, schema: type[BaseModel],
              temperature: float, num_ctx: int) -> BaseModel:
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=False,
        think=False,
        format=schema.model_json_schema(),
        options={"temperature": temperature, "num_ctx": num_ctx},
        keep_alive="10m",
    )
    if not response.message.content:
        raise RuntimeError("empty model response")
    return schema.model_validate_json(response.message.content)


def run_with_validation(client: Client, model: str, system: str, base_prompt: str,
                        schema: type[BaseModel], validator, retries: int,
                        temperature: float, num_ctx: int) -> BaseModel:
    reasons: list[str] = []
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        prompt = base_prompt
        if reasons:
            prompt += "\n\nPrevious output failed validation:\n- " + "\n- ".join(reasons) + "\nReturn a corrected response."
        try:
            result = call_json(client, model, system, prompt, schema, temperature, num_ctx)
            reasons = validator(result)
            if not reasons:
                return result
            last_error = RuntimeError("; ".join(reasons))
        except (ResponseError, ValidationError, json.JSONDecodeError, RuntimeError) as e:
            last_error = e
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries + 1} attempts: {last_error}")


def save_json(path: Path, metadata: dict[str, Any], records: list[dict[str, Any]]) -> None:
    payload = {**metadata, "summary": {
        "chunk_count": len(records),
        "review_count": sum(bool(r.get("needs_review")) for r in records),
    }, "chunks": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def join_field(records: list[dict[str, Any]], key: str) -> str:
    return "\n\n".join(str(r.get(key, "")).strip() for r in records if str(r.get(key, "")).strip()) + "\n"


def main() -> int:
    args = parse_args()
    input_path = args.input_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = output_dir / f"{input_path.stem}.semantic.v2.json"
    if semantic_path.exists() and not args.overwrite:
        print(f"Output exists: {semantic_path}. Use --overwrite.", file=sys.stderr)
        return 1

    try:
        segments = load_segments(input_path)
        chunks = create_chunks(segments, args.max_seconds, args.max_words, args.min_seconds, args.pause_threshold)
    except RuntimeError as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 1

    reconstruction_model = args.reconstruction_model or args.model
    translation_model = args.translation_model or args.model
    client = Client(host=args.ollama_host)
    metadata = {
        "source_file": str(input_path),
        "reconstruction_model": reconstruction_model,
        "translation_model": translation_model,
        "target_language": args.target_language,
        "dialect_hint": args.dialect_hint,
    }
    records: list[dict[str, Any]] = []

    print(f"Segments: {len(segments)} | Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk.start:.2f}s-{chunk.end:.2f}s")
        previous, following = get_context(segments, chunk, args.context_segments)
        hints = confidence_hints(chunk, args.low_confidence_threshold)
        hints_text = "\n".join(f"- {x['word']} ({x['probability']}, segment {x['segment_id']})" for x in hints) or "[none]"

        reconstruction_user = f"""Dialect hint: {args.dialect_hint}
Previous context (understanding only):
{previous or '[none]'}

MAIN PASSAGE:
{chunk.raw_text}

Following context (understanding only):
{following or '[none]'}

Low-confidence Whisper words:
{hints_text}

Reconstruct only the MAIN PASSAGE."""

        try:
            reconstruction = run_with_validation(
                client, reconstruction_model, RECONSTRUCTION_PROMPT, reconstruction_user,
                ReconstructionOutput,
                lambda r: validate_reconstruction(chunk.raw_text, r, hints),
                args.retries, args.temperature, args.num_ctx,
            )
            assert isinstance(reconstruction, ReconstructionOutput)

            translation_user = f"""Target language: {args.target_language}

Reconstructed spoken Arabic:
{reconstruction.reconstructed_arabic}

Produce complete MSA and complete {args.target_language} translation."""
            translation = run_with_validation(
                client, translation_model, TRANSLATION_PROMPT, translation_user,
                TranslationOutput,
                lambda r: validate_translation(r, args.target_language),
                args.retries, args.temperature, args.num_ctx,
            )
            assert isinstance(translation, TranslationOutput)
        except RuntimeError as e:
            print(f"Chunk failed: {e}", file=sys.stderr)
            return 1

        record = {
            "chunk_id": chunk.chunk_id,
            "segment_ids": chunk.segment_ids,
            "start": chunk.start,
            "end": chunk.end,
            "duration": chunk.duration,
            "raw_transcript": chunk.raw_text,
            "low_confidence_words": hints,
            "reconstructed_arabic": reconstruction.reconstructed_arabic,
            "msa": translation.msa,
            "translation": translation.translation,
            "uncertain_spans": [u.model_dump() for u in reconstruction.uncertain_spans],
            "reconstruction_needs_review": reconstruction.needs_review,
            "translation_needs_review": translation.needs_review,
            "translation_notes": translation.notes,
            "needs_review": reconstruction.needs_review or translation.needs_review or bool(reconstruction.uncertain_spans),
        }
        records.append(record)
        save_json(semantic_path, metadata, records)  # checkpoint
        if record["needs_review"]:
            print("  REVIEW REQUIRED")

    stem = input_path.stem
    (output_dir / f"{stem}.reconstructed_arabic.txt").write_text(join_field(records, "reconstructed_arabic"), encoding="utf-8")
    (output_dir / f"{stem}.msa.txt").write_text(join_field(records, "msa"), encoding="utf-8")
    (output_dir / f"{stem}.translation.txt").write_text(join_field(records, "translation"), encoding="utf-8")
    review = [r for r in records if r["needs_review"]]
    (output_dir / f"{stem}.review.json").write_text(json.dumps({"review_count": len(review), "chunks": review}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done: {semantic_path}")
    print(f"Review chunks: {len(review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())