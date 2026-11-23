#!/usr/bin/env python3
"""
Arabic ASR Reconstruction and Translation Pipeline — v3

Pipeline:
    1. Whisper transcript -> reconstructed spoken Arabic
    2. Reconstructed Arabic -> semantic MSA normalization
    3. MSA + reconstructed Arabic -> target-language translation
    4. Arabic/translation comparison -> semantic validation

Additional protections:
- Reads faster-whisper word probabilities
- Creates short pause-aware chunks
- Passes neighboring context
- Uses dialect-specific few-shot repair examples
- Detects identical dialect/MSA output
- Detects incomplete output
- Detects Arabic leakage into English
- Detects suspicious polarity reversals
- Detects unsupported additions with a validator model
- Forces review for unresolved low-confidence spans
- Retries failed stages
- Saves checkpoints after every chunk

Requirements:
    pip install ollama pydantic

Recommended:
    ollama pull qwen3:14b

Example:
    python reconstruct_translate_v3.py \
        randa.json \
        output \
        --model qwen3:14b \
        --dialect-hint "Palestinian Arabic" \
        --target-language English
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypeVar

from ollama import Client, ResponseError
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class UncertainSpan(BaseModel):
    raw_text: str
    proposed_reconstruction: str | None = None
    possible_interpretations: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ReconstructionOutput(BaseModel):
    reconstructed_arabic: str
    uncertain_spans: list[UncertainSpan] = Field(default_factory=list)
    needs_review: bool


class MSAOutput(BaseModel):
    msa: str
    unresolved_points: list[str] = Field(default_factory=list)
    needs_review: bool


class TranslationOutput(BaseModel):
    translation: str
    translation_notes: list[str] = Field(default_factory=list)
    needs_review: bool


class CriticalError(BaseModel):
    source_span: str
    output_span: str
    problem: str
    suggested_correction: str | None = None
    severity: str = Field(
        default="major",
        description="minor, major, or critical",
    )


class ValidationOutput(BaseModel):
    meaning_preserved: bool
    polarity_preserved: bool
    rhetorical_structure_preserved: bool
    names_numbers_preserved: bool
    target_language_correct: bool
    critical_errors: list[CriticalError] = Field(default_factory=list)
    needs_review: bool


# ---------------------------------------------------------------------------
# Transcript data structures
# ---------------------------------------------------------------------------

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
        return "\n".join(
            segment.text.strip()
            for segment in self.segments
            if segment.text.strip()
        )

    @property
    def segment_ids(self) -> list[int]:
        return [segment.segment_id for segment in self.segments]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RECONSTRUCTION_SYSTEM_PROMPT = """
You are an expert in reconstructing Arabic speech from noisy Whisper ASR transcripts.
Your task is NOT proofreading.
Your task is to recover what the speaker MOST LIKELY actually said.
The transcript comes from spontaneous spoken Palestinian Arabic.

Whisper frequently makes mistakes such as:

- incorrect words
- missing words
- merged words
- repeated words
- homophone substitutions
- broken grammar
- impossible Arabic expressions

Your job is to reconstruct the speaker's intended utterance.

Use the COMPLETE surrounding context.

Rules:

1. Preserve the speaker's meaning exactly.
2. Preserve Palestinian dialect.Do NOT convert to MSA.
3. Preserve political terminology, names, idioms, emotions and rhetorical questions.
4. Correct Whisper mistakes whenever the intended wording is highly supported by context.
5. NEVER invent information that is not supported by the transcript.
6. If multiple reconstructions are plausible, choose the most likely one.
8. Do NOT explain your reasoning.

Return only valid JSON matching the schema.
""".strip()


MSA_SYSTEM_PROMPT = """
You are an expert in Modern Standard Arabic (MSA) normalization.
Express the complete intended meaning in clear Modern Standard Arabic.
Repair fragmented spoken syntax where necessary. Preserve all meaningful
details, claims, uncertainty, emphasis, negation, names, quantities, dates,
religious references, and rhetorical structure.

Return only valid JSON matching the schema.
""".strip()


TRANSLATION_SYSTEM_PROMPT = """
You translate Arabic into the requested target language.

Use BOTH:
1. the reconstructed spoken Arabic, and
2. the meaning-faithful MSA.

The MSA is a semantic guide. The reconstructed Arabic preserves tone and
dialectal nuance.

Rules:
- Translate the intended meaning, not the ASR errors.
- Preserve polarity, negation, rhetorical questions, names, numbers, dates,
  emotional intensity, religious expressions, political tone, and emphasis.
- Translate idioms by meaning.
- Do not summarize.
- Do not omit clauses.
- Do not add unsupported events or explanations.
- Do not censor, soften, or intensify.
- Do not produce Arabic when the requested target language is English.
- Set needs_review=true if a material ambiguity remains.

Return only valid JSON matching the schema.
""".strip()


VALIDATION_SYSTEM_PROMPT = """
You are a strict bilingual semantic validator.

Compare:
- reconstructed Arabic,
- MSA,
- target-language translation.

Your job is not to rewrite the whole passage. Your job is to detect meaning
errors.

Check for:
- wrong polarity or negation,
- mistranslated idioms,
- unsupported additions,
- omitted clauses,
- wrong names or numbers,
- wrong rhetorical question structure,
- wrong target language,
- mistranslated body parts, places, or relationships,
- incomplete endings,
- semantic drift between the Arabic and translation.


Set needs_review=true when there is any major or critical error.
Return only valid JSON matching the schema.
""".strip()


# ---------------------------------------------------------------------------
# Constants and regexes
# ---------------------------------------------------------------------------

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
ENDING_PUNCTUATION_RE = re.compile(r"[.!؟?!…]\s*$")

INCOMPLETE_ENDINGS = {

}

DIALECT_MARKERS = {

}

POLARITY_PAIRS = [
  
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Four-stage Arabic ASR reconstruction, MSA normalization, "
            "translation, and validation pipeline."
        )
    )

    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_dir", type=Path)

    parser.add_argument(
        "--model",
        default="qwen3:14b",
        help="Default Ollama model for all stages.",
    )
    parser.add_argument("--reconstruction-model", default=None)
    parser.add_argument("--msa-model", default=None)
    parser.add_argument("--translation-model", default=None)
    parser.add_argument("--validation-model", default=None)

    parser.add_argument("--target-language", default="English")
    parser.add_argument(
        "--dialect-hint",
        default="unknown or mixed Arabic",
    )

    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-words", type=int, default=95)
    parser.add_argument("--min-seconds", type=float, default=7.0)
    parser.add_argument("--pause-threshold", type=float, default=0.55)
    parser.add_argument("--context-segments", type=int, default=4)
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.10,
    )

    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Stop on any validator-detected major/critical error instead of "
            "saving the chunk for review."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Top-level JSON value must be an object.")

    return payload


def extract_segments(payload: dict[str, Any]) -> list[WhisperSegment]:
    raw_segments = payload.get("segments")

    if raw_segments is None and isinstance(payload.get("files"), list):
        files = payload["files"]
        if len(files) == 1 and isinstance(files[0], dict):
            raw_segments = files[0].get("segments")

    if not isinstance(raw_segments, list):
        raise RuntimeError("No 'segments' array found.")

    segments: list[WhisperSegment] = []

    for position, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            continue

        text = str(raw.get("text", "")).strip()
        if not text:
            continue

        words: list[WhisperWord] = []

        if isinstance(raw.get("words"), list):
            for raw_word in raw["words"]:
                if not isinstance(raw_word, dict):
                    continue

                word_text = str(raw_word.get("word", "")).strip()
                probability = raw_word.get("probability")

                try:
                    parsed_probability = (
                        float(probability)
                        if probability is not None
                        else None
                    )
                except (TypeError, ValueError):
                    parsed_probability = None

                if word_text:
                    words.append(
                        WhisperWord(
                            text=word_text,
                            probability=parsed_probability,
                        )
                    )

        try:
            segments.append(
                WhisperSegment(
                    segment_id=int(raw.get("id", position)),
                    start=float(raw.get("start", 0.0)),
                    end=float(raw.get("end", 0.0)),
                    text=text,
                    words=words,
                )
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid segment at position {position}: {exc}"
            ) from exc

    if not segments:
        raise RuntimeError("No usable transcript segments found.")

    return segments


# ---------------------------------------------------------------------------
# Chunking and context
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len(text.split())


def has_sentence_ending(text: str) -> bool:
    return bool(ENDING_PUNCTUATION_RE.search(text.strip()))


def create_semantic_chunks(
    segments: list[WhisperSegment],
    max_seconds: float,
    max_words: int,
    min_seconds: float,
    pause_threshold: float,
) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    current: list[WhisperSegment] = []
    current_words = 0

    for segment in segments:
        if not current:
            current = [segment]
            current_words = count_words(segment.text)
            continue

        previous = current[-1]
        pause = max(0.0, segment.start - previous.end)

        current_duration = previous.end - current[0].start
        candidate_duration = segment.end - current[0].start
        candidate_words = current_words + count_words(segment.text)

        natural_boundary = (
            has_sentence_ending(previous.text)
            or pause >= pause_threshold
        )

        hard_limit = (
            candidate_duration > max_seconds
            or candidate_words > max_words
        )

        soft_limit = (
            current_duration >= min_seconds
            and natural_boundary
            and (
                current_duration >= max_seconds * 0.65
                or current_words >= max_words * 0.65
            )
        )

        if hard_limit or soft_limit:
            chunks.append(
                TranscriptChunk(
                    chunk_id=len(chunks),
                    segments=current,
                )
            )
            current = [segment]
            current_words = count_words(segment.text)
        else:
            current.append(segment)
            current_words = candidate_words

    if current:
        chunks.append(
            TranscriptChunk(
                chunk_id=len(chunks),
                segments=current,
            )
        )

    return chunks


def surrounding_context(
    all_segments: list[WhisperSegment],
    chunk: TranscriptChunk,
    count: int,
) -> tuple[str, str]:
    positions = {
        segment.segment_id: index
        for index, segment in enumerate(all_segments)
    }

    first_index = positions[chunk.segments[0].segment_id]
    last_index = positions[chunk.segments[-1].segment_id]

    previous = all_segments[max(0, first_index - count):first_index]
    following = all_segments[last_index + 1:last_index + 1 + count]

    return (
        "\n".join(segment.text for segment in previous),
        "\n".join(segment.text for segment in following),
    )


def low_confidence_words(
    chunk: TranscriptChunk,
    threshold: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for segment in chunk.segments:
        for word in segment.words:
            if (
                word.probability is not None
                and word.probability < threshold
            ):
                result.append(
                    {
                        "word": word.text,
                        "probability": round(word.probability, 4),
                        "segment_id": segment.segment_id,
                    }
                )

    return result


def format_confidence_hints(
    hints: list[dict[str, Any]],
) -> str:
    if not hints:
        return "[none]"

    return "\n".join(
        f"- {item['word']} "
        f"(confidence={item['probability']}, "
        f"segment={item['segment_id']})"
        for item in hints
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def arabic_letter_ratio(text: str) -> float:
    arabic = len(ARABIC_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = arabic + latin
    return 0.0 if total == 0 else arabic / total


def normalize_tokens(text: str) -> list[str]:
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text.lower())
    return [token for token in text.split() if token]


def token_similarity(a: str, b: str) -> float:
    a_tokens = normalize_tokens(a)
    b_tokens = normalize_tokens(b)

    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    a_set = set(a_tokens)
    b_set = set(b_tokens)

    return len(a_set & b_set) / max(len(a_set), len(b_set))


def contains_dialect_markers(text: str) -> list[str]:
    return [
        marker
        for marker in DIALECT_MARKERS
        if marker in text
    ]


def ends_incomplete(text: str) -> bool:
    stripped = text.strip().rstrip(".,!?؟…").strip().lower()

    if not stripped:
        return True

    if any(stripped.endswith(item.lower()) for item in INCOMPLETE_ENDINGS):
        return True

    final_words = stripped.split()
    if not final_words:
        return True

    return final_words[-1] in {
        "but",
        "and",
        "because",
        "however",
        "though",
        "لكن",
        "و",
        "لأن",
    }


def check_reconstruction(
    raw_text: str,
    result: ReconstructionOutput,
    confidence_hints: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []

    if not result.reconstructed_arabic.strip():
        failures.append("Reconstruction is empty.")
        return failures

    if arabic_letter_ratio(result.reconstructed_arabic) < 0.60:
        failures.append("Reconstruction is not predominantly Arabic.")

    similarity = token_similarity(
        raw_text,
        result.reconstructed_arabic,
    )

    if len(confidence_hints) >= 3 and similarity > 0.97:
        failures.append(
            "Reconstruction copied almost all raw text despite several "
            "low-confidence words."
        )

    if (
        len(confidence_hints) >= 5
        and not result.uncertain_spans
        and similarity > 0.90
    ):
        failures.append(
            "Many low-confidence words remain but no uncertainty was reported."
        )

    return failures


def check_msa(
    reconstructed: str,
    result: MSAOutput,
) -> list[str]:
    failures: list[str] = []

    if not result.msa.strip():
        failures.append("MSA output is empty.")
        return failures

    if arabic_letter_ratio(result.msa) < 0.60:
        failures.append("MSA output is not predominantly Arabic.")

    similarity = token_similarity(reconstructed, result.msa)
    dialect_markers = contains_dialect_markers(result.msa)

    if similarity > 0.96 and dialect_markers:
        failures.append(
            "MSA is almost identical to the dialect reconstruction."
        )

    if dialect_markers:
        failures.append(
            "MSA still contains dialect markers: "
            + ", ".join(sorted(dialect_markers))
        )

    for source_phrase, wrong_phrase in POLARITY_PAIRS:
        if source_phrase in reconstructed and wrong_phrase in result.msa:
            failures.append(
                f"Possible polarity reversal: '{source_phrase}' became "
                f"something containing '{wrong_phrase}'."
            )

    if ends_incomplete(result.msa):
        failures.append("MSA appears to end with an incomplete clause.")

    return failures


def target_language_is_correct(
    text: str,
    target_language: str,
) -> bool:
    target = target_language.strip().lower()

    if target in {"english", "en"}:
        return (
            bool(LATIN_RE.search(text))
            and arabic_letter_ratio(text) < 0.25
        )

    if target in {"arabic", "ar"}:
        return arabic_letter_ratio(text) > 0.65

    return bool(text.strip())


def check_translation(
    result: TranslationOutput,
    target_language: str,
) -> list[str]:
    failures: list[str] = []

    if not result.translation.strip():
        failures.append("Translation is empty.")
        return failures

    if not target_language_is_correct(
        result.translation,
        target_language,
    ):
        failures.append(
            f"Translation does not appear to be in {target_language}."
        )

    if ends_incomplete(result.translation):
        failures.append("Translation ends with an incomplete clause.")

    return failures


def check_validator(
    result: ValidationOutput,
) -> list[str]:
    failures: list[str] = []

    if not result.meaning_preserved:
        failures.append("Validator reports that meaning was not preserved.")

    if not result.polarity_preserved:
        failures.append("Validator reports polarity/negation damage.")

    if not result.rhetorical_structure_preserved:
        failures.append(
            "Validator reports rhetorical structure damage."
        )

    if not result.names_numbers_preserved:
        failures.append(
            "Validator reports name/number changes."
        )

    if not result.target_language_correct:
        failures.append(
            "Validator reports wrong target language."
        )

    if any(
        error.severity in {"major", "critical"}
        for error in result.critical_errors
    ):
        failures.append(
            "Validator found major or critical semantic errors."
        )

    return failures


# ---------------------------------------------------------------------------
# Ollama calls and retry wrapper
# ---------------------------------------------------------------------------

ModelT = TypeVar("ModelT", bound=BaseModel)


def call_ollama_json(
    client: Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[ModelT],
    temperature: float,
    num_ctx: int,
) -> ModelT:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        think=False,
        format=schema.model_json_schema(),
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
        keep_alive="10m",
    )

    content = response.message.content
    if not content:
        raise RuntimeError("Ollama returned empty content.")

    return schema.model_validate_json(content)


def run_with_validation(
    *,
    client: Client,
    model: str,
    system_prompt: str,
    base_user_prompt: str,
    schema: type[ModelT],
    local_validator,
    temperature: float,
    num_ctx: int,
    retries: int,
) -> tuple[ModelT, list[str]]:
    failures: list[str] = []
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        prompt = base_user_prompt

        if failures:
            prompt += (
                "\n\nThe previous result failed these checks:\n- "
                + "\n- ".join(failures)
                + "\nReturn a corrected result."
            )

        try:
            result = call_ollama_json(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                schema=schema,
                temperature=temperature,
                num_ctx=num_ctx,
            )

            failures = local_validator(result)

            if not failures:
                return result, []

            last_error = RuntimeError("; ".join(failures))

        except (
            ResponseError,
            ValidationError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Stage failed after {retries + 1} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_reconstruction(
    *,
    client: Client,
    model: str,
    chunk: TranscriptChunk,
    previous_context: str,
    following_context: str,
    dialect_hint: str,
    confidence_hints: list[dict[str, Any]],
    temperature: float,
    num_ctx: int,
    retries: int,
) -> ReconstructionOutput:
    prompt = f"""
Dialect hint:
{dialect_hint}

Previous context — understanding only:
{previous_context or "[none]"}

MAIN PASSAGE:
{chunk.raw_text}

Following context — understanding only:
{following_context or "[none]"}

Lower-confidence Whisper words:
{format_confidence_hints(confidence_hints)}

Reconstruct only the MAIN PASSAGE.
""".strip()

    result, _ = run_with_validation(
        client=client,
        model=model,
        system_prompt=RECONSTRUCTION_SYSTEM_PROMPT,
        base_user_prompt=prompt,
        schema=ReconstructionOutput,
        local_validator=lambda value: check_reconstruction(
            chunk.raw_text,
            value,
            confidence_hints,
        ),
        temperature=temperature,
        num_ctx=num_ctx,
        retries=retries,
    )
    return result


def stage_msa(
    *,
    client: Client,
    model: str,
    reconstructed_arabic: str,
    previous_context: str,
    following_context: str,
    temperature: float,
    num_ctx: int,
    retries: int,
) -> MSAOutput:
    prompt = f"""
Previous context — understanding only:
{previous_context or "[none]"}

Reconstructed spoken Arabic:
{reconstructed_arabic}

Following context — understanding only:
{following_context or "[none]"}

Produce complete, grammatical, meaning-faithful MSA for the reconstructed
passage only.
""".strip()

    result, _ = run_with_validation(
        client=client,
        model=model,
        system_prompt=MSA_SYSTEM_PROMPT,
        base_user_prompt=prompt,
        schema=MSAOutput,
        local_validator=lambda value: check_msa(
            reconstructed_arabic,
            value,
        ),
        temperature=temperature,
        num_ctx=num_ctx,
        retries=retries,
    )
    return result


def stage_translation(
    *,
    client: Client,
    model: str,
    reconstructed_arabic: str,
    msa: str,
    target_language: str,
    temperature: float,
    num_ctx: int,
    retries: int,
) -> TranslationOutput:
    prompt = f"""
Target language:
{target_language}

Reconstructed spoken Arabic:
{reconstructed_arabic}

Meaning-faithful MSA:
{msa}

Translate the complete intended meaning into {target_language}.
""".strip()

    result, _ = run_with_validation(
        client=client,
        model=model,
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        base_user_prompt=prompt,
        schema=TranslationOutput,
        local_validator=lambda value: check_translation(
            value,
            target_language,
        ),
        temperature=temperature,
        num_ctx=num_ctx,
        retries=retries,
    )
    return result


def stage_validation(
    *,
    client: Client,
    model: str,
    reconstructed_arabic: str,
    msa: str,
    translation: str,
    target_language: str,
    temperature: float,
    num_ctx: int,
    retries: int,
) -> ValidationOutput:
    prompt = f"""
Target language:
{target_language}

Reconstructed Arabic:
{reconstructed_arabic}

MSA:
{msa}

Translation:
{translation}

Compare these outputs strictly and identify semantic errors.
""".strip()

    result, _ = run_with_validation(
        client=client,
        model=model,
        system_prompt=VALIDATION_SYSTEM_PROMPT,
        base_user_prompt=prompt,
        schema=ValidationOutput,
        local_validator=lambda value: [],
        temperature=temperature,
        num_ctx=num_ctx,
        retries=retries,
    )
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_record(
    *,
    chunk: TranscriptChunk,
    confidence_hints: list[dict[str, Any]],
    reconstruction: ReconstructionOutput,
    msa_result: MSAOutput,
    translation_result: TranslationOutput,
    validation_result: ValidationOutput,
    local_failures: list[str],
) -> dict[str, Any]:
    validator_failures = check_validator(validation_result)

    needs_review = any([
        reconstruction.needs_review,
        bool(reconstruction.uncertain_spans),
        msa_result.needs_review,
        bool(msa_result.unresolved_points),
        translation_result.needs_review,
        validation_result.needs_review,
        bool(validator_failures),
        bool(local_failures),
    ])

    return {
        "chunk_id": chunk.chunk_id,
        "segment_ids": chunk.segment_ids,
        "start": chunk.start,
        "end": chunk.end,
        "duration": chunk.duration,
        "raw_transcript": chunk.raw_text,
        "low_confidence_words": confidence_hints,
        "reconstructed_arabic": reconstruction.reconstructed_arabic,
        "msa": msa_result.msa,
        "translation": translation_result.translation,
        "uncertain_spans": [
            item.model_dump()
            for item in reconstruction.uncertain_spans
        ],
        "msa_unresolved_points": msa_result.unresolved_points,
        "translation_notes": translation_result.translation_notes,
        "validation": validation_result.model_dump(),
        "local_validation_failures": local_failures,
        "validator_failures": validator_failures,
        "needs_review": needs_review,
    }


def join_field(
    records: Iterable[dict[str, Any]],
    field_name: str,
) -> str:
    values = [
        str(record.get(field_name, "")).strip()
        for record in records
        if str(record.get(field_name, "")).strip()
    ]

    return "\n\n".join(values).strip() + "\n"


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_checkpoint(
    *,
    path: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    save_json(
        path,
        {
            **metadata,
            "summary": {
                "chunk_count": len(records),
                "review_count": sum(
                    bool(record.get("needs_review"))
                    for record in records
                ),
            },
            "chunks": records,
        },
    )


def write_final_outputs(
    *,
    input_path: Path,
    output_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    semantic_path = output_dir / f"{stem}.semantic.v3.json"
    reconstructed_path = output_dir / f"{stem}.reconstructed_arabic.txt"
    msa_path = output_dir / f"{stem}.msa.txt"
    translation_path = output_dir / f"{stem}.translation.txt"
    review_path = output_dir / f"{stem}.review.v3.json"

    write_checkpoint(
        path=semantic_path,
        metadata=metadata,
        records=records,
    )

    reconstructed_path.write_text(
        join_field(records, "reconstructed_arabic"),
        encoding="utf-8",
    )
    msa_path.write_text(
        join_field(records, "msa"),
        encoding="utf-8",
    )
    translation_path.write_text(
        join_field(records, "translation"),
        encoding="utf-8",
    )

    review_records = [
        record
        for record in records
        if record.get("needs_review")
    ]

    save_json(
        review_path,
        {
            "source_file": str(input_path),
            "review_count": len(review_records),
            "chunks": review_records,
        },
    )

    return {
        "semantic": semantic_path,
        "reconstructed": reconstructed_path,
        "msa": msa_path,
        "translation": translation_path,
        "review": review_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_arguments()

    input_path = args.input_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    models = {
        "reconstruction": args.reconstruction_model or args.model,
        "msa": args.msa_model or args.model,
        "translation": args.translation_model or args.model,
        "validation": args.validation_model or args.model,
    }

    semantic_output = (
        output_dir / f"{input_path.stem}.semantic.v3.json"
    )

    if semantic_output.exists() and not args.overwrite:
        print(
            f"Output already exists: {semantic_output}\n"
            "Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    try:
        payload = load_json(input_path)
        segments = extract_segments(payload)
        chunks = create_semantic_chunks(
            segments=segments,
            max_seconds=args.max_seconds,
            max_words=args.max_words,
            min_seconds=args.min_seconds,
            pause_threshold=args.pause_threshold,
        )
    except RuntimeError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_file": str(input_path),
        "models": models,
        "target_language": args.target_language,
        "dialect_hint": args.dialect_hint,
        "settings": {
            "max_seconds": args.max_seconds,
            "max_words": args.max_words,
            "min_seconds": args.min_seconds,
            "pause_threshold": args.pause_threshold,
            "context_segments": args.context_segments,
            "low_confidence_threshold": (
                args.low_confidence_threshold
            ),
            "strict": args.strict,
        },
    }

    print(f"Input: {input_path}")
    print(f"Segments: {len(segments)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Models: {models}")
    print(f"Target language: {args.target_language}")

    client = Client(host=args.ollama_host)
    records: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"\n[{index}/{len(chunks)}] "
            f"{chunk.start:.2f}s–{chunk.end:.2f}s "
            f"({chunk.duration:.2f}s)"
        )

        previous_context, following_context = surrounding_context(
            all_segments=segments,
            chunk=chunk,
            count=args.context_segments,
        )

        confidence_hints = low_confidence_words(
            chunk=chunk,
            threshold=args.low_confidence_threshold,
        )

        print(f"  Low-confidence words: {len(confidence_hints)}")

        try:
            reconstruction = stage_reconstruction(
                client=client,
                model=models["reconstruction"],
                chunk=chunk,
                previous_context=previous_context,
                following_context=following_context,
                dialect_hint=args.dialect_hint,
                confidence_hints=confidence_hints,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                retries=args.retries,
            )
            print("  Stage 1: reconstruction complete")

            msa_result = stage_msa(
                client=client,
                model=models["msa"],
                reconstructed_arabic=(
                    reconstruction.reconstructed_arabic
                ),
                previous_context=previous_context,
                following_context=following_context,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                retries=args.retries,
            )
            print("  Stage 2: MSA complete")

            translation_result = stage_translation(
                client=client,
                model=models["translation"],
                reconstructed_arabic=(
                    reconstruction.reconstructed_arabic
                ),
                msa=msa_result.msa,
                target_language=args.target_language,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                retries=args.retries,
            )
            print("  Stage 3: translation complete")

            validation_result = stage_validation(
                client=client,
                model=models["validation"],
                reconstructed_arabic=(
                    reconstruction.reconstructed_arabic
                ),
                msa=msa_result.msa,
                translation=translation_result.translation,
                target_language=args.target_language,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                retries=args.retries,
            )
            print("  Stage 4: semantic validation complete")

        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 1

        local_failures: list[str] = []
        local_failures.extend(
            check_reconstruction(
                chunk.raw_text,
                reconstruction,
                confidence_hints,
            )
        )
        local_failures.extend(
            check_msa(
                reconstruction.reconstructed_arabic,
                msa_result,
            )
        )
        local_failures.extend(
            check_translation(
                translation_result,
                args.target_language,
            )
        )

        record = build_record(
            chunk=chunk,
            confidence_hints=confidence_hints,
            reconstruction=reconstruction,
            msa_result=msa_result,
            translation_result=translation_result,
            validation_result=validation_result,
            local_failures=local_failures,
        )

        records.append(record)

        if record["needs_review"]:
            print("  REVIEW REQUIRED")

        validator_failures = check_validator(validation_result)

        if args.strict and validator_failures:
            write_checkpoint(
                path=semantic_output,
                metadata=metadata,
                records=records,
            )
            print(
                "  Strict mode stopped because semantic validation failed.",
                file=sys.stderr,
            )
            return 2

        write_checkpoint(
            path=semantic_output,
            metadata=metadata,
            records=records,
        )

    paths = write_final_outputs(
        input_path=input_path,
        output_dir=output_dir,
        metadata=metadata,
        records=records,
    )

    print("\nCompleted")
    print(f"Structured JSON: {paths['semantic']}")
    print(f"Reconstructed:   {paths['reconstructed']}")
    print(f"MSA:             {paths['msa']}")
    print(f"Translation:     {paths['translation']}")
    print(f"Review file:     {paths['review']}")
    print(
        "Review chunks:   "
        f"{sum(bool(record['needs_review']) for record in records)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())