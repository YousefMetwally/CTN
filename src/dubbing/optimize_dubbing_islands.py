#!/usr/bin/env python3
"""
optimize_dubbing_islands.py

Anchored local timing optimizer for documentary dubbing.

Inputs
------
1) Original dubbing JSON with:
   segment, speaker, start, end, duration_seconds, text

2) timing_report.json with:
   segment, generated_duration_seconds, ...

Output
------
A new dubbing JSON whose timestamps are optimized LOCALLY.

Core design
-----------
- Speaker changes are HARD ANCHORS.
- Each same-speaker turn is optimized independently.
- Long turns are subdivided into synchronization islands.
- Island start/end are fixed to the original video timeline.
- Inside an island, time is redistributed to reduce unnecessary speed-up.
- Clips are NEVER slowed below 1.0x just to consume spare time.
- Any unavoidable spare time becomes explicit natural pause time.
- Spare time is distributed across internal boundaries rather than dumped
  into one giant silence whenever possible.
- No timing can drift outside its island.

This is specifically meant to prevent:
    "segment gets sped up, then a later segment has a huge silence"

Requirements:
    pip install numpy

No scipy required.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


EPS = 1e-9


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def ts_to_sec(ts: str) -> float:
    hh, mm, ss = ts.replace(",", ".").split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def sec_to_ts(sec: float) -> str:
    sec = max(0.0, sec)
    total_ms = int(round(sec * 1000))
    hh, rem = divmod(total_ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1_000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Load / validate
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_rows(original, report):
    rep = {int(x["segment"]): x for x in report}
    rows = []

    for item in original:
        seg = int(item["segment"])
        if seg not in rep:
            raise ValueError(f"Segment {seg} missing from timing report.")

        start = ts_to_sec(item["start"])
        end = ts_to_sec(item["end"])
        if end <= start:
            raise ValueError(f"Segment {seg}: invalid start/end.")

        generated = float(rep[seg]["generated_duration_seconds"])
        if generated <= 0:
            raise ValueError(f"Segment {seg}: generated duration must be > 0.")

        rows.append({
            "segment": seg,
            "speaker": str(item.get("speaker", "")).strip(),
            "start": start,
            "end": end,
            "original_duration": end - start,
            "generated": generated,
            "text": item.get("text", ""),
            "original": item,
        })

    rows.sort(key=lambda r: r["start"])

    for i in range(1, len(rows)):
        if rows[i]["start"] < rows[i-1]["start"] - 1e-6:
            raise ValueError("Original JSON is not chronological.")

    return rows


# ---------------------------------------------------------------------------
# Turn / island construction
# ---------------------------------------------------------------------------

def build_same_speaker_turns(rows):
    """
    Returns list of lists of row indices.
    Speaker changes are hard anchors.
    """
    if not rows:
        return []

    turns = []
    current = [0]

    for i in range(1, len(rows)):
        if rows[i]["speaker"] == rows[i-1]["speaker"]:
            current.append(i)
        else:
            turns.append(current)
            current = [i]

    turns.append(current)
    return turns


def split_turn_into_islands(rows, turn_indices, max_island_seconds=35.0,
                            max_segments=8, gap_anchor_seconds=1.5):
    """
    Split a same-speaker turn into smaller islands.

    Hard-ish split triggers:
    - island becomes longer than max_island_seconds
    - island contains more than max_segments
    - original gap between adjacent segments is >= gap_anchor_seconds

    The island boundary itself remains fixed to the original timeline.
    """
    if not turn_indices:
        return []

    islands = []
    current = [turn_indices[0]]
    island_start = rows[turn_indices[0]]["start"]

    for pos in range(1, len(turn_indices)):
        idx = turn_indices[pos]
        prev_idx = turn_indices[pos - 1]

        prev_end = rows[prev_idx]["end"]
        curr_start = rows[idx]["start"]
        original_gap = max(0.0, curr_start - prev_end)

        projected_end = rows[idx]["end"]
        projected_span = projected_end - island_start

        should_split = (
            len(current) >= max_segments
            or projected_span > max_island_seconds
            or original_gap >= gap_anchor_seconds
        )

        if should_split:
            islands.append(current)
            current = [idx]
            island_start = rows[idx]["start"]
        else:
            current.append(idx)

    islands.append(current)
    return islands


# ---------------------------------------------------------------------------
# Optimization logic
# ---------------------------------------------------------------------------

def waterfill_durations(generated, available):
    """
    Allocate speech-window durations within one island.

    Objective:
    - Never allocate > generated duration (no slowdown).
    - If speech total > available:
        minimize maximum speed-up by using a uniform ratio:
            d_i = g_i / ratio
    - If speech total <= available:
        every clip gets natural duration d_i = g_i
        and the remainder is unavoidable pause time.

    Returns:
        speech_durations, unavoidable_pause
    """
    g = np.array(generated, dtype=float)
    total_g = float(g.sum())

    if total_g > available + EPS:
        ratio = total_g / available
        d = g / ratio
        pause = 0.0
    else:
        ratio = 1.0
        d = g.copy()
        pause = available - total_g

    return d, pause, ratio


def original_internal_gaps(rows, indices):
    """
    Gaps between segment i and i+1 based on original timing.
    Negative overlap is treated as zero.
    """
    gaps = []
    for a, b in zip(indices[:-1], indices[1:]):
        gaps.append(max(0.0, rows[b]["start"] - rows[a]["end"]))
    return gaps


def pause_weights(rows, indices, mode="balanced"):
    """
    Weight where unavoidable pause should go.

    We do NOT want one huge gap unless the original timing strongly supports it.

    balanced:
      - baseline weight at every internal boundary
      - larger original gaps get more weight
      - sentence-ending punctuation gets slight preference

    equal:
      - distribute evenly

    original:
      - strongly follow original gaps, with tiny floor elsewhere
    """
    n_boundaries = len(indices) - 1
    if n_boundaries <= 0:
        return np.array([], dtype=float)

    gaps = np.array(original_internal_gaps(rows, indices), dtype=float)

    if mode == "equal":
        w = np.ones(n_boundaries, dtype=float)

    elif mode == "original":
        w = gaps + 0.05

    else:
        # balanced
        w = np.ones(n_boundaries, dtype=float)

        if gaps.max(initial=0.0) > 0:
            norm = gaps / max(gaps.max(), 1e-9)
            w += 2.0 * norm

        # Prefer pauses after complete/rhetorical sentences
        for j, idx in enumerate(indices[:-1]):
            txt = str(rows[idx].get("text", "")).strip()
            if txt.endswith((".", "!", "?", "…", "\"", "”", "'")):
                w[j] += 0.50
            if txt.endswith((",", ";", ":")):
                w[j] += 0.20

    return w


def distribute_pause(total_pause, weights, edge_pause_fraction=0.15):
    """
    Split unavoidable pause among:
      - leading pause
      - internal pauses
      - trailing pause

    edge_pause_fraction is the MAX fraction for each edge, not guaranteed.
    We keep most pause internally so it doesn't accumulate at one end.
    """
    if total_pause <= EPS:
        return 0.0, np.zeros(len(weights), dtype=float), 0.0

    n = len(weights)

    # Single-segment island: cannot redistribute internally.
    # Keep pause centered around the clip.
    if n == 0:
        lead = total_pause * 0.5
        trail = total_pause - lead
        return lead, np.array([], dtype=float), trail

    edge_each = total_pause * edge_pause_fraction
    internal_total = total_pause - 2 * edge_each

    # If pause is tiny, don't force edge gaps.
    if total_pause < 0.6:
        edge_each = 0.0
        internal_total = total_pause

    if weights.sum() <= EPS:
        internal = np.full(n, internal_total / n)
    else:
        internal = internal_total * (weights / weights.sum())

    return edge_each, internal, edge_each


def cap_and_redistribute_pauses(internal, max_single_pause):
    """
    Prevent one internal pause from becoming enormous.

    If a pause exceeds max_single_pause, redistribute excess across
    other internal pauses. If all are capped, leftover is returned.

    Returns:
        capped_pauses, leftover
    """
    p = np.array(internal, dtype=float)

    if len(p) == 0 or max_single_pause <= 0:
        return p, 0.0

    leftover = 0.0

    for _ in range(100):
        over = p > max_single_pause + EPS
        if not np.any(over):
            break

        excess = float(np.sum(p[over] - max_single_pause))
        p[over] = max_single_pause

        under = p < max_single_pause - EPS
        if not np.any(under):
            leftover += excess
            break

        capacity = np.maximum(0.0, max_single_pause - p[under])
        cap_sum = float(capacity.sum())

        if cap_sum <= EPS:
            leftover += excess
            break

        add = np.minimum(capacity, excess * capacity / cap_sum)
        p[under] += add

        remaining = excess - float(add.sum())
        if remaining > EPS:
            leftover += remaining
            break

    return p, leftover


def optimize_island(rows, indices, pause_mode="balanced",
                    max_single_pause=2.0, edge_pause_fraction=0.10):
    """
    Optimize one fixed island.

    Island start = original start of first segment.
    Island end   = original end of last segment.

    Returns segment timing records plus island diagnostics.
    """
    first = indices[0]
    last = indices[-1]

    island_start = rows[first]["start"]
    island_end = rows[last]["end"]
    available = island_end - island_start

    generated = [rows[i]["generated"] for i in indices]
    speech_durations, unavoidable_pause, common_ratio = waterfill_durations(
        generated, available
    )

    weights = pause_weights(rows, indices, mode=pause_mode)

    lead_pause, internal_pauses, trail_pause = distribute_pause(
        unavoidable_pause,
        weights,
        edge_pause_fraction=edge_pause_fraction,
    )

    internal_pauses, leftover = cap_and_redistribute_pauses(
        internal_pauses,
        max_single_pause=max_single_pause,
    )

    # If internal pauses are fully capped, put leftover at the edges.
    if leftover > EPS:
        lead_pause += leftover / 2.0
        trail_pause += leftover / 2.0

    # Numerical correction
    allocated = (
        float(np.sum(speech_durations))
        + lead_pause
        + float(np.sum(internal_pauses))
        + trail_pause
    )
    correction = available - allocated
    trail_pause += correction

    # Construct timestamps.
    cursor = island_start + lead_pause
    segment_results = []

    for local_i, idx in enumerate(indices):
        dur = float(speech_durations[local_i])
        start = cursor
        end = start + dur

        generated_dur = rows[idx]["generated"]
        speed_ratio = generated_dur / max(dur, EPS)

        segment_results.append({
            "row_index": idx,
            "start": start,
            "end": end,
            "duration": dur,
            "speed_ratio": speed_ratio,
            "pause_before": lead_pause if local_i == 0 else float(internal_pauses[local_i - 1]),
            "pause_after": (
                trail_pause if local_i == len(indices) - 1
                else float(internal_pauses[local_i])
            ),
        })

        cursor = end
        if local_i < len(indices) - 1:
            cursor += float(internal_pauses[local_i])

    # Final numerical sanity
    expected_last = island_end - trail_pause
    if segment_results:
        delta = expected_last - segment_results[-1]["end"]
        segment_results[-1]["end"] += delta
        segment_results[-1]["duration"] += delta
        segment_results[-1]["speed_ratio"] = (
            rows[indices[-1]]["generated"]
            / max(segment_results[-1]["duration"], EPS)
        )

    diag = {
        "speaker": rows[first]["speaker"],
        "first_segment": rows[first]["segment"],
        "last_segment": rows[last]["segment"],
        "start": sec_to_ts(island_start),
        "end": sec_to_ts(island_end),
        "available_seconds": round(available, 3),
        "generated_seconds": round(sum(generated), 3),
        "unavoidable_pause_seconds": round(unavoidable_pause, 3),
        "common_speed_ratio_if_overfull": round(common_ratio, 6),
        "common_speed_percent_if_overfull": round((common_ratio - 1) * 100, 3),
        "leading_pause_seconds": round(lead_pause, 3),
        "trailing_pause_seconds": round(trail_pause, 3),
        "max_internal_pause_seconds": round(
            float(max(internal_pauses)) if len(internal_pauses) else 0.0, 3
        ),
    }

    return segment_results, diag


# ---------------------------------------------------------------------------
# Full optimization
# ---------------------------------------------------------------------------

def optimize_all(rows, max_island_seconds, max_segments, gap_anchor_seconds,
                 pause_mode, max_single_pause, edge_pause_fraction):
    turns = build_same_speaker_turns(rows)

    all_results = {}
    island_diags = []

    for turn in turns:
        islands = split_turn_into_islands(
            rows,
            turn,
            max_island_seconds=max_island_seconds,
            max_segments=max_segments,
            gap_anchor_seconds=gap_anchor_seconds,
        )

        for island in islands:
            results, diag = optimize_island(
                rows,
                island,
                pause_mode=pause_mode,
                max_single_pause=max_single_pause,
                edge_pause_fraction=edge_pause_fraction,
            )

            island_diags.append(diag)

            for r in results:
                all_results[r["row_index"]] = r

    return all_results, island_diags


def build_output(rows, optimized):
    out = []

    for i, row in enumerate(rows):
        opt = optimized[i]
        item = dict(row["original"])

        item["start"] = sec_to_ts(opt["start"])
        item["end"] = sec_to_ts(opt["end"])
        item["duration_seconds"] = round(opt["duration"], 3)

        # Useful downstream metadata
        item["generated_duration_seconds"] = round(row["generated"], 3)
        item["planned_speed_ratio"] = round(opt["speed_ratio"], 6)
        item["planned_speed_percent"] = round(
            (opt["speed_ratio"] - 1.0) * 100.0, 3
        )
        item["planned_pause_before_seconds"] = round(opt["pause_before"], 3)
        item["planned_pause_after_seconds"] = round(opt["pause_after"], 3)
        item["timing_strategy"] = "anchored_local_island"

        out.append(item)

    return out


def summarize(rows, optimized, island_diags):
    speeds = [optimized[i]["speed_ratio"] for i in range(len(rows))]
    pauses = []

    for i in range(len(rows)):
        pauses.append(optimized[i]["pause_before"])
        if i == len(rows) - 1:
            pauses.append(optimized[i]["pause_after"])

    total_original_target = sum(r["original_duration"] for r in rows)
    total_generated = sum(r["generated"] for r in rows)

    return {
        "segments": len(rows),
        "islands": len(island_diags),
        "total_original_segment_windows_seconds": round(total_original_target, 3),
        "total_generated_speech_seconds": round(total_generated, 3),
        "maximum_planned_speed_ratio": round(max(speeds), 6),
        "maximum_planned_speed_percent": round((max(speeds) - 1) * 100, 3),
        "segments_over_1_05x": sum(s > 1.05 + EPS for s in speeds),
        "segments_over_1_10x": sum(s > 1.10 + EPS for s in speeds),
        "segments_over_1_15x": sum(s > 1.15 + EPS for s in speeds),
        "total_unavoidable_pause_seconds": round(
            sum(x["unavoidable_pause_seconds"] for x in island_diags), 3
        ),
        "maximum_internal_pause_seconds": round(
            max(
                (x["max_internal_pause_seconds"] for x in island_diags),
                default=0.0
            ),
            3
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Anchored local optimizer for documentary dubbing timing."
    )

    ap.add_argument("original_json")
    ap.add_argument("timing_report")

    ap.add_argument(
        "-o", "--output",
        default="dubbing_locally_optimized.json"
    )
    ap.add_argument(
        "--diagnostics",
        default=None,
        help="Optional diagnostics JSON path. Default: sibling *_diagnostics.json"
    )

    ap.add_argument(
        "--max-island-seconds",
        type=float,
        default=35.0,
        help="Maximum approximate island span before adding a hard anchor."
    )
    ap.add_argument(
        "--max-segments",
        type=int,
        default=8,
        help="Maximum number of segments per island."
    )
    ap.add_argument(
        "--gap-anchor-seconds",
        type=float,
        default=1.5,
        help="Original same-speaker gap at/above this becomes an island boundary."
    )
    ap.add_argument(
        "--max-single-pause",
        type=float,
        default=2.0,
        help="Try not to place more than this many seconds at one internal pause."
    )
    ap.add_argument(
        "--edge-pause-fraction",
        type=float,
        default=0.10,
        help="Fraction of unavoidable island pause reserved for each edge."
    )
    ap.add_argument(
        "--pause-mode",
        choices=["balanced", "equal", "original"],
        default="balanced",
        help="How unavoidable pause is distributed."
    )

    args = ap.parse_args()

    original = load_json(args.original_json)
    report = load_json(args.timing_report)

    rows = prepare_rows(original, report)

    optimized, island_diags = optimize_all(
        rows,
        max_island_seconds=args.max_island_seconds,
        max_segments=args.max_segments,
        gap_anchor_seconds=args.gap_anchor_seconds,
        pause_mode=args.pause_mode,
        max_single_pause=args.max_single_pause,
        edge_pause_fraction=args.edge_pause_fraction,
    )

    output = build_output(rows, optimized)
    summary = summarize(rows, optimized, island_diags)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.diagnostics:
        diagnostics_path = Path(args.diagnostics)
    else:
        diagnostics_path = output_path.with_name(
            output_path.stem + "_diagnostics.json"
        )

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    diagnostics = {
        "summary": summary,
        "settings": {
            "max_island_seconds": args.max_island_seconds,
            "max_segments": args.max_segments,
            "gap_anchor_seconds": args.gap_anchor_seconds,
            "max_single_pause": args.max_single_pause,
            "edge_pause_fraction": args.edge_pause_fraction,
            "pause_mode": args.pause_mode,
        },
        "islands": island_diags,
    }

    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("ANCHORED LOCAL DUBBING OPTIMIZATION")
    print("=" * 80)
    print(f"Segments:                  {summary['segments']}")
    print(f"Islands:                   {summary['islands']}")
    print(
        f"Max planned speed-up:      "
        f"{summary['maximum_planned_speed_ratio']:.4f}x "
        f"({summary['maximum_planned_speed_percent']:.2f}%)"
    )
    print(
        f"Segments > 1.05x:          {summary['segments_over_1_05x']}"
    )
    print(
        f"Segments > 1.10x:          {summary['segments_over_1_10x']}"
    )
    print(
        f"Segments > 1.15x:          {summary['segments_over_1_15x']}"
    )
    print(
        f"Unavoidable pause total:   "
        f"{summary['total_unavoidable_pause_seconds']:.3f}s"
    )
    print(
        f"Max internal pause:        "
        f"{summary['maximum_internal_pause_seconds']:.3f}s"
    )
    print("-" * 80)
    print(f"Optimized JSON: {output_path.resolve()}")
    print(f"Diagnostics:    {diagnostics_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
