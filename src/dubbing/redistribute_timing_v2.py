#!/usr/bin/env python3
"""
redistribute_timing.py

Inputs
------
1) Original dubbing JSON:
   [
     {
       "segment": 1,
       "speaker": "RANDA",
       "start": "00:00:00.000",
       "end": "00:00:06.560",
       "duration_seconds": 6.56,
       "text": "..."
     },
     ...
   ]

2) timing_report.json from the Chatterbox generation pass:
   [
     {
       "segment": 1,
       "speaker": "RANDA",
       "target_duration_seconds": 6.56,
       "generated_duration_seconds": 8.88,
       ...
     },
     ...
   ]

Output
------
A new dubbing JSON with redistributed start/end/duration values.

Optimization goals
------------------
- Preserve the overall start and end of the dubbing timeline.
- Never allocate more time to a segment than its generated WAV duration.
  Therefore the downstream dubbing pass never needs to leave spare silence.
- Minimize the maximum speed-up required by any segment.
- Keep boundary movement local.
- Permit less movement at speaker changes than within one speaker's turn.
- Among equally good solutions, keep boundaries as close as possible
  to their original positions.

Requires:
    pip install scipy
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


DEFAULT_MAX_SHIFT = 2.0
DEFAULT_SPEAKER_SHIFT = 0.50
BINARY_SEARCH_ITERS = 45
EPS = 1e-8


def ts_to_sec(ts: str) -> float:
    """HH:MM:SS.mmm -> seconds"""
    hh, mm, ss = ts.replace(",", ".").split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def sec_to_ts(sec: float) -> str:
    """seconds -> HH:MM:SS.mmm"""
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    hh, rem = divmod(total_ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1_000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_prepare(original, report):
    rep = {int(x["segment"]): x for x in report}

    rows = []
    for item in original:
        seg = int(item["segment"])
        if seg not in rep:
            raise ValueError(f"Segment {seg} is missing from timing report.")

        start = ts_to_sec(item["start"])
        end = ts_to_sec(item["end"])
        if end <= start:
            raise ValueError(f"Segment {seg}: end must be after start.")

        generated = float(rep[seg]["generated_duration_seconds"])
        if generated <= 0:
            raise ValueError(f"Segment {seg}: invalid generated duration.")

        rows.append({
            "segment": seg,
            "speaker": str(item.get("speaker", "")).strip(),
            "start": start,
            "end": end,
            "generated": generated,
            "original": item,
            "report": rep[seg],
        })

    # Require chronological, touching/near-touching dubbing windows.
    # Small original gaps are absorbed into the adjacent timing space.
    for i in range(1, len(rows)):
        if rows[i]["start"] < rows[i-1]["start"] - EPS:
            raise ValueError("Original JSON is not chronological.")

    return rows


def make_original_boundaries(rows):
    """
    We optimize one continuous dubbing interval from first start to last end.
    Internal original boundaries are midpoint-style handoffs:
      boundary i = original start of segment i+1 when touching,
      otherwise midpoint of previous end and next start.
    This preserves local placement while allowing original gaps to be redistributed.
    """
    n = len(rows)
    B = np.zeros(n + 1, dtype=float)
    B[0] = rows[0]["start"]
    B[-1] = rows[-1]["end"]

    for j in range(1, n):
        prev_end = rows[j-1]["end"]
        next_start = rows[j]["start"]
        if abs(next_start - prev_end) <= 0.020:
            B[j] = (prev_end + next_start) / 2.0
        else:
            B[j] = (prev_end + next_start) / 2.0

    if np.any(np.diff(B) <= 0):
        raise ValueError("Could not construct increasing original boundaries.")

    return B


def boundary_shift_limits(rows, max_shift, speaker_shift):
    """
    Internal boundary j lies between segment j and j+1.
    Use tighter allowance when speaker changes.
    """
    n = len(rows)
    limits = np.zeros(n + 1, dtype=float)
    limits[0] = 0.0
    limits[-1] = 0.0

    for j in range(1, n):
        left = rows[j-1]["speaker"]
        right = rows[j]["speaker"]
        limits[j] = speaker_shift if left != right else max_shift

    return limits


def feasibility_lp(rows, B, shift_limits, ratio):
    """
    Find durations d_i such that:
      generated_i / ratio <= d_i <= generated_i
      sum d_i = total timeline
      every internal cumulative boundary stays within allowed local drift
    """
    n = len(rows)
    g = np.array([r["generated"] for r in rows], dtype=float)
    total = B[-1] - B[0]

    lower = g / ratio
    upper = g.copy()

    # Immediate total feasibility.
    if lower.sum() > total + 1e-7:
        return None
    if upper.sum() < total - 1e-7:
        return None

    A = []
    b = []

    # cumulative durations through segment j-1 define internal boundary j
    # new absolute boundary = B[0] + cumulative durations
    for j in range(1, n):
        row = np.zeros(n)
        row[:j] = 1.0

        desired_cum = B[j] - B[0]
        lim = shift_limits[j]

        # cumulative <= desired + lim
        A.append(row.copy())
        b.append(desired_cum + lim)

        # cumulative >= desired - lim  => -cumulative <= -(desired-lim)
        A.append(-row.copy())
        b.append(-(desired_cum - lim))

    res = linprog(
        c=np.zeros(n),
        A_ub=np.array(A) if A else None,
        b_ub=np.array(b) if b else None,
        A_eq=np.ones((1, n)),
        b_eq=np.array([total]),
        bounds=list(zip(lower, upper)),
        method="highs",
    )

    if not res.success:
        return None
    return res.x


def find_min_shift_scale(rows, B, base_shift_limits, max_scale=100.0):
    """
    Find the smallest multiplier for the configured boundary-shift limits
    that makes ANY zero-spare solution feasible.

    This is needed when a local speaker block contains less generated speech
    than its original timeline allocation. In that case, zero silence cannot
    be achieved unless one or more neighboring boundaries are allowed to move.

    The relative protection remains intact:
      - same-speaker boundaries use --max-shift * scale
      - speaker-change boundaries use --speaker-shift * scale
    """
    # At a very large speed ratio, lower duration bounds approach zero.
    # Feasibility then depends mainly on:
    #   1) total generated duration being enough to fill the timeline, and
    #   2) whether boundary shift limits permit the redistribution.
    test_ratio = 1000.0

    if feasibility_lp(rows, B, base_shift_limits, test_ratio) is not None:
        return 1.0

    lo = 1.0
    hi = 2.0

    while hi <= max_scale:
        scaled = base_shift_limits * hi
        if feasibility_lp(rows, B, scaled, test_ratio) is not None:
            break
        lo = hi
        hi *= 2.0

    if hi > max_scale:
        scaled = base_shift_limits * max_scale
        if feasibility_lp(rows, B, scaled, test_ratio) is None:
            raise RuntimeError(
                "No zero-spare redistribution is feasible even after "
                f"relaxing boundary shifts by {max_scale:.1f}x. "
                "This usually means the current timeline cannot be filled "
                "using the generated clips without allowing slowdown/silence."
            )
        hi = max_scale

    for _ in range(50):
        mid = (lo + hi) / 2.0
        scaled = base_shift_limits * mid
        if feasibility_lp(rows, B, scaled, test_ratio) is not None:
            hi = mid
        else:
            lo = mid

    return hi


def describe_boundary_limits(rows, shift_limits):
    same = []
    changes = []
    for j in range(1, len(rows)):
        left = rows[j-1]["speaker"]
        right = rows[j]["speaker"]
        if left == right:
            same.append(shift_limits[j])
        else:
            changes.append(shift_limits[j])

    return {
        "same_speaker_limit": max(same) if same else 0.0,
        "speaker_change_limit": max(changes) if changes else 0.0,
    }

def find_min_ratio(rows, B, shift_limits):
    g = np.array([r["generated"] for r in rows], dtype=float)
    total = B[-1] - B[0]

    if g.sum() < total - 1e-7:
        raise ValueError(
            "\nZERO-SPARE SOLUTION IS IMPOSSIBLE with the current rule 'never slow clips'.\n"
            f"Generated speech total: {g.sum():.3f}s\n"
            f"Timeline total:        {total:.3f}s\n"
            f"Shortfall:              {total - g.sum():.3f}s\n"
            "Either allow some silence, allow slight slowdown, or shorten the protected timeline."
        )

    lo = 1.0
    hi = max(1.01, float(g.max() / max(np.diff(B).min(), 0.001)))

    # Increase upper bound until feasible.
    while feasibility_lp(rows, B, shift_limits, hi) is None:
        hi *= 1.5
        if hi > 10:
            raise RuntimeError(
                "No feasible local redistribution found. "
                "Increase --max-shift and/or --speaker-shift."
            )

    best = None
    for _ in range(BINARY_SEARCH_ITERS):
        mid = (lo + hi) / 2.0
        sol = feasibility_lp(rows, B, shift_limits, mid)
        if sol is not None:
            hi = mid
            best = sol
        else:
            lo = mid

    # Re-solve at final ratio.
    best = feasibility_lp(rows, B, shift_limits, hi)
    return hi, best


def refine_boundaries(rows, B, shift_limits, ratio):
    """
    At the already-minimal maximum speed ratio, choose the solution
    whose cumulative boundaries stay closest to the original boundaries.

    LP variables:
      d_0..d_(n-1)  segment durations
      z_1..z_(n-1)  absolute boundary deviations

    Objective: minimize sum(z_j), with a tiny penalty encouraging durations
    closer to generated lengths where possible.
    """
    n = len(rows)
    g = np.array([r["generated"] for r in rows], dtype=float)
    total = B[-1] - B[0]

    nd = n
    nz = n - 1
    N = nd + nz

    c = np.zeros(N)
    # Main objective: keep boundaries local.
    c[nd:] = 1.0

    A = []
    b = []

    for j in range(1, n):
        cum = np.zeros(N)
        cum[:j] = 1.0
        zidx = nd + (j - 1)

        desired = B[j] - B[0]
        lim = shift_limits[j]

        # Hard allowed shift
        A.append(cum.copy())
        b.append(desired + lim)

        A.append(-cum.copy())
        b.append(-(desired - lim))

        # z >= cumulative - desired
        r1 = cum.copy()
        r1[zidx] = -1.0
        A.append(r1)
        b.append(desired)

        # z >= desired - cumulative
        r2 = -cum.copy()
        r2[zidx] = -1.0
        A.append(r2)
        b.append(-desired)

    Aeq = np.zeros((1, N))
    Aeq[0, :n] = 1.0

    lower = g / ratio
    upper = g.copy()

    bounds = [(lower[i], upper[i]) for i in range(n)]
    bounds += [(0.0, None) for _ in range(nz)]

    res = linprog(
        c=c,
        A_ub=np.array(A) if A else None,
        b_ub=np.array(b) if b else None,
        A_eq=Aeq,
        b_eq=np.array([total]),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError("Final refinement LP failed: " + res.message)

    return res.x[:n]


def build_output(rows, durations, B, max_ratio):
    out = []
    cursor = B[0]

    for row, dur in zip(rows, durations):
        start = cursor
        end = start + float(dur)
        generated = row["generated"]
        speed_ratio = generated / dur

        item = dict(row["original"])
        item["start"] = sec_to_ts(start)
        item["end"] = sec_to_ts(end)
        item["duration_seconds"] = round(float(dur), 3)

        # Extra metadata is ignored by the normal dubbing script,
        # but useful for checking the redistribution.
        item["generated_duration_seconds"] = round(generated, 3)
        item["planned_speed_ratio"] = round(speed_ratio, 6)
        item["planned_speed_percent"] = round((speed_ratio - 1.0) * 100.0, 3)
        item["timing_redistributed"] = True

        out.append(item)
        cursor = end

    # Force exact final timestamp after rounding accumulation.
    out[-1]["end"] = sec_to_ts(B[-1])
    out[-1]["duration_seconds"] = round(
        ts_to_sec(out[-1]["end"]) - ts_to_sec(out[-1]["start"]), 3
    )

    return out


def print_report(rows, output, B, max_ratio):
    print("\n" + "=" * 78)
    print("LOCAL TIMING REDISTRIBUTION")
    print("=" * 78)
    print(f"Segments:                  {len(rows)}")
    print(f"Timeline start:            {sec_to_ts(B[0])}")
    print(f"Timeline end:              {sec_to_ts(B[-1])}")
    print(f"Timeline duration:         {B[-1] - B[0]:.3f}s")
    print(f"Generated speech total:    {sum(r['generated'] for r in rows):.3f}s")
    print(f"Minimum max speed ratio:   {max_ratio:.6f}x")
    print(f"Maximum speed-up:          {(max_ratio - 1) * 100:.3f}%")
    print()

    speeds = [x["planned_speed_ratio"] for x in output]
    exactish = sum(1 for s in speeds if s <= 1.005)
    under5 = sum(1 for s in speeds if s <= 1.05)
    under10 = sum(1 for s in speeds if s <= 1.10)

    print(f"Segments <= 0.5% speed-up: {exactish}")
    print(f"Segments <= 5% speed-up:   {under5}")
    print(f"Segments <= 10% speed-up:  {under10}")
    print(f"Segments above 10%:        {len(speeds) - under10}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original_json", help="Original dubbing JSON with start/end timestamps")
    ap.add_argument("timing_report", help="timing_report.json from generated clips")
    ap.add_argument(
        "-o", "--output",
        default="dubbing_redistributed.json",
        help="Output JSON filename"
    )
    ap.add_argument(
        "--max-shift",
        type=float,
        default=DEFAULT_MAX_SHIFT,
        help="Maximum internal boundary movement within the same speaker turn (seconds)"
    )
    ap.add_argument(
        "--speaker-shift",
        type=float,
        default=DEFAULT_SPEAKER_SHIFT,
        help="Maximum movement at a speaker change (seconds)"
    )
    ap.add_argument(
        "--no-auto-relax",
        action="store_true",
        help=(
            "Disable automatic relaxation of boundary-shift limits. "
            "Without this flag, the script finds the smallest multiplier "
            "needed to make zero-spare redistribution feasible."
        )
    )
    args = ap.parse_args()

    original = load_json(args.original_json)
    report = load_json(args.timing_report)

    rows = validate_and_prepare(original, report)
    B = make_original_boundaries(rows)
    base_limits = boundary_shift_limits(rows, args.max_shift, args.speaker_shift)

    if args.no_auto_relax:
        limits = base_limits
        shift_scale = 1.0
    else:
        shift_scale = find_min_shift_scale(rows, B, base_limits)
        limits = base_limits * shift_scale

        if shift_scale > 1.000001:
            d = describe_boundary_limits(rows, limits)
            print("\nAUTO-RELAX")
            print("-" * 78)
            print(
                f"Original local shift limits were infeasible for zero-spare timing."
            )
            print(f"Smallest feasible shift multiplier: {shift_scale:.4f}x")
            print(
                f"Effective same-speaker limit:  ±{d['same_speaker_limit']:.3f}s"
            )
            print(
                f"Effective speaker-change limit: ±{d['speaker_change_limit']:.3f}s"
            )
            print("-" * 78)

    ratio, _ = find_min_ratio(rows, B, limits)
    durations = refine_boundaries(rows, B, limits, ratio)

    output = build_output(rows, durations, B, ratio)

    # Add top-level metadata without breaking downstream readers that expect
    # a plain list: write the normal list to the requested output, and a
    # sibling diagnostics file with optimizer details.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    diagnostic_path = str(Path(args.output).with_suffix("")) + "_diagnostics.json"
    d = describe_boundary_limits(rows, limits)
    diagnostics = {
        "configured_max_shift_seconds": args.max_shift,
        "configured_speaker_shift_seconds": args.speaker_shift,
        "auto_relax_enabled": not args.no_auto_relax,
        "applied_shift_scale": shift_scale,
        "effective_same_speaker_shift_seconds": d["same_speaker_limit"],
        "effective_speaker_change_shift_seconds": d["speaker_change_limit"],
        "minimum_max_speed_ratio": ratio,
        "minimum_max_speed_percent": (ratio - 1.0) * 100.0,
    }
    with open(diagnostic_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print_report(rows, output, B, ratio)
    print(f"\nSaved:       {Path(args.output).resolve()}")
    print(f"Diagnostics: {Path(diagnostic_path).resolve()}")


if __name__ == "__main__":
    main()
