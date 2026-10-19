#!/usr/bin/env python3
"""Summarize an ANE_FIELDFARE_GPU_TRACE timeline.

usage: summarize.py trace.json [--phase decode|verify|prefill|draft] [--top N]
                    [--within PREFIX] [--step NAME]

For each phase window (host spans `decode_token`, `verify N`, `prefill N`),
reports, per step:

- wall time, GPU busy time (union of encoder intervals) and GPU idle time;
- GPU time per kernel, with every instant of busy time split evenly among
  the encoders running at that instant, so the column sums to busy time even
  when encoders from consecutive command buffers overlap; the raw encoder
  duration is shown beside it;
- GPU idle time by what the host was doing meanwhile (the innermost host
  span covering it), which separates the GPU waiting on the host from the
  host waiting on the GPU.

`--within PREFIX` keeps only steps inside host spans whose name starts with
PREFIX, e.g. the bench's "case code spec". `--step NAME` keeps only steps
named exactly NAME, e.g. "verify 8".

Kernel names are encoder call sites (`File.function:line`).
"""
import argparse
import bisect
import collections
import json
import sys

PHASE_SPANS = {
    "decode": lambda n: n == "decode_token",
    "verify": lambda n: n.startswith("verify "),
    "prefill": lambda n: n.startswith("prefill "),
    "draft": lambda n: n == "draft_propose",
}


def load(path):
    events = [e for e in json.load(open(path))["traceEvents"] if e.get("ph") == "X"]
    for e in events:
        e["t0"] = e["ts"]
        e["t1"] = e["ts"] + e["dur"]
    return events


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def clip(intervals, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi]


def clip_to_steps(events, steps):
    """Each event's parts inside the step windows, as (t0, t1, name)."""
    starts = [a for a, _ in steps]
    out = []
    for e in events:
        i = max(0, bisect.bisect_right(starts, e["t0"]) - 1)
        while i < len(steps) and steps[i][0] < e["t1"]:
            a, b = steps[i]
            lo, hi = max(a, e["t0"]), min(b, e["t1"])
            if hi > lo:
                out.append((lo, hi, e["name"]))
            i += 1
    return out


def shared_time(parts):
    """Busy time per name, each instant split evenly among the active parts."""
    points = sorted([(lo, 1, n) for lo, _, n in parts] + [(hi, -1, n) for _, hi, n in parts],
                    key=lambda p: (p[0], p[1]))
    active = collections.Counter()
    share = collections.Counter()
    last = None
    for t, kind, name in points:
        if last is not None and active and t > last:
            n = sum(active.values())
            for k, c in active.items():
                share[k] += (t - last) * c / n
        last = t
        active[name] += kind
        if active[name] == 0:
            del active[name]
    return share


def innermost(spans):
    """Flatten nested host spans to segments owned by the shortest covering span."""
    points = sorted({x for e in spans for x in (e["t0"], e["t1"])})
    by_start = sorted(spans, key=lambda e: e["t0"])
    segments, active, j = [], [], 0
    for s, t in zip(points, points[1:]):
        while j < len(by_start) and by_start[j]["t0"] <= s:
            active.append(by_start[j])
            j += 1
        active = [e for e in active if e["t1"] > s]
        if active:
            segments.append((s, t, min(active, key=lambda e: e["dur"])["name"]))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--phase", default="decode", choices=PHASE_SPANS)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--within", help="only steps inside host spans named PREFIX...")
    ap.add_argument("--step", help="only steps named exactly STEP, e.g. 'verify 8'")
    args = ap.parse_args()

    events = load(args.trace)
    gpu = [e for e in events if e["cat"] == "gpu"]
    host = [e for e in events if e["cat"] == "host"]
    match = PHASE_SPANS[args.phase]
    steps = sorted((e["t0"], e["t1"]) for e in host
                   if match(e["name"]) and (args.step is None or e["name"] == args.step))
    if args.within:
        outer = [(e["t0"], e["t1"]) for e in host if e["name"].startswith(args.within)]
        steps = [(a, b) for a, b in steps if any(lo <= a and b <= hi for lo, hi in outer)]
    if not steps:
        sys.exit(f"no {args.phase} spans in {args.trace}")
    n = len(steps)
    wall = sum(b - a for a, b in steps)

    busy_iv = clip_to_steps([{"t0": a, "t1": b, "name": ""}
                             for a, b in union([(e["t0"], e["t1"]) for e in gpu])], steps)
    busy_iv = [(lo, hi) for lo, hi, _ in busy_iv]
    busy = sum(b - a for a, b in busy_iv)

    print(f"{args.phase}: {n} steps, {wall / n:8.1f} us/step wall, "
          f"GPU busy {busy / n:8.1f} us ({100 * busy / wall:.1f}%), idle {(wall - busy) / n:8.1f} us")

    # Kernels.
    parts = clip_to_steps(gpu, steps)
    share = shared_time(parts)
    raw = collections.Counter()
    count = collections.Counter()
    for lo, hi, name in parts:
        raw[name] += hi - lo
        count[name] += 1
    print(f"\n{'GPU time per step (us)':<52} {'shared':>8} {'raw':>8} {'calls':>6} {'share':>6}")
    total = sum(share.values())
    for name, t in share.most_common(args.top):
        print(f"  {name:<50} {t / n:8.1f} {raw[name] / n:8.1f} {count[name] / n:6.1f} {100 * t / total:5.1f}%")
    rest = total - sum(t for _, t in share.most_common(args.top))
    if rest > 0:
        print(f"  {'(other)':<50} {rest / n:8.1f}")

    # GPU idle, by the innermost host span covering it.
    idle = []
    for a, b in steps:
        cursor = a
        for lo, hi in busy_iv:
            if hi <= a or lo >= b:
                continue
            if lo > cursor:
                idle.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < b:
            idle.append((cursor, b))
    phase_spans = {e["name"] for e in host if match(e["name"])}
    # Bench case spans enclose whole runs; time only they cover is untraced.
    segments = innermost([e for e in host
                          if e["name"] not in phase_spans and not e["name"].startswith("case ")])
    blame = collections.Counter()
    k = 0
    for lo, hi in idle:
        while k < len(segments) and segments[k][1] <= lo:
            k += 1
        cursor, j = lo, k
        while j < len(segments) and segments[j][0] < hi:
            s0, s1, name = segments[j]
            if s0 > cursor:
                blame["host (untraced)"] += min(s0, hi) - cursor
            blame[name] += min(s1, hi) - max(s0, cursor)
            cursor = max(cursor, min(s1, hi))
            j += 1
        if cursor < hi:
            blame["host (untraced)"] += hi - cursor
    print(f"\n{'GPU idle per step, by host activity (us)':<52} {'idle':>8}")
    for name, t in blame.most_common():
        print(f"  {name:<50} {t / n:8.1f}")


if __name__ == "__main__":
    main()
