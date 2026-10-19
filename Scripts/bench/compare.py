#!/usr/bin/env python3
"""Statistics, token agreement and an HTML report for ANEFieldfareBench results.

usage: compare.py --a A1.json [A2.json ...] [--b B1.json ...] [--html report.html]

Each side pools all of its files (rounds) and reps. With --b, every case's
throughput and every step's latency is compared as a B/A ratio with a
bootstrap 95% interval. A change is reported only when the interval excludes 1
and the ratio clears the A/A noise floor: the largest spread between two of
A's own rounds, the analogue of the 1-ulp KL floor vLLM PR #61867 judged its
accuracy against. Stdlib only.
"""
import argparse
import html
import json
import math
import random
import statistics as st
import sys

RNG = random.Random(0)
BOOT = 2000


def load(paths):
    runs = [json.load(open(p)) for p in paths]
    for run, path in zip(runs, paths):
        run["_path"] = path
    return runs


def case_samples(runs, key="decode_tps"):
    out = {}
    for run in runs:
        for c in run["cases"]:
            out.setdefault((c["prompt"], c["mode"]), []).append(c[key])
    return out


def step_samples(runs):
    out = {}
    for run in runs:
        for s in run.get("steps", []):
            out.setdefault(s["rows"], []).extend(s["ms"])
    return out


def boot_ratio(a, b, stat=st.mean):
    """B/A ratio of `stat` and its bootstrap 95% interval."""
    point = stat(b) / stat(a)
    draws = []
    for _ in range(BOOT):
        ra = [RNG.choice(a) for _ in a]
        rb = [RNG.choice(b) for _ in b]
        draws.append(stat(rb) / stat(ra))
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def boot_ci(xs, stat=st.mean):
    draws = sorted(stat([RNG.choice(xs) for _ in xs]) for _ in range(BOOT))
    return stat(xs), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def noise_floor(runs):
    """Largest |ratio - 1| between two of one build's own rounds, per case mean."""
    if len(runs) < 2:
        return None
    per_run = [case_samples([r]) for r in runs]
    worst = 0.0
    for i in range(len(per_run)):
        for j in range(i + 1, len(per_run)):
            for key in per_run[i].keys() & per_run[j].keys():
                worst = max(worst, abs(st.mean(per_run[j][key]) / st.mean(per_run[i][key]) - 1))
    return worst


def verdict(point, lo, hi, floor):
    if lo > 1 and (floor is None or point - 1 > floor):
        return "faster"
    if hi < 1 and (floor is None or 1 - point > floor):
        return "slower"
    return "no change"


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def agreement(runs_a, runs_b):
    """Per case: does B reproduce A's tokens, and is each side self-consistent."""
    def ids(runs):
        out = {}
        for run in runs:
            for c in run["cases"]:
                out.setdefault((c["prompt"], c["mode"]), []).append(c["generated_ids"])
        return out
    ia, ib = ids(runs_a), ids(runs_b) if runs_b else {}
    reference = {}
    for (prompt, mode), seqs in ia.items():
        if mode == "plain":
            reference[prompt] = seqs[0]
    rows = []
    for key in sorted(ia):
        prompt, mode = key
        ref = reference.get(prompt, ia[key][0])
        divs_a = sorted({first_divergence(ref, s) for s in ia[key]}, key=lambda d: (d is None, d))
        divs_b = sorted({first_divergence(ref, s) for s in ib.get(key, [])}, key=lambda d: (d is None, d))
        rows.append((prompt, mode, divs_a, divs_b))
    return rows


def breakdown(runs):
    """Mean ms per decoded token by phase, plain cases only."""
    out = {}
    for run in runs:
        for c in run["cases"]:
            if c["mode"] != "plain" or c["new_tokens"] < 2:
                continue
            n, k = c["new_tokens"], c["counters"]
            parts = {
                "attention GPU": k["decode_attention_gpu_s"],
                "shared FFN GPU": k["decode_shared_gpu_s"],
                "routed GPU": k["decode_routed_gpu_s"],
                "storage wait": k["decode_io_s"],
            }
            total = c["decode_seconds"]
            parts = {name: 1000 * v / n for name, v in parts.items()}
            parts["host and other"] = max(0.0, 1000 * total / n - sum(parts.values()))
            for name, v in parts.items():
                out.setdefault(c["prompt"], {}).setdefault(name, []).append(v)
    return {p: {name: st.mean(v) for name, v in d.items()} for p, d in out.items()}


def spec_summary(runs):
    out = {}
    for run in runs:
        for c in run["cases"]:
            s = c.get("speculation")
            if not s:
                continue
            steps = max(1, s["speculative_steps"])
            drafted_blocks = s["speculative_steps"] + s["truncated_steps"]
            out.setdefault(c["prompt"], []).append((
                s["accepted_drafts"] / steps,
                s["plain_steps"],
                s["truncated_steps"] / max(1, drafted_blocks),
                s.get("acceptance_rate", 0)))
    return {p: tuple(st.mean(x[i] for x in v) for i in range(4)) for p, v in out.items()}


# ---------- text report ----------

def fmt_pct(r):
    return f"{(r - 1) * 100:+.1f}%"


def text_report(A, B, floor):
    lines = []
    sa, sb = case_samples(A), case_samples(B) if B else {}
    lines.append(f"A: {len(A)} round(s) {[r['_path'] for r in A]}")
    if B:
        lines.append(f"B: {len(B)} round(s) {[r['_path'] for r in B]}")
    lines.append(f"A/A noise floor: {'n/a (one round)' if floor is None else fmt_pct(1 + floor).lstrip('+')}")
    lines.append("")
    lines.append(f"{'case':16s} {'A tok/s':>10s} {'B tok/s':>10s} {'B/A':>8s} {'95% CI':>18s}  verdict")
    for key in sorted(sa):
        a = sa[key]
        name = f"{key[0]} {key[1]}"
        if key in sb:
            p, lo, hi = boot_ratio(a, sb[key])
            lines.append(f"{name:16s} {st.mean(a):10.1f} {st.mean(sb[key]):10.1f} {fmt_pct(p):>8s} "
                         f"[{fmt_pct(lo)}, {fmt_pct(hi)}]  {verdict(p, lo, hi, floor)}")
        else:
            m, lo, hi = boot_ci(a)
            lines.append(f"{name:16s} {m:10.1f} {'':>10s} {'':>8s} [{lo:.1f}, {hi:.1f}]")
    pa, pb = step_samples(A), step_samples(B) if B else {}
    if pa:
        lines.append("")
        lines.append(f"{'step':16s} {'A ms':>10s} {'B ms':>10s} {'B/A':>8s} {'95% CI':>18s}")
        for rows in sorted(pa):
            name = "decode" if rows == 0 else f"verify {rows} row" + ("s" if rows > 1 else "")
            if rows in pb:
                p, lo, hi = boot_ratio(pa[rows], pb[rows], st.median)
                lines.append(f"{name:16s} {st.median(pa[rows]):10.1f} {st.median(pb[rows]):10.1f} "
                             f"{fmt_pct(p):>8s} [{fmt_pct(lo)}, {fmt_pct(hi)}]")
            else:
                lines.append(f"{name:16s} {st.median(pa[rows]):10.1f}")
    lines.append("")
    lines.append("token agreement with A's plain greedy output (first differing token; - = identical):")
    for prompt, mode, da, db in agreement(A, B):
        show = lambda ds: ", ".join("-" if d is None else str(d) for d in ds) or "n/a"
        lines.append(f"  {prompt:8s} {mode:5s} A: {show(da):14s}" + (f" B: {show(db)}" if B else ""))
    return "\n".join(lines)


# ---------- HTML report ----------

CSS = """
.viz-root { color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --good:#006300; --bad:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#f0efec; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --good:#0ca30c; --bad:#e66767;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#f0efec; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --good:#0ca30c; --bad:#e66767;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; }
body { margin:0; background:var(--plane); }
.viz-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--text-primary);
  background:var(--plane); padding:24px 16px; max-width:1040px; margin:0 auto; }
h1 { font-size:20px; font-weight:600; margin:0 0 4px; } h2 { font-size:16px; font-weight:600; margin:0 0 4px; }
.sub { color:var(--text-secondary); font-size:13px; margin:0 0 16px; }
.card { background:var(--surface-1); border:1px solid rgba(127,127,127,.18); border-radius:10px;
  padding:16px; margin:0 0 16px; overflow-x:auto; }
.hero { font-size:48px; font-weight:600; line-height:1.1; }
.tiles { display:flex; flex-wrap:wrap; gap:24px; }
.tile .label { color:var(--text-secondary); font-size:13px; } .tile .value { font-size:22px; font-weight:600; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--text-secondary); margin:4px 0 8px; }
.key { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.keybar { display:inline-block; width:12px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }
table { border-collapse:collapse; font-size:13px; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } th { color:var(--text-secondary); font-weight:500; }
details { margin-top:8px; } summary { cursor:pointer; color:var(--text-secondary); font-size:13px; }
.faster { color:var(--good); } .slower { color:var(--bad); } .same { color:var(--text-secondary); }
svg text { font-family: system-ui, -apple-system, sans-serif; }
.tick { fill:var(--muted); font-size:11px; } .lab { fill:var(--text-secondary); font-size:12px; }
.grid { stroke:var(--grid); stroke-width:1; } .axis { stroke:var(--axis); stroke-width:1; }
.hit:hover + .mark, .mark:hover { opacity:.85; }
pre { font-size:12px; color:var(--text-secondary); white-space:pre-wrap; }
"""


def esc(x):
    return html.escape(str(x))


def nice_max(v):
    if v <= 0:
        return 1
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def dot_plot(sa, sb):
    """Throughput per case: mean with bootstrap 95% interval, A blue, B orange."""
    keys = sorted(sa)
    row_h, left, right, top = 34, 130, 24, 12
    width = 720
    height = top + row_h * len(keys) + 30
    vmax = nice_max(max(max(v) for v in list(sa.values()) + list(sb.values())) * 1.05)
    x = lambda v: left + (width - left - right) * v / vmax
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Decode throughput per case">']
    for i in range(6):
        v = vmax * i / 5
        parts.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{height - 26}"/>')
        parts.append(f'<text class="tick" x="{x(v):.1f}" y="{height - 10}" text-anchor="middle">'
                     f'{v:g}{" tok/s" if i == 5 else ""}</text>')
    for r, key in enumerate(keys):
        y = top + row_h * r + row_h / 2
        parts.append(f'<text class="lab" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">'
                     f'{esc(key[0])} · {esc(key[1])}</text>')
        for side, samples, color, dy in (("A", sa, "var(--s1)", -6), ("B", sb, "var(--s2)", 6)):
            if key not in samples:
                continue
            m, lo, hi = boot_ci(samples[key])
            yy = y + (dy if sb else 0)
            tip = f"{side} {key[0]} {key[1]}: {m:.1f} tok/s (95% CI {lo:.1f}–{hi:.1f}, n={len(samples[key])})"
            parts.append(f'<g><title>{esc(tip)}</title>'
                         f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                         f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                         f'<circle cx="{x(m):.1f}" cy="{yy:.1f}" r="5" fill="{color}" '
                         f'stroke="var(--surface-1)" stroke-width="2"/>'
                         f'<rect x="{x(lo) - 6:.1f}" y="{yy - 8:.1f}" width="{x(hi) - x(lo) + 12:.1f}" '
                         f'height="16" fill="transparent"/></g>')
    parts.append("</svg>")
    return "".join(parts)


def ratio_chart(sa, sb, floor):
    """B/A change per case in percent with its bootstrap 95% interval, around
    zero, over the A/A noise-floor band."""
    keys = sorted(sa.keys() & sb.keys())
    stats = {k: boot_ratio(sa[k], sb[k]) for k in keys}
    lim = max([abs(v - 1) for p in stats.values() for v in p] + [floor or 0]) * 100 * 1.15
    lim = nice_max(max(lim, 1.0))
    row_h, left, right, top = 34, 130, 24, 12
    width = 720
    height = top + row_h * len(keys) + 34
    x = lambda pct: left + (width - left - right) * (pct + lim) / (2 * lim)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Throughput change per case, B versus A">']
    if floor:
        f = floor * 100
        parts.append(f'<rect x="{x(-f):.1f}" y="{top}" width="{x(f) - x(-f):.1f}" '
                     f'height="{height - top - 30}" fill="var(--grid)" opacity="0.7">'
                     f'<title>A/A noise floor ±{f:.1f}%</title></rect>')
    for i in range(5):
        v = -lim + 2 * lim * i / 4
        parts.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{height - 30}"/>')
        parts.append(f'<text class="tick" x="{x(v):.1f}" y="{height - 14}" text-anchor="middle">{v:+g}%</text>')
    parts.append(f'<line class="axis" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top}" y2="{height - 30}"/>')
    for r, key in enumerate(keys):
        y = top + row_h * r + row_h / 2
        p, lo, hi = stats[key]
        v = verdict(p, lo, hi, floor)
        color = {"faster": "var(--good)", "slower": "var(--bad)"}.get(v, "var(--s2)")
        parts.append(f'<text class="lab" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">'
                     f'{esc(key[0])} · {esc(key[1])}</text>')
        tip = (f"{key[0]} {key[1]}: {fmt_pct(p)} (95% CI {fmt_pct(lo)} to {fmt_pct(hi)}), {v}; "
               f"A {st.mean(sa[key]):.1f}, B {st.mean(sb[key]):.1f} tok/s")
        parts.append(f'<g><title>{esc(tip)}</title>'
                     f'<line x1="{x((lo - 1) * 100):.1f}" x2="{x((hi - 1) * 100):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                     f'<circle cx="{x((p - 1) * 100):.1f}" cy="{y:.1f}" r="5" fill="{color}" '
                     f'stroke="var(--surface-1)" stroke-width="2"/>'
                     f'<rect x="{left}" y="{y - row_h / 2:.1f}" width="{width - left - right}" height="{row_h}" '
                     f'fill="transparent"/></g>')
        parts.append(f'<text class="tick" x="{width - right}" y="{y + 4:.1f}" text-anchor="end">{esc(v)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def step_chart(pa, pb):
    rows = sorted(pa)
    width, height, left, right, top, bottom = 720, 260, 64, 24, 16, 40
    vmax = nice_max(max(st.median(v) for v in list(pa.values()) + list(pb.values())) * 1.1)
    xs = {r: left + (width - left - right) * i / max(1, len(rows) - 1) for i, r in enumerate(rows)}
    y = lambda v: top + (height - top - bottom) * (1 - v / vmax)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Step latency by rows">']
    for i in range(6):
        v = vmax * i / 5
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">'
                     f'{v:g}{" ms" if i == 5 else ""}</text>')
    for r in rows:
        parts.append(f'<text class="tick" x="{xs[r]:.1f}" y="{height - 18}" text-anchor="middle">'
                     f'{"decode" if r == 0 else f"{r} row" + ("s" if r > 1 else "")}</text>')
    for side, samples, color in (("A", pa, "var(--s1)"), ("B", pb, "var(--s2)")):
        pts = [(xs[r], y(st.median(samples[r])), r) for r in rows if r in samples]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py, _) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        for px, py, r in pts:
            s = sorted(samples[r])
            tip = (f"{side} {'decode' if r == 0 else f'verify {r} rows'}: median {st.median(s):.1f} ms, "
                   f"p10 {s[len(s) // 10]:.1f}, p90 {s[len(s) * 9 // 10]:.1f} (n={len(s)})")
            parts.append(f'<g><title>{esc(tip)}</title><circle cx="{px:.1f}" cy="{py:.1f}" r="5" '
                         f'fill="{color}" stroke="var(--surface-1)" stroke-width="2"/>'
                         f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="transparent"/></g>')
    parts.append("</svg>")
    return "".join(parts)


PHASES = ["attention GPU", "shared FFN GPU", "routed GPU", "storage wait", "host and other"]


def breakdown_chart(ba, bb):
    prompts = sorted(ba)
    bars = [(p, "A", ba[p]) for p in prompts] + [(p, "B", bb[p]) for p in prompts if p in bb]
    bars.sort(key=lambda t: (t[0], t[1]))
    width, left, right, bar_h, gap, top = 720, 130, 60, 18, 12, 8
    height = top + len(bars) * (bar_h + gap) + 28
    vmax = nice_max(max(sum(d.values()) for _, _, d in bars) * 1.05)
    x = lambda v: left + (width - left - right) * v / vmax
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Decode time per token by phase">']
    for i in range(6):
        v = vmax * i / 5
        parts.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{height - 24}"/>')
        parts.append(f'<text class="tick" x="{x(v):.1f}" y="{height - 8}" text-anchor="middle">'
                     f'{v:g}{" ms/token" if i == 5 else ""}</text>')
    for i, (prompt, side, d) in enumerate(bars):
        y0 = top + i * (bar_h + gap)
        parts.append(f'<text class="lab" x="{left - 10}" y="{y0 + bar_h - 5}" text-anchor="end">'
                     f'{esc(prompt)} · {side}</text>')
        acc = 0.0
        segs = [(n, d[n]) for n in PHASES if d.get(n, 0) > 0]
        for j, (name, v) in enumerate(segs):
            x0, x1 = x(acc), x(acc + v)
            w = max(0.0, x1 - x0 - (2 if j < len(segs) - 1 else 0))   # 2px surface gap
            rx = 4 if j == len(segs) - 1 else 0
            color = f"var(--s{PHASES.index(name) + 1})"
            tip = f"{prompt} {side}: {name} {v:.2f} ms/token"
            if rx:
                path = (f"M{x0:.1f},{y0} H{x0 + w - rx:.1f} Q{x0 + w:.1f},{y0} {x0 + w:.1f},{y0 + rx} "
                        f"V{y0 + bar_h - rx} Q{x0 + w:.1f},{y0 + bar_h} {x0 + w - rx:.1f},{y0 + bar_h} "
                        f"H{x0:.1f} Z")
                parts.append(f'<path d="{path}" fill="{color}"><title>{esc(tip)}</title></path>')
            else:
                parts.append(f'<rect x="{x0:.1f}" y="{y0}" width="{w:.1f}" height="{bar_h}" fill="{color}">'
                             f'<title>{esc(tip)}</title></rect>')
            acc += v
        parts.append(f'<text class="lab" x="{x(acc) + 6:.1f}" y="{y0 + bar_h - 5}">{acc:.1f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def html_report(A, B, floor):
    sa, sb = case_samples(A), case_samples(B) if B else {}
    pa, pb = step_samples(A), step_samples(B) if B else {}
    ba, bb = breakdown(A), breakdown(B) if B else {}
    env = A[0]["environment"]
    out = [f'<!doctype html><html><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>Bench report</title><style>{CSS}</style></head><body><div class="viz-root">']
    out.append("<h1>ANE Fieldfare bench</h1>")
    meta = f"{esc(env.get('hardware'))}, {esc(env.get('os'))}, {esc(env.get('date'))}"
    out.append(f'<p class="sub">A: {esc(A[0]["label"])} (git {esc(env.get("git"))}, {len(A)} round(s))'
               + (f' · B: {esc(B[0]["label"])} (git {esc(B[0]["environment"].get("git"))}, {len(B)} round(s))'
                  if B else "") + f' · {meta}</p>')

    # Headline: geometric mean B/A over cases.
    if B:
        keys = sorted(sa.keys() & sb.keys())
        ratios = [st.mean(sb[k]) / st.mean(sa[k]) for k in keys]
        geo = math.exp(st.mean(math.log(r) for r in ratios))
        draws = []
        for _ in range(500):
            rs = [st.mean([RNG.choice(sb[k]) for _ in sb[k]]) / st.mean([RNG.choice(sa[k]) for _ in sa[k]])
                  for k in keys]
            draws.append(math.exp(st.mean(math.log(r) for r in rs)))
        draws.sort()
        lo, hi = draws[12], draws[487]
        v = verdict(geo, lo, hi, floor)
        cls = {"faster": "faster", "slower": "slower"}.get(v, "same")
        out.append('<div class="card"><div class="tiles">')
        headline = fmt_pct(geo) if v != "no change" else "No change"
        reason = "" if v != "no change" else (
            " · within the A/A floor" if lo > 1 or hi < 1 else " · interval includes 0")
        out.append(f'<div class="tile"><div class="label">B vs A, geometric mean decode throughput</div>'
                   f'<div class="hero {cls}">{headline}</div>'
                   f'<div class="label">estimate {fmt_pct(geo)}, 95% CI {fmt_pct(lo)} to {fmt_pct(hi)}'
                   f'{reason}</div></div>')
        out.append(f'<div class="tile"><div class="label">A/A noise floor</div><div class="value">'
                   f'{"n/a" if floor is None else f"±{floor * 100:.1f}%"}</div>'
                   f'<div class="label">needs 2+ A rounds</div></div>')
        out.append("</div></div>")

    legend = ('<div class="legend"><span><span class="key" style="background:var(--s1)"></span>A '
              f'{esc(A[0]["label"])}</span>' + (f'<span><span class="key" style="background:var(--s2)"></span>B '
              f'{esc(B[0]["label"])}</span>' if B else "") + "</div>")

    # Throughput.
    if B:
        out.append('<div class="card"><h2>Throughput change per case, B vs A</h2>'
                   '<p class="sub">Change in mean decode tok/s with bootstrap 95% interval; the shaded band is '
                   'the A/A noise floor. Absolute values are in the table.</p>')
        out.append(ratio_chart(sa, sb, floor))
    else:
        out.append('<div class="card"><h2>Decode throughput per case</h2>'
                   '<p class="sub">Mean tok/s with bootstrap 95% interval over all reps and rounds.</p>')
        out.append(legend + dot_plot(sa, sb))
    out.append("<details><summary>Table</summary><table><tr><th>case</th><th>A tok/s</th><th>n</th>"
               + ("<th>B tok/s</th><th>n</th><th>B/A</th><th>95% CI</th><th>verdict</th>" if B else "") + "</tr>")
    for key in sorted(sa):
        row = f"<tr><td>{esc(key[0])} · {esc(key[1])}</td><td>{st.mean(sa[key]):.1f}</td><td>{len(sa[key])}</td>"
        if key in sb:
            p, lo, hi = boot_ratio(sa[key], sb[key])
            v = verdict(p, lo, hi, floor)
            cls = {"faster": "faster", "slower": "slower"}.get(v, "same")
            row += (f"<td>{st.mean(sb[key]):.1f}</td><td>{len(sb[key])}</td><td>{fmt_pct(p)}</td>"
                    f"<td>{fmt_pct(lo)} to {fmt_pct(hi)}</td><td class='{cls}'>{esc(v)}</td>")
        out.append(row + "</tr>")
    out.append("</table></details></div>")

    # Steps.
    if pa:
        out.append('<div class="card"><h2>Step latency by verify width</h2>'
                   '<p class="sub">Median ms at a fixed context: one decode step, then verifies of 1–8 rows.</p>')
        out.append(legend + step_chart(pa, pb))
        out.append("<details><summary>Table</summary><table><tr><th>step</th><th>A median ms</th>"
                   + ("<th>B median ms</th><th>B/A</th><th>95% CI</th>" if B else "") + "</tr>")
        for rows in sorted(pa):
            name = "decode" if rows == 0 else f"verify {rows} row" + ("s" if rows > 1 else "")
            row = f"<tr><td>{name}</td><td>{st.median(pa[rows]):.1f}</td>"
            if rows in pb:
                p, lo, hi = boot_ratio(pa[rows], pb[rows], st.median)
                row += f"<td>{st.median(pb[rows]):.1f}</td><td>{fmt_pct(p)}</td><td>{fmt_pct(lo)} to {fmt_pct(hi)}</td>"
            out.append(row + "</tr>")
        out.append("</table></details></div>")

    # Breakdown.
    if ba:
        out.append('<div class="card"><h2>Where a plain decode token goes</h2><p class="sub">GPU time by command '
                   'buffer, host time waiting on storage, and the remainder (host work and GPU bubbles). Command '
                   'buffers can overlap, so the remainder is a floor.</p>')
        out.append('<div class="legend">' + "".join(
            f'<span><span class="keybar" style="background:var(--s{i + 1})"></span>{esc(n)}</span>'
            for i, n in enumerate(PHASES)) + "</div>")
        out.append(breakdown_chart(ba, bb))
        out.append("<details><summary>Table</summary><table><tr><th>prompt · build</th>"
                   + "".join(f"<th>{esc(n)}</th>" for n in PHASES) + "<th>total</th></tr>")
        for side, data in (("A", ba), ("B", bb)):
            for p in sorted(data):
                d = data[p]
                out.append(f"<tr><td>{esc(p)} · {side}</td>" + "".join(f"<td>{d.get(n, 0):.2f}</td>" for n in PHASES)
                           + f"<td>{sum(d.values()):.2f}</td></tr>")
        out.append("</table></details></div>")

    # Speculation + agreement.
    spa, spb = spec_summary(A), spec_summary(B) if B else {}
    if spa or spb:
        out.append('<div class="card"><h2>Speculation</h2><table><tr><th>prompt · build</th>'
                   "<th>accepted drafts / verify</th><th>plain steps</th><th>drafted blocks truncated</th>"
                   "<th>planner acceptance</th></tr>")
        for side, data in (("A", spa), ("B", spb)):
            for p in sorted(data):
                acc, plain, trunc, rate = data[p]
                out.append(f"<tr><td>{esc(p)} · {side}</td><td>{acc:.2f}</td><td>{plain:.0f}</td>"
                           f"<td>{trunc * 100:.0f}%</td><td>{rate:.2f}</td></tr>")
        out.append("</table></div>")
    out.append('<div class="card"><h2>Token agreement</h2><p class="sub">First token that differs from A\'s first '
               'plain greedy run of the same prompt, per run; "identical" means every run matched.</p>'
               "<table><tr><th>case</th><th>A runs</th>" + ("<th>B runs</th>" if B else "") + "</tr>")
    for prompt, mode, da, db in agreement(A, B):
        show = lambda ds: "identical" if ds == [None] else ", ".join("same" if d is None else str(d) for d in ds)
        out.append(f"<tr><td>{esc(prompt)} · {esc(mode)}</td><td>{show(da)}</td>"
                   + (f"<td>{show(db) if db else 'n/a'}</td>" if B else "") + "</tr>")
    out.append("</table></div>")
    out.append(f"<details><summary>Text summary</summary><pre>{esc(text_report(A, B, floor))}</pre></details>")
    out.append("</div></body></html>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="*", default=[])
    ap.add_argument("--html")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b) if args.b else []
    floor = noise_floor(A)
    print(text_report(A, B, floor))
    if args.html:
        open(args.html, "w").write(html_report(A, B, floor))
        print(f"\nreport: {args.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
