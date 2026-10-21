#!/usr/bin/env python3
"""Draws the README's speed chart as two SVGs, one per GitHub theme.

    python3 Scripts/readme/charts.py

writes assets/speed-light.svg and assets/speed-dark.svg. The numbers
are the README's comparison table (same M5, 16 GB, measured 2026-10-20); change
them there and here together.
"""

from pathlib import Path

SYSTEMS = ["ANE Fieldfare", "Turbo Fieldfare", "llama.cpp"]

# (group title, [(system index, low, high, estimated)])
GENERATION = [
    ("Greedy, math prompt", [(0, 96.6, 96.6, False), (1, 24, 25, False), (2, 5, 9, False)]),
    ("Temperature 0.2", [(0, 36, 53, False), (1, 21, 25, False)]),
]
FIRST_TOKEN = [
    ("3,015-token prompt, model load included",
     [(0, 10, 10, False), (1, 48, 48, False), (2, 110, 110, True)]),
]

THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "page": "#ffffff",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70"],
        "text": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d", "page": "#0d1117",
    },
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
WIDTH = 880
BAR = 18
PITCH = 28
HEADER = 26
NAME_COLUMN = 112
TOP = 62


def panel(x, width, title, subtitle, groups, axis_max, ticks, unit, theme):
    """One bar panel with its own axis; returns (svg parts, bottom y)."""
    t = THEMES[theme]
    bar_left = x + NAME_COLUMN
    bar_width = width - NAME_COLUMN - 96

    def scale(v):
        return bar_left + v / axis_max * bar_width

    out = [
        f'<text x="{x}" y="{TOP - 30}" class="title">{title}</text>',
        f'<text x="{x}" y="{TOP - 12}" class="muted">{subtitle}</text>',
    ]
    y = TOP
    rows = []
    spans = []  # gridlines run beside the bars, never through a group title
    for group, entries in groups:
        y += HEADER
        rows.append(f'<text x="{x}" y="{y - 8}" class="group">{group}</text>')
        spans.append((y - 5, y + len(entries) * PITCH - PITCH + BAR + 5))
        for system, low, high, estimated in entries:
            color = t["series"][system]
            mid = y + BAR / 2
            rows.append(f'<text x="{bar_left - 10}" y="{mid + 4.5}" class="name" '
                        f'text-anchor="end">{SYSTEMS[system]}</text>')
            fill = f'url(#hatch-{system})' if estimated else color
            if high > low:
                # Solid to the lowest result, a lighter extension to the highest.
                rows.append(f'<rect x="{bar_left}" y="{y}" width="{scale(low) - bar_left - 1:.1f}" '
                            f'height="{BAR}" fill="{fill}"/>')
                rows.append(rounded_end(scale(low) + 1, y, scale(high) - scale(low) - 1,
                                        blend(color, t["page"], 0.4)))
            else:
                rows.append(rounded_end(bar_left, y, scale(high) - bar_left, fill))
            if estimated:
                label = f"~{high}{unit} (est.)"
            elif high > low:
                label = f"{low}–{high}{unit}"
            else:
                label = f"{high}{unit}"
            weight = ' class="value strong"' if system == 0 else ' class="value"'
            rows.append(f'<text x="{scale(high) + 8:.1f}" y="{mid + 4.5}"{weight}>{label}</text>')
            y += PITCH
    bottom = y + 4
    for tick in ticks:
        tx = scale(tick)
        for top, end in spans:
            out.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{end}" class="grid"/>')
        out.append(f'<text x="{tx:.1f}" y="{bottom + 16}" class="tick" text-anchor="middle">{tick}</text>')
    return out + rows, bottom + 16


def blend(color, page, alpha):
    """`color` at `alpha` over GitHub's page colour, as a solid fill the
    gridlines cannot show through."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(page[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * alpha + y * (1 - alpha)):02x}" for x, y in zip(a, b))


def rounded_end(x, y, width, fill):
    """A bar square at its start with a 4 px rounded data end."""
    r = min(4.0, width / 2)
    if width <= 0:
        return ""
    path = (f"M{x:.1f},{y} h{width - r:.1f} a{r},{r} 0 0 1 {r},{r} "
            f"v{BAR - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{width - r:.1f} z")
    return f'<path d="{path}" fill="{fill}"/>'


def chart(theme):
    t = THEMES[theme]
    left, left_bottom = panel(
        0, 470, "Generation speed", "tokens per second, higher is better",
        GENERATION, 100, [0, 25, 50, 75, 100], "", theme)
    right, right_bottom = panel(
        490, 390, "Time to first token", "seconds, lower is better",
        FIRST_TOKEN, 120, [0, 30, 60, 90, 120], " s", theme)
    height = max(left_bottom, right_bottom) + 34
    hatches = "".join(
        f'<pattern id="hatch-{i}" width="6" height="6" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="6" height="6" fill="{blend(c, t["page"], 0.25)}"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{c}" stroke-width="3"/></pattern>'
        for i, c in enumerate(t["series"]))
    note = ("Same M5 MacBook, 16 GB, macOS 27, measured 2026-10-20. Ranges span the benchmark prompts. "
            "llama.cpp’s first token is estimated from its prefill rate.")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="t d">
<title id="t">ANE Fieldfare against Turbo Fieldfare and llama.cpp</title>
<desc id="d">Greedy generation on a math prompt: ANE Fieldfare 96.6 tokens per second, Turbo Fieldfare 24 to 25, llama.cpp 5 to 9. Temperature 0.2: ANE Fieldfare 36 to 53, Turbo Fieldfare 21 to 25. First token on a 3,015-token prompt: ANE Fieldfare 10 seconds, Turbo Fieldfare 48, llama.cpp about 110 (estimated).</desc>
<defs>{hatches}</defs>
<style>
text {{ font-family: {FONT}; fill: {t["text"]}; }}
.title {{ font-size: 16px; font-weight: 600; }}
.group {{ font-size: 13px; font-weight: 600; }}
.name, .value {{ font-size: 13px; }}
.strong {{ font-weight: 600; }}
.muted, .tick, .note {{ fill: {t["muted"]}; }}
.muted {{ font-size: 13px; }}
.tick {{ font-size: 11px; }}
.note {{ font-size: 11px; }}
.grid {{ stroke: {t["grid"]}; stroke-width: 1; }}
</style>
{chr(10).join(left)}
{chr(10).join(right)}
<text x="0" y="{height - 8}" class="note">{note}</text>
</svg>
"""


def main():
    out = Path(__file__).resolve().parents[2] / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"speed-{theme}.svg").write_text(chart(theme), encoding="utf-8")
        print(out / f"speed-{theme}.svg")


if __name__ == "__main__":
    main()
