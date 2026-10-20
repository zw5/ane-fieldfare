#!/usr/bin/env python3
"""Reject Swift escape hatches that silence the compiler instead of satisfying it.

Concurrency hatches (`@unchecked Sendable`, `nonisolated(unsafe)`,
`@preconcurrency`, `assumeIsolated`, `DispatchQueue`, `Task.detached`) are
allowed only with a `// SAFETY:` comment stating the invariant the compiler
cannot check. The comment goes in the comment block directly above the line,
or directly below a declaration line that opens a body.

Crash hatches (`try!`, `as!`, force unwraps) are rejected in Sources/ and
allowed in Tests/, where a crash is a failed test.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONCURRENCY_HATCHES = [
    (re.compile(r"@unchecked\s+Sendable"), "@unchecked Sendable"),
    (re.compile(r"nonisolated\(unsafe\)"), "nonisolated(unsafe)"),
    (re.compile(r"@preconcurrency"), "@preconcurrency"),
    (re.compile(r"\bassumeIsolated\b"), "assumeIsolated"),
    (re.compile(r"\bDispatchQueue\b"), "DispatchQueue"),
    (re.compile(r"\bTask\.detached\b"), "Task.detached"),
]
CRASH_HATCHES = [
    (re.compile(r"\btry!"), "try!"),
    (re.compile(r"\bas!"), "as!"),
    # A postfix `!` after an identifier, call or subscript, but not `!=`/`!==`
    # and not the `try!`/`as!` already reported above.
    (re.compile(r"(?<!\btr)(?<!\ba)[\w)\]]!(?!=)"), "force unwrap"),
]
SAFETY = "SAFETY:"


def strip_code(line: str, in_multiline: bool) -> tuple[str, bool]:
    """Blank out string literals and trailing comments so only code is matched."""
    out = []
    i = 0
    while i < len(line):
        if in_multiline:
            end = line.find('"""', i)
            if end < 0:
                return "".join(out), True
            i = end + 3
            in_multiline = False
            continue
        if line.startswith('"""', i):
            in_multiline = True
            i += 3
            continue
        if line.startswith("//", i):
            break
        if line[i] == '"':
            j = i + 1
            while j < len(line) and line[j] != '"':
                j += 2 if line[j] == "\\" else 1
            out.append('""')
            i = j + 1
            continue
        out.append(line[i])
        i += 1
    return "".join(out), in_multiline


def justified(lines: list[str], index: int) -> bool:
    j = index - 1
    while j >= 0 and lines[j].strip().startswith("//"):
        if SAFETY in lines[j]:
            return True
        j -= 1
    if lines[index].rstrip().endswith("{"):
        j = index + 1
        while j < len(lines) and lines[j].strip().startswith("//"):
            if SAFETY in lines[j]:
                return True
            j += 1
    return False


def lint(path: Path, check_crash_hatches: bool) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    problems = []
    in_multiline = False
    for index, raw in enumerate(lines):
        if raw.lstrip().startswith("//"):
            continue
        code, in_multiline = strip_code(raw, in_multiline)
        for pattern, name in CONCURRENCY_HATCHES:
            if pattern.search(code) and not justified(lines, index):
                rel = path.relative_to(ROOT)
                problems.append(f"{rel}:{index + 1}: {name} without a // {SAFETY} justification"
                                f"\n    {raw.strip()}")
        for pattern, name in CRASH_HATCHES if check_crash_hatches else []:
            if pattern.search(code):
                rel = path.relative_to(ROOT)
                problems.append(f"{rel}:{index + 1}: {name} in Sources/\n    {raw.strip()}")
    return problems


def main() -> int:
    problems = []
    for swift in sorted((ROOT / "Sources").rglob("*.swift")):
        problems += lint(swift, check_crash_hatches=True)
    for swift in sorted((ROOT / "Tests").rglob("*.swift")):
        problems += lint(swift, check_crash_hatches=False)
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} escape hatch problem(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
