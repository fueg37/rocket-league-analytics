#!/usr/bin/env python3
"""Fail when newly-added raw hex color literals appear in app.py chart call sites."""

from __future__ import annotations

import re
import subprocess

HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")
CHART_LINE_RE = re.compile(r"(marker_color\s*=|line\s*=\s*dict\(|fill_color\s*=|colorscale\s*=|color_discrete_sequence\s*=|gauge\s*=)")


def _added_app_lines() -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["git", "diff", "--unified=0", "--", "app.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = proc.stdout.splitlines()
    added: list[tuple[int, str]] = []
    lineno = 0

    for line in lines:
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                lineno = int(m.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append((lineno, line[1:]))
            lineno += 1
        elif not line.startswith("-"):
            lineno += 1

    return added


def main() -> int:
    violations: list[str] = []
    for lineno, content in _added_app_lines():
        if not CHART_LINE_RE.search(content):
            continue
        for hex_code in HEX_RE.findall(content):
            violations.append(f"app.py:{lineno}: {hex_code} -> use semantic_color()/factory helper")

    if violations:
        print("Found newly-added chart hex literals in app.py:")
        for item in violations:
            print(f"  - {item}")
        return 1

    print("OK: no newly-added app.py chart hex literals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
