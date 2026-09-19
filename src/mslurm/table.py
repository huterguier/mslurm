"""Tiny dynamic-width table printer."""

from __future__ import annotations


def format_table(headers: list[str], rows: list[list[str]], max_width: int = 40) -> str:
    cells = [[_clip(str(c), max_width) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    for row in cells:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
    return "\n".join(lines)


def print_table(headers: list[str], rows: list[list[str]], max_width: int = 40) -> None:
    print(format_table(headers, rows, max_width))


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
