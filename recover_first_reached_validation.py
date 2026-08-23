#!/usr/bin/env python3
"""Recover the first validation observed when each checkpoint was reached."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path


VALIDATION = re.compile(
    r"step:(?P<step>\d+).*? - "
    r"(?P<key>val-core/[^ ]+/(?:acc|reward)/mean@1):"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def recover(log_path: Path, expected_step: int, interval: int) -> tuple[list[dict], dict[int, int]]:
    values: dict[int, tuple[str, float]] = {}
    occurrences: dict[int, int] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = VALIDATION.search(line)
        if match is None:
            continue
        step = int(match.group("step"))
        item = (match.group("key"), float(match.group("value")))
        if not math.isfinite(item[1]):
            raise RuntimeError(f"non-finite validation at step {step}")
        occurrences[step] = occurrences.get(step, 0) + 1
        values.setdefault(step, item)

    expected = list(range(0, expected_step + 1, interval))
    if sorted(values) != expected:
        raise RuntimeError(f"validation steps do not match {expected}: {sorted(values)}")
    rows = [
        {"step": step, "data": {values[step][0]: values[step][1]}}
        for step in expected
    ]
    return rows, occurrences


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-step", type=int, default=150)
    parser.add_argument("--interval", type=int, default=25)
    args = parser.parse_args()
    rows, occurrences = recover(args.log, args.expected_step, args.interval)
    atomic_write(args.output, rows)
    print(json.dumps({
        "rows": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "occurrences": occurrences,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
