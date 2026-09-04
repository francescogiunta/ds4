#!/usr/bin/env python3
"""Generate a deterministic ~65,536-token GB10 fixture for the DSpark probe.

Same structure as long_context_story_prompt_gb10.txt (repeated canonical body,
one final fact-recall query, identical assignment facts), with the copy count
and prefix paragraphs parameterized so the fixture can be calibrated to the
65,536-token frontier.
"""

import sys
from pathlib import Path

from generate_gb10_long_context_fixture import (
    ASSISTANT,
    BOS,
    FACTS,
    USER,
    FINAL_MARKER,
    make_story,
)


def make_fixture(full_copies: int, prefix_paragraphs: int) -> str:
    story = make_story()
    body, marker, final_tail = story.partition(FINAL_MARKER)
    if not marker:
        raise RuntimeError("canonical story is missing the final-task marker")

    paragraphs = body.split("\n\n")
    if prefix_paragraphs > len(paragraphs):
        raise RuntimeError("prefix paragraph count exceeds canonical body")

    parts = []
    for index in range(full_copies):
        parts.append(
            "\n\n===== IDENTICAL CANONICAL LEDGER COPY "
            f"{index + 1} OF {full_copies} =====\n\n"
        )
        parts.append(body)

    if prefix_paragraphs:
        parts.append(
            "\n\n===== DETERMINISTIC PREFIX OF CANONICAL LEDGER COPY "
            f"{full_copies + 1} =====\n\n"
        )
        parts.append("\n\n".join(paragraphs[:prefix_paragraphs]))

    system = (
        "You are a careful assistant. Read the repeated ledger, remember the "
        "assignments, and answer the single final task exactly. Repeated copies "
        "contain identical assignments and never create a new value."
    )
    fixture = (
        BOS
        + system
        + USER
        + "".join(parts)
        + FINAL_MARKER
        + final_tail
        + ASSISTANT
        + "</think>"
    )
    if fixture.count(FINAL_MARKER) != 1:
        raise RuntimeError("extended fixture must contain exactly one final task")
    for name, word, _value in FACTS:
        assignment = f"{name} was assigned the number {word}."
        if fixture.count(assignment) < full_copies:
            raise RuntimeError(f"extended fixture is missing repeated fact: {name}")
    return fixture


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: generate_gb10_65k_fixture.py COPIES PARAGRAPHS OUT_FILE")
        sys.exit(1)
    full_copies = int(sys.argv[1])
    prefix_paragraphs = int(sys.argv[2])
    output = Path(sys.argv[3])
    output.write_text(make_fixture(full_copies, prefix_paragraphs), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
