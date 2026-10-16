from pathlib import Path

import pytest

from nhl_rules.ingest.parser import (
    Line,
    Span,
    join_spans,
    parse_glossary,
    parse_pdf,
    parse_rules,
)

BODY = 10.0  # regular text size used in these synthetic pages


def span(text, x0, bold=False, size=BODY, width=None):
    return Span(text, bold, size, x0, x0 + (width if width is not None else 4 * len(text)))


def line(y, *spans, page=1):
    return Line(page, spans[0].x0, y, list(spans))


def section(y, text):
    return line(y, span(text, 40, bold=True, size=11.8))


def rule(y, text):
    return line(y, span(text, 40, bold=True, size=10.7))


def sub(y, number, title, rest):
    n = span(number, 40)
    t = span(title, 60, bold=True)
    return line(y, n, t, span(rest, t.x1))


def body(y, text, x=60):
    return line(y, span(text, x))


def by_number(chunks):
    return {c.rule_number: c for c in chunks}


def test_join_spans_adds_space_only_at_visible_gaps():
    assert join_spans([span("(see", 0, width=16), span("Rule 19", 20)]) == "(see Rule 19"
    assert join_spans([span("abc", 0, width=12), span("def", 12)]) == "abcdef"


def test_parse_rules_structure_and_paragraphs():
    chunks = by_number(
        parse_rules(
            [
                section(10, "SECTION 7 – RESTRAINING INFRACTIONS"),
                rule(20, "Rule 55 – Hooking"),
                sub(30, "55.1", "Hooking", " - Hooking is the act of using the stick"),
                body(40, "to restrain an opponent."),
                body(50, "When a player is checking another,", x=72),  # indented: new paragraph
                body(60, "it is not hooking."),
                sub(70, "55.2", "Minor Penalty", " – A minor penalty shall be imposed:"),
                body(80, "(i) on a player who hooks;"),
                body(90, "(ii) on a player who"),
                body(100, "impedes an opponent."),
            ]
        )
    )
    assert chunks["55"].title == "Hooking"
    assert chunks["55"].parent_number is None
    assert chunks["55.1"].parent_number == "55"
    assert chunks["55.1"].section == "Restraining Infractions"
    assert chunks["55.1"].text == (
        "Hooking is the act of using the stick to restrain an opponent.\n"
        "When a player is checking another, it is not hooking."
    )
    assert chunks["55.2"].title == "Minor Penalty"
    assert chunks["55.2"].text.split("\n") == [
        "A minor penalty shall be imposed:",
        "(i) on a player who hooks;",
        "(ii) on a player who impedes an opponent.",
    ]


def test_subsection_titles_that_wrap_or_hold_the_separator():
    chunks = by_number(
        parse_rules(
            [
                rule(10, "Rule 38 – Coach’s Challenge"),
                line(
                    20, span("38.10", 40), span("Applicable Standards for “Missed", 60, bold=True)
                ),
                line(
                    30,
                    span("Offensive Zone” Challenge", 60, bold=True),
                    span(" – The standard", 160),
                ),
                body(40, "for overturning."),
                line(
                    50,
                    span("38.11", 40),
                    span("Procedure –", 60, bold=True),
                    span(" As soon as", 110),
                ),
            ]
        )
    )
    assert chunks["38.10"].title == "Applicable Standards for “Missed Offensive Zone” Challenge"
    assert chunks["38.10"].text == "The standard for overturning."
    assert (chunks["38.11"].title, chunks["38.11"].text) == ("Procedure", "As soon as")


def test_cross_reference_note_and_repeated_rule_are_skipped():
    chunks = by_number(
        parse_rules(
            [
                rule(10, "Rule 60 – High-sticking"),
                sub(20, "60.7", "Fines", " - Supplementary discipline may apply."),
                rule(30, "Hooking"),  # a note header, not a rule
                body(40, "Hooking is placed in Section 7 as Rule 55."),
                rule(50, "Rule 61 – Slashing"),
                sub(60, "61.1", "Slashing", " - Slashing is swinging the stick."),
            ]
        )
    )
    assert chunks["60.7"].text == "Supplementary discipline may apply."
    assert set(chunks) == {"60", "60.7", "61", "61.1"}


def test_bold_cross_reference_at_line_start_is_not_a_subsection():
    chunks = by_number(
        parse_rules(
            [
                rule(10, "Rule 16 – Minor Penalties"),
                sub(20, "16.2", "Short-handed", " - Means below strength"),
                line(30, span("17.2", 40), span("applies too.", 60)),  # not bold: not a header
            ]
        )
    )
    assert set(chunks) == {"16", "16.2"}
    assert chunks["16.2"].text == "Means below strength 17.2 applies too."


def test_right_margin_rule_references_move_to_paragraph_end():
    chunks = by_number(
        parse_rules(
            [
                rule(10, "Rule 32 – Linespersons"),
                sub(20, "32.5", "Stoppages", " - The Linesperson shall stop play:"),
                line(30, span("(vi) When the puck is struck above the", 60), span("Rule 80", 234)),
                body(40, "normal height of the shoulders"),
            ]
        )
    )
    assert chunks["32.5"].text.split("\n")[1] == (
        "(vi) When the puck is struck above the normal height of the shoulders (Rule 80)"
    )


def test_side_by_side_columns_are_read_left_then_right():
    chunks = by_number(
        parse_rules(
            [
                rule(10, "Rule 73 – Refusing to Start Play"),
                sub(20, "73.2", "Procedure", " - The Referee shall warn the Captain."),
                line(30, span("First Violation", 80), span("Second Violation", 210)),
                line(40, span("(i) Warn the", 70), span("(i) Warn the", 185)),
                line(50, span("Captain.", 85), span("Captain.", 200)),
                line(60, span("(ii) Assess a game misconduct.", 185)),
            ]
        )
    )
    assert chunks["73.2"].text.split("\n") == [
        "The Referee shall warn the Captain.",
        "First Violation",
        "(i) Warn the Captain.",
        "Second Violation",
        "(i) Warn the Captain.",
        "(ii) Assess a game misconduct.",
    ]


def test_signals_table():
    chunks = by_number(
        parse_rules(
            [
                rule(10, "Rule 29 – Signals"),
                body(40, "Striking the clenched fist", x=126),
                line(50, span("29.1", 49), span("Boarding", 81), span("into the palm.", 126)),
                body(100, "No signal in the National", x=126),
                line(110, span("29.2", 49), span("Kicking", 81), span("NO SIGNAL", 227)),
                body(120, "Hockey League.", x=126),
                rule(150, "Rule 30 – Appointment of Officials"),
            ]
        )
    )
    assert (chunks["29.1"].title, chunks["29.1"].text) == (
        "Boarding",
        "Striking the clenched fist into the palm.",
    )
    assert chunks["29.2"].text == "No signal in the National Hockey League."
    assert chunks["29.2"].parent_number == "29"
    assert "30" in chunks


def test_glossary_table():
    lines = [
        line(55, span("TERM", 61, bold=True), span("DESCRIPTION", 137, bold=True)),
        line(
            69,
            span("Attacking Zone", 45),
            span("Between the opponent's", 101),
            span("Rule 72", 258),
        ),
        line(78, span("blue line and the end boards.", 101), span("Rule 1", 227)),
        line(100, span("Change of", 45), span("Rebounds off the pads", 101)),
        line(108, span("Control", 45), span("are not a loss of control.", 101)),
    ]
    chunks = parse_glossary(lines)
    assert [c.title for c in chunks] == ["Attacking Zone", "Change of Control"]
    assert chunks[0].rule_number == "Glossary: Attacking Zone"
    assert chunks[0].text == (
        "Between the opponent's blue line and the end boards.\nRule references: Rule 1, Rule 72"
    )
    assert chunks[1].text == "Rebounds off the pads are not a loss of control."


# --- against the real rulebook (not committed; skipped when absent) ---------

PDF = Path(__file__).parent.parent / "2026-27Rules.pdf"


@pytest.fixture(scope="module")
def rulebook():
    if not PDF.exists():
        pytest.skip(f"{PDF.name} not present")
    return by_number(parse_pdf(PDF))


def test_rulebook_has_every_rule_with_contiguous_subsections(rulebook):
    top = [
        c for c in rulebook.values() if c.parent_number is None and c.section != "Glossary of Terms"
    ]
    assert sorted(int(c.rule_number) for c in top) == list(range(1, 88))
    for c in top:
        subs = sorted(
            int(k.split(".")[1]) for k, v in rulebook.items() if v.parent_number == c.rule_number
        )
        assert subs == list(range(1, len(subs) + 1)), c.rule_number
    assert all(c.text for k, c in rulebook.items() if c.parent_number)


@pytest.mark.parametrize(
    ("number", "title", "section", "starts"),
    [
        ("1.1", "Rink", "Playing Area", "National Hockey League games shall be played"),
        ("29.25", "Misconduct", "Types of Penalties", "Both hands on hips."),
        ("38.1", "General", "Officials", "The video review mechanism triggered"),
        ("55.1", "Hooking", "Restraining Infractions", "Hooking is the act of using the stick"),
        ("74.1", "Too Many Men on the Ice", "Other Infractions", "Players may be changed at any"),
        ("83.1", "Off-side", "Game Flow", "Players of the attacking team must not precede"),
    ],
)
def test_rulebook_known_rules(rulebook, number, title, section, starts):
    c = rulebook[number]
    assert (c.title, c.section) == (title, section)
    assert c.text.startswith(starts)


def test_rulebook_layout_edge_cases(rulebook):
    # The Section 8 "Hooking" note is not glued onto 60.7.
    assert rulebook["60.7"].text.endswith("(refer to Rule 28).")
    # 73.2's two columns are not interleaved.
    assert "15 seconds (ii)" not in rulebook["73.2"].text
    assert rulebook["73.2"].text.index("First Violation") < rulebook["73.2"].text.index(
        "Second Violation"
    )
    # 56.1's definition list keeps one paragraph per label.
    assert "\nPick: A “pick” is" in rulebook["56.1"].text
    assert len([c for c in rulebook.values() if c.section == "Glossary of Terms"]) == 14
