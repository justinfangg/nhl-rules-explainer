"""Parse the NHL rulebook PDF into rule/subsection chunks and glossary definitions.

The PDF is read span by span (not as plain text) because fonts carry the structure:

- section header:    bold ~11.8pt  "SECTION 6 – PHYSICAL INFRACTIONS"
- rule header:       bold ~10.7pt  "Rule 41 – Boarding"
- subsection header: at the left margin, "41.1" then a bold title, then " - " and the text

Rule 29 (Signals) is laid out as a table and gets its own parser.

Extraction is split from parsing so the parsing logic can be tested on synthetic lines.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# 1-based PDF page ranges for the 2026-27 rulebook.
RULES_PAGES = range(12, 148)
GLOSSARY_PAGES = range(222, 224)

# Running headers/footers sit outside this vertical band (page height is 612pt).
BODY_TOP, BODY_BOTTOM = 38, 570
NAV_TEXT = {"Previous Page", "Next Page", "Table of Contents"}

SECTION_SIZE, RULE_SIZE = 11.8, 10.7
RULE_HEADER = re.compile(r"^Rule (\d+)\s*[–-]\s*(.+)$")
SECTION_HEADER = re.compile(r"^SECTION (\d+)\s*[–-]\s*(.+)$", re.IGNORECASE)
SUBSECTION_NUMBER = re.compile(r"^(\d+)\.(\d+)$")
LIST_MARKER = re.compile(r"^(\((?:[ivx]+|[a-z]|\d+)\)|•)")
SENTENCE_END = re.compile(r"[.:;!?”\")]$")
LABEL = re.compile(r"^[A-Z][A-Za-z ]{1,30}:(\s|$)")  # "Body Position:" in 56.1
SPAN_GAP = 1.0  # points of horizontal gap between spans that reads as a space


@dataclass
class Span:
    text: str
    bold: bool
    size: float
    x0: float = 0.0
    x1: float = 0.0


@dataclass
class Line:
    page: int
    x0: float
    y0: float
    spans: list[Span]
    refs: list[str] = field(default_factory=list)  # right-margin "Rule N" references

    @property
    def text(self) -> str:
        return join_spans(self.spans)


@dataclass
class Chunk:
    rule_number: str
    parent_number: str | None
    title: str
    section: str
    paragraphs: list[str] = field(default_factory=list)
    # Right-margin rule references, keyed by paragraph index; appended to that paragraph.
    para_refs: dict[int, list[str]] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(
            f"{p} ({', '.join(self.para_refs[i])})" if i in self.para_refs else p
            for i, p in enumerate(self.paragraphs)
        )


def join_spans(spans: list[Span]) -> str:
    """Concatenate spans, adding a space where the PDF leaves a visible gap between two
    spans but no space character (e.g. "(see" + "Rule 19", or "(i)" + "when")."""
    out = ""
    prev: Span | None = None
    for s in spans:
        if (
            prev is not None
            and out
            and not out[-1].isspace()
            and not s.text[:1].isspace()
            and s.x0 - prev.x1 > SPAN_GAP
        ):
            out += " "
        out += s.text
        prev = s
    return out


# --- extraction -------------------------------------------------------------


def extract_lines(pdf_path: Path, pages: range) -> list[Line]:
    """Body lines from the given 1-based pages, in reading order.

    Spans whose baselines are within a few points are merged into one line, so a list
    marker like "(i)" set slightly lower than its text stays on the same line.
    """
    doc = pymupdf.open(pdf_path)
    lines: list[Line] = []
    for pn in pages:
        raw: list[tuple[float, float, Span]] = []
        for block in doc[pn - 1].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for s in line["spans"]:
                    if not s["text"].strip():
                        continue
                    x0, y0, x1 = s["bbox"][0], s["bbox"][1], s["bbox"][2]
                    if not BODY_TOP <= y0 <= BODY_BOTTOM:
                        continue
                    bold = "Bold" in s["font"]
                    raw.append((y0, x0, Span(s["text"], bold, round(s["size"], 1), x0, x1)))
        raw.sort(key=lambda r: (r[0], r[1]))

        page_lines: list[list[tuple[float, float, Span]]] = []
        for item in raw:
            if page_lines and abs(item[0] - page_lines[-1][0][0]) <= 4:
                page_lines[-1].append(item)
            else:
                page_lines.append([item])
        for group in page_lines:
            group.sort(key=lambda r: r[1])
            line = Line(pn, group[0][1], group[0][0], [g[2] for g in group])
            if line.text.strip() not in NAV_TEXT:
                lines.append(line)
    return lines


# --- rules ------------------------------------------------------------------


def _is_header(line: Line, size: float) -> bool:
    first = line.spans[0]
    return first.bold and abs(first.size - size) < 0.3


def _subsection_start(line: Line, rule: str | None) -> tuple[str, str, str] | None:
    """If the line opens a subsection of `rule`, return (number, title, rest of text)."""
    if rule is None or len(line.spans) < 2:
        return None
    m = SUBSECTION_NUMBER.match(line.spans[0].text.strip())
    if not m or m.group(1) != rule or line.spans[0].bold:
        return None
    title_spans: list[Span] = []
    i = 1
    while i < len(line.spans) and line.spans[i].bold:
        title_spans.append(line.spans[i])
        i += 1
    if not title_spans:
        return None
    rest = join_spans(line.spans[i:])
    return f"{m.group(1)}.{m.group(2)}", *_split_title(title_spans, rest)


def _split_title(title_spans: list[Span], rest: str) -> tuple[str, str]:
    """Clean a bold title and the text after it; the " - " separator may sit in either."""
    title = re.sub(r"\s*[–-]\s*$", "", join_spans(title_spans).strip())
    return title, re.sub(r"^\s*[–-]\s*", "", rest)


def _title_case(s: str) -> str:
    """'TYPES OF PENALTIES' -> 'Types of Penalties' (matches the table of contents)."""
    small = {"of", "the", "and", "on", "to", "a"}
    words = s.strip().lower().split()
    return " ".join(w if i and w in small else w.capitalize() for i, w in enumerate(words))


class _ParagraphBuilder:
    """Accumulates wrapped lines into paragraphs for the current chunk."""

    def add(self, chunk: Chunk, line: Line, text: str, body_x: float) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            self._add_text(chunk, line, text, body_x)
        if line.refs and chunk.paragraphs:
            chunk.para_refs.setdefault(len(chunk.paragraphs) - 1, []).extend(line.refs)

    def _add_text(self, chunk: Chunk, line: Line, text: str, body_x: float) -> None:
        prev_text = chunk.paragraphs[-1] if chunk.paragraphs else ""
        new_para = (
            not chunk.paragraphs
            or LIST_MARKER.match(text)
            or ((line.x0 > body_x + 6 or LABEL.match(text)) and SENTENCE_END.search(prev_text))
        )
        if new_para:
            chunk.paragraphs.append(text)
        elif prev_text.endswith("-") and not prev_text.endswith(" -"):
            chunk.paragraphs[-1] = prev_text + text
        else:
            chunk.paragraphs[-1] = prev_text + " " + text


def _body_x(lines: list[Line]) -> dict[int, float]:
    """Most common left edge per page: the body-text margin (pages alternate)."""
    counts: dict[int, dict[float, int]] = {}
    for ln in lines:
        c = counts.setdefault(ln.page, {})
        c[round(ln.x0)] = c.get(round(ln.x0), 0) + 1
    return {p: max(c, key=c.get) for p, c in counts.items()}


MARGIN_REF = re.compile(r"^Rule \d+$")
MARGIN_REF_X = 220  # right-margin references start here; body text never has "Rule N" alone
COLUMN_SPLIT_X = 150  # a span starting right of this, after a wide gap, is a second column
COLUMN_GAP = 15


def _column_split(line: Line) -> int | None:
    """Index of the first span in a right-hand column, if the line has one."""
    for i in range(1, len(line.spans)):
        a, b = line.spans[i - 1], line.spans[i]
        if b.x0 - a.x1 > COLUMN_GAP and b.x0 > COLUMN_SPLIT_X:
            return i
    return None


def _split_margin_refs(line: Line) -> Line:
    """Move right-aligned "Rule N" cross-references (used in some lists, e.g. 32.5)
    out of the text so they don't land mid-sentence."""
    spans, refs = [], []
    for s in line.spans:
        text = s.text.strip()
        if s.x0 > MARGIN_REF_X and MARGIN_REF.match(text):
            refs.append(text)
        else:
            spans.append(s)
    if not refs:
        return line
    return Line(line.page, spans[0].x0 if spans else line.x0, line.y0, spans, refs)


def _unweave_columns(lines: list[Line]) -> list[Line]:
    """Side-by-side columns (e.g. 73.2 "First Violation | Second Violation") get merged
    line by line during extraction. Re-emit such a region as the left column, then the
    right column. A region starts at the first line with a clear column gap; the divider
    is the middle of that gap, and the region runs while lines have right-column text."""
    out: list[Line] = []
    i = 0
    while i < len(lines):
        first = _column_split(lines[i])
        if first is None:
            out.append(lines[i])
            i += 1
            continue
        spans = lines[i].spans
        divider = (spans[first - 1].x1 + spans[first].x0) / 2
        left: list[Line] = []
        right: list[Line] = []
        while i < len(lines):
            line = lines[i]
            lhs = [s for s in line.spans if s.x0 < divider]
            rhs = [s for s in line.spans if s.x0 >= divider]
            if not rhs:
                break
            if lhs:
                left.append(Line(line.page, lhs[0].x0, line.y0, lhs))
            right.append(Line(line.page, rhs[0].x0, line.y0, rhs))
            i += 1
        out.extend(left + right)
    return out


def _prepare(lines: list[Line]) -> list[Line]:
    """Layout fixes before parsing. Rule 29's table is left alone (its own parser)."""
    out: list[Line] = []
    region: list[Line] = []
    rule: str | None = None
    for line in lines:
        if line.spans and _is_header(line, RULE_SIZE):
            out.extend(_unweave_columns(region) if rule != SIGNALS_RULE else region)
            region = []
            m = RULE_HEADER.match(line.text.strip())
            rule = m.group(1) if m else None
            out.append(line)
            continue
        region.append(line if rule == SIGNALS_RULE else _split_margin_refs(line))
    out.extend(_unweave_columns(region) if rule != SIGNALS_RULE else region)
    return out


def parse_rules(lines: list[Line]) -> list[Chunk]:
    """Turn body lines into one chunk per rule (header + intro text) and per subsection."""
    lines = _prepare(lines)
    body_x = _body_x(lines)
    chunks: list[Chunk] = []
    seen: set[str] = set()
    section = ""
    rule: str | None = None
    current: Chunk | None = None  # None means "skip lines until the next header"
    para = _ParagraphBuilder()
    signal_lines: list[Line] = []
    title_open = False  # the last subsection title may wrap onto the next line

    for line in lines:
        text = line.text.strip()
        if not line.spans:  # only margin references were on this line
            if current is not None:
                para.add(current, line, "", body_x[line.page])
            continue

        is_header = _is_header(line, RULE_SIZE) or _is_header(line, SECTION_SIZE)
        if rule == SIGNALS_RULE and not is_header:
            signal_lines.append(line)
            continue

        if _is_header(line, SECTION_SIZE) and (m := SECTION_HEADER.match(text)):
            section = _title_case(m.group(2))
            continue

        if _is_header(line, RULE_SIZE):
            m = RULE_HEADER.match(text)
            if m and m.group(1) not in seen:
                rule = m.group(1)
                seen.add(rule)
                current = Chunk(rule, None, m.group(2).strip(), section)
                chunks.append(current)
            else:
                # A non-rule header (e.g. the "Hooking" cross-reference note in
                # Section 8) or a repeated rule: skip until the next rule.
                rule, current = None, None
            continue

        if title_open and current is not None and line.spans[0].bold:
            # e.g. 38.10 "Applicable Standards for “Missed Game Stoppage Event in the"
            #            "Offensive Zone” Challenge – The standard for ..."
            n = next((i for i, sp in enumerate(line.spans) if not sp.bold), len(line.spans))
            more, rest = _split_title(line.spans[:n], join_spans(line.spans[n:]))
            current.title = f"{current.title} {more}"
            title_open = not rest.strip()
            para.add(current, line, rest, body_x[line.page])
            continue
        title_open = False

        if sub := _subsection_start(line, rule):
            number, title, rest = sub
            if number in seen:
                current = None
                continue
            seen.add(number)
            current = Chunk(number, rule, title, section)
            chunks.append(current)
            para.add(current, line, rest, body_x[line.page])
            title_open = not rest.strip()
            continue

        if current is not None:
            para.add(current, line, line.text, body_x[line.page])

    chunks.extend(_parse_signals(signal_lines, signals_section(chunks)))
    return chunks


# --- signals table (Rule 29) ------------------------------------------------

SIGNALS_RULE = "29"
SIGNAL_NAME_X, SIGNAL_DESC_X, SIGNAL_IMAGE_X = 70, 115, 215
SIGNAL_ROW_GAP = 15
# Rows where the PDF puts the signal name and its description in a single span.
SIGNAL_FIXUPS = {"Misconduct Both hands on hips.": ("Misconduct", "Both hands on hips.")}
# Names hyphenated by line wrapping, or split across two cells.
SIGNAL_NAME_FIXUPS = {
    "Unsports-manlike conduct": "Unsportsmanlike conduct",
    "Icing (a) Icing (b)": "Icing",
}


def signals_section(chunks: list[Chunk]) -> str:
    return next((c.section for c in chunks if c.rule_number == SIGNALS_RULE), "")


def _parse_signals(lines: list[Line], section: str) -> list[Chunk]:
    """Rule 29 is a table: number | signal name | description | picture. A row's
    cells are vertically centred, so rows are found by vertical gaps between lines."""
    rows: list[dict[str, list[str]]] = []
    prev: Line | None = None
    for line in lines:
        if prev is None or line.page != prev.page or line.y0 - prev.y0 > SIGNAL_ROW_GAP:
            rows.append({"number": [], "name": [], "desc": []})
        prev = line
        for span in line.spans:
            text = span.text.strip()
            if span.x0 < SIGNAL_NAME_X:
                col = "number"
            elif span.x0 < SIGNAL_DESC_X:
                col = "name"
            elif span.x0 < SIGNAL_IMAGE_X:
                col = "desc"
            else:
                continue  # "NO SIGNAL" placeholder in the picture column
            rows[-1][col].append(text)

    chunks: list[Chunk] = []
    for row in rows:
        number = " ".join(row["number"])
        name = _join_wrapped(row["name"])
        desc = _join_wrapped(row["desc"])
        if name in SIGNAL_FIXUPS and not desc:
            name, desc = SIGNAL_FIXUPS[name]
        name = SIGNAL_NAME_FIXUPS.get(name, name)
        if SUBSECTION_NUMBER.match(number):
            chunks.append(Chunk(number, SIGNALS_RULE, name, section, [desc]))
        elif chunks and (name or desc):
            # A row without a number continues the previous signal (e.g. across a page).
            chunks[-1].paragraphs.append(_join_wrapped([name, desc]))
    return chunks


def _join_wrapped(parts: list[str]) -> str:
    out = ""
    for part in filter(None, parts):
        if out.endswith("-"):
            out += part
        else:
            out += (" " if out else "") + part
    return out


# --- glossary ---------------------------------------------------------------

GLOSSARY_DESC_X, GLOSSARY_REFS_X = 95, 220


def parse_glossary(lines: list[Line]) -> list[Chunk]:
    """Glossary of Terms: a three-column table (term, description, rule references)."""
    entries: list[dict] = []
    last_term_y: tuple[int, float] | None = None

    for line in lines:
        text = line.text.strip()
        if text in {"TERM", "DESCRIPTION", "RULE", "REFERENCES"} or line.spans[0].bold:
            continue
        for col, part in _split_columns(line):
            if col == "term":
                continuation = (
                    last_term_y is not None
                    and last_term_y[0] == line.page
                    and line.y0 - last_term_y[1] < 12
                )
                if continuation:
                    entries[-1]["term"].append(part)
                else:
                    entries.append({"term": [part], "desc": [], "refs": []})
                last_term_y = (line.page, line.y0)
            elif entries:
                entries[-1][col].append(part)

    chunks = []
    for e in entries:
        term = " ".join(e["term"])
        desc = " ".join(e["desc"])
        desc = re.sub(r"(\w)- (\w)", r"\1-\2", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        # References sit in two side-by-side columns, so sort them back into order.
        refs = sorted({int(n) for n in re.findall(r"Rule (\d+)", " ".join(e["refs"]))})
        paragraphs = [desc]
        if refs:
            paragraphs.append("Rule references: " + ", ".join(f"Rule {n}" for n in refs))
        chunks.append(Chunk(f"Glossary: {term}", None, term, "Glossary of Terms", paragraphs))
    return chunks


def _split_columns(line: Line) -> list[tuple[str, str]]:
    """extract_lines merges spans that share a baseline across columns; split them back."""
    parts: dict[str, list[str]] = {}
    for s in line.spans:
        if s.x0 < GLOSSARY_DESC_X:
            col = "term"
        elif s.x0 < GLOSSARY_REFS_X:
            col = "desc"
        else:
            col = "refs"
        parts.setdefault(col, []).append(s.text.strip())
    return [(c, " ".join(t)) for c, t in parts.items()]


def parse_pdf(pdf_path: Path) -> list[Chunk]:
    rules = parse_rules(extract_lines(pdf_path, RULES_PAGES))
    glossary = parse_glossary(extract_lines(pdf_path, GLOSSARY_PAGES))
    return rules + glossary
