"""Verify stage: deterministic checks that an answer only cites what was retrieved."""

import re

from nhl_rules.answer import GeneratedAnswer

# Typographic variants the model may normalize when copying text. Mapping them is not a
# paraphrase allowance: the words must still match exactly.
_CHARS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_CHARS)).strip()


def verify(answer: GeneratedAnswer, retrieved: dict[str, str]) -> list[str]:
    """Return a list of problems (empty if the answer passes).

    `retrieved` maps each retrieved rule number to its text. Every cited rule must be in
    it, and every excerpt must be a verbatim substring of that rule's text (ignoring
    whitespace and quote/dash styles).
    """
    errors = []
    texts = {n: normalize(t) for n, t in retrieved.items()}
    for c in answer.citations():
        if c.number not in texts:
            errors.append(f"Rule {c.number} was cited but is not one of the provided rules.")
            continue
        excerpt = normalize(c.excerpt).strip(" .…\"'")
        if not excerpt:
            errors.append(f"Rule {c.number}: the excerpt is empty.")
        elif excerpt not in texts[c.number]:
            errors.append(
                f'Rule {c.number}: the excerpt "{c.excerpt}" is not a word-for-word quote '
                "of that rule's text."
            )
    return errors
