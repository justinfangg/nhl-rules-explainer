# Eval set

`cases.json` is the hand-written eval set. It was written before retrieval was
built; change it deliberately (and note why in the commit), not to make a
retrieval or prompt change look better.

## Case format

```json
{
  "id": "hook-001",
  "question": "why did he get 2 minutes for pulling the guy back with his stick",
  "type": "situation",
  "style": "vague",
  "expected_rules": ["55"],
  "acceptable_rules": ["54", "56"],
  "notes": "Holding (54) if he used his hands."
}
```

- `type`: what the Understand stage should classify the question as:
  `situation`, `definition`, `judgment` or `out_of_scope`.
- `style`: `plain` (rulebook language), `slang` (fan language) or `vague`
  (several rules could apply; the answer should give alternatives). Used to
  slice metrics, since slang is where the glossary rewrite should help.
- `expected_rules`: the rule(s) the primary answer should cite. Any one counts.
- `acceptable_rules`: rules that are fine as alternatives, or that a reasonable
  answer might cite instead.
- Rule numbers are as loaded by `nhl ingest`: `55`, `55.2`, or
  `Glossary: Attacking Zone` for the rulebook's own glossary.
- `out_of_scope` cases list no rules.

## Matching

A retrieved or cited rule matches an expected one if it is the same rule, its
parent, or one of its subsections: `55` and `55.2` match each other; `55.1` and
`55.2` don't (`nhl_rules.evals.cases.rule_matches`).

## Checks

`uv run pytest tests/test_eval_cases.py` validates the file and checks that
every cited rule exists in the rulebook.
