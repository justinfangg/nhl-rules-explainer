"""Write parsed rulebook chunks to Postgres."""

from pgvector.psycopg import register_vector

from nhl_rules.db import apply_schema, connect
from nhl_rules.embeddings import embed_passages
from nhl_rules.ingest.parser import Chunk


def embedding_input(chunk: Chunk, titles: dict[str, str]) -> str:
    """Text the embedding model sees: rule number and titles give short chunks context."""
    if chunk.parent_number:
        heading = f"Rule {chunk.rule_number}: {titles[chunk.parent_number]} – {chunk.title}"
    elif chunk.rule_number.startswith("Glossary"):
        heading = f"Definition: {chunk.title}"
    else:
        heading = f"Rule {chunk.rule_number}: {chunk.title}"
    return f"{heading}\n{chunk.text}".strip()


def load(chunks: list[Chunk], season: str, model_name: str) -> int:
    """Replace all rows for `season` with `chunks`, embedded with `model_name`."""
    titles = {c.rule_number: c.title for c in chunks if c.parent_number is None}
    vectors = embed_passages([embedding_input(c, titles) for c in chunks], model_name)

    apply_schema()
    with connect() as conn:
        register_vector(conn)
        with conn.transaction():
            conn.execute("DELETE FROM rules WHERE season = %s", (season,))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO rules (season, rule_number, parent_number, title, section,
                                       text, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            season,
                            c.rule_number,
                            c.parent_number,
                            c.title,
                            c.section,
                            c.text,
                            vec,
                            model_name,
                        )
                        for c, vec in zip(chunks, vectors, strict=True)
                    ],
                )
    return len(chunks)
