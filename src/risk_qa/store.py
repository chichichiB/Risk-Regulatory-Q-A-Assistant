"""Transactional immutable corpus storage and exact pgvector search."""

import hashlib
import json

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from risk_qa.contracts import CorpusManifest, EvidencePassage, PageRecord, PassageRecord
from risk_qa.corpus import validate_passage_offsets
from risk_qa.embeddings import validate_vector


class CorpusStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5)

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("""CREATE TABLE IF NOT EXISTS rq_releases (
                release_id text PRIMARY KEY, manifest jsonb NOT NULL,
                pages jsonb NOT NULL, signature text NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS rq_passages (
                release_id text REFERENCES rq_releases(release_id), passage_id text,
                authority text NOT NULL, data jsonb NOT NULL, embedding vector(384) NOT NULL,
                PRIMARY KEY(release_id, passage_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS rq_active (
                singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
                release_id text NOT NULL REFERENCES rq_releases(release_id))""")

    def ingest_release(self, manifest: CorpusManifest, passages: list[PassageRecord],
                       vectors: list[list[float]], pages: list[PageRecord] | None = None) -> str:
        if not passages or len(passages) != len(vectors):
            raise ValueError("Every passage must have exactly one vector")
        vectors = [validate_vector(v) for v in vectors]
        source_versions = {(s.doc_id, s.version_id, s.authority, s.pdf_url) for s in manifest.sources}
        if any((p.doc_id, p.version_id, p.authority, p.pdf_url) not in source_versions for p in passages):
            raise ValueError("Passage provenance is absent from the source manifest")
        if len({p.passage_id for p in passages}) != len(passages):
            raise ValueError("Duplicate passage IDs")
        if pages is not None:
            by_page = {(p.version_id, p.pdf_page_index): p for p in pages}
            for p in passages:
                validate_passage_offsets(p, by_page[(p.version_id, p.pdf_page_index)])
        payload = [p.model_dump(mode="json", include=set(PassageRecord.model_fields)) for p in passages]
        page_data = [p.model_dump(mode="json") for p in pages or []]
        sig = hashlib.sha256(json.dumps([manifest.model_dump(mode="json"), payload, vectors,
                                        page_data], sort_keys=True).encode()).hexdigest()
        release = manifest.release_id
        with self.connect() as conn:
            # Serialize publishers, so a conflict cannot overwrite immutable rows.
            conn.execute("SELECT pg_advisory_xact_lock(710621)")
            existing = conn.execute("SELECT signature FROM rq_releases WHERE release_id=%s",
                                    (release,)).fetchone()
            if existing and existing["signature"] != sig:
                raise ValueError("Release is immutable; input changed")
            if not existing:
                conn.execute("INSERT INTO rq_releases VALUES (%s,%s,%s,%s)",
                             (release, Jsonb(manifest.model_dump(mode="json")), Jsonb(page_data), sig))
                with conn.cursor() as cur:
                    cur.executemany("INSERT INTO rq_passages VALUES (%s,%s,%s,%s,%s::vector)",
                                    [(release, p.passage_id, p.authority, Jsonb(d), str(v))
                                     for p, d, v in zip(passages, payload, vectors, strict=True)])
            conn.execute("""INSERT INTO rq_active VALUES (true,%s)
                         ON CONFLICT(singleton) DO UPDATE SET release_id=excluded.release_id""",
                         (release,))
        return release

    def active_release_id(self) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT release_id FROM rq_active WHERE singleton").fetchone()
        if row is None:
            raise RuntimeError("No corpus has been ingested")
        return row["release_id"]

    def manifest(self, release_id: str) -> CorpusManifest:
        with self.connect() as conn:
            row = conn.execute("SELECT manifest FROM rq_releases WHERE release_id=%s",
                               (release_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown corpus release")
        return CorpusManifest.model_validate(row["manifest"])

    def all_passages(self, release_id: str) -> list[EvidencePassage]:
        with self.connect() as conn:
            rows = conn.execute("SELECT data FROM rq_passages WHERE release_id=%s ORDER BY passage_id",
                                (release_id,)).fetchall()
        if not rows:
            raise ValueError("Unknown or empty release")
        return [EvidencePassage(**r["data"], release_id=release_id) for r in rows]

    def search_dense(self, query_vector: list[float], release_id: str,
                     authority: str | None, limit: int) -> list[EvidencePassage]:
        vector = str(validate_vector(query_vector))
        with self.connect() as conn:
            rows = conn.execute("""SELECT data, 1-(embedding <=> %s::vector) AS score
                FROM rq_passages WHERE release_id=%s AND (%s::text IS NULL OR authority=%s)
                ORDER BY embedding <=> %s::vector, passage_id LIMIT %s""",
                (vector, release_id, authority, authority, vector, limit)).fetchall()
        return [EvidencePassage(**r["data"], release_id=release_id, score=r["score"]) for r in rows]
