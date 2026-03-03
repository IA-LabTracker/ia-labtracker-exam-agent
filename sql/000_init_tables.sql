CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS questions (
    id BIGSERIAL PRIMARY KEY,
    institution TEXT NOT NULL DEFAULT 'unknown',
    year INTEGER,
    raw_text TEXT NOT NULL,
    tema_normalized TEXT,
    subtema_normalized TEXT,
    embedding vector(384),
    source_file TEXT,
    content_hash TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_questions_embedding ON questions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

ALTER TABLE
    questions
ADD
    COLUMN IF NOT EXISTS fts tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'portuguese',
            coalesce(tema_normalized, '') || ' ' || coalesce(subtema_normalized, '') || ' ' || coalesce(raw_text, '')
        )
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_questions_fts ON questions USING gin(fts);

CREATE TABLE IF NOT EXISTS themes (
    id BIGSERIAL PRIMARY KEY,
    tema TEXT NOT NULL,
    subtema TEXT,
    aliases TEXT [] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(tema, subtema)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    row_count INTEGER DEFAULT 0,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);