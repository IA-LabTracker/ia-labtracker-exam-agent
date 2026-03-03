-- migration: create theme_stats table and indexes (schema only)
CREATE TABLE IF NOT EXISTS theme_stats (
    id BIGSERIAL PRIMARY KEY,
    institution TEXT NOT NULL,
    ranking INTEGER,
    category TEXT NOT NULL,
    tema TEXT NOT NULL,
    subtema TEXT,
    percentage FLOAT NOT NULL DEFAULT 0.0,
    num_questions INTEGER NOT NULL DEFAULT 0,
    cor TEXT NOT NULL DEFAULT 'verde',
    cor_hex TEXT NOT NULL DEFAULT '#22C55E',
    source_file TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(institution, category, tema, subtema)
);

CREATE INDEX IF NOT EXISTS idx_theme_stats_institution ON theme_stats(institution);

CREATE INDEX IF NOT EXISTS idx_theme_stats_tema ON theme_stats(tema);

CREATE INDEX IF NOT EXISTS idx_theme_stats_cor ON theme_stats(cor);

ALTER TABLE
    theme_stats
ADD
    COLUMN IF NOT EXISTS fts tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'portuguese',
            coalesce(tema, '') || ' ' || coalesce(subtema, '')
        )
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_theme_stats_fts ON theme_stats USING gin(fts);