-- theme_stats_all: canonical table with pre-aggregated data across all institutions.
-- This is a standalone table (NOT a view of theme_stats) populated via its own INSERT file.
-- Queries in get_questions_by_institution() look here first for num_questions per institution.
CREATE TABLE IF NOT EXISTS theme_stats_all (
    id BIGSERIAL PRIMARY KEY,
    institution TEXT NOT NULL,
    ranking INTEGER,
    category TEXT NOT NULL DEFAULT '',
    tema TEXT NOT NULL,
    subtema TEXT,
    percentage FLOAT NOT NULL DEFAULT 0.0,
    num_questions INTEGER NOT NULL DEFAULT 0,
    cor TEXT NOT NULL DEFAULT 'verde',
    cor_hex TEXT NOT NULL DEFAULT '#22C55E',
    source_file TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_theme_stats_all_institution ON theme_stats_all(institution);
CREATE INDEX IF NOT EXISTS idx_theme_stats_all_tema ON theme_stats_all(lower(tema));
CREATE INDEX IF NOT EXISTS idx_theme_stats_all_subtema ON theme_stats_all(lower(subtema));
