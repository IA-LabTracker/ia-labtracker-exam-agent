-- Add embedding column to theme_stats for semantic search
ALTER TABLE theme_stats
ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Create vector index for cosine similarity search on theme_stats
CREATE INDEX IF NOT EXISTS idx_theme_stats_embedding
    ON theme_stats USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Function: semantic search on theme_stats
CREATE OR REPLACE FUNCTION semantic_search_theme_stats(
    query_embedding vector(768),
    query_text text,
    match_count int DEFAULT 5,
    alpha double precision DEFAULT 0.7,
    beta double precision DEFAULT 0.3
) RETURNS TABLE (
    id bigint,
    institution text,
    ranking integer,
    category text,
    tema text,
    subtema text,
    percentage double precision,
    num_questions integer,
    cor text,
    cor_hex text,
    similarity double precision,
    fts_score double precision,
    hybrid_score double precision
) LANGUAGE plpgsql STABLE AS $func$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            ts.id,
            ts.institution,
            ts.ranking,
            ts.category,
            ts.tema,
            ts.subtema,
            ts.percentage::double precision,
            ts.num_questions,
            ts.cor,
            ts.cor_hex,
            1 - (ts.embedding <=> query_embedding) AS similarity
        FROM
            theme_stats ts
        WHERE
            ts.embedding IS NOT NULL
        ORDER BY
            ts.embedding <=> query_embedding
        LIMIT
            match_count * 3
    ), fts_results AS (
        SELECT
            ts.id,
            ts.institution,
            ts.ranking,
            ts.category,
            ts.tema,
            ts.subtema,
            ts.percentage::double precision,
            ts.num_questions,
            ts.cor,
            ts.cor_hex,
            ts_rank_cd(ts.fts, plainto_tsquery('portuguese', query_text)) AS fts_score
        FROM
            theme_stats ts
        WHERE
            ts.fts @@ plainto_tsquery('portuguese', query_text)
        LIMIT
            match_count * 3
    ), combined AS (
        SELECT
            COALESCE(v.id, f.id) AS id,
            COALESCE(v.institution, f.institution) AS institution,
            COALESCE(v.ranking, f.ranking) AS ranking,
            COALESCE(v.category, f.category) AS category,
            COALESCE(v.tema, f.tema) AS tema,
            COALESCE(v.subtema, f.subtema) AS subtema,
            COALESCE(v.percentage, f.percentage) AS percentage,
            COALESCE(v.num_questions, f.num_questions) AS num_questions,
            COALESCE(v.cor, f.cor) AS cor,
            COALESCE(v.cor_hex, f.cor_hex) AS cor_hex,
            COALESCE(v.similarity, 0.0)::double precision AS similarity,
            COALESCE(f.fts_score, 0.0)::double precision AS fts_score,
            (alpha * COALESCE(v.similarity, 0.0)::double precision + beta * COALESCE(f.fts_score, 0.0)::double precision)::double precision AS hybrid_score
        FROM
            vector_results v
        FULL OUTER JOIN
            fts_results f ON v.id = f.id
    )
    SELECT
        combined.id,
        combined.institution,
        combined.ranking,
        combined.category,
        combined.tema,
        combined.subtema,
        combined.percentage,
        combined.num_questions,
        combined.cor,
        combined.cor_hex,
        combined.similarity,
        combined.fts_score,
        combined.hybrid_score
    FROM
        combined
    ORDER BY
        combined.hybrid_score DESC
    LIMIT
        match_count;
END;
$func$;
