CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(768),
    query_text text,
    match_count int DEFAULT 5,
    alpha double precision DEFAULT 0.7,
    beta double precision DEFAULT 0.3
) RETURNS TABLE (
    id bigint,
    tema_normalized text,
    subtema_normalized text,
    raw_text text,
    similarity double precision,
    fts_score double precision,
    hybrid_score double precision
) LANGUAGE plpgsql STABLE AS $func$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            q.id,
            q.tema_normalized,
            q.subtema_normalized,
            q.raw_text,
            1 - (q.embedding <=> query_embedding) AS similarity
        FROM
            questions q
        WHERE
            q.embedding IS NOT NULL
        ORDER BY
            q.embedding <=> query_embedding
        LIMIT
            match_count * 3
    ), fts_results AS (
        SELECT
            q.id,
            q.tema_normalized,
            q.subtema_normalized,
            q.raw_text,
            ts_rank_cd(q.fts, plainto_tsquery('portuguese', query_text)) AS fts_score
        FROM
            questions q
        WHERE
            q.fts @@ plainto_tsquery('portuguese', query_text)
        LIMIT
            match_count * 3
    ), combined AS (
        SELECT
            COALESCE(v.id, f.id) AS id,
            COALESCE(v.tema_normalized, f.tema_normalized) AS tema_normalized,
            COALESCE(v.subtema_normalized, f.subtema_normalized) AS subtema_normalized,
            COALESCE(v.raw_text, f.raw_text) AS raw_text,
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
        combined.tema_normalized,
        combined.subtema_normalized,
        combined.raw_text,
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