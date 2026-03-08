ALTER TABLE theme_stats
ADD COLUMN IF NOT EXISTS cor_hex TEXT NOT NULL DEFAULT '#22C55E';
UPDATE theme_stats
SET cor_hex = CASE
    WHEN num_questions >= 6 THEN '#EF4444'
    WHEN num_questions >= 4 THEN '#F97316'
    WHEN num_questions >= 2 THEN '#EAB308'
    WHEN num_questions >= 1 THEN '#22C55E'
    ELSE '#3B82F6'
END;
