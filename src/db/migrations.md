Run SQL files in `sql/` folder against your Postgres instance. Example:

```bash
psql $DATABASE_URL -f sql/000_init_tables.sql
psql $DATABASE_URL -f sql/001_hybrid_search_function.sql
psql $DATABASE_URL -f sql/002_theme_stats.sql  # additional stats table + seed/ upsert data
```

When using Supabase, open Query Editor and paste contents.
