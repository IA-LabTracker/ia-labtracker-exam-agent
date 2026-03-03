from typing import Optional
import psycopg
from psycopg.rows import dict_row
from src.config import settings


class DBClient:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or settings.DATABASE_URL
        self.conn = psycopg.connect(self.dsn, row_factory=dict_row)

    def execute(self, sql: str, *args, **kwargs):
        with self.conn.cursor() as cur:
            cur.execute(sql, *args, **kwargs)
            return cur

    def fetch(self, sql: str, *args, **kwargs):
        with self.conn.cursor() as cur:
            cur.execute(sql, *args, **kwargs)
            return cur.fetchall()

    def close(self):
        self.conn.close()


# simple supabase wrapper if keys provided
from supabase import create_client


class SupabaseClient:
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("Supabase credentials not set")
        self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    def table(self, name: str):
        return self.client.table(name)

    def rpc(self, name: str, params: dict):
        return self.client.rpc(name, params)
