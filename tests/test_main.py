from __future__ import annotations

import io
import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _make_excel(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


def test_reconcile_missing_file_returns_422(caplog):
    caplog.set_level(logging.ERROR)
    response = client.post("/reconcile", data={})
    assert response.status_code == 422
    assert "field required" in response.text.lower()


def test_reconcile_invalid_excel_returns_400(caplog, monkeypatch):
    df = pd.DataFrame({"foo": [1, 2, 3]})
    buf = _make_excel(df)
    files = {
        "file": (
            "bad.xlsx",
            buf,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    caplog.set_level(logging.INFO)
    resp = client.post("/reconcile", files=files)
    assert resp.status_code == 400
    assert "could not parse excel" in resp.json()["detail"].lower()
    assert "failed to read Excel input" in caplog.text


def test_reconcile_valid_file_success(caplog, monkeypatch):
    monkeypatch.setattr("src.main.reconcile_all", lambda rows, emb, db: [])

    df = pd.DataFrame({"tema": ["A", "B"]})
    buf = _make_excel(df)
    files = {
        "file": (
            "good.xlsx",
            buf,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    caplog.set_level(logging.INFO)
    resp = client.post("/reconcile", files=files)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "saved uploaded file" in caplog.text
