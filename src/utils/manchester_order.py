"""Manchester ordering utility.

Loads Manchesters.xlsx to derive (semana, row_position) for each (tema, subtema) pair.
Used to sort the output spreadsheet following the Manchester curriculum order.
"""

from __future__ import annotations

from pathlib import Path
from unicodedata import normalize as unorm

_MANCHESTER_PATH = Path(__file__).parent.parent.parent / "Manchesters.xlsx"
_cached_index: dict[tuple[str, str], tuple[int, int]] | None = None

# Sentinel for rows not found in Manchester
SEMANA_UNKNOWN = 999
POS_UNKNOWN = 99_999


def _norm(text: str) -> str:
    """NFC-normalize, lowercase, strip — for key comparison."""
    return unorm("NFC", str(text or "")).lower().strip()


def get_manchester_index() -> dict[tuple[str, str], tuple[int, int]]:
    """Load and cache Manchester ordering.

    Returns dict: (tema_norm, subtema_norm) → (semana_int, row_position)
    Row position is the original row index (0-based from data start), used as
    a tiebreaker within the same semana.
    """
    global _cached_index
    if _cached_index is not None:
        return _cached_index

    try:
        import openpyxl

        wb = openpyxl.load_workbook(_MANCHESTER_PATH, read_only=True, data_only=True)
        ws = wb.active
        index: dict[tuple[str, str], tuple[int, int]] = {}
        for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True)):
            semana_raw = row[0]
            tema = str(row[3] or "")
            subtema = str(row[4] or "")
            if not semana_raw or not tema:
                continue
            try:
                semana_int = int(float(str(semana_raw).strip()))
            except (ValueError, TypeError):
                semana_int = SEMANA_UNKNOWN

            key = (_norm(tema), _norm(subtema))
            if key not in index:
                index[key] = (semana_int, row_idx)

            # Also index tema-only key (subtema="") as fallback
            key_tema_only = (_norm(tema), "")
            if key_tema_only not in index:
                index[key_tema_only] = (semana_int, row_idx)

        wb.close()
        _cached_index = index
    except Exception:
        _cached_index = {}

    return _cached_index


def lookup_semana(
    normalized_tema: str | None,
    normalized_subtema: str | None,
    input_tema_raw: str | None = None,
    input_subtema_raw: str | None = None,
) -> tuple[int, int]:
    """Return (semana_int, row_position) for a given tema/subtema pair.

    Tries multiple key variants for matching — input fields first (closer to
    Manchester source text), then normalized DB fields.
    Returns (SEMANA_UNKNOWN, POS_UNKNOWN) when no match is found.
    """
    index = get_manchester_index()

    candidates: list[tuple[str, str]] = []

    # Input fields match best since Manchester uses the same source vocabulary
    if input_tema_raw:
        it = _norm(input_tema_raw)
        ist = _norm(input_subtema_raw or "")
        candidates.append((it, ist))
        candidates.append((it, ""))

    # DB-normalized fields as fallback
    if normalized_tema:
        nt = _norm(normalized_tema)
        ns = _norm(normalized_subtema or "")
        candidates.append((nt, ns))
        candidates.append((nt, ""))

    for key in candidates:
        if key in index:
            return index[key]

    return (SEMANA_UNKNOWN, POS_UNKNOWN)


def format_semana(semana_int: int) -> str:
    """Format semana integer as zero-padded string, e.g. 1 → '01'."""
    if semana_int >= SEMANA_UNKNOWN:
        return ""
    return f"{semana_int:02d}"
