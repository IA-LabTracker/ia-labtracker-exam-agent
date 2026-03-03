from typing import List, Dict
import pandas as pd


def write_excel(rows: List[Dict], path: str):
    # convert any list values to comma‑separated strings so Excel cells are readable
    norm_rows = []
    for r in rows:
        nr = {
            k: (",".join(map(str, v)) if isinstance(v, list) else v)
            for k, v in r.items()
        }
        norm_rows.append(nr)
    df = pd.DataFrame(norm_rows)
    df.to_excel(path, index=False)
