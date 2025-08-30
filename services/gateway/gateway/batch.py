import argparse, asyncio, csv
import pandas as pd
from gateway.client import infer  # this calls the classifier

FIELDS = ["id", "title", "requires_geo_logic", "reasoning", "regulations"]

def _to_docs(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    s = str(v).strip()
    if not s:
        return []
    # support pipe- or semicolon-separated docs in CSVs
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    return [s]

async def _infer_one(row):
    payload = {
        "title": row.get("title", "") or "",
        "description": row.get("description", "") or row.get("desc", "") or "",
        "docs": _to_docs(row.get("docs")),
    }
    res = await infer(payload)
    regs = res.get("regulations") or []
    if not isinstance(regs, (list, tuple)):
        regs = [regs] if regs else []
    regs = sorted({str(x).strip() for x in regs if str(x).strip()})
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "requires_geo_logic": _tri(res.get("requires_geo_logic")),
        "reasoning": res.get("reasoning", ""),
        # serialize list → semicolon-separated string
        "regulations": ";".join(regs),
    }

def _tri(v):
    return "true" if v is True else "false" if v is False else "null"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    rows = df.to_dict(orient="records")

    out = asyncio.run(_run_all(rows))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow(r)

async def _run_all(rows):
    tasks = [ _infer_one(r) for r in rows ]
    # run sequentially to keep rate limits happy; switch to gather for speed
    out = []
    for t in tasks:
        out.append(await t)
    return out

if __name__ == "__main__":
    main()