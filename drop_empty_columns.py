"""
Remove empty columns from every CSV under a folder (e.g. the NEON output).
A column is "empty" if it has no data at all (every value blank/NaN).
Works on huge files: reads in chunks, so RAM stays low.

Edit the CONFIG block and press Run.
"""

from pathlib import Path
import pandas as pd

# ============================== CONFIG ======================================
DATA_DIR      = r"neon_data"   # folder to scan (subfolders included)
MIN_COVERAGE  = 0.0            # 0.0 = drop only fully empty columns
                               # e.g. 0.05 = also drop columns with <5% data
IN_PLACE      = True           # True = overwrite files; False = write *_clean.csv
UPDATE_PARQUET = True          # also rewrite wide_30min.parquet if it exists
SKIP_FILES    = {"download_report.csv", "dropped_columns_log.csv"}
CHUNK_ROWS    = 50_000
# ============================================================================


def find_keep_columns(path: Path) -> tuple[list[str], list[str], int]:
    """Pass 1: count non-null values per column, chunk by chunk."""
    counts, n_rows, cols = None, 0, None
    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False):
        cols = list(chunk.columns)
        c = chunk.notna().sum()
        counts = c if counts is None else counts.add(c, fill_value=0)
        n_rows += len(chunk)
    if cols is None:
        return [], [], 0
    first = cols[0]  # always keep the first column (timestamp / index)
    keep, drop = [], []
    for col in cols:
        cov = counts[col] / n_rows if n_rows else 0
        has_data = counts[col] > 0 and cov >= MIN_COVERAGE
        (keep if col == first or has_data else drop).append(col)
    return keep, drop, n_rows


def rewrite(path: Path, keep: list[str]) -> Path:
    """Pass 2: stream the kept columns into a new file."""
    out = path.with_name(path.stem + ("_tmp.csv" if IN_PLACE else "_clean.csv"))
    first = True
    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, usecols=keep, low_memory=False):
        chunk[keep].to_csv(out, mode="w" if first else "a", header=first, index=False)
        first = False
    if IN_PLACE:
        out.replace(path)
        return path
    return out


def main() -> None:
    root = Path(DATA_DIR)
    files = [p for p in root.rglob("*.csv")
             if p.name not in SKIP_FILES and not p.stem.endswith(("_clean", "_tmp"))]
    if not files:
        print(f"No CSV files found under {root.resolve()}")
        return

    total_dropped, log = 0, []
    for path in sorted(files):
        try:
            keep, drop, n = find_keep_columns(path)
        except Exception as e:
            print(f"[fail] {path}: {e}")
            continue
        rel = path.relative_to(root)
        if not drop:
            print(f"[ok  ] {rel}: nothing to drop ({len(keep)} cols)")
            continue
        out = rewrite(path, keep)
        total_dropped += len(drop)
        log += [(str(rel), c) for c in drop]
        print(f"[drop] {rel}: removed {len(drop)} of {len(keep) + len(drop)} cols -> {out.name}")

        # keep the parquet twin in sync with the cleaned wide CSV
        pq = path.with_suffix(".parquet")
        if UPDATE_PARQUET and pq.exists():
            try:
                df = pd.read_parquet(pq)
                df.drop(columns=[c for c in drop if c in df.columns]).to_parquet(
                    pq if IN_PLACE else pq.with_name(pq.stem + "_clean.parquet"))
                print(f"       parquet updated: {pq.name}")
            except Exception as e:
                print(f"       [warn] parquet not updated: {e}")

    if log:
        pd.DataFrame(log, columns=["file", "dropped_column"]).to_csv(
            root / "dropped_columns_log.csv", index=False)
    print(f"\nDone. Dropped {total_dropped} columns across {len(files)} files."
          + (f" Log: {root / 'dropped_columns_log.csv'}" if log else ""))


if __name__ == "__main__":
    main()
