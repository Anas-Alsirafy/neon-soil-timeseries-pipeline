"""
NEON historical soil + weather time-series downloader
=====================================================
Downloads every continuous soil and meteorological sensor product available at a
NEON site, filters bad readings with NEON's quality flags, pivots each
(plot x depth) sensor into its own column, and merges everything into one wide
30-minute DataFrame. Periodic soil chemistry (pH, C/N, ...) is saved separately.

Install:
    pip install neonutilities pandas requests pyarrow

Easiest: edit the CONFIG block below (token, site, dates) and press Run in your IDE.

Terminal usage also works (flags override the CONFIG block):
    python neon_soil_downloader.py --site SRER --start 2019-01 --end 2023-12
    python neon_soil_downloader.py --site SRER --start 2022-01 --end 2022-12 --provisional

REQUIRED (since June 2026): a free NEON API token
    python neon_soil_downloader.py --token YOUR_TOKEN ...
    or: setx NEON_TOKEN "your-token"   (then open a NEW terminal)

Outputs (in --out/<SITE>/):
    wide_30min.parquet / .csv   -> merged, QC-filtered, one column per sensor channel
    per_product/*.csv           -> each product's wide table
    metadata/*.csv              -> sensor positions (depths!), variable units, issue logs
    chemistry/*.csv             -> periodic soil lab data (pH, moisture, C/N) if available
    download_report.csv         -> what succeeded / failed / was skipped
"""

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

import pandas as pd
import requests
import neonutilities as nu

# ============================================================================
# >>> EDIT THESE, THEN JUST PRESS "RUN" (VS Code / PyCharm / Spyder / Jupyter) <<<
# ============================================================================
NEON_TOKEN    = ""            # paste your token from neonscience.org -> My Account
SITE          = "SRER"        # 4-letter NEON site code (SRER = semi-arid Arizona)
START         = "auto"        # YYYY-MM, or "auto" = earliest available
END           = "auto"        # YYYY-MM, or "auto" = latest available
OUT_DIR       = "neon_data"   # output folder (created next to where you run)
PROVISIONAL   = False         # True = include newest, not-yet-released months
KEEP_FLAGGED  = False         # True = do NOT mask quality-flagged values
SKIP_CHEMISTRY = False        # True = skip periodic soil lab data (pH, C/N)
# ============================================================================

# ----------------------------------------------------------------------------
# Continuous sensor products (30-min). short name -> data product ID
# ----------------------------------------------------------------------------
SENSOR_PRODUCTS = {
    # --- soil ---
    "soil_temp":      "DP1.00041.001",  # soil temperature profile
    "soil_water":     "DP1.00094.001",  # soil water content + salinity (ion content)
    "soil_co2":       "DP1.00095.001",  # soil CO2 concentration
    "soil_heatflux":  "DP1.00040.001",  # soil heat flux plates
    "ir_surf_temp":   "DP1.00005.001",  # IR biological (surface) temperature
    # --- weather ---
    "air_temp":       "DP1.00003.001",  # triple-aspirated air temperature
    "rel_humidity":   "DP1.00098.001",  # relative humidity, dew point
    "precip":         "DP1.00006.001",  # precipitation (primary/secondary/throughfall)
    "radiation":      "DP1.00023.001",  # shortwave + longwave radiation
    "par":            "DP1.00024.001",  # photosynthetically active radiation
    "wind":           "DP1.00001.001",  # 2D wind speed and direction
    "pressure":       "DP1.00004.001",  # barometric pressure
}

# Periodic (lab) soil chemistry - not a time series, saved as-is
CHEMISTRY_PRODUCTS = {
    "soil_chem_periodic": "DP1.10086.001",  # pH, moisture, C/N a few times per year
}

NEON_API = "https://data.neonscience.org/api/v0"
POSITION_COLS = ["horizontalPosition", "verticalPosition"]
TIME_COL = "startDateTime"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def available_products(site: str, token: str | None) -> dict[str, list[str]]:
    """Ask the NEON API which products (and months) exist at this site."""
    headers = {"X-API-Token": token} if token else {}
    r = requests.get(f"{NEON_API}/sites/{site}", headers=headers, timeout=30)
    r.raise_for_status()
    prods = r.json()["data"]["dataProducts"]
    return {p["dataProductCode"]: p.get("availableMonths", []) for p in prods}


def months_in_range(months: list[str], start: str, end: str) -> int:
    return sum(start <= m <= end for m in months)


def is_30min_table(name: str, df) -> bool:
    return (
        isinstance(df, pd.DataFrame)
        and TIME_COL in df.columns
        and re.search(r"30", name) is not None
    )


def value_and_qf_columns(df: pd.DataFrame) -> list[tuple[str, str | None]]:
    """
    Pair each measurement column with its final quality flag.
    Measurement columns end in 'Mean' (most products) or 'Bulk' (precipitation).
    NEON QF naming is inconsistent: '<base>FinalQF' or a single 'finalQF'.
    """
    lower_map = {c.lower(): c for c in df.columns}
    generic_qf = lower_map.get("finalqf")
    pairs = []
    for col in df.columns:
        if col.endswith("StdErMean"):  # standard error of the mean, not a measurement
            continue
        m = re.match(r"^(.*?)(Mean|Bulk)$", col)
        if not m:
            continue
        base = m.group(1)
        qf = lower_map.get(f"{base.lower()}finalqf", generic_qf)
        pairs.append((col, qf))
    return pairs


def tidy_table(df: pd.DataFrame, prefix: str, keep_flagged: bool) -> pd.DataFrame:
    """QC-filter and pivot a long NEON table into wide format (one col per channel)."""
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    pairs = value_and_qf_columns(df)
    if not pairs:
        return pd.DataFrame()

    # Mask values whose final QF == 1 (failed quality checks)
    if not keep_flagged:
        for val, qf in pairs:
            if qf is not None and qf in df.columns:
                df.loc[df[qf] == 1, val] = pd.NA

    value_cols = [v for v, _ in pairs]
    pos_cols = [c for c in POSITION_COLS if c in df.columns]

    if pos_cols:
        for c in pos_cols:
            df[c] = df[c].astype(str).str.zfill(3)
        wide = df.pivot_table(index=TIME_COL, columns=pos_cols,
                              values=value_cols, aggfunc="mean", dropna=False)
        names = []
        for var, *pos in wide.columns:
            if len(pos) == 2:
                names.append(f"{prefix}__{var}__h{pos[0]}_v{pos[1]}")  # plot h, depth v
            else:
                names.append(f"{prefix}__{var}__{pos[0]}")
        wide.columns = names
    else:
        wide = df.groupby(TIME_COL)[value_cols].mean()
        wide.columns = [f"{prefix}__{c}" for c in wide.columns]

    return wide.sort_index()


def save_metadata(tables: dict, short: str, meta_dir: Path) -> None:
    for name, obj in tables.items():
        if isinstance(obj, pd.DataFrame) and re.match(
            r"^(sensor_positions|variables|issueLog)", name
        ):
            obj.to_csv(meta_dir / f"{short}__{name}.csv", index=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Download NEON soil + weather history")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--provisional", action="store_true", default=PROVISIONAL)
    ap.add_argument("--keep-flagged", action="store_true", default=KEEP_FLAGGED)
    ap.add_argument("--no-chem", action="store_true", default=SKIP_CHEMISTRY)
    ap.add_argument("--token", default=None)
    args, _ = ap.parse_known_args()  # ignores extra args injected by Jupyter/IDEs

    token = (args.token or NEON_TOKEN or os.environ.get("NEON_TOKEN") or "").strip() or None
    if not token:
        sys.exit(
            "NEON requires an API token for downloads.\n"
            "  1. Sign in at https://www.neonscience.org -> My Account -> generate API token\n"
            "  2. Paste it into NEON_TOKEN = '...' at the top of this script, then run again"
        )
    site = args.site.upper()
    root = Path(args.out) / site
    per_dir, meta_dir, chem_dir = root / "per_product", root / "metadata", root / "chemistry"
    for d in (per_dir, meta_dir, chem_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Site {site} | {args.start} -> {args.end} (auto = full record) | token: {'yes' if token else 'no'}")
    try:
        avail = available_products(site, token)
    except Exception as e:
        print(f"[warn] could not query site availability ({e}); trying all products")
        avail = None

    report, wide_tables = [], []

    # ---- continuous sensors ----
    # START/END = "auto" -> each product downloads its full record at this site.
    # Downloads happen one year at a time to keep RAM low on long ranges.
    for short, dpid in SENSOR_PRODUCTS.items():
        months = sorted(avail.get(dpid, [])) if avail is not None else []
        p_start = (months[0] if months else None) if args.start == "auto" else args.start
        p_end = (months[-1] if months else None) if args.end == "auto" else args.end
        if p_start is None or p_end is None:
            if avail is None:
                sys.exit("'auto' dates need the site availability query, which failed - use fixed dates")
            print(f"[skip] {short:14s} {dpid}: no data at {site}")
            report.append((short, dpid, "skipped: unavailable", 0))
            continue
        if avail is not None:
            n = months_in_range(months, p_start, p_end)
            if n == 0:
                print(f"[skip] {short:14s} {dpid}: no data at {site} in range")
                report.append((short, dpid, "skipped: unavailable", 0))
                continue
            print(f"[get ] {short:14s} {dpid}: {p_start} -> {p_end} ({n} months available)")

        yearly = []
        for y in range(int(p_start[:4]), int(p_end[:4]) + 1):
            c_start = max(p_start, f"{y}-01")
            c_end = min(p_end, f"{y}-12")
            if avail is not None and months_in_range(months, c_start, c_end) == 0:
                continue
            try:
                tables = nu.load_by_product(
                    dpid=dpid, site=site,
                    startdate=c_start, enddate=c_end,
                    timeindex="30", package="basic",
                    include_provisional=args.provisional,
                    check_size=False, progress=True, token=token,
                )
                save_metadata(tables, short, meta_dir)  # overwritten each year = latest version
                parts = [
                    tidy_table(df, f"{short}_{name}", args.keep_flagged)
                    for name, df in tables.items() if is_30min_table(name, df)
                ]
                parts = [p for p in parts if not p.empty]
                if parts:
                    yearly.append(pd.concat(parts, axis=1))
                    print(f"       {y}: {len(yearly[-1])} rows")
                del tables
            except Exception as e:
                if "token" in str(e).lower():
                    sys.exit(f"[auth] NEON rejected the token: {e}\nCheck it is copied fully and not expired.")
                print(f"[fail] {short} {y}: {e}")
                traceback.print_exc(limit=1)
                report.append((short, dpid, f"failed {y}: {e}", 0))

        if not yearly:
            report.append((short, dpid, "no 30-min data found", 0))
            continue
        prod_wide = pd.concat(yearly, axis=0).sort_index()
        prod_wide = prod_wide[~prod_wide.index.duplicated(keep="last")]
        prod_wide.to_csv(per_dir / f"{short}.csv")
        wide_tables.append(prod_wide)
        report.append((short, dpid, f"ok {p_start}->{p_end}", prod_wide.shape[1]))
        print(f"       -> {prod_wide.shape[1]} channels, {len(prod_wide)} rows total")

    # ---- periodic chemistry ----
    if not args.no_chem:
        for short, dpid in CHEMISTRY_PRODUCTS.items():
            try:
                tables = nu.load_by_product(
                    dpid=dpid, site=site,
                    startdate=None if args.start == "auto" else args.start,
                    enddate=None if args.end == "auto" else args.end,
                    package="basic", include_provisional=args.provisional,
                    check_size=False, token=token,
                )
                n = 0
                for name, df in tables.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(chem_dir / f"{short}__{name}.csv", index=False)
                        n += 1
                report.append((short, dpid, "ok (tables saved)", n))
            except Exception as e:
                print(f"[fail] {short}: {e}")
                report.append((short, dpid, f"failed: {e}", 0))

    pd.DataFrame(report, columns=["product", "dpid", "status", "channels_or_tables"]) \
      .to_csv(root / "download_report.csv", index=False)

    if not wide_tables:
        sys.exit("No sensor data downloaded - check site code, dates, and download_report.csv")

    # ---- merge on a regular 30-min grid ----
    merged = pd.concat(wide_tables, axis=1).sort_index()
    grid = pd.date_range(merged.index.min(), merged.index.max(), freq="30min", tz="UTC")
    merged = merged.reindex(grid)
    merged.index.name = "timestamp_utc"

    merged.to_csv(root / "wide_30min.csv")
    try:
        merged.to_parquet(root / "wide_30min.parquet")
    except Exception:
        print("[info] pyarrow not installed; skipped parquet")

    coverage = merged.notna().mean().sort_values()
    print(f"\nDone: {merged.shape[1]} channels x {len(merged)} timesteps -> {root}")
    print("Lowest-coverage channels:\n", coverage.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
