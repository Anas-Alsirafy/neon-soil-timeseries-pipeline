"""
Generate a human-readable data dictionary for wide_30min columns.

For every column it writes: product, variable description, units, soil plot,
sensor depth/height (m), and data coverage (% non-empty).
Uses the metadata the downloader saved (variables_* and sensor_positions_*).

Outputs (next to the data):
    column_dictionary.txt   -> readable, grouped by product
    column_dictionary.csv   -> same info as a table (handy for renaming/filtering)

Edit CONFIG and press Run.
"""

from pathlib import Path
import re
import pandas as pd

# ============================== CONFIG ======================================
SITE_DIR   = r"neon_data/SRER"                 # folder with wide_30min.* and metadata/
DATA_FILE  = "wide_30min.parquet"              # or "wide_30min.csv"
# ============================================================================

PRODUCTS = {  # short name -> (readable name, NEON id)
    "soil_temp":     ("Soil temperature",               "DP1.00041.001"),
    "soil_water":    ("Soil water content & salinity",  "DP1.00094.001"),
    "soil_co2":      ("Soil CO2 concentration",         "DP1.00095.001"),
    "soil_heatflux": ("Soil heat flux",                 "DP1.00040.001"),
    "ir_surf_temp":  ("IR surface temperature",         "DP1.00005.001"),
    "air_temp":      ("Air temperature",                "DP1.00003.001"),
    "rel_humidity":  ("Relative humidity",              "DP1.00098.001"),
    "precip":        ("Precipitation",                  "DP1.00006.001"),
    "radiation":     ("Shortwave/longwave radiation",   "DP1.00023.001"),
    "par":           ("Photosynthetically active rad.", "DP1.00024.001"),
    "wind":          ("2D wind",                        "DP1.00001.001"),
    "pressure":      ("Barometric pressure",            "DP1.00004.001"),
}

# Used only when a variable is missing from the downloaded metadata
FALLBACK = {
    "soilTempMean": ("Mean soil temperature", "celsius"),
    "VSWCMean": ("Mean volumetric soil water content", "m3/m3 (fraction)"),
    "VSICMean": ("Mean volumetric soil ion content (salinity proxy)", "dimensionless index"),
    "soilCO2concentrationMean": ("Mean soil CO2 concentration", "ppm"),
    "SHFMean": ("Mean soil heat flux", "W/m2"),
    "bioTempMean": ("Mean IR surface (biological) temperature", "celsius"),
    "tempTripleMean": ("Mean air temperature (triple-aspirated)", "celsius"),
    "RHMean": ("Mean relative humidity", "percent"),
    "tempRHMean": ("Mean air temperature at RH sensor", "celsius"),
    "dewTempMean": ("Mean dew point temperature", "celsius"),
    "priPrecipBulk": ("Total precipitation, primary gauge", "mm"),
    "secPrecipBulk": ("Total precipitation, secondary gauge", "mm"),
    "TFPrecipBulk": ("Total throughfall precipitation", "mm"),
    "inSWMean": ("Mean incoming shortwave radiation", "W/m2"),
    "outSWMean": ("Mean outgoing shortwave radiation", "W/m2"),
    "inLWMean": ("Mean incoming longwave radiation", "W/m2"),
    "outLWMean": ("Mean outgoing longwave radiation", "W/m2"),
    "PARMean": ("Mean photosynthetically active radiation", "umol/m2/s"),
    "windSpeedMean": ("Mean wind speed", "m/s"),
    "windDirMean": ("Mean wind direction", "degrees"),
    "staPresMean": ("Mean station barometric pressure", "kPa"),
    "corPres": ("Sea-level corrected barometric pressure", "kPa"),
}


def load_metadata(meta_dir: Path):
    """Collect variable descriptions/units and sensor offsets from saved metadata."""
    var_info, positions = {}, {}
    for f in meta_dir.glob("*.csv"):
        short = f.name.split("__")[0]
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "__variables" in f.name and {"fieldName", "description"} <= set(df.columns):
            for _, r in df.iterrows():
                var_info.setdefault(r["fieldName"], (str(r["description"]),
                                                     str(r.get("units", "") or "")))
        elif "__sensor_positions" in f.name and "HOR.VER" in df.columns:
            df = df.copy()
            if "positionEndDateTime" in df.columns:
                # current position first (no end date), then latest end date
                df["_order"] = df["positionEndDateTime"].isna()
                df = df.sort_values(["_order", "positionEndDateTime"], na_position="last")
            for hv, g in df.groupby(df["HOR.VER"].astype(str)):
                z = pd.to_numeric(g.get("zOffset"), errors="coerce").dropna()
                desc = g.get("sensorLocationDescription")
                positions[(short, hv)] = (
                    z.iloc[-1] if len(z) else None,
                    (z.min(), z.max()) if len(z) > 1 and z.min() != z.max() else None,
                    str(desc.iloc[-1]) if desc is not None and len(desc) else "",
                )
    return var_info, positions


def normalise_hv(h: str, v: str) -> list[str]:
    """NEON writes HOR.VER as '001.502' or '1.502' depending on the version."""
    return [f"{h}.{v}", f"{int(h)}.{v}" if h.isdigit() else f"{h}.{v}",
            f"{h.zfill(3)}.{v.zfill(3)}"]


def describe(col: str, var_info, positions, coverage: float) -> dict:
    parts = col.split("__")
    row = {"column": col, "product": "", "variable": "", "description": "",
           "units": "", "plot": "", "depth_or_height_m": "", "position_note": "",
           "coverage_pct": round(coverage * 100, 1)}
    if len(parts) < 2:
        row["description"] = "timestamp / index" if "time" in col.lower() else ""
        return row

    head, var = parts[0], parts[1]
    short = next((s for s in sorted(PRODUCTS, key=len, reverse=True)
                  if head.startswith(s + "_") or head == s), head)
    row["product"] = PRODUCTS.get(short, (short,))[0]
    row["variable"] = var

    desc, units = var_info.get(var) or FALLBACK.get(var, ("", ""))
    row["description"], row["units"] = desc, units if units != "nan" else ""

    if len(parts) >= 3:
        m = re.match(r"h(\w+?)_v(\w+)$", parts[2])
        if m:
            h, v = m.groups()
            row["plot"] = str(int(h)) if h.isdigit() else h
            for key in normalise_hv(h, v):
                if (short, key) in positions:
                    z, zrange, note = positions[(short, key)]
                    if z is not None:
                        where = "below surface" if z < 0 else "above ground"
                        row["depth_or_height_m"] = f"{abs(z):.2f} m {where}"
                        if zrange:
                            row["depth_or_height_m"] += \
                                f" (sensor moved over time: {abs(zrange[1]):.2f}-{abs(zrange[0]):.2f} m)"
                    row["position_note"] = note if note != "nan" else ""
                    break
            else:
                row["position_note"] = f"HOR.VER {h}.{v} (no position metadata found)"
        else:
            row["position_note"] = parts[2]
    return row


def main() -> None:
    site = Path(SITE_DIR)
    path = site / DATA_FILE
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    if df.index.name:  # timestamp stored as index
        df = df.reset_index()

    var_info, positions = load_metadata(site / "metadata")
    coverage = df.notna().mean()
    rows = [describe(c, var_info, positions, coverage[c]) for c in df.columns]
    table = pd.DataFrame(rows)
    table.to_csv(site / "column_dictionary.csv", index=False)

    ts = df.iloc[:, 0]
    lines = [
        f"DATA DICTIONARY - {path.name}",
        f"Site: {site.name} | {len(df.columns)} columns | {len(df)} rows (30-min steps, UTC)",
        f"Time range: {ts.min()} -> {ts.max()}",
        "",
        "Column name format:  <product>_<NEONtable>__<variable>__h<plot>_v<depth index>",
        "  plot  = soil plot / sensor location number",
        "  depth = negative zOffset -> below ground; positive -> height above ground",
        "  coverage = % of rows that have data (low = gaps or late installation)",
        "  Missing descriptions/units: check metadata/*__variables_*.csv",
        "=" * 90,
    ]
    for product, g in table.groupby("product", sort=False):
        lines += ["", f"### {product or 'Timestamp'}  ({len(g)} columns)", "-" * 90]
        for _, r in g.iterrows():
            lines.append(r["column"])
            info = [f"what: {r['description'] or r['variable'] or '-'}"]
            if r["units"]:
                info.append(f"units: {r['units']}")
            if r["plot"]:
                info.append(f"plot: {r['plot']}")
            if r["depth_or_height_m"]:
                info.append(f"position: {r['depth_or_height_m']}")
            info.append(f"coverage: {r['coverage_pct']}%")
            lines.append("    " + " | ".join(info))
            if r["position_note"]:
                lines.append(f"    note: {r['position_note']}")

    (site / "column_dictionary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {site / 'column_dictionary.txt'} and column_dictionary.csv "
          f"({len(table)} columns)")
    missing = table.loc[(table["description"] == "") & (table["product"] != ""), "variable"]
    if len(missing):
        print(f"[info] {missing.nunique()} variables had no description in metadata: "
              f"{sorted(missing.unique())[:8]}")


if __name__ == "__main__":
    main()
