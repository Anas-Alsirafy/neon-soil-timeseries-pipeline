"""
Derived soil & weather features for the NEON SRER dataset
=========================================================
Computes every feature group from the dataset document:

  1. Atmosphere & humidity    es, ea, VPD, specific humidity, dew-point depression, gamma, GDD
  2. Radiation & energy       net SW/LW, Rn, albedo, radiometric Ts, Rn-G, G/Rn, PAR fraction,
                              solar elevation, day flag, extraterrestrial radiation, clearness index
  3. Evapotranspiration       u2, ASCE/FAO-56 hourly reference ET0, Priestley-Taylor ET
  4. Soil water & balance     profile storage S, dS, water-balance residual, relative wetness,
                              plant-available water, API, days since rain, rolling rain,
                              salinity proxy, wetting-front lags, drydown time constants (events)
  5. Soil temperature & heat  vertical gradients, diurnal range, phase lag, thermal diffusivity,
                              surface-soil and surface-air differences
  6. Soil CO2                 gradients, flux-gradient efflux (Millington-Quirk), Q10,
                              rain-pulse response, CO2 x moisture x temperature bins
  7. Encodings & time series  wind u/v, cyclic time, cross-plot mean/std, rolling stats,
                              slopes, differences, lags, climatology anomalies, x-corr lags

Inputs : wide_30min.parquet (or .csv) + metadata/ from neon_soil_downloader.py
Outputs: derived/derived_30min.parquet   30-min features (plus raw cross-plot means)
         derived/derived_daily.parquet   daily features (GDD, diurnal range, kappa, balance)
         derived/events.csv              rain events: wetting-front lags, drydown tau, CO2 pulse
         derived/summary_*.csv           Q10 fits, CO2 bins, cross-correlation lags
         derived/feature_dictionary.csv  name, description, units for every derived column

Edit CONFIG and press Run.  Requires: pandas, numpy, pyarrow
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN windows, log(0)

# ============================== CONFIG ======================================
SITE_DIR   = r"neon_data/SRER"          # folder with wide_30min.* and metadata/
DATA_FILE  = "wide_30min.parquet"       # or "wide_30min.csv"
OUT_SUBDIR = "derived"

# Site info: used only if the metadata has no reference lat/lon/elevation
SITE_LAT, SITE_LON, SITE_ELEV_M = 31.911, -110.835, 997.0   # SRER (approx.) - verify

# Wind sensor height (m) if not found in metadata
WIND_HEIGHT_M_DEFAULT = 8.0

# Soil properties (set from megapit / chemistry data when you have them)
BULK_DENSITY_G_CM3 = 1.5      # -> total porosity = 1 - BD / 2.65
FIELD_CAPACITY     = None     # m3/m3, e.g. 0.25  (None = skip plant-available water)
WILTING_POINT      = None     # m3/m3, e.g. 0.08

# Thresholds
GDD_BASE_C          = 10.0    # growing-degree-day base temperature
RAIN_STEP_MM        = 0.2     # a 30-min step counts as "raining" above this
RAIN_EVENT_MIN_MM   = 2.0     # minimum event total
RAIN_EVENT_GAP_H    = 6       # dry hours that separate two events
API_K_PER_DAY       = 0.9     # antecedent precipitation index decay
DAY_SW_THRESHOLD    = 10.0    # W/m2 incoming SW = daytime
EMISSIVITY          = 0.95    # surface emissivity for radiometric temperature
MIN_STEPS_PER_DAY   = 40      # of 48; daily stats need this many valid steps

# Generic time-series features are applied to these derived columns (if present).
# Keep the list short: every column gets ~15 extra features.
TS_FEATURES = [
    "vpd_kpa", "air_temp_c", "rn_wm2", "et0_mm", "precip_mm",
    "storage_mm_sitemean", "theta_shallow_sitemean", "soil_temp_shallow_sitemean",
    "co2_flux_umol_m2_s_sitemean",
]
ROLL_WINDOWS = {"3h": 6, "24h": 48, "7d": 336}       # name -> 30-min steps
LAGS         = {"30min": 1, "1d": 48, "1w": 336}
XCORR_MAX_LAG_STEPS = 96                              # 48 h
# ============================================================================

# Physical constants
SIGMA = 5.670374419e-8   # Stefan-Boltzmann, W/m2/K4
R_GAS = 8.314462618      # J/mol/K
LAMBDA_MJ_KG = 2.45      # latent heat of vaporisation
SOLAR_CONST = 1367.0     # W/m2
STEP_H = 0.5             # hours per step


# ----------------------------------------------------------------------------
# Feature registry
# ----------------------------------------------------------------------------
class Registry:
    def __init__(self):
        self.cols: dict[str, pd.Series] = {}
        self.meta: list[tuple[str, str, str, str]] = []

    def add(self, name: str, s: pd.Series, desc: str, units: str, group: str):
        self.cols[name] = s.astype("float32") if s.dtype != bool else s.astype("float32")
        self.meta.append((name, group, desc, units))

    def get(self, name: str) -> pd.Series | None:
        return self.cols.get(name)

    def frame(self, index) -> pd.DataFrame:
        return pd.DataFrame(self.cols, index=index)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Column discovery
# ----------------------------------------------------------------------------
def parse_columns(cols) -> pd.DataFrame:
    """Split '<product>_<table>__<var>__h001_v502' into parts."""
    rows = []
    for c in cols:
        parts = c.split("__")
        if len(parts) < 2:
            continue
        h = v = None
        if len(parts) >= 3:
            m = re.match(r"h(\d+)_v(\d+)$", parts[2])
            if m:
                h, v = int(m.group(1)), int(m.group(2))
        rows.append({"col": c, "head": parts[0], "var": parts[1], "h": h, "v": v})
    return pd.DataFrame(rows)


def short_name(head: str) -> str:
    for s in ["soil_temp", "soil_water", "soil_co2", "soil_heatflux", "ir_surf_temp",
              "air_temp", "rel_humidity", "precip", "radiation", "par", "wind", "pressure"]:
        if head == s or head.startswith(s + "_"):
            return s
    return head


def load_positions(meta_dir: Path):
    """(product, h, v) -> zOffset in m (latest position); also site reference coords."""
    pos, ref = {}, {}
    for f in meta_dir.glob("*__sensor_positions*.csv"):
        short = f.name.split("__")[0]
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "HOR.VER" not in df.columns or "zOffset" not in df.columns:
            continue
        if "positionEndDateTime" in df.columns:
            df = df.assign(_cur=df["positionEndDateTime"].isna()) \
                   .sort_values(["_cur", "positionEndDateTime"], na_position="last")
        for hv, g in df.groupby(df["HOR.VER"].astype(str)):
            try:
                h, v = (int(float(x)) for x in hv.split("."))
            except ValueError:
                continue
            z = pd.to_numeric(g["zOffset"], errors="coerce").dropna()
            if len(z):
                pos[(short, h, v)] = float(z.iloc[-1])
        for key, col in [("lat", "referenceLatitude"), ("lon", "referenceLongitude"),
                         ("elev", "referenceElevation")]:
            if col in df.columns and key not in ref:
                val = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(val):
                    ref[key] = float(val.median())
    return pos, ref


class Columns:
    """Helper to find raw columns by variable name."""

    def __init__(self, df: pd.DataFrame, positions: dict):
        self.df = df
        self.info = parse_columns(df.columns)
        self.info["short"] = self.info["head"].map(short_name)
        self.info["coverage"] = [df[c].notna().mean() for c in self.info["col"]]
        self.info["z"] = [
            positions.get((s, h, v)) if h is not None else None
            for s, h, v in zip(self.info["short"], self.info["h"], self.info["v"])
        ]

    def by_var(self, var: str) -> pd.DataFrame:
        return self.info[self.info["var"] == var]

    def best(self, var: str, prefer_top: bool = True) -> pd.Series | None:
        """Single site-level series: highest-coverage column (ties -> top level)."""
        sub = self.by_var(var)
        if sub.empty:
            return None
        sub = sub.sort_values(["coverage", "v"], ascending=[False, not prefer_top])
        return self.df[sub["col"].iloc[0]]

    def best_z(self, var: str) -> float | None:
        sub = self.by_var(var).sort_values("coverage", ascending=False)
        if sub.empty or pd.isna(sub["z"].iloc[0]):
            return None
        return float(sub["z"].iloc[0])

    def profiles(self, var: str) -> dict[int, list[tuple[float, str]]]:
        """plot -> [(depth_m positive down, column)] sorted shallow->deep.
        Uses metadata depth; if missing, falls back to depth index order."""
        out: dict[int, list] = {}
        for _, r in self.by_var(var).iterrows():
            if r["h"] is None or pd.isna(r["h"]):
                continue
            depth = abs(r["z"]) if r["z"] is not None and not pd.isna(r["z"]) else None
            out.setdefault(int(r["h"]), []).append((depth, r["col"], int(r["v"])))
        clean = {}
        for h, lst in out.items():
            if any(d is None for d, _, _ in lst):
                lst = [(None, c, v) for _, c, v in lst]
                lst.sort(key=lambda t: t[2])
            else:
                lst.sort(key=lambda t: t[0])
            clean[h] = [(d, c) for d, c, _ in lst]
        return clean


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
def daily(s: pd.Series, how: str = "mean") -> pd.Series:
    g = s.resample("1D")
    n = g.count()
    out = getattr(g, how)()
    return out.where(n >= MIN_STEPS_PER_DAY)


def rolling_slope(x: pd.Series, w: int) -> pd.Series:
    """Least-squares slope per hour over a trailing window, NaN-aware."""
    t = pd.Series(np.arange(len(x), dtype=float), index=x.index).where(x.notna())
    mp = max(3, w // 2)
    mt, mx = t.rolling(w, mp).mean(), x.rolling(w, mp).mean()
    mtx, mtt = (t * x).rolling(w, mp).mean(), (t * t).rolling(w, mp).mean()
    slope_per_step = (mtx - mt * mx) / (mtt - mt * mt)
    return slope_per_step / STEP_H


def sat_vp(t_c: pd.Series) -> pd.Series:
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def solar_geometry(index: pd.DatetimeIndex, lat: float, lon: float):
    """Solar elevation (deg) and extraterrestrial irradiance (W/m2) at step midpoints."""
    t = index + pd.Timedelta(minutes=15)
    doy = t.dayofyear.values
    hour_utc = t.hour.values + t.minute.values / 60
    b = 2 * np.pi * (doy - 81) / 364
    eot_h = (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60
    solar_time = hour_utc + lon / 15 + eot_h
    omega = np.deg2rad(15 * (solar_time - 12))
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    phi = np.deg2rad(lat)
    sin_h = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    elev = np.rad2deg(np.arcsin(np.clip(sin_h, -1, 1)))
    ra = SOLAR_CONST * dr * np.clip(sin_h, 0, None)
    return pd.Series(elev, index=index), pd.Series(ra, index=index)


def fix_fraction(s: pd.Series) -> pd.Series:
    """Water content should be m3/m3; convert if it looks like percent."""
    return s / 100 if s.median(skipna=True) > 1.5 else s


# ----------------------------------------------------------------------------
# 1. Atmosphere
# ----------------------------------------------------------------------------
def atmosphere(C: Columns, R: Registry, elev: float):
    T = C.best("tempTripleMean")
    if T is None:
        T = C.best("tempRHMean")
    RH = C.best("RHMean")
    P = C.best("staPresMean")
    if T is None:
        log("  atmosphere: no air temperature - skipped")
        return
    R.add("air_temp_c", T, "Air temperature (best tower level)", "degC", "atmosphere")
    if P is None:
        P = pd.Series(101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26, index=T.index)
        log("  atmosphere: no pressure sensor - using elevation-based pressure")
    R.add("pressure_kpa", P, "Station barometric pressure", "kPa", "atmosphere")

    es = sat_vp(T)
    R.add("es_kpa", es, "Saturation vapour pressure (FAO-56)", "kPa", "atmosphere")
    R.add("delta_kpa_c", 4098 * es / (T + 237.3) ** 2,
          "Slope of saturation vapour pressure curve", "kPa/degC", "atmosphere")
    R.add("gamma_kpa_c", 0.000665 * P, "Psychrometric constant", "kPa/degC", "atmosphere")
    if RH is not None:
        T_rh = C.best("tempRHMean")
        es_rh = sat_vp(T_rh) if T_rh is not None else es
        ea = (es_rh * RH.clip(0, 100) / 100)
        R.add("rh_pct", RH, "Relative humidity", "%", "atmosphere")
        R.add("ea_kpa", ea, "Actual vapour pressure", "kPa", "atmosphere")
        R.add("vpd_kpa", (es - ea).clip(lower=0), "Vapour pressure deficit", "kPa", "atmosphere")
        R.add("specific_humidity_g_kg", 1000 * 0.622 * ea / (P - 0.378 * ea),
              "Specific humidity", "g/kg", "atmosphere")
    Td = C.best("dewTempMean")
    if Td is not None:
        R.add("dewpoint_depression_c", T - Td, "Air temperature minus dew point", "degC", "atmosphere")


# ----------------------------------------------------------------------------
# 2. Radiation & energy
# ----------------------------------------------------------------------------
def radiation(C: Columns, R: Registry, lat: float, lon: float):
    idx = C.df.index
    elev_deg, ra = solar_geometry(idx, lat, lon)
    R.add("solar_elevation_deg", elev_deg, "Solar elevation angle", "deg", "radiation")
    R.add("extraterrestrial_sw_wm2", ra, "Top-of-atmosphere irradiance on horizontal", "W/m2", "radiation")

    sw_in, sw_out = C.best("inSWMean"), C.best("outSWMean")
    lw_in, lw_out = C.best("inLWMean"), C.best("outLWMean")

    day = (sw_in > DAY_SW_THRESHOLD) if sw_in is not None else (elev_deg > 0)
    R.add("is_day", day.astype(float), "Daytime flag (SW_in > threshold, else sun up)", "0/1", "radiation")

    if sw_in is not None:
        R.add("sw_in_wm2", sw_in, "Incoming shortwave", "W/m2", "radiation")
        R.add("clearness_index", (sw_in / ra).where(ra > 50).clip(0, 1.2),
              "Incoming SW / extraterrestrial SW (cloud proxy)", "-", "radiation")
    if sw_in is not None and sw_out is not None:
        R.add("sw_net_wm2", sw_in - sw_out, "Net shortwave", "W/m2", "radiation")
        R.add("albedo", (sw_out / sw_in).where(sw_in > 50).clip(0, 1),
              "Surface albedo (daytime only)", "-", "radiation")
    if lw_in is not None and lw_out is not None:
        R.add("lw_net_wm2", lw_in - lw_out, "Net longwave", "W/m2", "radiation")
    if lw_out is not None:
        R.add("ts_radiometric_c", (lw_out.clip(lower=0) / (EMISSIVITY * SIGMA)) ** 0.25 - 273.15,
              "Surface temperature from outgoing longwave", "degC", "radiation")
    if all(s is not None for s in (sw_in, sw_out, lw_in, lw_out)):
        R.add("rn_wm2", (sw_in - sw_out) + (lw_in - lw_out), "Net radiation", "W/m2", "radiation")

    # soil heat flux: mean over plots
    shf_cols = C.by_var("SHFMean")["col"].tolist()
    if not shf_cols:
        shf_cols = C.info[C.info["short"] == "soil_heatflux"]["col"].tolist()
    if shf_cols:
        G = C.df[shf_cols].mean(axis=1)
        R.add("g_wm2", G, "Soil heat flux, mean over plots", "W/m2", "radiation")
        rn = R.get("rn_wm2")
        if rn is not None:
            R.add("available_energy_wm2", rn - G, "Rn - G", "W/m2", "radiation")
            R.add("g_over_rn", (G / rn).where(rn.abs() > 50).clip(-1, 1),
                  "Ground heat fraction G/Rn (|Rn|>50)", "-", "radiation")

    par = C.best("PARMean", prefer_top=True)
    if par is not None and sw_in is not None:
        R.add("par_over_sw", (par / sw_in).where(sw_in > 50),
              "PAR / incoming SW (umol/J)", "umol/J", "radiation")


# ----------------------------------------------------------------------------
# 3. Evapotranspiration
# ----------------------------------------------------------------------------
def evapotranspiration(C: Columns, R: Registry):
    T, rn = R.get("air_temp_c"), R.get("rn_wm2")
    if T is None or rn is None:
        log("  ET: needs air temperature and net radiation - skipped")
        return
    G = R.get("g_wm2")
    if G is None:
        G = (rn * np.where(R.get("is_day") > 0, 0.1, 0.5))   # FAO-56 hourly approximation
        log("  ET: no soil heat flux - using G = 0.1 Rn (day) / 0.5 Rn (night)")
    delta, gamma = R.get("delta_kpa_c"), R.get("gamma_kpa_c")
    rn_mj, g_mj = rn * 0.0036, G * 0.0036            # W/m2 -> MJ/m2/h

    pt = 1.26 * delta / (delta + gamma) * (rn_mj - g_mj) / LAMBDA_MJ_KG * STEP_H
    R.add("et_pt_mm", pt.clip(lower=0), "Priestley-Taylor ET per 30 min", "mm", "evapotranspiration")

    ws = C.best("windSpeedMean", prefer_top=True)
    vpd = R.get("vpd_kpa")
    if ws is None or vpd is None:
        log("  ET: no wind or VPD - ET0 skipped (Priestley-Taylor kept)")
        return
    z = C.best_z("windSpeedMean") or WIND_HEIGHT_M_DEFAULT
    z = max(z, 2.0)
    u2 = ws * 4.87 / np.log(67.8 * z - 5.42)
    R.add("u2_ms", u2, f"Wind speed adjusted to 2 m (from {z:.1f} m)", "m/s", "evapotranspiration")
    cd = np.where(R.get("is_day") > 0, 0.24, 0.96)
    et0_h = (0.408 * delta * (rn_mj - g_mj) + gamma * 37 / (T + 273) * u2 * vpd) / \
            (delta + gamma * (1 + cd * u2))
    R.add("et0_mm", (et0_h * STEP_H).clip(lower=0),
          "ASCE/FAO-56 hourly reference ET per 30 min", "mm", "evapotranspiration")


# ----------------------------------------------------------------------------
# 4. Soil water & water balance
# ----------------------------------------------------------------------------
def layer_thickness(depths: list[float]) -> np.ndarray:
    d = np.array(depths)
    bounds = np.concatenate([[0], (d[:-1] + d[1:]) / 2,
                             [d[-1] + (d[-1] - d[-2]) / 2 if len(d) > 1 else d[-1] * 2]])
    return np.diff(bounds)


def soil_water(C: Columns, R: Registry, events: pd.DataFrame | None):
    prof = C.profiles("VSWCMean")
    if not prof:
        log("  soil water: no VSWCMean columns - skipped")
        return
    storages, shallow = [], []
    for h, lst in prof.items():
        cols = [c for _, c in lst]
        theta = C.df[cols].apply(fix_fraction)
        shallow.append(theta.iloc[:, 0])
        # relative wetness per sensor
        for col in cols:
            q01, q99 = theta[col].quantile(0.01), theta[col].quantile(0.99)
            if q99 > q01:
                d_lab = C.info.loc[C.info["col"] == col, "v"].iloc[0]
                R.add(f"rel_wetness_h{h}_v{d_lab}", ((theta[col] - q01) / (q99 - q01)).clip(0, 1),
                      f"Relative wetness (1-99 pct scaling), plot {h}, depth index {d_lab}",
                      "0-1", "soil_water")
                if FIELD_CAPACITY and WILTING_POINT:
                    R.add(f"paw_frac_h{h}_v{d_lab}",
                          ((theta[col] - WILTING_POINT) / (FIELD_CAPACITY - WILTING_POINT)).clip(0, 1.2),
                          f"Plant-available water fraction, plot {h}, depth index {d_lab}",
                          "0-1", "soil_water")
        depths = [d for d, _ in lst]
        if all(d is not None for d in depths) and len(depths) >= 2:
            thick = layer_thickness(depths)
            S = (theta * thick).sum(axis=1, min_count=len(cols)) * 1000
            R.add(f"storage_mm_h{h}", S, f"Profile water storage to {sum(thick):.2f} m, plot {h}",
                  "mm", "soil_water")
            storages.append(S)
    if storages:
        S = pd.concat(storages, axis=1).mean(axis=1)
        R.add("storage_mm_sitemean", S, "Profile water storage, mean over plots", "mm", "soil_water")
        R.add("storage_change_mm", S.diff(), "Change in storage per 30 min", "mm", "soil_water")
    else:
        log("  soil water: no depth metadata - storage skipped (relative wetness kept)")
    R.add("theta_shallow_sitemean", pd.concat(shallow, axis=1).mean(axis=1),
          "Shallowest water content, mean over plots", "m3/m3", "soil_water")

    # salinity proxy (VSIC / VSWC), same plot & depth
    for _, r in C.by_var("VSICMean").iterrows():
        twin = r["col"].replace("VSICMean", "VSWCMean")
        if twin in C.df.columns:
            th = fix_fraction(C.df[twin])
            R.add(f"salinity_proxy_h{r['h']}_v{r['v']}", (C.df[r["col"]] / th).where(th > 0.02),
                  f"Ion content / water content (pore-water salinity proxy), plot {r['h']}, "
                  f"depth index {r['v']}", "index", "soil_water")


def precipitation(C: Columns, R: Registry) -> pd.Series | None:
    for var in ["secPrecipBulk", "priPrecipBulk", "TFPrecipBulk"]:
        p = C.best(var)
        if p is not None and p.notna().mean() > 0.05:
            R.add("precip_mm", p.clip(lower=0), f"Precipitation per 30 min ({var})", "mm", "soil_water")
            break
    else:
        log("  precipitation: none found - rain features skipped")
        return None
    p = R.get("precip_mm")
    for name, w in {"24h": 48, "7d": 336, "30d": 1440}.items():
        R.add(f"precip_sum_{name}_mm", p.rolling(w, 1).sum(), f"Rolling precipitation sum {name}",
              "mm", "soil_water")
    k = API_K_PER_DAY ** (1 / 48)
    api = np.zeros(len(p))
    vals = p.fillna(0).to_numpy()
    for i in range(1, len(vals)):
        api[i] = k * api[i - 1] + vals[i]
    R.add("api_mm", pd.Series(api, index=p.index), f"Antecedent precipitation index (k={API_K_PER_DAY}/day)",
          "mm", "soil_water")
    rained = p > RAIN_STEP_MM
    last = pd.Series(np.where(rained, np.arange(len(p)), np.nan), index=p.index).ffill()
    R.add("days_since_rain", (np.arange(len(p)) - last) * STEP_H / 24,
          "Days since last rain step", "days", "soil_water")
    return p


def rain_events(p: pd.Series) -> pd.DataFrame:
    rained = (p > RAIN_STEP_MM).to_numpy()
    idx = np.flatnonzero(rained)
    if len(idx) == 0:
        return pd.DataFrame()
    gap = int(RAIN_EVENT_GAP_H / STEP_H)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1)
    rows = []
    for g in groups:
        total = p.iloc[g[0]:g[-1] + 1].sum()
        if total >= RAIN_EVENT_MIN_MM:
            rows.append({"start_i": g[0], "end_i": g[-1], "start": p.index[g[0]],
                         "end": p.index[g[-1]], "total_mm": round(float(total), 2)})
    ev = pd.DataFrame(rows)
    if not ev.empty:
        ev["next_start_i"] = ev["start_i"].shift(-1).fillna(len(p)).astype(int)
    return ev


def event_responses(C: Columns, R: Registry, ev: pd.DataFrame) -> pd.DataFrame:
    """Wetting-front arrival lag per plot/depth, drydown tau (shallow), CO2 pulse."""
    if ev.empty:
        return ev
    prof = C.profiles("VSWCMean")
    co2_shallow = None
    co2p = C.profiles("soilCO2concentrationMean")
    if co2p:
        co2_shallow = pd.concat([C.df[lst[0][1]] for lst in co2p.values()], axis=1).mean(axis=1)
    out = ev.copy()
    for h, lst in prof.items():
        for k, (d, col) in enumerate(lst):
            th = fix_fraction(C.df[col]).to_numpy()
            lags, taus = [], []
            for _, e in ev.iterrows():
                s, nxt = e["start_i"], e["next_start_i"]
                pre = np.nanmean(th[max(0, s - 6):s]) if s > 0 else np.nan
                win = th[s:min(s + 144, nxt)]                       # up to 72 h
                hit = np.flatnonzero(win > pre + 0.01) if not np.isnan(pre) else []
                lags.append(hit[0] * STEP_H if len(hit) else np.nan)
                if k == 0:                                          # drydown on shallowest
                    seg = th[e["end_i"]:min(e["end_i"] + 480, nxt)]   # up to 10 days
                    taus.append(drydown_tau(seg))
            tag = f"h{h}_d{k}"
            out[f"wetting_lag_h_{tag}"] = lags
            if k == 0:
                out[f"drydown_tau_h_h{h}"] = taus
    if co2_shallow is not None:
        c = co2_shallow.to_numpy()
        out["co2_pulse_ppm"] = [
            np.nanmax(c[e["start_i"]:e["start_i"] + 48]) - np.nanmean(c[max(0, e["start_i"] - 12):e["start_i"]])
            for _, e in ev.iterrows()
        ]
    return out.drop(columns=["start_i", "end_i", "next_start_i"])


def drydown_tau(seg: np.ndarray) -> float:
    seg = seg[~np.isnan(seg)]
    if len(seg) < 24:
        return np.nan
    peak = int(np.argmax(seg[:12]))
    seg = seg[peak:]
    t = np.arange(len(seg)) * STEP_H
    rng_ = seg.max() - seg.min()
    if rng_ < 0.005:                                   # no real drydown
        return np.nan
    # theta(t) = theta_res + A exp(-t/tau): grid-search theta_res, keep best log-linear fit
    best_r2, best_tau = -np.inf, np.nan
    for f in np.r_[np.logspace(-3, 0, 25), 2.0]:
        resid = seg.min() - f * rng_
        y = np.log(seg - resid)
        slope, icpt = np.polyfit(t, y, 1)
        if slope >= 0:
            continue
        r2 = 1 - np.sum((y - (slope * t + icpt)) ** 2) / np.sum((y - y.mean()) ** 2)
        if r2 > best_r2:
            best_r2, best_tau = r2, -1 / slope
    return float(best_tau) if best_r2 > 0.8 else np.nan


# ----------------------------------------------------------------------------
# 5. Soil temperature & heat
# ----------------------------------------------------------------------------
def soil_heat(C: Columns, R: Registry) -> pd.DataFrame:
    prof = C.profiles("soilTempMean")
    daily_cols = {}
    if not prof:
        log("  soil heat: no soilTempMean columns - skipped")
        return pd.DataFrame()
    shallow = []
    omega = 2 * np.pi / 86400
    for h, lst in prof.items():
        shallow.append(C.df[lst[0][1]])
        have_depth = all(d is not None for d, _ in lst)
        for (d1, c1), (d2, c2) in zip(lst[:-1], lst[1:]):
            if have_depth and d2 > d1:
                R.add(f"soil_temp_grad_c_m_h{h}_{d1:.2f}_{d2:.2f}",
                      (C.df[c1] - C.df[c2]) / (d2 - d1),
                      f"Soil temperature gradient {d1:.2f}-{d2:.2f} m, plot {h} (+ = heat flows down)",
                      "degC/m", "soil_heat")
        for k, (d, c) in enumerate(lst):
            dmax, dmin = daily(C.df[c], "max"), daily(C.df[c], "min")
            label = f"{d:.2f}m" if d is not None else f"d{k}"
            daily_cols[f"soil_temp_range_c_h{h}_{label}"] = dmax - dmin
        if len(lst) >= 2:
            (d1, c1), (d2, c2) = lst[0], lst[1]
            a1 = (daily(C.df[c1], "max") - daily(C.df[c1], "min")) / 2
            a2 = (daily(C.df[c2], "max") - daily(C.df[c2], "min")) / 2
            tmax1 = C.df[c1].resample("1D").apply(lambda s: s.idxmax().hour + s.idxmax().minute / 60
                                                   if s.notna().sum() >= MIN_STEPS_PER_DAY else np.nan)
            tmax2 = C.df[c2].resample("1D").apply(lambda s: s.idxmax().hour + s.idxmax().minute / 60
                                                   if s.notna().sum() >= MIN_STEPS_PER_DAY else np.nan)
            lag = (tmax2 - tmax1) % 24
            daily_cols[f"phase_lag_h_h{h}"] = lag
            if have_depth:
                ratio = np.log(a1 / a2)
                kappa = omega * (d2 - d1) ** 2 / (2 * ratio ** 2)
                daily_cols[f"thermal_diffusivity_m2_s_h{h}"] = kappa.where((ratio > 0.05) & (kappa < 5e-6))
    sh = pd.concat(shallow, axis=1).mean(axis=1)
    R.add("soil_temp_shallow_sitemean", sh, "Shallowest soil temperature, mean over plots", "degC", "soil_heat")
    ir = C.best("bioTempMean")
    if ir is not None:
        R.add("ir_surface_temp_c", ir, "IR surface temperature (best-coverage sensor)", "degC", "soil_heat")
        R.add("surface_minus_soil_c", ir - sh, "IR surface temp - shallow soil temp", "degC", "soil_heat")
        if R.get("air_temp_c") is not None:
            R.add("surface_minus_air_c", ir - R.get("air_temp_c"), "IR surface temp - air temp", "degC",
                  "soil_heat")
    return pd.DataFrame(daily_cols)


# ----------------------------------------------------------------------------
# 6. Soil CO2
# ----------------------------------------------------------------------------
def soil_co2(C: Columns, R: Registry) -> tuple[pd.DataFrame, pd.DataFrame]:
    prof = C.profiles("soilCO2concentrationMean")
    if not prof:
        log("  CO2: no soil CO2 columns - skipped")
        return pd.DataFrame(), pd.DataFrame()
    tprof, wprof = C.profiles("soilTempMean"), C.profiles("VSWCMean")
    P = R.get("pressure_kpa")
    porosity = 1 - BULK_DENSITY_G_CM3 / 2.65
    fluxes, q10_rows, shallow_c, shallow_t, shallow_w = [], [], [], [], []

    def nearest(profile, h, depth):
        lst = profile.get(h)
        if not lst:
            return None
        if depth is None or any(d is None for d, _ in lst):
            return C.df[lst[0][1]]
        return C.df[min(lst, key=lambda t: abs(t[0] - depth))[1]]

    for h, lst in prof.items():
        have_depth = all(d is not None for d, _ in lst)
        for (d1, c1), (d2, c2) in zip(lst[:-1], lst[1:]):
            if have_depth and d2 > d1:
                R.add(f"co2_grad_ppm_m_h{h}_{d1:.2f}_{d2:.2f}", (C.df[c2] - C.df[c1]) / (d2 - d1),
                      f"Soil CO2 gradient {d1:.2f}-{d2:.2f} m, plot {h} (+ = increases with depth)",
                      "ppm/m", "soil_co2")
        (d1, c1) = lst[0]
        Ts, th = nearest(tprof, h, d1), nearest(wprof, h, d1)
        shallow_c.append(C.df[c1])
        if Ts is not None:
            shallow_t.append(Ts)
            ok = C.df[c1].notna() & Ts.notna() & (C.df[c1] > 0)
            if ok.sum() > 200:
                b = np.polyfit(Ts[ok], np.log(C.df[c1][ok]), 1)[0]
                q10_rows.append({"plot": h, "depth_m": d1, "q10": float(np.exp(10 * b)),
                                 "n": int(ok.sum())})
        if th is not None:
            shallow_w.append(fix_fraction(th))
        if have_depth and len(lst) >= 2 and Ts is not None and th is not None and P is not None:
            d2, c2 = lst[1]
            th = fix_fraction(th)
            T_k = Ts + 273.15
            eps_a = (porosity - th).clip(lower=0)
            d0 = 1.47e-5 * (T_k / 293.15) ** 1.75 * (101.3 / P)               # m2/s
            ds = d0 * eps_a ** (10 / 3) / porosity ** 2                         # Millington-Quirk
            to_umol_m3 = P * 1000 / (R_GAS * T_k)                               # ppm -> umol/m3
            flux = ds * (C.df[c2] - C.df[c1]) * to_umol_m3 / (d2 - d1)
            R.add(f"co2_flux_umol_m2_s_h{h}", flux,
                  f"Soil CO2 efflux, flux-gradient {d1:.2f}-{d2:.2f} m, plot {h} "
                  f"(porosity {porosity:.2f})", "umol/m2/s", "soil_co2")
            fluxes.append(flux)
    if fluxes:
        R.add("co2_flux_umol_m2_s_sitemean", pd.concat(fluxes, axis=1).mean(axis=1),
              "Soil CO2 efflux, mean over plots", "umol/m2/s", "soil_co2")

    bins = pd.DataFrame()
    if shallow_c and shallow_t and shallow_w:
        df = pd.DataFrame({"co2": pd.concat(shallow_c, axis=1).mean(axis=1),
                           "t": pd.concat(shallow_t, axis=1).mean(axis=1),
                           "w": pd.concat(shallow_w, axis=1).mean(axis=1)}).dropna()
        if len(df) > 100:
            df["t_bin"] = pd.cut(df["t"], bins=np.arange(-10, 61, 5))
            df["w_bin"] = pd.qcut(df["w"], q=5, duplicates="drop")
            bins = df.pivot_table(index="w_bin", columns="t_bin", values="co2",
                                  aggfunc="mean", observed=True).round(0)
    return pd.DataFrame(q10_rows), bins


# ----------------------------------------------------------------------------
# 7. Encodings, cross-plot aggregation, generic time-series features
# ----------------------------------------------------------------------------
def encodings(C: Columns, R: Registry):
    idx = C.df.index
    hour = idx.hour + idx.minute / 60
    doy = idx.dayofyear
    R.add("hour_sin", pd.Series(np.sin(2 * np.pi * hour / 24), index=idx), "Hour of day, sine (UTC)", "-", "encoding")
    R.add("hour_cos", pd.Series(np.cos(2 * np.pi * hour / 24), index=idx), "Hour of day, cosine (UTC)", "-", "encoding")
    R.add("doy_sin", pd.Series(np.sin(2 * np.pi * doy / 365.25), index=idx), "Day of year, sine", "-", "encoding")
    R.add("doy_cos", pd.Series(np.cos(2 * np.pi * doy / 365.25), index=idx), "Day of year, cosine", "-", "encoding")
    ws, wd = C.best("windSpeedMean", True), C.best("windDirMean", True)
    if ws is not None and wd is not None:
        rad = np.deg2rad(wd)
        R.add("wind_u_ms", -ws * np.sin(rad), "Wind u component (toward east +)", "m/s", "encoding")
        R.add("wind_v_ms", -ws * np.cos(rad), "Wind v component (toward north +)", "m/s", "encoding")


def cross_plot(C: Columns, R: Registry):
    info = C.info.dropna(subset=["h"])
    soil = info[info["short"].isin(["soil_temp", "soil_water", "soil_co2", "soil_heatflux"])]
    for (var, v), g in soil.groupby(["var", "v"]):
        if len(g) < 2:
            continue
        block = C.df[g["col"]]
        if var == "VSWCMean":
            block = block.apply(fix_fraction)
        R.add(f"{var}_v{int(v)}_plotmean", block.mean(axis=1), f"{var}, depth index {int(v)}, mean of "
              f"{len(g)} plots", "as raw", "cross_plot")
        R.add(f"{var}_v{int(v)}_plotstd", block.std(axis=1), f"{var}, depth index {int(v)}, std across plots",
              "as raw", "cross_plot")


def time_series_features(R: Registry, index) -> pd.DataFrame:
    out, meta = {}, []
    for name in TS_FEATURES:
        x = R.get(name)
        if x is None or x.notna().sum() < 500:
            continue
        x = x.astype("float64")
        for wn, w in ROLL_WINDOWS.items():
            r = x.rolling(w, max(2, w // 2))
            out[f"{name}__mean_{wn}"], out[f"{name}__std_{wn}"] = r.mean(), r.std()
            out[f"{name}__min_{wn}"], out[f"{name}__max_{wn}"] = r.min(), r.max()
            out[f"{name}__slope_per_h_{wn}"] = rolling_slope(x, w)
        out[f"{name}__diff"] = x.diff()
        for ln, l in LAGS.items():
            out[f"{name}__lag_{ln}"] = x.shift(l)
        clim = x.groupby([index.month, index.hour])
        mu, sd = clim.transform("mean"), clim.transform("std")
        out[f"{name}__anomaly_z"] = (x - mu) / sd.replace(0, np.nan)
        meta.append(name)
    df = pd.DataFrame(out, index=index).astype("float32")
    for c in df.columns:
        base, feat = c.split("__", 1)
        R.meta.append((c, "time_series", f"{feat} of {base}", "see base / per hour for slopes"))
    log(f"  time-series features for: {meta}")
    return df


def xcorr_lags(C: Columns, R: Registry) -> pd.DataFrame:
    rows = []

    def best_lag(driver: pd.Series, resp: pd.Series, name: str):
        a, b = driver.astype(float), resp.astype(float)
        cors = [a.corr(b.shift(-k)) for k in range(XCORR_MAX_LAG_STEPS + 1)]
        cors = np.array(cors, dtype=float)
        if np.all(np.isnan(cors)):
            return
        # first local peak (daily cycles repeat, so the global max can be a day late)
        k = int(np.nanargmax(cors))
        for i in range(len(cors) - 1):
            left = cors[i - 1] if i > 0 else -np.inf
            if cors[i] >= left and cors[i] >= cors[i + 1] and cors[i] > 0.3 * np.nanmax(cors):
                k = i
                break
        rows.append({"pair": name, "best_lag_h": k * STEP_H, "corr": round(float(cors[k]), 3)})

    T, p = R.get("air_temp_c"), R.get("precip_mm")
    for var, driver, dname in [("soilTempMean", T, "air_temp"), ("VSWCMean", p, "precip")]:
        if driver is None:
            continue
        for _, g in C.info[C.info["var"] == var].groupby("v"):
            resp = C.df[g["col"]].mean(axis=1)
            depth = g["z"].dropna()
            dlab = f"{abs(depth.mean()):.2f} m" if len(depth) else f"v{int(g['v'].iloc[0])}"
            best_lag(driver, resp, f"{dname} -> {var} @ {dlab}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    site = Path(SITE_DIR)
    out = site / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)

    path = site / DATA_FILE
    log(f"Loading {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else \
        pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).iloc[:, 1:]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    df = df.loc[:, ~df.columns.str.contains("StdErMean")]
    log(f"  {df.shape[1]} raw columns x {len(df)} steps "
        f"({df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d})")

    positions, ref = load_positions(site / "metadata")
    lat, lon, elev = ref.get("lat", SITE_LAT), ref.get("lon", SITE_LON), ref.get("elev", SITE_ELEV_M)
    log(f"  site lat {lat:.4f}, lon {lon:.4f}, elev {elev:.0f} m | {len(positions)} sensor positions")

    C, R = Columns(df, positions), Registry()

    steps = [
        ("1 atmosphere", lambda: atmosphere(C, R, elev)),
        ("2 radiation & energy", lambda: radiation(C, R, lat, lon)),
        ("3 evapotranspiration", lambda: evapotranspiration(C, R)),
        ("4a precipitation", lambda: precipitation(C, R)),
        ("4b soil water", lambda: soil_water(C, R, None)),
        ("5 soil heat", lambda: soil_heat(C, R)),
        ("6 soil CO2", lambda: soil_co2(C, R)),
        ("7a encodings", lambda: encodings(C, R)),
        ("7b cross-plot", lambda: cross_plot(C, R)),
    ]
    results = {}
    for name, fn in steps:
        log(f"Step {name}")
        try:
            results[name] = fn()
        except Exception as e:                       # one failed group never kills the run
            log(f"  [fail] {name}: {e}")
            results[name] = None

    # events (need precipitation)
    events = pd.DataFrame()
    p = R.get("precip_mm")
    if p is not None:
        log("Step 4c rain events")
        try:
            events = event_responses(C, R, rain_events(p))
            log(f"  {len(events)} rain events")
        except Exception as e:
            log(f"  [fail] events: {e}")

    log("Step 7c time-series features")
    ts = time_series_features(R, df.index)
    wide = pd.concat([R.frame(df.index), ts], axis=1)

    # daily table
    log("Building daily table")
    day = {}
    for name, how in [("air_temp_c", "mean"), ("vpd_kpa", "mean"), ("rn_wm2", "mean"),
                      ("storage_mm_sitemean", "mean"), ("soil_temp_shallow_sitemean", "mean")]:
        if R.get(name) is not None:
            day[f"{name}_daily_{how}"] = daily(R.get(name), how)
    for name in ["et0_mm", "et_pt_mm", "precip_mm"]:
        if R.get(name) is not None:
            day[f"{name}_daily_sum"] = daily(R.get(name), "sum")
    if R.get("air_temp_c") is not None:
        tmax, tmin = daily(R.get("air_temp_c"), "max"), daily(R.get("air_temp_c"), "min")
        gdd = ((tmax + tmin) / 2 - GDD_BASE_C).clip(lower=0)
        day["gdd"] = gdd
        day["gdd_cumulative_year"] = gdd.fillna(0).groupby(gdd.index.year).cumsum()
        day["air_temp_range_c"] = tmax - tmin
    if R.get("storage_mm_sitemean") is not None and R.get("precip_mm") is not None:
        s_day = R.get("storage_mm_sitemean").resample("1D").last()
        day["storage_change_daily_mm"] = s_day.diff()
        day["water_balance_residual_mm"] = daily(R.get("precip_mm"), "sum") - s_day.diff()
    heat_daily = results.get("5 soil heat")
    daily_df = pd.concat([pd.DataFrame(day), heat_daily if isinstance(heat_daily, pd.DataFrame) else None],
                         axis=1)
    daily_desc = {
        "gdd": ("Growing degree days, base %.0f degC" % GDD_BASE_C, "degC-day"),
        "gdd_cumulative_year": ("Cumulative GDD within calendar year", "degC-day"),
        "air_temp_range_c": ("Daily air temperature range", "degC"),
        "storage_change_daily_mm": ("Daily change in profile storage", "mm"),
        "water_balance_residual_mm": ("P - dS = ET + drainage + runoff (approx.)", "mm"),
    }
    for c in daily_df.columns:
        desc, units = daily_desc.get(c, (c.replace("_", " "), "see name"))
        R.meta.append((c, "daily", desc, units))

    # save
    log("Saving")
    wide.to_parquet(out / "derived_30min.parquet")
    daily_df.to_parquet(out / "derived_daily.parquet")
    if not events.empty:
        events.to_csv(out / "events.csv", index=False)
    co2_res = results.get("6 soil CO2")
    if co2_res:
        q10, bins = co2_res
        if not q10.empty:
            q10.to_csv(out / "summary_q10.csv", index=False)
        if not bins.empty:
            bins.to_csv(out / "summary_co2_by_moisture_temp.csv")
    xc = xcorr_lags(C, R)
    if not xc.empty:
        xc.to_csv(out / "summary_xcorr_lags.csv", index=False)
    pd.DataFrame(R.meta, columns=["feature", "group", "description", "units"]) \
      .drop_duplicates("feature").to_csv(out / "feature_dictionary.csv", index=False)

    groups = pd.Series([g for _, g, _, _ in R.meta]).value_counts()
    log(f"Done in {time.time() - t0:.0f} s -> {out}")
    print(f"  30-min features: {wide.shape[1]}  | daily features: {daily_df.shape[1]}  | events: {len(events)}")
    print("  features per group:\n" + groups.to_string())


if __name__ == "__main__":
    main()
