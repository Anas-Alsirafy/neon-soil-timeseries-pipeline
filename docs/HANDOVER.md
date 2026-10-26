# Handover: Soil Sensor Time-Series Data Pipeline (NEON)

Oct 26, 2026 · Anas Alsirafy

This project built a working pipeline that downloads multi-year soil and weather sensor data from NEON's Santa Rita site (SRER, Arizona), cleans it, documents every column, and computes about 300 derived features. The data supports the sensor-based soil-health classification work.

| Item | State at handover |
| --- | --- |
| Data download (`neon_soil_downloader.py`) | Working; full SRER record downloaded (12 sensor products + soil chemistry) |
| Cleaning (`drop_empty_columns.py`, StdErMean removal) | Working; empty and standard-error columns removed |
| Column documentation (`make_column_dictionary.py`) | Working; produces `column_dictionary.txt` / `.csv` |
| Derived features (`compute_derived_features.py`) | Written and tested on synthetic NEON-format data; **not yet run on the real downloaded data** |
| Dataset description | Complete: [DATASET.md](DATASET.md) and [PDF](NEON_SRER_Soil_Weather_Dataset.pdf) |
| Model training on these features | Not started |

All four scripts are plain Python, configured by editing a CONFIG block at the top of each file and pressing Run in an IDE. No terminal is needed.

## Project context and data sources evaluated

The goal was a public, multi-parameter soil time series to support the lab's own labelled soil-health classifier (26 sensors). The public data was meant for testing the feature pipeline, pretraining, and adding weather context. NEON was chosen because it is the only free source with many continuous soil and weather channels, research-grade quality flags, and a full multi-year history.

| Source | What it offers | Decision |
| --- | --- | --- |
| NEON (US National Ecological Observatory Network) | Soil temperature, water content, salinity index, CO₂, heat flux, IR surface temperature + full tower weather; 30-min; ~2017 to present; quality flags | **Chosen.** Site SRER picked as the closest semi-arid analogue to Egyptian conditions. |
| ISMN (International Soil Moisture Network) | In-situ soil moisture and temperature, global | Kept as a backup / pretraining source; mostly moisture and temperature only |
| Open-Meteo (Forecast + ERA5 archive) | Modelled soil moisture/temperature and weather for any coordinates; live and historical; no API key | Recommended for live context at the Egyptian field site, not for training (model output, not sensors) |
| Kaggle soil datasets (e.g. WSU Cook Agronomy Farm sensor network) | Moisture and temperature time series; tabular NPK snapshots | Rejected: too few variables; NPK sets are not time series |
| AmeriFlux / FLUXNET | Half-hourly fluxes and soil/met channels | Not pursued; ecosystem-focused |
| IEEE DataPort / Hugging Face soil sets | pH, EC, NPK with health labels | Rejected as training data: point samples, some look synthetic |

No public source has continuous pH/NPK or soil-health labels. Those can only come from the lab's own sensors.

## Pipeline overview

The pipeline runs in four steps plus one manual cleanup, each reading the previous step's output:

1. **Download:** `neon_soil_downloader.py` pulls 12 sensor products year by year, masks quality-flagged values, pivots each plot × depth sensor into its own column, and merges everything onto one 30-min UTC grid. Output: `wide_30min.parquet` / `.csv`, plus metadata and chemistry.
2. **Remove empty columns:** `drop_empty_columns.py` deletes channels with no data from every CSV and keeps the parquet in sync.
3. **Remove standard-error columns:** a one-off snippet (in How to run) drops `...StdErMean` columns from the data downloaded before the downloader fix. New downloads no longer contain them.
4. **Document columns:** `make_column_dictionary.py` writes a readable description, units, plot, real depth and coverage for every column.
5. **Derive features:** `compute_derived_features.py` computes the physics-based and time-series features into a `derived/` folder.

## Scripts reference

Every script has a CONFIG block at the top. Edit it, then press Run in VS Code, PyCharm, Spyder or Jupyter.

| Script | Purpose | Reads | Writes | Key settings |
| --- | --- | --- | --- | --- |
| `neon_soil_downloader.py` | Download, QC-mask, pivot and merge NEON soil + weather data | NEON API (token required) | `neon_data/<SITE>/wide_30min.*`, `per_product/`, `metadata/`, `chemistry/`, `download_report.csv` | `NEON_TOKEN`, `SITE` (SRER), `START`/`END` (`"auto"` = full record), `PROVISIONAL`, `KEEP_FLAGGED`, `SKIP_CHEMISTRY` |
| `drop_empty_columns.py` | Remove all-empty columns from every CSV; sync parquet | All CSVs under `DATA_DIR` | Cleaned files in place, `dropped_columns_log.csv` | `DATA_DIR`, `MIN_COVERAGE` (0 = only fully empty), `IN_PLACE`, `UPDATE_PARQUET`, `CHUNK_ROWS` |
| `make_column_dictionary.py` | Human-readable data dictionary | `wide_30min.parquet`, `metadata/` | `column_dictionary.txt`, `column_dictionary.csv` | `SITE_DIR`, `DATA_FILE` |
| `compute_derived_features.py` | ~300 derived features in 7 groups | `wide_30min.parquet`, `metadata/` | `derived/derived_30min.parquet`, `derived_daily.parquet`, `events.csv`, `summary_*.csv`, `feature_dictionary.csv` | `SITE_DIR`, site lat/lon/elevation fallback, `BULK_DENSITY_G_CM3`, `FIELD_CAPACITY`, `WILTING_POINT`, rain thresholds, `TS_FEATURES` |

The derived feature groups are:

- **Atmosphere:** vapour pressure deficit, specific humidity, dew-point depression, psychrometric constant, growing degree days.
- **Radiation:** net radiation, albedo, radiometric surface temperature, G/Rn, solar elevation, clearness index.
- **Evapotranspiration:** FAO-56/ASCE hourly reference ET₀, Priestley–Taylor ET.
- **Soil water:** profile storage, water-balance residual, relative wetness, antecedent precipitation index, days since rain, salinity proxy, wetting-front lag and drydown time constant per rain event.
- **Soil heat:** temperature gradients, diurnal range, phase lag, thermal diffusivity.
- **Soil CO₂:** gradients, flux-gradient efflux, Q10, rain-pulse response.
- **Encodings and time series:** wind u/v, cyclic time, cross-plot mean/std, rolling stats, slopes, lags, climatology anomalies.

The formulas behind each feature are in [DATASET.md](DATASET.md).

## How to run end to end

A full SRER download takes hours and tens of GB, so test with a short date range first.

1. **Install** (Windows, once): `pip install -r requirements.txt`
2. **Get a NEON API token.** NEON has required one for downloads since June 2026. Sign in at neonscience.org → My Account → generate API token. The token is personal: the next person must create their own, and it must never be committed to Git or shared inside the script.
3. **Download:** in `neon_soil_downloader.py`, paste the token into `NEON_TOKEN`, set `SITE`, `START`, `END` (try `"2022-01"`/`"2022-03"` first, then `"auto"` for the full record), and Run. Check `download_report.csv` for skipped or failed products.
4. **Remove empty columns:** set `DATA_DIR` in `drop_empty_columns.py` to the same output folder and Run. For a dry run, set `IN_PLACE = False` to write `*_clean.csv` copies instead.
5. **Remove standard-error columns** (only for data downloaded before the fix):

```python
import pandas as pd
p = "neon_data/SRER/wide_30min.parquet"
df = pd.read_parquet(p)
df = df.loc[:, ~df.columns.str.contains("StdErMean")]
df.to_parquet(p)
```

6. **Document columns:** run `make_column_dictionary.py` (set `SITE_DIR`). Read `column_dictionary.txt` to choose channels.
7. **Derive features:** in `compute_derived_features.py` set `SITE_DIR`, and ideally `BULK_DENSITY_G_CM3`, `FIELD_CAPACITY` and `WILTING_POINT` from the NEON megapit data. Then Run. Each feature group logs `[fail] <group>: <error>` instead of stopping the run, so check the log for failures.

If a full path is needed, use a raw string such as `r"C:\data\neon_data"` and the same path in all four scripts, because relative paths resolve against the IDE's working directory.

## Outputs and folder layout

```
neon_data/
  dropped_columns_log.csv          columns removed by drop_empty_columns.py
  SRER/
    wide_30min.parquet / .csv      main merged table, one column per sensor channel (train from this)
    download_report.csv            per-product status: ok / skipped / failed
    column_dictionary.txt / .csv   readable description, units, plot, depth, coverage per column
    per_product/*.csv              one wide table per NEON product
    metadata/                      sensor_positions (depths), variables (units), issueLog per product
    chemistry/*.csv                periodic lab soil samples: pH, moisture, C, N
    derived/
      derived_30min.parquet        30-min derived + time-series features
      derived_daily.parquet        daily features (GDD, ET0 sums, diffusivity, water balance)
      events.csv                   one row per rain event (wetting lag, drydown tau, CO2 pulse)
      summary_q10.csv              CO2 temperature sensitivity per plot
      summary_co2_by_moisture_temp.csv
      summary_xcorr_lags.csv       driver -> soil response lags by depth
      feature_dictionary.csv       name, group, description, units of every derived feature
```

Column names follow `<product>_<NEONtable>__<variable>__h<plot>_v<depth index>`. For example, `soil_temp_ST_30_minute__soilTempMean__h001_v502` is soil temperature in plot 1 at depth index 502. Real depths in metres are in `column_dictionary.txt`. All timestamps are UTC.

The data itself is not in this repository (it is regenerated by the downloader and excluded by `.gitignore`). Record where the downloaded `neon_data/` folder is kept in the checklist below.

## Decisions made and known issues

### Decisions

| Decision | Reason |
| --- | --- |
| Site SRER (Santa Rita, Arizona) | Semi-arid, hot climate: the closest NEON analogue to Egyptian field conditions |
| 30-min resolution, `basic` package | 1-min data is far larger with little gain for this use; basic package keeps the mean and quality flag |
| Mask values with final quality flag = 1 | Removes sensor faults before training; set `KEEP_FLAGGED = True` to keep them |
| Drop `StdErMean` (standard error) columns | Measurement uncertainty, not soil state; halved the column count; not available from the lab's own sensors |
| Download year by year | Keeps RAM low on multi-year runs; one failed year does not lose the rest |
| Released data only by default | Reproducible and citable; provisional months may still change |
| CONFIG block + IDE Run instead of command line | Easier to run without a terminal; command-line flags still work and override the block |

### Known issues and limitations

- **Derived features untested on real data.** `compute_derived_features.py` was verified on synthetic NEON-format data (it recovered Q10, thermal diffusivity, drydown τ and lags correctly). Real NEON variable names or metadata may differ. Check the run log for `[fail]` lines.
- **No health labels.** NEON has no soil-health or disease labels, so it supports forecasting, anomaly detection, pretraining and pipeline testing, not direct supervised training of the health classifier.
- **Gaps and uneven start dates.** Products began at different times and have maintenance gaps. Trim to a window where the chosen channels overlap (see coverage in `column_dictionary.txt`).
- **Sensor depth changes.** Metadata is saved once per product-year and overwritten each year, so the dictionary shows the latest sensor positions. Earlier reinstallations at other depths may be missed.
- **Salinity is an index.** `VSICMean` is an ion-content index from the water sensor, not calibrated EC in dS/m.
- **Soil property defaults.** CO₂ flux uses bulk density 1.5 g/cm³ by default, and plant-available water is skipped until field capacity and wilting point are set.
- **Large files.** `wide_30min.csv` is several GB; use the parquet file.
- **Timestamps are UTC.** Convert the lab's own sensor timestamps to UTC before aligning with NEON.

## Related documents and next steps

- [DATASET.md](DATASET.md): what each parameter means, data quality notes, and the formula for every derived feature. A PDF copy is [NEON_SRER_Soil_Weather_Dataset.pdf](NEON_SRER_Soil_Weather_Dataset.pdf).
- NEON data portal and API docs: data.neonscience.org (product IDs are listed in the downloader).

### Handover checklist

- [ ] Copy (or record the location of) the downloaded `neon_data/` folder
- [ ] Confirm `NEON_TOKEN` is blank in every copy of the downloader
- [ ] Share this repository with the supervisor / next owner
- [ ] Note who takes over the soil-health classification work

### Suggested next steps for whoever continues

1. Run `compute_derived_features.py` on the real SRER data and fix any `[fail]` groups.
2. Set soil properties from the NEON megapit data (bulk density, field capacity, wilting point) and rerun.
3. Pick a training window with more than 80% coverage for the chosen channels; average plots per depth or keep the best-covered plot.
4. Map each of the lab's 26 sensors to its closest NEON channel or derived feature, and keep scale-free features (relative wetness, anomalies, slopes) for transfer.
5. Use NEON for pretraining or pipeline testing, then train the health classifier on the lab's labelled data.
