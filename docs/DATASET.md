# NEON SRER Soil & Weather Dataset — Description and Derived Features

Oct 2, 2026 · Anas Alsirafy

## Overview

The dataset is a multivariate, 30-minute soil and weather time series from one NEON site, cleaned and pivoted into one wide table for ML.

| Item | Value |
| --- | --- |
| Source | US National Ecological Observatory Network (NEON), released data via the NEON API |
| Site | SRER — Santa Rita Experimental Range, Arizona (semi-arid desert grassland/shrubland) |
| Time span | Full available record per product ("auto" mode); exact start/end per channel in column_dictionary.txt |
| Resolution | 30-minute averages (totals for precipitation), UTC timestamps |
| Spatial layout | 5 soil plots, sensors at several depths per plot; weather sensors on the site tower |
| Products | 12 continuous sensor products + periodic soil lab chemistry |
| Labels | None — no soil health or disease labels |

Pipeline: `neon_soil_downloader.py` (download, QC masking, pivot, merge) → `drop_empty_columns.py` (remove empty channels) → StdErMean columns removed → `make_column_dictionary.py` (column descriptions, depths, coverage) → `compute_derived_features.py` (derived features).

## Files and structure

Train from `wide_30min.parquet`; everything else supports or explains it.

| File | Contents |
| --- | --- |
| `wide_30min.parquet` / `.csv` | All products merged on one regular 30-min UTC grid; one column per sensor channel |
| `per_product/*.csv` | Same wide format, one file per product, own date range |
| `metadata/*__sensor_positions_*.csv` | Plot and depth/height (zOffset, m) for each HOR.VER position |
| `metadata/*__variables_*.csv` | Variable descriptions and units |
| `metadata/*__issueLog_*.csv` | NEON's known sensor issues and outages |
| `chemistry/*.csv` | Periodic lab soil samples: pH, moisture, C and N |
| `column_dictionary.txt` / `.csv` | Readable description, units, plot, depth and coverage per column |
| `download_report.csv` / `dropped_columns_log.csv` | What downloaded, failed, or was removed |

Column names follow `<product>_<NEONtable>__<variable>__h<plot>_v<depth index>`. Example: `soil_temp_ST_30_minute__soilTempMean__h001_v502` = soil temperature, plot 1, depth index 502. Weather columns have no plot/depth suffix.

## Parameters measured

Twelve sensor products give about 20 physical variables; soil variables repeat across 5 plots and several depths, giving most of the columns.

| Product (NEON ID) | Variables (column) | Units | Position |
| --- | --- | --- | --- |
| Soil temperature (DP1.00041.001) | soilTempMean | °C | 5 plots × up to 9 depths, to ~2 m |
| Soil water & salinity (DP1.00094.001) | VSWCMean (water content), VSICMean (ion content, salinity proxy) | m³/m³; index | 5 plots × up to 8 depths |
| Soil CO₂ (DP1.00095.001) | soilCO2concentrationMean | ppm | 5 plots × 3 shallow depths |
| Soil heat flux (DP1.00040.001) | SHFMean | W/m² | 5 plots, ~8 cm below surface |
| IR surface temperature (DP1.00005.001) | bioTempMean | °C | Soil/canopy surface, tower and plots |
| Air temperature (DP1.00003.001) | tempTripleMean | °C | Tower top |
| Relative humidity (DP1.00098.001) | RHMean, tempRHMean, dewTempMean | %, °C | Tower |
| Precipitation (DP1.00006.001) | priPrecipBulk / secPrecipBulk / TFPrecipBulk | mm per 30 min | Gauges (availability varies) |
| Radiation (DP1.00023.001) | inSWMean, outSWMean, inLWMean, outLWMean | W/m² | Tower |
| PAR (DP1.00024.001) | PARMean | µmol/m²/s | Tower levels |
| Wind (DP1.00001.001) | windSpeedMean, windDirMean | m/s, degrees | Tower levels |
| Barometric pressure (DP1.00004.001) | staPresMean | kPa | Tower |
| Soil chemistry, periodic (DP1.10086.001) | pH (water, CaCl₂), moisture, C, N | various | Lab samples, a few times per year |

Exact depths, heights and units per column are in `column_dictionary.txt`; variable names above may differ slightly by NEON release.

## Data quality and caveats

The data is research grade but gappy; trim to a common window and handle NaNs before training.

- **QC masking:** values with NEON final quality flag = 1 were set to NaN. Remaining values passed NEON's automated checks.
- **Gaps:** maintenance, calibration, sensor swaps and late installation leave NaN runs. The coverage % per column is in `column_dictionary.txt`.
- **Uneven start dates:** products began at different times, so early rows are mostly empty. Pick the window where your chosen channels overlap.
- **Sensor moves:** some soil sensors were reinstalled at slightly different depths; the dictionary flags the latest position and any recorded range.
- **Time zone:** all timestamps are UTC. Arizona local time is UTC−7; Egypt is UTC+2 (UTC+3 in summer).
- **No labels:** there is no soil health or disease ground truth. Use the data for forecasting, anomaly detection, pretraining or pipeline testing.
- **Salinity index:** VSICMean is an ion-content index from the water sensor, not a calibrated EC in dS/m.
- **Chemistry:** pH and C/N are sparse lab points, not a time series.

## Derived and calculated features

The raw channels support about 40 physically meaningful derived features, from vapour pressure deficit to soil water storage and thermal diffusivity. T = temperature in °C, P = pressure in kPa, θ = volumetric water content, z = depth in m.

### Atmosphere and humidity

| Feature | Calculation | Inputs | Why it matters |
| --- | --- | --- | --- |
| Saturation vapour pressure eₛ | FAO-56 formula below | tempTripleMean | Basis for all humidity features |
| Actual vapour pressure eₐ | eₛ × RH / 100 | RHMean, air temp | Absolute moisture in air |
| Vapour pressure deficit (VPD) | eₛ − eₐ | RH, air temp | Drives evaporation and plant water stress |
| Specific humidity q | 0.622 eₐ / (P − 0.378 eₐ) | RH, temp, staPresMean | Pressure-independent moisture measure |
| Dew point depression | T − T_dew | tempRHMean, dewTempMean | Condensation and dew risk on leaves |
| Psychrometric constant γ | 0.000665 × P | staPresMean | Needed for evapotranspiration |
| Growing degree days | Σ max(0, T_daily_mean − T_base), base often 10 °C | air temp | Crop and pathogen development stage |

**Formula:** eₛ(T) = 0.6108 × exp( 17.27 T / (T + 237.3) ), in kPa

### Radiation and surface energy balance

| Feature | Calculation | Inputs | Why it matters |
| --- | --- | --- | --- |
| Net shortwave | inSW − outSW | inSWMean, outSWMean | Energy absorbed by the surface |
| Albedo | outSW / inSW (daytime only, inSW > 50 W/m²) | inSW, outSW | Changes with soil wetness and vegetation cover |
| Net longwave | inLW − outLW | inLWMean, outLWMean | Night cooling, cloud effect |
| Net radiation Rₙ | net SW + net LW | all 4 radiation channels | Main energy input to soil and plants |
| Radiometric surface temperature | (outLW / (εσ))^0.25 − 273.15, ε ≈ 0.95 | outLWMean | Cross-check against IR surface temperature |
| Available energy | Rₙ − G | Rₙ, SHFMean | Energy left for evaporation and heating air |
| Ground heat fraction | G / Rₙ | Rₙ, SHFMean | Indicates soil dryness and vegetation cover |
| PAR fraction | PAR / inSW | PARMean, inSW | Light quality for photosynthesis |
| Day/night flag, solar elevation | from timestamp + site lat/lon, or inSW > 10 W/m² | timestamp, inSW | Separates day and night behaviour |
| Clearness index | inSW / extraterrestrial radiation | inSW, solar geometry | Cloud cover proxy |

### Evapotranspiration

| Feature | Calculation | Inputs | Why it matters |
| --- | --- | --- | --- |
| Wind at 2 m, u₂ | u_z × 4.87 / ln(67.8 z − 5.42) | windSpeedMean, sensor height z | Standardises wind for ET formulas |
| Reference ET₀ (FAO-56 Penman–Monteith, hourly) | formula below, summed to daily | Rₙ, G, T, VPD, u₂, γ | Atmospheric water demand; irrigation scheduling |
| Priestley–Taylor ET | 1.26 × Δ / (Δ + γ) × (Rₙ − G) / λ | Rₙ, G, T, P | Simpler ET when wind is missing |

**Formula:** ET₀ = [ 0.408 Δ (Rₙ − G) + γ × 37 / (T + 273) × u₂ (eₛ − eₐ) ] / [ Δ + γ (1 + 0.34 u₂) ]

Hourly form, with Rₙ and G in MJ/m²/h (W/m² × 0.0036); Δ is the slope of the eₛ curve; the 0.34 coefficient becomes 0.24 by day and 0.96 by night in the ASCE hourly standard.

### Soil water and water balance

| Feature | Calculation | Inputs | Why it matters |
| --- | --- | --- | --- |
| Profile water storage S | Σ θᵢ × Δzᵢ × 1000 (mm) | VSWCMean at all depths, layer thicknesses | Total water available in the root zone |
| Storage change ΔS | S(t) − S(t−1) | S | Wetting and drying events |
| Water balance residual | P − ΔS ≈ ET + drainage + runoff | precip, S | Estimates water loss rate |
| Relative soil wetness | (θ − θ_min) / (θ_max − θ_min), per depth | VSWCMean history | Scale-free; transfers across sensors and sites |
| Plant-available water | (θ − θ_wp) / (θ_fc − θ_wp) | VSWCMean + field capacity and wilting point (megapit texture) | Crop water stress |
| Wetting-front arrival lag | time from rain start to θ rise at each depth | precip, VSWCMean by depth | Infiltration speed; soil structure |
| Drydown time constant τ | fit θ(t) = θ_res + A e^(−t/τ) after rain | VSWCMean | Drainage and evaporation behaviour |
| Antecedent precipitation index | API_t = k × API_(t−1) + P_t, k ≈ 0.9 per day | precip | Memory of recent rain |
| Days since last rain, cumulative rain | event counters, rolling sums (24 h, 7 d, 30 d) | precip | Simple, strong predictors |
| Pore-water salinity proxy | VSIC / θ | VSICMean, VSWCMean | Salt concentrates as soil dries |

### Soil temperature and heat

| Feature | Calculation | Inputs | Why it matters |
| --- | --- | --- | --- |
| Vertical gradient dT/dz | (T₁ − T₂) / (z₂ − z₁) | soilTempMean at adjacent depths | Direction of heat flow |
| Diurnal range per depth | daily max − daily min | soilTempMean | Damps with depth and with wetter soil |
| Phase lag between depths | time of daily max at depth 2 − depth 1 | soilTempMean | Heat transport speed |
| Thermal diffusivity κ | amplitude method below (or phase-lag method) | daily amplitudes at two depths | Changes with soil moisture; a soil state indicator |
| Surface–soil difference | IR surface temp − shallowest soil temp | bioTempMean, soilTempMean | Surface heating and drying |
| Surface–air difference | IR surface temp − air temp | bioTempMean, tempTripleMean | Canopy water stress (crop water stress index idea) |

**Formula:** κ = ω (z₂ − z₁)² / ( 2 [ ln(A₁ / A₂) ]² ), with ω = 2π / 86400 s and A = daily temperature amplitude at each depth

### Soil CO₂ and biological activity

| Feature | Calculation | Inputs | Why it matters |
| --- | --- | --- | --- |
| CO₂ gradient dC/dz | between the 3 measured depths | soilCO2concentrationMean | Direction and strength of CO₂ production |
| Soil CO₂ efflux (flux-gradient) | F = −Dₛ dC/dz, Dₛ from air-filled porosity (Millington–Quirk) | CO₂ profile, θ, porosity (megapit bulk density) | Soil respiration estimate |
| Temperature sensitivity Q₁₀ | fit C = a × Q₁₀^(T/10) | CO₂, soil temp | Biological activity response to heat |
| Rain pulse response | CO₂ rise after rain on dry soil | CO₂, precip, θ | Microbial activation (Birch effect) |
| Moisture–CO₂ interaction | CO₂ binned by θ and T | CO₂, θ, T | Separates water and temperature limits |

### Encodings, aggregation and generic time-series features

| Feature | Calculation | Why it matters |
| --- | --- | --- |
| Wind components u, v | u = −ws sin(dir), v = −ws cos(dir) | Removes the 0°/360° jump in direction |
| Cyclic time | sin/cos of hour-of-day and day-of-year | Lets models learn daily and seasonal cycles |
| Cross-plot mean and spread | mean and std over the 5 plots per depth | Fewer columns; robust to one failed sensor |
| Rolling statistics | mean, std, min, max over 3 h, 24 h, 7 d | Smooths noise; captures trends |
| Slopes and differences | linear-fit slope over a window; first difference | Rate of drying, warming, salinity build-up |
| Daily aggregates | daily min, max, mean, range | Reduces 48 steps/day to compact features |
| Lags | values at t−1, t−48 (1 day), t−336 (1 week) | Persistence and memory |
| Climatology anomalies | z-score vs same hour and day-of-year | Flags unusual conditions, sensor faults |
| Cross-correlation lags | lag of max correlation between depths or drivers | Response time of soil to weather |

## Suggested uses and next steps

Without labels, the dataset is best for forecasting, anomaly detection and pretraining, then transfer to the labelled 26-sensor soil health data.

| Use | Target or method | Key features |
| --- | --- | --- |
| Soil moisture forecasting | θ or profile storage S, 1–48 h ahead | precip, API, ET₀, VPD, lags |
| Soil temperature forecasting | soilTempMean per depth | radiation, air temp, θ, cyclic time |
| Anomaly / sensor-fault detection | autoencoder or isolation forest on residuals | climatology anomalies, cross-plot spread |
| Self-supervised pretraining | masked-channel or next-step prediction | all channels; fine-tune on own labels |
| Stress-event detection (weak labels) | rule-based labels, e.g. low relative wetness + high VPD | relative wetness, VPD, surface–air difference |
| Pipeline and edge testing | run the full feature code on Raspberry Pi / Jetson | rolling stats, slopes, ET₀ (all cheap to compute online) |

To transfer to your own sensors, keep only features both datasets can produce, and prefer scale-free versions (relative wetness, anomalies, slopes) over raw values, since probes, soils and climate differ.

- [ ] Pick a training window where the chosen channels have more than 80% coverage
- [ ] Average plots per depth, or keep the plot with the best coverage
- [ ] Implement the derived features in one feature module shared by training and the edge device
- [ ] Pull field capacity, wilting point and bulk density from the megapit data for plant-available water and CO₂ flux
- [ ] Map each of the 26 own sensors to its closest NEON channel or derived feature
