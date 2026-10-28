# Literature Review: Time-Series Data in Soil and Plant Monitoring

Oct 15, 2026 · Anas Alsirafy

Sensor time series are well established for soil moisture forecasting. They are still emerging for detecting plant stress, soil health and nutrient status, where labelled data is scarce and models transfer poorly between sites.

This review covers 25 sources (2014–2026) on in-situ soil and plant sensor time series: data networks, forecasting, stress and disease detection, nutrient sensing, methods, and edge deployment. It is weighted toward multi-sensor soil-health classification on low-cost edge hardware. Remote-sensing imagery and leaf-image disease detection are out of scope.

Main findings:

- **Data:** environment data is abundant and standardised (ISMN: 2,842 stations; NEON; SAPFLUXNET: 2,714 plants). Plant-health labels exist only in small studies of 16 plants to one orchard.
- **Forecasting:** LSTM-family models are the established baseline for soil moisture. Attention, Transformer and GAN variants add modest gains, and foundation models win only at sub-daily horizons.
- **Classification:** feature-based gradient boosting and ROCKET-family classifiers match or beat deep networks on small sensor datasets at much lower cost.
- **Nutrient sensing:** low-cost NPK and EC probes are moisture-confounded and drift. Most results show R² below 0.5 against lab analysis, so calibration matters more than model choice.
- **Main weakness:** models degrade sharply on unseen sites, regions or classes, and few studies test this.
- **Edge deployment:** running models on the node cut latency and bandwidth about 4× in one orchard trial, but resource use and agronomic impact are rarely reported.

## Data sources

Research-grade soil and plant time series come from a few large networks; most published ML work trains on them or on small, self-built IoT deployments.

| Source | What it holds | Scale | Notes |
| --- | --- | --- | --- |
| [ISMN](https://hess.copernicus.org/articles/25/5749/2021/) (Dorigo et al., 2021) | In-situ soil moisture, plus soil/air temperature and precipitation at some stations | 71 networks, 2,842 stations; data from 1952 | Harmonised to hourly UTC; automated QC flags (range, geophysical consistency, spectral); ~900 stations near-real-time; >1,000 papers use it |
| [NEON soil sensors](https://www.neonscience.org/data-collection/soil-sensors) | Soil temperature (to 2 m), soil moisture (6–200 cm), soil CO₂ (2–20 cm), soil heat flux, plus near-surface radiation and humidity | 5 soil plots per terrestrial site | Sampled at 0.1 Hz, published as 1- and 30-min means; the dataset already downloaded for this project |
| [SAPFLUXNET](https://essd.copernicus.org/articles/13/2607/2021/) (Poyatos et al., 2021) | Sub-daily sap flow (plant transpiration) with drivers: radiation, VPD, soil water content (56% of datasets) | 202 datasets, 121 sites, 174 species, 2,714 plants, 1995–2018 | Mostly trees; the main open source for plant water-use time series |
| Self-built IoT nodes | Low-cost capacitive moisture, NPK/EC/pH probes, DHT air sensors | Usually 1 site, weeks to months | Most applied papers; small, noisy, rarely shared |
| Plant electrophysiology nodes (e.g. [PhytoNode](https://arxiv.org/abs/2604.28038)) | Stem electrical potential at ~10 Hz | Tens of plants, greenhouse | Emerging; direct plant-state signal rather than environment |

The pattern: environment-side data (soil, weather) is abundant and standardized, while plant-side data with health labels is scarce and small.

## Soil moisture and soil-state forecasting

Soil moisture forecasting is the most mature use of soil time series. LSTM-family models are the standard baseline. Attention and Transformer variants give modest, data-dependent gains, and physics-constrained models are still at the proof-of-concept stage.

| Study | Data | Models | Key result |
| --- | --- | --- | --- |
| [Fang et al., 2017](https://arxiv.org/abs/1707.06611) | SMAP satellite soil moisture with NLDAS-2 forcings, Noah land model output, soil properties (CONUS) | One continental LSTM vs lasso, autoregressive, 1-layer NN | LSTM test RMSE below 0.035 m³/m³ over most of CONUS and lowest error of all models. An early landmark that showed one LSTM can generalise across a continent. |
| [Wang et al., 2024](https://hess.copernicus.org/articles/28/917/2024/) (HESS) | ISMN in-situ soil moisture with NASA POWER weather; 4-day input window, 1/3/7-day horizons | 3 ML (RF, ELM, SVR) vs 10 DL (CNN, LSTM, Transformer, CNN-LSTM, attention LSTMs, GAN-LSTM) | LSTM beat all three ML models on accuracy and stability; feature-attention and GAN-LSTM improved further. RF was the most stable ML model, and all ML models degraded at 7 days. Code is open source. |
| [Wang & Zha, 2024](https://doaj.org/article/9682b0e6d3c24bf0b6ccec0f998be5d6) (Agric. Water Manag.) | Groundwater level + meteorology, shallow-groundwater area | Transformer, LSTM, coupled Transformer-LSTM | Average R² of 0.523 (Transformer) vs 0.485 (LSTM); coupling with LSTM improved stability. Groundwater was a key predictor. |
| [Bischof et al., 2024](https://meetingorganizer.copernicus.org/EGU24/EGU24-9931.html) (EGU abstract) | Long-term in-situ TDR soil moisture, Attert basins, Luxembourg | LSTM, with clustering and network analysis of the series | LSTMs promising for catchment soil water storage and dry-event early warning; the authors flag the limits of purely data-driven models. No metrics reported. |
| [Xie et al., 2024](https://arxiv.org/html/2403.08154v1) | Synthetic 3D soil, 75 sensor series with noise | Physics-informed NN with the Richards equation and van Genuchten model | Full-batch Adam gave a 0.5% relative error in pressure head. Shows physics constraints can be trained from sparse sensors, but tested on simulation only. |

Common design choices across this work:

- **Inputs:** lagged soil moisture plus precipitation, air temperature, radiation, humidity and wind. Soil temperature and groundwater help where available.
- **Horizons:** 1–7 days. Accuracy drops sharply beyond about 3 days for ML models without memory.
- **Evaluation:** mostly per-site or temporal holdout. Cross-site transfer (train on some stations, test on unseen ones) is reported far less often, and it matters most for a new deployment.

## Plant stress and disease detection from time series

This work splits into two approaches:

- **Environment-driven risk models:** weather data predicts the conditions under which disease develops. This is the older, larger and more operational body of work.
- **Plant-signal models:** sensors on the plant itself (electrical potential, stem diameter, sap flow) detect stress directly. This is newer, small-scale, and closer to soil-health classification.

### Environment-driven disease risk

| Study | Data | Method | Result |
| --- | --- | --- | --- |
| [Baker et al., 2014/2015](https://www.cambridge.org/core/journals/journal-of-agricultural-science/article/abs/improved-weatherbased-late-blight-risk-management-comparing-models-with-a-ten-year-forecast-archive/5FD6B54862FDE7CB47D018C54BC9A74A) (J. Agric. Sci.) | 10 years of hourly weather forecasts, 26 locations, Great Lakes region | Artificial neural network for potato late blight risk; new hourly-optimised model vs 2007 model | New model more accurate in nearly every month, lead time and region, especially under weather conducive to severe disease |
| [Solís & Rojas-Herrera, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8161455) (Biomimetics) | Coffee plantations, 6 regions, 13 weather variables at 15–30 min, 2016–2020 | XGBoost: regression vs per-step wet/dry classification | Leaf wetness duration error about 60 min/day. Hourly or per-step models beat daily aggregates, and 15-min sampling beat 30-min. When a region was held out of training, error rose above 120–350 min. |
| [Yuan et al., 2026](https://www.mdpi.com/2227-7390/14/2/225) (Mathematics) | Hourly weather, 54 stations, 6 Chinese provinces, 2016–2022; 18.7% blight-positive | PLB-GPT: GPT-2 backbone with conv encoder, RevIN, patching; vs ARIMA, LSTM, Informer, CARAH | 5-day prediction accuracy 0.875, F1 0.847; beat all baselines. One of the first pretrained-transformer applications to crop disease. |

### Plant-signal stress detection

| Study | Plants and sensor | Method | Result |
| --- | --- | --- | --- |
| [Buss, Aust & Hamann, 2026](https://arxiv.org/abs/2604.28038) (arXiv) | 16 greenhouse tomatoes, 4 irrigation levels; stem electrical potential at 10 Hz (PhytoNode) | ~700 tsfresh features with feature selection and gradient boosting; also CNN, InceptionTime, Mamba | 30-min window: 83% test accuracy, AUPRC 0.87, generalised to an unseen plant. Stress flagged around day 4 of deficit. Multiclass (over- vs under-watered) did not generalise. |
| [De Oliveira et al., 2023](https://journals.flvc.org/FLAIRS/article/view/133180) (FLAIRS) | IoT-sensed plant electrical signals (electrome) | Series reduction (PAA, SAX, MCB) then ANN, KNN, SVM | PAA+SAX with SVM best; shows compression before classification works for IoT nodes. No accuracy figures given. |
| [Hanan et al., 2026](https://cris.bgu.ac.il/en/publications/stress-detection-and-classification-using-dendrometer-based-sdvs--2/) (Agriculture) | Citrus under N, P, K deficiency and excess; hourly stem diameter (dendrometers) | Temporal features, hierarchical classifier (stress → deficit/excess → nutrient), temporal voting | Over 70% precision for the specific nutrient, about 90% with a modified grouping. Stable after 1–2 weeks of data; hierarchical beat flat classifiers. |

Two results recur across these studies:

- **Within-site models work; transfer is weak.** Both the leaf-wetness and electrophysiology studies saw performance drop on unseen regions or classes.
- **Days of history are needed.** Reliable stress classification needed between 30 minutes and 2 weeks of history. It improves with temporal aggregation or voting, not just bigger models.

## Soil health and nutrient monitoring over time

Continuous nutrient monitoring is the weakest link in the literature. Low-cost NPK, EC and pH probes give trends at best, depend strongly on soil moisture, and drift over time. Calibration and drift handling matter more than the choice of forecasting model.

| Study | Focus | Method | Result |
| --- | --- | --- | --- |
| [CIAT / Alliance Bioversity, 2025](https://alliancebioversityciat.org/publications-data/digital-transformation-accelerator-technical-report-probe-low-cost-portable-soil) (technical report) | 3 commercial low-cost NPK probes vs lab analysis, Guatemala (10 samples) and Colombia (63 samples) | Linear and polynomial regression against lab values; moisture test 0–40% | R² up to 0.25 (Guatemala), adjusted R² 0.05–0.48 (Colombia); potassium best (0.54), N and P weakest. NPK and EC readings strongly moisture-dependent; not recommended for fertiliser management. |
| [Lin et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647256) (Sensors) | Calibrating low-cost soil EC sensors, strawberry greenhouse | SensorTalk3: XGBoost + RF ensemble with linear meta-learner, trained on the edge; dual-sensor trigger for recalibration | MAPE cut from 11.2% to 3.2% and from 7.8% to 1.7%, beating a lookup table. Trains and infers on a Raspberry Pi 4 within the 20-s sampling interval. |
| [Aguiar & Barros, 2023](https://sol.sbc.org.br/index.php/sbesc/article/view/27185) (SBESC) | Custom soil pH sensor + IoT system | LSTM forecasting of periodic pH at the edge | Low error vs commercial sensors and lab methods, reported qualitatively; no numbers given |
| [Zheng & Paiva, 2021](https://arxiv.org/abs/2109.04356) (KDD AIoT workshop) | Drift in IoT chemical sensors (gas array, 36 months, 16 sensors) | Self-training, domain adaptation (DRCA, LDSP), mean-centering | Self-training made drift worse. Domain adaptation helped but was parameter-sensitive; mean-centering held 70–75% accuracy for about 18 months, then failed. |

Consequences for soil-health work:

- **Moisture is a confound.** Nutrient and EC readings should be modelled together with soil moisture and temperature, or normalised by them (as the salinity proxy in the NEON feature set does).
- **Drift needs its own handling.** Use reference-sensor recalibration on the edge, and track the input distribution over time. Simple self-training on unlabelled data can reinforce errors.
- **Soil biological activity is better measured than nutrients.** Research networks such as NEON track soil CO₂ profiles continuously, from which soil respiration is derived ([NEON](https://www.neonscience.org/data-collection/soil-sensors)). This gives a time-series proxy for soil biological health that low-cost nutrient probes cannot.

## Methods landscape

No single architecture dominates. What works depends on the task:

- **Classification:** feature-based and convolution-kernel methods (tsfresh + boosting, ROCKET family) match or beat deep networks at a fraction of the cost.
- **Forecasting:** LSTM-family models remain the default.
- **Foundation models:** help mainly at short horizons on new sites with little data.

| Method family | Typical use in soil/plant work | Strengths | Weaknesses | Evidence |
| --- | --- | --- | --- | --- |
| Hand-crafted features + gradient boosting (tsfresh, rolling stats, slopes; XGBoost, HGB, RF) | Stress and disease classification, leaf wetness, sensor calibration | Strong on small data, interpretable, cheap on edge devices | Feature design effort; windows must be chosen | Electrophysiology stress at 83% ([Buss et al., 2026](https://arxiv.org/abs/2604.28038)); leaf wetness ([Solís & Rojas-Herrera, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8161455)); EC calibration ([Lin et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647256)) |
| Random-convolution-kernel classifiers (ROCKET, MiniROCKET, MultiROCKET-Hydra) | General time-series classification; little used yet in soil work | Top accuracy on 112 UCR datasets with very low training cost; MiniROCKET more than 10× faster than ROCKET at equal accuracy | Less interpretable; mainly benchmarked on univariate, equal-length series | [Middlehurst et al., 2023/2024](https://arxiv.org/abs/2304.13029) |
| HIVE-COTE 2 (ensemble of representations) | Benchmark ceiling for classification | Joint best accuracy with MultiROCKET-Hydra | Slow; impractical for edge deployment | [Middlehurst et al.](https://arxiv.org/abs/2304.13029) |
| CNN / InceptionTime / LiteTime | Classification with enough data | H-InceptionTime is the best deep classifier; LiteTime matches InceptionTime at much lower cost | Needs more data; unstable across window sizes in small plant studies | [Middlehurst et al.](https://arxiv.org/abs/2304.13029); [Buss et al.](https://arxiv.org/abs/2604.28038) |
| LSTM / GRU and hybrids (CNN-LSTM, attention LSTM) | Soil moisture and temperature forecasting | Best-established forecaster; handles weather drivers and lags | Gains over RF shrink at short horizons; slower to train | [Wang et al., 2024](https://hess.copernicus.org/articles/28/917/2024/); [Fang et al., 2017](https://arxiv.org/abs/1707.06611) |
| Transformers (vanilla, Informer, patching) | Multistep forecasting, disease risk | Modest gains over LSTM (R² 0.52 vs 0.49); coupling with LSTM adds stability | Data-hungry; benefits vary by site | [Wang & Zha, 2024](https://doaj.org/article/9682b0e6d3c24bf0b6ccec0f998be5d6); [Yuan et al., 2026](https://www.mdpi.com/2227-7390/14/2/225) |
| Time-series foundation models (TimesFM, Chronos, Moirai), zero-shot | Forecasting at newly instrumented sites | TimesFM beat rule-based baselines by 21–76% at sub-daily horizons with no site training | Lost by 6–79% beyond one day; weaker on long periods, low SNR, many components | [Talaei et al., 2026](https://neurips.cc/virtual/2026/174613); [Datta Gupta](https://arxiv.org/abs/2501.00889) |
| Physics-informed / physics-constrained NNs | Soil water flow from sparse sensors | Respects the Richards equation; recovers soil hydraulic properties | So far mostly synthetic tests; sensitive to optimiser and training setup | [Xie et al., 2024](https://arxiv.org/html/2403.08154v1) |

### Gaps and missing data

Field series always have gaps, and how they are filled affects downstream results.

- **Short gaps:** interpolation is adequate.
- **Long gaps (months):** model-based imputation from co-measured variables beats univariate methods. In [Park et al.](https://arxiv.org/abs/2202.12441), an MLP using supporting variables beat ImputeTS on 12–15-month soil moisture gaps. A multivariate statistical method (mtsdi) still won some cases, and errors accumulated toward the end of the gap.
- **QC first:** network data such as ISMN flags out-of-range, dubious and missing values rather than deleting them ([Dorigo et al., 2021](https://hess.copernicus.org/articles/25/5749/2021/)). Masking flagged values measurably improved agreement with reference products.

## Edge and IoT deployment

Inference is moving onto field nodes, but training mostly stays in the cloud, and few systems are validated across seasons or sites. Running models on the edge clearly cuts latency and bandwidth. Agronomic benefit (yield, water saved) is rarely measured.

| Study | Setup | Result |
| --- | --- | --- |
| [Cambra Baseca et al., 2025](https://www.mdpi.com/1424-8220/25/22/7079) (Sensors) | Almond orchard deficit irrigation. Arduino Edge Control, soil moisture probes at 10–50 cm, weather station, LoRaWAN (ChirpStack), solar power. LSTM on daily weather, soil moisture and ET₀ predicts next-week water demand. | RMSE 0.30 mm/day, R² 0.985. Edge vs cloud: latency 450 → 103 ms, bandwidth 12.4 → 3.1 MB/day, about 20 mAh/day over 197 days. Single site, short period; water savings not measured. |
| [Lin et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647256) (Sensors) | Raspberry Pi 4, no GPU; EC sensor calibration ensemble | Trains and predicts on the node within the 20-s sampling interval (about 0.014 s per prediction); training on the edge is feasible for tree ensembles |
| [Samanta & Saha, 2026](https://arxiv.org/abs/2603.15085) (review of 28 deployments, 2023–2026) | Low-cost edge AI and TinyML in precision agriculture | Microcontrollers (ESP32, STM32) dominate. Quantisation is used in about half the works; pruning and hardware-aware architecture search are rare. Reporting of memory, latency and energy is inconsistent, field validation is limited, and open edge-ready soil datasets are scarce. |
| [Abushahla et al., 2025/2026](https://arxiv.org/abs/2508.15008) (survey) | Neural network quantisation for microcontrollers | INT8 cuts memory 4× vs FP32, and INT8 multiply-accumulate uses about 20× less energy. Post-training quantisation can cost accuracy on deeper models; quantisation-aware training reduces the loss. Typical TinyML footprint is 64–256 KB. |

Two practical points carry over to Raspberry Pi and Jetson deployments:

- **Split the pipeline by device.** Light feature extraction and tree ensembles can run on the sensor node or a Raspberry Pi, even with on-device retraining. Deep sequence models suit a Jetson-class gateway or need INT8 quantisation.
- **Plan for low bandwidth.** LoRa gives long range at low bandwidth. Sending features or predictions instead of raw 10-s samples is what makes the bandwidth and energy savings above possible.

## Gaps and opportunities

The clearest open space is labelled, multi-sensor soil-health classification that transfers across sites and runs on the edge. Most published work does only one of these.

| Gap in the literature | Evidence | Opportunity for a 26-sensor soil-health project |
| --- | --- | --- |
| Few labelled, multivariate soil/plant health datasets | Large networks (ISMN, NEON, SAPFLUXNET) have no health labels; plant-signal studies use 16 plants or one orchard ([Buss et al.](https://arxiv.org/abs/2604.28038), [Hanan et al.](https://cris.bgu.ac.il/en/publications/stress-detection-and-classification-using-dendrometer-based-sdvs--2/)) | A labelled 26-channel dataset is itself a contribution; publishing it (even partially) adds value |
| Weak cross-site transfer | Leaf-wetness error rose from ~60 to over 120–350 min on unseen regions ([Solís & Rojas-Herrera](https://pmc.ncbi.nlm.nih.gov/articles/PMC8161455)); multiclass stress failed to generalise ([Buss et al.](https://arxiv.org/abs/2604.28038)) | Pretrain on NEON/ISMN, fine-tune on own data. Use scale-free features (relative wetness, anomalies, slopes). Report leave-one-site or leave-one-plot-out results. |
| Strong classifiers not yet tried in soil work | ROCKET-family and HIVE-COTE 2 lead general benchmarks ([Middlehurst et al.](https://arxiv.org/abs/2304.13029)) but rarely appear in agricultural papers | Benchmark MiniROCKET/MultiROCKET and tsfresh + boosting against LSTM/InceptionTime; MiniROCKET is cheap enough for a Raspberry Pi |
| Low-cost nutrient sensors are unreliable and moisture-confounded | Low-cost NPK R² mostly below 0.5 vs lab analysis ([CIAT, 2025](https://alliancebioversityciat.org/publications-data/digital-transformation-accelerator-technical-report-probe-low-cost-portable-soil)) | Model nutrient channels jointly with moisture and temperature; add recalibration and drift detection on the edge ([Lin et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647256), [Zheng & Paiva](https://arxiv.org/abs/2109.04356)) |
| Foundation models only help at short horizons | TimesFM won at sub-daily horizons, lost beyond one day ([Talaei et al.](https://neurips.cc/virtual/2026/174613)) | Use them for short-term anomaly scores or cold-start at new sites, not as the main classifier |
| Edge studies rarely report resources or field impact | Inconsistent memory, latency and energy reporting; little multi-season validation ([Samanta & Saha](https://arxiv.org/abs/2603.15085)) | Report flash, RAM, latency and energy per inference on Raspberry Pi and Jetson Xavier NX, plus a multi-week field run |

A defensible study design that follows from this:

1. Build a shared feature module (rolling statistics, slopes, VPD, ET₀, relative wetness, anomalies) used in both training and on the device.
2. Pretrain or validate the pipeline on NEON SRER, then train on the labelled 26-sensor data.
3. Compare feature + boosting, MiniROCKET, and one deep model (InceptionTime/LiteTime or LSTM) under leave-one-plot-out evaluation.
4. Deploy the best model with INT8 quantisation where needed; send predictions, not raw data, over LoRa; log resource use.

## Sources

All pages below were opened for this review. Several are preprints or conference abstracts, marked as such; findings from those are less settled than journal papers.

**Data networks**

1. Dorigo, W. et al. (2021). [The International Soil Moisture Network: serving Earth system science for over a decade](https://hess.copernicus.org/articles/25/5749/2021/). Hydrology and Earth System Sciences, 25, 5749–5804.
2. Poyatos, R. et al. (2021). [Global transpiration data from sap flow measurements: the SAPFLUXNET database](https://essd.copernicus.org/articles/13/2607/2021/). Earth System Science Data, 13, 2607–2649.
3. NEON. [Soil sensors](https://www.neonscience.org/data-collection/soil-sensors). National Ecological Observatory Network.

**Soil moisture forecasting**

4. Fang, K., Shen, C., Kifer, D., Yang, X. (2017). [Prolongation of SMAP to spatio-temporally seamless coverage of continental US using a deep learning neural network](https://arxiv.org/abs/1707.06611). arXiv:1707.06611.
5. Wang, Y., Shi, L. et al. (2024). [A comprehensive study of deep learning for soil moisture prediction](https://hess.copernicus.org/articles/28/917/2024/). Hydrology and Earth System Sciences, 28, 917–943.
6. Wang, Y., Zha, Y. (2024). [Comparison of transformer, LSTM and coupled algorithms for soil moisture prediction in shallow-groundwater-level areas with interpretability analysis](https://doaj.org/article/9682b0e6d3c24bf0b6ccec0f998be5d6). Agricultural Water Management, 305, 109120.
7. Bischof, B., Zehe, E., Loritz, R. (2024). [Using neural networks for predicting soil water storage based in situ soil moisture observations](https://meetingorganizer.copernicus.org/EGU24/EGU24-9931.html). EGU General Assembly abstract EGU24-9931.
8. Xie, J., Yao, B., Jiang, Z. (2024). [The effect of different optimization strategies to physics-constrained deep learning for soil moisture estimation](https://arxiv.org/html/2403.08154v1). arXiv:2403.08154 (IISE Annual Conference 2023).

**Plant stress and disease**

9. Baker, K. M. et al. (2015). [Improved weather-based late blight risk management: comparing models with a ten year forecast archive](https://www.cambridge.org/core/journals/journal-of-agricultural-science/article/abs/improved-weatherbased-late-blight-risk-management-comparing-models-with-a-ten-year-forecast-archive/5FD6B54862FDE7CB47D018C54BC9A74A). Journal of Agricultural Science, 153(2), 245–256.
10. Solís, M., Rojas-Herrera, V. (2021). [Approaches for the prediction of leaf wetness duration with machine learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8161455). Biomimetics, 6(2), 29.
11. Yuan, P. et al. (2026). [PLB-GPT: Potato late blight prediction with generative pretrained transformer and optimizing](https://www.mdpi.com/2227-7390/14/2/225). Mathematics, 14(2), 225.
12. Buss, E., Aust, T., Hamann, H. (2026). [Early detection of water stress by plant electrophysiology: machine learning for irrigation management](https://arxiv.org/abs/2604.28038). arXiv:2604.28038 (preprint).
13. De Oliveira Jr, M., Sedrez, G., Cavalheiro, G. G. H. (2023). [ML-based plant stress detection from IoT-sensed reduced electromes](https://journals.flvc.org/FLAIRS/article/view/133180). FLAIRS Conference Proceedings.
14. Hanan, O., Sperling, O., Raveh, E., Shani, G. (2026). [Stress detection and classification using dendrometer-based SDVs in citrus trees](https://cris.bgu.ac.il/en/publications/stress-detection-and-classification-using-dendrometer-based-sdvs--2/). Agriculture, 16(6), 683.

**Soil nutrients, calibration and drift**

15. Alliance of Bioversity International and CIAT (2025). [Low-cost portable soil fertility sensors: myth or reality?](https://alliancebioversityciat.org/publications-data/digital-transformation-accelerator-technical-report-probe-low-cost-portable-soil) Technical report.
16. Lin, Y.-W., Lin, Y.-B., Chang, T. C.-Y., Lu, B.-X. (2023). [An edge transfer learning approach for calibrating soil electrical conductivity sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647256). Sensors, 23(21), 8710.
17. Aguiar, S., Barros, E. (2023). [A soil pH sensor and a based on time-series prediction IoT system for agriculture](https://sol.sbc.org.br/index.php/sbesc/article/view/27185). SBESC 2023, 108–113.
18. Zheng, H., Paiva, A. R. (2021). [Assessing machine learning approaches to address IoT sensor drift](https://arxiv.org/abs/2109.04356). KDD AIoT Workshop.

**Methods**

19. Middlehurst, M., Schäfer, P., Bagnall, A. (2023/2024). [Bake off redux: a review and experimental evaluation of recent time series classification algorithms](https://arxiv.org/abs/2304.13029). arXiv:2304.13029.
20. Park, J., Müller, J. et al. [Long-term missing value imputation for time series data using deep neural networks](https://arxiv.org/abs/2202.12441). arXiv:2202.12441.
21. Talaei, M., Goudarzi, M., Wong, V. N. L., Chades, I. (2026). [Zero-shot forecasting of in-situ soil moisture and temperature with a time-series foundation model](https://neurips.cc/virtual/2026/174613). NeurIPS 2026 Workshop on Tackling Climate Change with ML (poster).
22. Datta Gupta, S. [Evaluating time series foundation models on noisy periodic time series](https://arxiv.org/abs/2501.00889). arXiv:2501.00889.

**Edge deployment**

23. Cambra Baseca, C., Dionísio, R., Ribeiro, F., Metrôlho, J. (2025). [Edge-computing smart irrigation controller using LoRaWAN and LSTM for predictive controlled deficit irrigation](https://www.mdpi.com/1424-8220/25/22/7079). Sensors, 25(22), 7079.
24. Samanta, R., Saha, B. (2026). [Affordable precision agriculture: a deployment-oriented review of low-cost, low-power edge AI and TinyML](https://arxiv.org/abs/2603.15085). arXiv:2603.15085 (preprint).
25. Abushahla, H. A., Varam, D., Panopio, A. J. N., AlHajri, M. I. (2025). [Neural network quantization for microcontrollers: a comprehensive survey](https://arxiv.org/abs/2508.15008). arXiv:2508.15008.
