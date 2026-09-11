# Multisensor Data Fusion of UAV and Sentinel-2 for Precision Agriculture

**Undergraduate research — Ethan Ngo, advised by Dr. John Korah**
Department of Computer Science, California State Polytechnic University, Pomona

Satellite imagery covers whole regions but is coarse (Landsat 8 at 30 m, Sentinel-2 at 10–60 m). Drone imagery is sharp enough to see individual plants but is expensive and slow to collect, so it only ever covers a small plot. This project asks whether machine learning can bridge the two: **can we train a model on a small drone-surveyed area and use it to predict drone-resolution crop health across the much larger area that only satellites see?**

The target variable is **NDVI** (Normalized Difference Vegetation Index), the standard remote-sensing proxy for crop health, biomass, and water/disease stress.

## What was built

A regression pipeline that predicts high-resolution UAV NDVI from low-resolution satellite spectral bands:

- **Data collection** — Drone flights over a commercial strawberry field in Pomona, CA across 8 dates (Aug–Nov 2024), paired with matching Landsat 8 and Sentinel-2 scenes.
- **Preprocessing** — Orthomosaics built in Drone2Map, multispectral bands extracted, NDVI rasters computed for both UAV and satellite, satellite bands clipped to the field boundary, and pixels with NDVI < 0.2 (bare rock, soil, snow) masked out.
- **Resampling study** — Satellite pixels were resampled to the UAV grid three ways (**nearest neighbor**, **bilinear**, **cubic**) to test how interpolation choice affects downstream model accuracy.
- **Feature engineering** (`weather_script.py`) — Beyond raw bands, the pipeline derives vegetation indices (NDVI, GNDVI, NDWI, SAVI, EVI, RVI, VARI, NBR), cyclical date encodings, ERA5 daily weather from the Open-Meteo API (temperature, precipitation, solar radiation, humidity, ET₀, vapor pressure deficit), and SoilGrids soil properties (pH, organic carbon, clay/sand/silt, CEC).
- **Modeling** (`fusion.ipynb`, `new_fusion.ipynb`) — Linear Regression, Random Forest, Gradient Boosting, XGBoost, and stacked ensembles, evaluated with R² and RMSE. Splits were done both randomly and **spatially** (train on one half of the field, predict the unseen half) — the harder and more realistic test.

## Experiments and findings

| # | Setup | Result |
|---|-------|--------|
| 1 | Landsat 8 bands → UAV NDVI, 4 models × 3 resampling methods, 8 dates | Gradient Boosting + bilinear was best; RMSE 0.08 but **R² ≈ 0.1** — the model was essentially predicting the mean |
| 2 | Added vegetation indices, date, and weather features | R² 0.0599, RMSE 0.0810 — extra features alone did not fix it |
| 3 | Switched Landsat 8 → Sentinel-2 (10/20 m), downsampled UAV to match, random vs. spatial splits | More area and finer satellite resolution both improved results; tree-based models led |
| 4 | Swept UAV NDVI target resolution (0.5 m, 1 m, 3 m, 6 m) | **R² 0.5, RMSE 0.08** with Random Forest at 3 m — a 5× improvement over Experiment 1 |

The headline result is that **resolution ratio, not model choice, was the binding constraint.** Predicting NDVI at the drone's native 0.1–0.5 m from a 30 m satellite pixel is close to hopeless — the satellite simply doesn't carry that information. Aggregating the UAV target to ~3 m made the problem learnable, and accuracy plateaued beyond that, pointing to an **optimal fusion resolution** rather than "finer is always better."

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · GDAL/raster workflows · ArcGIS Drone2Map · Landsat 8 · Sentinel-2 · Open-Meteo (ERA5) API · SoilGrids API · Jupyter

## Repository layout

```
fusion.ipynb                    Experiments 1 & 3-4: model × resampling comparison, spatial splits, error maps
new_fusion.ipynb                Experiment 2: expanded feature set (indices + date + weather)
weather_script.py               Fetches ERA5 weather and SoilGrids soil features per flight date
ndvi_features_Walnut_CA.csv     Generated per-date weather/soil feature table
requirements.txt                Python dependencies
```

## Running it

```bash
pip install -r requirements.txt
python weather_script.py          # regenerates the weather/soil feature table
jupyter notebook fusion.ipynb
```

> **Note on data:** the UAV and satellite raster tables are several gigabytes and are not committed here. The notebooks read them from a local `Agriculture Data/` directory (`Bilinear Tables/`, `Cubic Tables/`, `Nearest Tables/`); update the `base` path at the top of each notebook to point at your copy.

## Future work

- Systematically sweep UAV NDVI resolution to pin down the optimal fusion ratio
- Extend to a study area larger than a single field, across more dates
- Incorporate bioclimatic variables and a Digital Elevation Model

## Key references

1. Allu & Mesapam, *Fusion of satellite and UAV imagery for crop monitoring*, ISPRS Annals, 2025.
2. Allu & Mesapam, *Impact of remote sensing data fusion on agriculture applications: A review*, European Journal of Agronomy, 2025.
3. Toosi et al., *Toward the optimal spatial resolution ratio for fusion of UAV and Sentinel-2 satellite imageries*, Advances in Space Research, 2025.
4. Maimaitijiang et al., *Crop monitoring using satellite/UAV data fusion and machine learning*, Remote Sensing, 2020.
