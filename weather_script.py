"""
NDVI Feature Extractor — Pomona, CA
====================================
Fetches for each date:
  - Daily weather (ERA5 via Open-Meteo Historical API):
      temperature mean/min/max, precipitation, solar radiation,
      wind speed, relative humidity, ET0, vapor pressure deficit
  - Soil properties (SoilGrids v2 REST API):
      pH, SOC, clay, sand, silt, bulk density, CEC

Outputs: ndvi_features_pomona.csv

Dependencies:
    pip install requests pandas

Citations:
  - Open-Meteo / ERA5: Hersbach et al. (2020). The ERA5 global reanalysis.
    Q. J. R. Meteorol. Soc., 146(730), 1999–2049.
    https://doi.org/10.1002/qj.3803
  - SoilGrids: Poggio et al. (2021). SoilGrids 2.0: producing soil information
    for the globe with quantified spatial uncertainty.
    SOIL, 7, 217–240. https://doi.org/10.5194/soil-7-217-2021
"""

import requests
import pandas as pd
import time

# ── Config ────────────────────────────────────────────────────────────────────
LAT  =  34.0219
LON  = -117.8653
LOCATION = "Walnut_CA"

DATE_MAP = {
    1: "2024-08-08",
    2: "2024-08-15",
    3: "2024-09-13",
    4: "2024-09-20",
    5: "2024-10-04",
    6: "2024-10-18",
    7: "2024-11-08",
    8: "2024-11-15",
}

# ── Step 1: Fetch ERA5 daily weather via Open-Meteo ───────────────────────────
def fetch_weather(dates: list[str]) -> pd.DataFrame:
    """
    Fetches ERA5 reanalysis daily data from Open-Meteo Historical API.
    One API call covering the full date range.
    """
    print("Fetching ERA5 weather data from Open-Meteo...")

    start = min(dates)
    end   = max(dates)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "start_date": start,
        "end_date":   end,
        "daily": ",".join([
            "temperature_2m_mean",         # Mean air temp at 2m (°C)
            "temperature_2m_min",           # Min air temp at 2m (°C)
            "temperature_2m_max",           # Max air temp at 2m (°C)
            "precipitation_sum",            # Total precipitation (mm)
            "shortwave_radiation_sum",      # Solar radiation (MJ/m²)
            "windspeed_10m_max",            # Max wind speed at 10m (km/h)
            "windgusts_10m_max",            # Max wind gusts (km/h)
            "relative_humidity_2m_mean",    # Mean relative humidity (%)
            "et0_fao_evapotranspiration",   # Reference ET0 (mm) — key NDVI driver
            "vapor_pressure_deficit_max",   # VPD max (kPa) — plant stress indicator
        ]),
        "timezone": "America/Los_Angeles",
        "models": "era5"   # force ERA5 reanalysis
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "daily" not in data:
        raise ValueError(f"Unexpected response: {data}")

    df = pd.DataFrame(data["daily"])
    df.rename(columns={"time": "date"}, inplace=True)

    # Filter to only our target dates
    df = df[df["date"].isin(dates)].reset_index(drop=True)

    # Rename columns to be descriptive
    df.rename(columns={
        "temperature_2m_mean":        "tavg_c",
        "temperature_2m_min":         "tmin_c",
        "temperature_2m_max":         "tmax_c",
        "precipitation_sum":          "precip_mm",
        "shortwave_radiation_sum":    "srad_mj_m2",
        "windspeed_10m_max":          "wind_max_kmh",
        "windgusts_10m_max":          "wind_gusts_kmh",
        "relative_humidity_2m_mean":  "rh_pct",
        "et0_fao_evapotranspiration": "et0_mm",
        "vapor_pressure_deficit_max": "vpd_max_kpa",
    }, inplace=True)

    print(f"  ✓ Retrieved weather for {len(df)} dates")
    return df


# ── Step 2: Fetch SoilGrids static properties ─────────────────────────────────
def fetch_soilgrids() -> dict:
    """
    Fetches soil properties at 0-5cm depth from SoilGrids v2 REST API.
    Returns a dict of {variable: value} — same for all dates (soil is static).

    Conversion factors applied (SoilGrids stores as integers with scale factor):
      phh2o:  /10     → actual pH
      soc:    /10     → g/kg
      clay:   /10     → g/kg (≈ %)
      sand:   /10     → g/kg
      silt:   /10     → g/kg
      bdod:   /100    → cg/cm³
      cec:    /10     → mmol(c)/kg
      nitrogen: /100  → cg/kg
    """
    print("Fetching soil properties from SoilGrids...")

    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    properties = ["phh2o", "soc", "clay", "sand", "silt", "bdod", "cec", "nitrogen"]

    params = []
    for p in properties:
        params.append(("property", p))
    params += [
        ("lon",   LON),
        ("lat",   LAT),
        ("depth", "0-5cm"),
        ("value", "mean"),
    ]

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Parse response
    scale_factors = {
        "phh2o":    0.1,
        "soc":      0.1,
        "clay":     0.1,
        "sand":     0.1,
        "silt":     0.1,
        "bdod":     0.01,
        "cec":      0.1,
        "nitrogen": 0.01,
    }
    units = {
        "phh2o":    "pH",
        "soc":      "g/kg",
        "clay":     "g/kg",
        "sand":     "g/kg",
        "silt":     "g/kg",
        "bdod":     "cg/cm3",
        "cec":      "mmol(c)/kg",
        "nitrogen": "cg/kg",
    }

    soil_vals = {}
    for layer in data["properties"]["layers"]:
        name = layer["name"]
        raw  = layer["depths"][0]["values"]["mean"]
        if raw is not None:
            val = round(raw * scale_factors.get(name, 1.0), 3)
        else:
            val = None
        col_name = f"soil_{name}_{units.get(name,'').replace('/','_per_').replace('(','').replace(')','')}"
        soil_vals[col_name] = val

    print(f"  ✓ Retrieved {len(soil_vals)} soil properties")
    return soil_vals


# ── Step 3: Combine and export ────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  NDVI Feature Extractor — {LOCATION}")
    print(f"  Lat: {LAT}, Lon: {LON}")
    print(f"  Dates: {len(DATE_MAP)}")
    print(f"{'='*55}\n")

    dates = list(DATE_MAP.values())

    # Fetch weather
    weather_df = fetch_weather(dates)

    # Fetch soil (static — same row repeated for all dates)
    soil_data = fetch_soilgrids()

    # Add image_id and soil columns
    date_to_id = {v: k for k, v in DATE_MAP.items()}
    weather_df.insert(0, "image_id", weather_df["date"].map(date_to_id))

    for col, val in soil_data.items():
        weather_df[col] = val

    # Add lat/lon
    weather_df["latitude"]  = LAT
    weather_df["longitude"] = LON

    # Sort by date
    weather_df.sort_values("image_id", inplace=True)

    # Save
    out_path = f"ndvi_features_{LOCATION}.csv"
    weather_df.to_csv(out_path, index=False)

    print(f"\n{'='*55}")
    print(weather_df.to_string(index=False))
    print(f"\n✓ Saved to: {out_path}")

    print("\nColumn descriptions:")
    descriptions = {
        "image_id":            "Your date index (1–8)",
        "date":                "Date (YYYY-MM-DD)",
        "tavg_c":              "Mean air temperature at 2m (°C) — ERA5",
        "tmin_c":              "Min air temperature at 2m (°C) — ERA5",
        "tmax_c":              "Max air temperature at 2m (°C) — ERA5",
        "precip_mm":           "Total precipitation (mm) — ERA5",
        "srad_mj_m2":          "Solar radiation sum (MJ/m²) — ERA5",
        "wind_max_kmh":        "Max wind speed at 10m (km/h) — ERA5",
        "wind_gusts_kmh":      "Max wind gusts (km/h) — ERA5",
        "rh_pct":              "Mean relative humidity (%) — ERA5",
        "et0_mm":              "Reference evapotranspiration (mm) — ERA5",
        "vpd_max_kpa":         "Max vapor pressure deficit (kPa) — ERA5",
        "soil_phh2o_pH":       "Soil pH in water (0–5cm) — SoilGrids",
        "soil_soc_g_per_kg":   "Soil organic carbon (g/kg, 0–5cm) — SoilGrids",
        "soil_clay_g_per_kg":  "Clay content (g/kg ≈ %, 0–5cm) — SoilGrids",
        "soil_sand_g_per_kg":  "Sand content (g/kg, 0–5cm) — SoilGrids",
        "soil_silt_g_per_kg":  "Silt content (g/kg, 0–5cm) — SoilGrids",
        "soil_bdod_cg_per_cm3":"Bulk density (cg/cm³, 0–5cm) — SoilGrids",
        "soil_cec_mmolc_per_kg":"Cation exchange capacity (mmol(c)/kg) — SoilGrids",
        "soil_nitrogen_cg_per_kg": "Total nitrogen (cg/kg, 0–5cm) — SoilGrids",
    }
    for col, desc in descriptions.items():
        print(f"  {col:<30} {desc}")

    print("\nCitations:")
    print("  ERA5/Open-Meteo: Hersbach et al. (2020). Q.J.R. Meteorol. Soc.")
    print("    https://doi.org/10.1002/qj.3803")
    print("  SoilGrids: Poggio et al. (2021). SOIL, 7, 217–240.")
    print("    https://doi.org/10.5194/soil-7-217-2021")

    return weather_df


if __name__ == "__main__":
    df = main()