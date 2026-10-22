"""Live data from Open-Meteo (free, no API key)."""
from datetime import date, timedelta
import httpx

GEO = "https://geocoding-api.open-meteo.com/v1/search"
FC = "https://api.open-meteo.com/v1/forecast"
AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
AR = "https://archive-api.open-meteo.com/v1/archive"
FEATS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "cloud_cover", "wind_speed_10m"]


def _get(url, params):
    r = httpx.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def geocode(q):
    j = _get(GEO, {"name": q, "count": 6, "language": "en"})
    return [{"name": x["name"], "admin": x.get("admin1", ""), "country": x.get("country", ""),
             "lat": x["latitude"], "lon": x["longitude"]} for x in j.get("results", [])]


def forecast(lat, lon):
    return _get(FC, {
        "latitude": lat, "longitude": lon, "timezone": "auto", "past_days": 2, "forecast_days": 7,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,"
                   "surface_pressure,cloud_cover,precipitation,is_day",
        "hourly": ",".join(FEATS) + ",precipitation_probability",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                 "sunrise,sunset,uv_index_max"})


def air(lat, lon):
    return _get(AQ, {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 1,
        "current": "pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,us_aqi",
        "hourly": "pm2_5"})


def history(lat, lon, days=730):
    end = date.today() - timedelta(days=3)
    start = end - timedelta(days=days)
    j = _get(AR, {"latitude": lat, "longitude": lon, "timezone": "auto",
                  "start_date": start.isoformat(), "end_date": end.isoformat(),
                  "hourly": ",".join(FEATS)})
    return j["hourly"]
