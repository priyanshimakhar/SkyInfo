import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import data, model

app = FastAPI(title="SkyInfo")
STATIC = Path(__file__).resolve().parent.parent / "static"
_cache = {}


def wc(c):
    """WMO weather code -> (label, emoji, gradient index)."""
    if c in (0, 1): return ("Clear sky", "☀️", 0)
    if c == 2: return ("Partly cloudy", "⛅", 1)
    if c == 3: return ("Overcast", "☁️", 2)
    if c in (45, 48): return ("Fog", "🌫️", 2)
    if c <= 57: return ("Drizzle", "🌦️", 3)
    if c <= 67: return ("Rain", "🌧️", 3)
    if c <= 77: return ("Snow", "❄️", 2)
    if c <= 82: return ("Showers", "🌦️", 3)
    return ("Thunderstorm", "⛈️", 4)


PM25 = [(30, 50), (60, 100), (90, 200), (120, 300), (250, 400), (380, 500)]
PM10 = [(50, 50), (100, 100), (250, 200), (350, 300), (430, 400), (510, 500)]


def _sub(x, bp):
    c0 = i0 = 0
    for c1, i1 in bp:
        if x <= c1:
            return i0 + (x - c0) * (i1 - i0) / (c1 - c0)
        c0, i0 = c1, i1
    return 500


def naqi(p25, p10):
    """Indian National AQI (PM2.5 / PM10 sub-indices)."""
    return round(max(_sub(p25 or 0, PM25), _sub(p10 or 0, PM10)))


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/search")
def search(q: str):
    try:
        return data.geocode(q)
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/forecast")
def forecast(lat: float, lon: float):
    key = (round(lat, 2), round(lon, 2))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    try:
        fc = data.forecast(lat, lon)
    except Exception as e:
        raise HTTPException(502, f"Weather provider error: {e}")
    try:
        aq = data.air(lat, lon)
    except Exception:
        aq = None

    c, H, D = fc["current"], fc["hourly"], fc["daily"]
    hk = c["time"][:13] + ":00"
    i = H["time"].index(hk)
    d0 = D["time"].index(c["time"][:10])
    label, icon, g = wc(c["weather_code"])

    st = model.status(lat, lon)
    pred = model.predict(lat, lon, H, i) if st["ready"] else None
    hourly = []
    for k in range(24):
        row = {"h": H["time"][i + 1 + k][11:16], "nwp": H["temperature_2m"][i + 1 + k]}
        if pred:
            row.update(ml=pred["mean"][k], lo=pred["lo"][k], hi=pred["hi"][k])
        hourly.append(row)

    daily = [{"day": "Today" if k == 0 else time.strftime("%a", time.strptime(D["time"][d0 + k], "%Y-%m-%d")),
              "icon": wc(D["weather_code"][d0 + k])[1], "hi": D["temperature_2m_max"][d0 + k],
              "lo": D["temperature_2m_min"][d0 + k], "rain": D["precipitation_probability_max"][d0 + k] or 0}
             for k in range(7)]

    air = None
    if aq and aq.get("current"):
        a = aq["current"]
        trend = "stable"
        try:
            ah = aq["hourly"]["pm2_5"]
            j = aq["hourly"]["time"].index(hk)
            nxt = [x for x in ah[j + 1:j + 7] if x is not None]
            if nxt and a["pm2_5"]:
                ratio = (sum(nxt) / len(nxt)) / max(a["pm2_5"], 1)
                trend = "likely to worsen" if ratio > 1.15 else "likely to improve" if ratio < 0.85 else "stable"
        except Exception:
            pass
        air = {"naqi": naqi(a["pm2_5"], a["pm10"]), "us": a.get("us_aqi"), "trend": trend,
               "pol": [["PM2.5", a["pm2_5"], 150], ["PM10", a["pm10"], 300], ["NO₂", a["nitrogen_dioxide"], 100],
                       ["O₃", a["ozone"], 120], ["SO₂", a["sulphur_dioxide"], 60],
                       ["CO", (a["carbon_monoxide"] or 0) / 1000, 4]]}

    out = {"updated": c["time"][11:16], "ml": st, "hourly": hourly, "daily": daily, "aqi": air,
           "now": {"temp": c["temperature_2m"], "feels": c["apparent_temperature"], "label": label, "icon": icon,
                   "g": g, "hi": daily[0]["hi"], "lo": daily[0]["lo"], "rain": H["precipitation_probability"][i] or 0,
                   "wind": c["wind_speed_10m"], "hum": c["relative_humidity_2m"], "pres": c["surface_pressure"],
                   "cloud": c["cloud_cover"], "precip": c["precipitation"], "uv": D["uv_index_max"][d0],
                   "rise": D["sunrise"][d0][11:16], "set": D["sunset"][d0][11:16]}}
    if st["ready"]:
        _cache[key] = (time.time(), out)
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
