# SkyInfo – live weather, AQI and ML forecast

FastAPI backend + single-page UI. Live data comes from Open-Meteo (no API key).
The ML model (gradient boosting, mean + 10%/90% quantiles) trains automatically
per location on ~2 years of hourly history the first time a city is opened
(about a minute); until then the UI shows the weather-model forecast.

## Run locally
    ./run.sh            # or: pip install -r requirements.txt && uvicorn app.main:app --reload
    open http://localhost:8000

Pre-train a city: `python -m app.model 22.72 75.86`

## Put it live (free)
**Render:** push this folder to a GitHub repo -> render.com -> New -> Blueprint
(uses render.yaml) -> Deploy. You get a public https URL.
**Hugging Face Spaces:** new Space -> SDK "Docker" -> upload these files; add
`ENV PORT=7860` to the Dockerfile.
Free tiers use temporary disk, so models retrain after a restart.

## API
GET /api/forecast?lat=..&lon=..   GET /api/search?q=..   GET /health
