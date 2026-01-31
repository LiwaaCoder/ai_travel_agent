import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch_weather_summary(city: str) -> str:
    """Fetch a simple weather summary for a city using Open-Meteo geocoding + forecast."""
    lat, lon = await _geocode_city(city)
    if lat is None or lon is None:
        return "Weather unavailable"

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    days = data.get("daily", {})
    temps_max = days.get("temperature_2m_max", [])
    temps_min = days.get("temperature_2m_min", [])
    if not temps_max or not temps_min:
        return "Weather unavailable"
    return f"Temp range next days: {min(temps_min):.0f}-{max(temps_max):.0f}°C"


async def _geocode_city(city: str) -> tuple[float | None, float | None]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params={"name": city, "count": 1})
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results") or []
    if not results:
        return None, None
    first = results[0]
    return first.get("latitude"), first.get("longitude")
