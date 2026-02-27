from typing import Tuple, Optional
import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def fetch_weather_summary(city: str) -> str:
    """Fetch weather summary - fast with tight timeout and fallback."""
    try:
        lat, lon = await _geocode_city(city)
        if lat is None or lon is None:
            return "Check local forecast"

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        }
        async with httpx.AsyncClient(timeout=4) as client:
            resp = await client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        days = data.get("daily", {})
        temps_max = days.get("temperature_2m_max", [])
        temps_min = days.get("temperature_2m_min", [])
        if not temps_max or not temps_min:
            return "Check local forecast"
        return f"Temp: {min(temps_min):.0f}-{max(temps_max):.0f}°C"
    except Exception:
        return "Check local forecast"


async def _geocode_city(city: str) -> Tuple[Optional[float], Optional[float]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(url, params={"name": city, "count": 1})
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results") or []
        if not results:
            return None, None
        first = results[0]
        return first.get("latitude"), first.get("longitude")
    except Exception:
        return None, None
