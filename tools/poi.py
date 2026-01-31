import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch_pois(city: str, preferences: str | None = None) -> list[str]:
    """Fetch a few POIs from Overpass/OSM; keep it lightweight."""
    query = (
        f"[out:json][timeout:10];"
        f"area['name'='{city}']->.a;"
        "(node(area.a)[tourism~'attraction|museum'];);"
        "out center 10;"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(OVERPASS_URL, data={"data": query})
        resp.raise_for_status()
        data = resp.json()
    elements = data.get("elements", [])
    names = []
    for el in elements:
        name = el.get("tags", {}).get("name")
        if name:
            names.append(name)
        if len(names) >= 10:
            break
    if preferences:
        # Simple preference filter substring check
        names = [n for n in names if preferences.lower() in n.lower()] or names
    return names
