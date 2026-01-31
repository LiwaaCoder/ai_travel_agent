import httpx

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


async def fetch_pois(city: str, preferences: str | None = None) -> list[str]:
    """Fetch POIs from Overpass/OSM - fast with tight timeout."""
    query = (
        f"[out:json][timeout:5];"
        f"area['name'='{city}']->.a;"
        "(node(area.a)[tourism~'attraction|museum'];);"
        "out center 10;"
    )
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            data = resp.json()
        elements = data.get("elements", [])
        names = []
        for el in elements:
            name = el.get("tags", {}).get("name")
            if name:
                names.append(name)
            if len(names) >= 8:
                break
        return names
    except Exception:
        # Return fallback POIs on timeout
        return [f"{city} Old Town", f"{city} Cathedral", "Local Market", "City Park"]
