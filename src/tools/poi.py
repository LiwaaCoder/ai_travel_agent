from typing import List, Optional
import httpx

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Preference to Overpass tag mapping
PREFERENCE_TAGS = {
    "tennis": "sport~'tennis'",
    "sports": "sport",
    "food": "amenity~'restaurant|cafe'",
    "restaurants": "amenity~'restaurant'",
    "cafe": "amenity~'cafe'",
    "art": "tourism~'museum|gallery|artwork'",
    "museum": "tourism~'museum'",
    "history": "historic",
    "architecture": "building~'cathedral|church|castle'",
    "shopping": "shop",
    "parks": "leisure~'park|garden'",
    "nature": "natural|leisure~'park|garden'",
    "beach": "natural~'beach'",
    "nightlife": "amenity~'bar|nightclub'",
}


async def fetch_pois(city: str, preferences: Optional[str] = None) -> List[str]:
    """Fetch POIs from Overpass/OSM - fast with preference-based filtering."""
    
    # Build query based on preferences
    tags = ["tourism~'attraction|museum'"]  # Default
    
    if preferences:
        pref_lower = preferences.lower()
        for keyword, tag in PREFERENCE_TAGS.items():
            if keyword in pref_lower:
                tags.append(tag)
    
    # Combine tags with OR logic
    tag_query = ";".join([f"node(area.a)[{tag}]" for tag in tags[:3]])  # Limit to 3 tag types
    
    query = (
        f"[out:json][timeout:3];"  # Reduced timeout
        f"area['name'='{city}']->.a;"
        f"({tag_query};);"
        "out center 12;"  # Get a few more results
    )
    
    try:
        async with httpx.AsyncClient(timeout=3.5) as client:  # Shorter timeout
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            data = resp.json()
        elements = data.get("elements", [])
        names = []
        for el in elements:
            name = el.get("tags", {}).get("name")
            if name and name not in names:  # Avoid duplicates
                names.append(name)
            if len(names) >= 10:
                break
        
        # Return results or fallback
        return names if names else [f"{city} Old Town", f"{city} Cathedral", "Local Market"]
    except Exception:
        # Quick fallback on timeout/error
        return [f"{city} Old Town", f"{city} Cathedral", "Local Market", "City Park"]
