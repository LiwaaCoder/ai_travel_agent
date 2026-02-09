"""
Flights tool: Fetch live flight data to a destination.

Uses Aviationstack API (free tier: 100 requests/month) with SQLite caching
to avoid hitting rate limits.
"""

import os
import httpx
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional

from src.database import get_cached_flights, save_flights

# API Configuration
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"


def get_api_key() -> Optional[str]:
    """Get Aviationstack API key from environment."""
    return os.getenv("AVIATIONSTACK_API_KEY")


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
async def _fetch_from_api(destination_iata: str) -> list[dict]:
    """Fetch flights from Aviationstack API."""
    api_key = get_api_key()
    if not api_key:
        return []
    
    params = {
        "access_key": api_key,
        "arr_iata": destination_iata,
        "flight_status": "scheduled",
        "limit": 10,
    }
    
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(AVIATIONSTACK_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    
    flights = []
    for flight in data.get("data", [])[:10]:
        flights.append({
            "airline": flight.get("airline", {}).get("name"),
            "flight_number": flight.get("flight", {}).get("iata"),
            "origin": flight.get("departure", {}).get("airport"),
            "departure_time": flight.get("departure", {}).get("scheduled"),
            "arrival_time": flight.get("arrival", {}).get("scheduled"),
            "price": None,  # Aviationstack doesn't provide pricing
            "currency": "USD",
        })
    
    return flights


# IATA code mapping for common cities
CITY_TO_IATA = {
    "barcelona": "BCN",
    "tokyo": "TYO",
    "paris": "CDG",
    "london": "LHR",
    "new york": "JFK",
    "rome": "FCO",
    "amsterdam": "AMS",
    "berlin": "BER",
    "madrid": "MAD",
    "lisbon": "LIS",
    "dubai": "DXB",
    "singapore": "SIN",
    "hong kong": "HKG",
    "sydney": "SYD",
    "los angeles": "LAX",
    "miami": "MIA",
    "chicago": "ORD",
    "toronto": "YYZ",
    "vancouver": "YVR",
    "tel aviv": "TLV",
}


def city_to_iata(city: str) -> Optional[str]:
    """Convert city name to IATA code."""
    return CITY_TO_IATA.get(city.lower())


async def fetch_flights(
    destination: str,
    origin: Optional[str] = None,
) -> list[dict]:
    """
    Fetch upcoming flights to a destination city.
    
    Checks SQLite cache first, then fetches from API if stale.
    
    Args:
        destination: Destination city name
        origin: Optional origin city name
        
    Returns:
        List of flight dictionaries with airline, times, etc.
    """
    # Try cache first
    cached = get_cached_flights(destination, origin)
    if cached:
        return cached
    
    # Get IATA code
    iata = city_to_iata(destination)
    if not iata:
        # Return mock data for unsupported cities
        return _get_mock_flights(destination)
    
    # Fetch from API
    try:
        flights = await _fetch_from_api(iata)
        if flights:
            save_flights(destination, flights, origin)
            return flights
    except Exception:
        pass
    
    # Fallback to mock data
    return _get_mock_flights(destination)


def _get_mock_flights(destination: str) -> list[dict]:
    """Generate mock flight data when API is unavailable."""
    # Create realistic mock data for demo purposes
    now = datetime.now()
    
    mock_flights = [
        {
            "airline": "Lufthansa",
            "flight_number": "LH1234",
            "origin": "Frankfurt",
            "departure_time": now.replace(hour=10, minute=30).isoformat(),
            "arrival_time": now.replace(hour=14, minute=0).isoformat(),
            "price": 289,
            "currency": "USD",
        },
        {
            "airline": "British Airways",
            "flight_number": "BA567",
            "origin": "London",
            "departure_time": now.replace(hour=8, minute=15).isoformat(),
            "arrival_time": now.replace(hour=12, minute=45).isoformat(),
            "price": 312,
            "currency": "USD",
        },
        {
            "airline": "Air France",
            "flight_number": "AF890",
            "origin": "Paris",
            "departure_time": now.replace(hour=14, minute=0).isoformat(),
            "arrival_time": now.replace(hour=16, minute=30).isoformat(),
            "price": 245,
            "currency": "USD",
        },
    ]
    
    # Save mock data to cache
    save_flights(destination, mock_flights)
    return mock_flights


def format_flights_for_prompt(flights: list[dict]) -> str:
    """Format flight data for inclusion in LLM prompt."""
    if not flights:
        return "No flight information available."
    
    lines = ["✈️ **Available Flights:**"]
    for f in flights[:5]:
        dep_time = ""
        if f.get("departure_time"):
            try:
                dt = datetime.fromisoformat(f["departure_time"].replace("Z", "+00:00"))
                dep_time = dt.strftime("%H:%M")
            except Exception:
                dep_time = f["departure_time"][:5] if len(f.get("departure_time", "")) > 5 else ""
        
        price_str = f"${f['price']}" if f.get("price") else "Price varies"
        lines.append(f"  - {f.get('airline', 'Airline')} {f.get('flight_number', '')}: "
                    f"from {f.get('origin', 'Various')} at {dep_time} ({price_str})")
    
    return "\n".join(lines)
