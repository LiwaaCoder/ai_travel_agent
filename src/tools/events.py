"""
Events tool: Fetch live event data (football matches, etc.) for a city.

Uses API-Football (free tier: 100 requests/day) with SQLite caching.
"""

import os
import httpx
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional

from src.database import get_cached_events, save_events

# API Configuration
FOOTBALL_API_URL = "https://v3.football-api.com/fixtures"
RAPIDAPI_FOOTBALL_URL = "https://api-football-v1.p.rapidapi.com/v3/fixtures"


def get_api_key() -> Optional[str]:
    """Get Football API key from environment."""
    return os.getenv("FOOTBALL_API_KEY")


# City to country/league mapping for football
CITY_TO_FOOTBALL = {
    "barcelona": {"country": "spain", "teams": ["Barcelona", "Espanyol"]},
    "madrid": {"country": "spain", "teams": ["Real Madrid", "Atletico Madrid"]},
    "london": {"country": "england", "teams": ["Arsenal", "Chelsea", "Tottenham", "West Ham"]},
    "manchester": {"country": "england", "teams": ["Manchester United", "Manchester City"]},
    "paris": {"country": "france", "teams": ["Paris Saint Germain"]},
    "rome": {"country": "italy", "teams": ["Roma", "Lazio"]},
    "milan": {"country": "italy", "teams": ["AC Milan", "Inter"]},
    "munich": {"country": "germany", "teams": ["Bayern Munich"]},
    "amsterdam": {"country": "netherlands", "teams": ["Ajax"]},
    "lisbon": {"country": "portugal", "teams": ["Benfica", "Sporting CP"]},
    "tel aviv": {"country": "israel", "teams": ["Maccabi Tel Aviv", "Hapoel Tel Aviv"]},
}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
async def _fetch_from_api(city: str) -> list[dict]:
    """Fetch upcoming fixtures from Football API."""
    api_key = get_api_key()
    if not api_key:
        return []
    
    city_info = CITY_TO_FOOTBALL.get(city.lower(), {})
    if not city_info:
        return []
    
    # Search for upcoming matches for teams in this city
    events = []
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    
    today = datetime.now().strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    async with httpx.AsyncClient(timeout=10) as client:
        for team in city_info.get("teams", [])[:2]:  # Limit to 2 teams
            params = {
                "team": team,
                "from": today,
                "to": next_week,
            }
            try:
                resp = await client.get(RAPIDAPI_FOOTBALL_URL, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                
                for fixture in data.get("response", [])[:3]:
                    events.append({
                        "name": f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}",
                        "venue": fixture.get("fixture", {}).get("venue", {}).get("name"),
                        "event_date": fixture.get("fixture", {}).get("date"),
                        "teams": [fixture["teams"]["home"]["name"], fixture["teams"]["away"]["name"]],
                        "league": fixture.get("league", {}).get("name"),
                    })
            except Exception:
                continue
    
    return events


async def fetch_events(
    city: str,
    event_type: str = "football",
) -> list[dict]:
    """
    Fetch upcoming events in a city.
    
    Checks SQLite cache first, then fetches from API if stale.
    
    Args:
        city: City name
        event_type: Type of event (currently only "football" supported)
        
    Returns:
        List of event dictionaries
    """
    # Try cache first
    cached = get_cached_events(city, event_type)
    if cached:
        return cached
    
    # Fetch from API
    try:
        if event_type == "football":
            events = await _fetch_from_api(city)
            if events:
                save_events(city, events, event_type)
                return events
    except Exception:
        pass
    
    # Fallback to mock data
    return _get_mock_events(city, event_type)


def _get_mock_events(city: str, event_type: str) -> list[dict]:
    """Generate mock event data when API is unavailable."""
    city_info = CITY_TO_FOOTBALL.get(city.lower(), {})
    teams = city_info.get("teams", ["Home Team", "Away Team"])
    
    now = datetime.now()
    
    mock_events = []
    
    if event_type == "football" and len(teams) >= 1:
        mock_events = [
            {
                "name": f"{teams[0]} vs Visiting Team",
                "venue": f"{city.title()} Stadium",
                "event_date": (now + timedelta(days=3)).isoformat(),
                "teams": [teams[0], "Visiting Team"],
                "league": "League Match",
            },
        ]
        
        if len(teams) >= 2:
            mock_events.append({
                "name": f"{teams[1]} vs Another Team",
                "venue": f"{city.title()} Arena",
                "event_date": (now + timedelta(days=5)).isoformat(),
                "teams": [teams[1], "Another Team"],
                "league": "League Match",
            })
    
    # Default fallback
    if not mock_events:
        mock_events = [
            {
                "name": f"Local Derby in {city.title()}",
                "venue": f"{city.title()} Stadium",
                "event_date": (now + timedelta(days=4)).isoformat(),
                "teams": ["Local Team A", "Local Team B"],
                "league": "Local League",
            }
        ]
    
    # Save mock data to cache
    save_events(city, mock_events, event_type)
    return mock_events


def format_events_for_prompt(events: list[dict]) -> str:
    """Format event data for inclusion in LLM prompt."""
    if not events:
        return "No upcoming events found."
    
    lines = ["⚽ **Upcoming Events:**"]
    for e in events[:5]:
        date_str = ""
        if e.get("event_date"):
            try:
                dt = datetime.fromisoformat(e["event_date"].replace("Z", "+00:00"))
                date_str = dt.strftime("%a %b %d, %H:%M")
            except Exception:
                date_str = e["event_date"][:10] if len(e.get("event_date", "")) > 10 else ""
        
        venue = e.get("venue", "TBD")
        league = e.get("league", "")
        league_str = f" ({league})" if league else ""
        
        lines.append(f"  - {e.get('name', 'Event')}: {date_str} at {venue}{league_str}")
    
    return "\n".join(lines)
