"""
Database module for caching live data (flights, events).

Uses SQLite with TTL-based cache invalidation to avoid
hitting API rate limits while keeping data fresh.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json

# Database location
DB_PATH = Path(__file__).parent.parent / "data" / "live_cache.db"

# Cache TTL settings (in seconds)
FLIGHTS_TTL = 3600  # 1 hour
EVENTS_TTL = 7200   # 2 hours


def get_connection() -> sqlite3.Connection:
    """Get SQLite connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Flights table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            destination TEXT NOT NULL,
            origin TEXT,
            airline TEXT,
            flight_number TEXT,
            departure_time TEXT,
            arrival_time TEXT,
            price REAL,
            currency TEXT DEFAULT 'USD',
            fetched_at TEXT NOT NULL,
            UNIQUE(destination, origin, flight_number, departure_time)
        )
    """)
    
    # Events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            event_type TEXT NOT NULL,
            name TEXT NOT NULL,
            venue TEXT,
            event_date TEXT,
            teams TEXT,  -- JSON array for sports matches
            league TEXT,
            fetched_at TEXT NOT NULL,
            UNIQUE(city, event_type, name, event_date)
        )
    """)
    
    # Index for faster lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_flights_dest ON flights(destination)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_city ON events(city)")
    
    conn.commit()
    conn.close()


def is_cache_valid(fetched_at: str, ttl_seconds: int) -> bool:
    """Check if cached data is still valid based on TTL."""
    fetch_time = datetime.fromisoformat(fetched_at)
    return datetime.now() - fetch_time < timedelta(seconds=ttl_seconds)


# =============================================================================
# Flights Cache Operations
# =============================================================================

def get_cached_flights(destination: str, origin: Optional[str] = None) -> list[dict]:
    """
    Get flights from cache if valid.
    
    Returns empty list if cache is stale or missing.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if origin:
        cursor.execute(
            "SELECT * FROM flights WHERE destination = ? AND origin = ? ORDER BY departure_time",
            (destination.lower(), origin.lower())
        )
    else:
        cursor.execute(
            "SELECT * FROM flights WHERE destination = ? ORDER BY departure_time",
            (destination.lower(),)
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    # Check if any row is still valid
    if not is_cache_valid(rows[0]["fetched_at"], FLIGHTS_TTL):
        return []
    
    return [dict(row) for row in rows]


def save_flights(destination: str, flights: list[dict], origin: Optional[str] = None) -> None:
    """Save flight data to cache."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Clear old data for this route
    if origin:
        cursor.execute(
            "DELETE FROM flights WHERE destination = ? AND origin = ?",
            (destination.lower(), origin.lower())
        )
    else:
        cursor.execute(
            "DELETE FROM flights WHERE destination = ?",
            (destination.lower(),)
        )
    
    # Insert new data
    fetched_at = datetime.now().isoformat()
    for flight in flights:
        cursor.execute("""
            INSERT OR REPLACE INTO flights 
            (destination, origin, airline, flight_number, departure_time, arrival_time, price, currency, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            destination.lower(),
            (origin or flight.get("origin", "")).lower(),
            flight.get("airline"),
            flight.get("flight_number"),
            flight.get("departure_time"),
            flight.get("arrival_time"),
            flight.get("price"),
            flight.get("currency", "USD"),
            fetched_at
        ))
    
    conn.commit()
    conn.close()


# =============================================================================
# Events Cache Operations
# =============================================================================

def get_cached_events(city: str, event_type: str = "football") -> list[dict]:
    """
    Get events from cache if valid.
    
    Returns empty list if cache is stale or missing.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM events WHERE city = ? AND event_type = ? ORDER BY event_date",
        (city.lower(), event_type.lower())
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    # Check if cache is still valid
    if not is_cache_valid(rows[0]["fetched_at"], EVENTS_TTL):
        return []
    
    # Parse teams JSON
    result = []
    for row in rows:
        event = dict(row)
        if event.get("teams"):
            event["teams"] = json.loads(event["teams"])
        result.append(event)
    
    return result


def save_events(city: str, events: list[dict], event_type: str = "football") -> None:
    """Save event data to cache."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Clear old data for this city/type
    cursor.execute(
        "DELETE FROM events WHERE city = ? AND event_type = ?",
        (city.lower(), event_type.lower())
    )
    
    # Insert new data
    fetched_at = datetime.now().isoformat()
    for event in events:
        teams = event.get("teams")
        if teams and isinstance(teams, list):
            teams = json.dumps(teams)
        
        cursor.execute("""
            INSERT OR REPLACE INTO events 
            (city, event_type, name, venue, event_date, teams, league, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            city.lower(),
            event_type.lower(),
            event.get("name"),
            event.get("venue"),
            event.get("event_date"),
            teams,
            event.get("league"),
            fetched_at
        ))
    
    conn.commit()
    conn.close()


def clear_cache() -> None:
    """Clear all cached data."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM flights")
    cursor.execute("DELETE FROM events")
    conn.commit()
    conn.close()


# Initialize database on module import
init_db()
