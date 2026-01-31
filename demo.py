#!/usr/bin/env python3
"""
Demo script for the AI Travel Agent.

Demonstrates the full RAG pipeline with LangGraph:
1. Builds the knowledge base (if needed)
2. Runs a sample query
3. Shows observability features
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n--- {text} ---\n")


async def run_demo():
    """Run the complete demo."""
    print_header("🌍 AI Travel Agent Demo")
    
    # 1. Check/Build Knowledge Base
    print_section("1. Knowledge Base Setup")
    
    from src.rag.pipeline import VECTOR_STORE_DIR, build_knowledge_base
    
    if not VECTOR_STORE_DIR.exists():
        print("📚 Building knowledge base for the first time...")
        build_knowledge_base()
    else:
        print("✅ Knowledge base already exists")
        print(f"   Location: {VECTOR_STORE_DIR}")
    
    # 2. Show Graph Structure
    print_section("2. LangGraph Workflow")
    
    from src.graph import generate_ascii_diagram
    print(generate_ascii_diagram())
    
    # 3. Initialize Agent with Logging
    print_section("3. Initializing Agent")
    
    from src.agents.graph import build_travel_agent
    from src.observability import AgentLogger, MetricsCollector
    
    agent = build_travel_agent()
    logger = AgentLogger(session_id="demo")
    metrics = MetricsCollector()
    
    print("✅ TravelAgent initialized")
    print("✅ Observability configured")
    
    # 4. Run Sample Queries
    print_section("4. Running Sample Queries")
    
    test_cases = [
        {
            "city": "Barcelona",
            "days": 3,
            "preferences": "food, architecture, art",
            "query": None,
        },
        {
            "city": "Tokyo",
            "days": 5,
            "preferences": "technology, traditional culture",
            "query": "What's the best way to get around?",
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📍 Query {i}: {test['days']}-day trip to {test['city']}")
        print(f"   Preferences: {test['preferences']}")
        if test['query']:
            print(f"   Question: {test['query']}")
        
        with metrics.timer(f"query_{i}"):
            try:
                result = await agent.plan_trip(
                    city=test['city'],
                    days=test['days'],
                    preferences=test['preferences'],
                    query=test['query'],
                )
                
                print(f"\n   ✅ Generated plan with {result.confidence:.0%} confidence")
                print(f"   📚 Sources: {', '.join(result.sources) or 'General knowledge'}")
                print(f"   🌤️  Weather: {result.weather}")
                print(f"   📍 POIs found: {len(result.pois)}")
                
                # Show first part of summary
                summary_preview = result.summary[:300] + "..." if len(result.summary) > 300 else result.summary
                print(f"\n   📋 Plan preview:\n   {summary_preview}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # 5. Show Metrics
    print_section("5. Performance Metrics")
    
    summary = metrics.summary()
    print("📊 Timing Summary:")
    for name, stats in summary.get("timers", {}).items():
        print(f"   {name}: {stats['mean_ms']:.0f}ms avg ({stats['count']} calls)")
    
    # 6. API Example
    print_section("6. API Usage Example")
    
    print("""
To use the REST API:

    # Start server
    uvicorn app:app --reload

    # Make request
    curl -X POST http://localhost:8000/plan \\
      -H "Content-Type: application/json" \\
      -d '{"city": "Paris", "days": 3, "preferences": "museums, food"}'
    
Response includes:
    {
        "plan": "Day-by-day itinerary...",
        "pois": ["Louvre", "Eiffel Tower", ...],
        "weather": "Mild, 15-20°C",
        "sources": ["destinations.md", "travel_tips.md"],
        "confidence": 0.85
    }
""")
    
    print_header("✅ Demo Complete!")
    
    print("""
Next steps:
    
    1. Try the CLI:
       python cli.py plan --city "London" --days 4
    
    2. Launch the web UI:
       streamlit run streamlit_app.py
    
    3. Enable LangSmith tracing:
       export LANGCHAIN_TRACING_V2=true
       export LANGCHAIN_API_KEY=your-key
    
    4. Add more knowledge:
       - Add .md files to data/knowledge/
       - Run: python cli.py build-kb --force
""")


if __name__ == "__main__":
    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Copy env.sample to .env and add your API key")
        sys.exit(1)
    
    asyncio.run(run_demo())
