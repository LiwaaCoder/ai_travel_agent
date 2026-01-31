import asyncio
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.agents.graph import build_travel_agent
from src.rag.pipeline import build_knowledge_base

load_dotenv()
cli = typer.Typer(help="AI Travel Agent CLI - RAG-powered trip planning")
console = Console()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@cli.command()
def plan(
    city: str = typer.Option(..., help="City to visit"),
    days: int = typer.Option(3, help="Number of days"),
    preferences: Optional[str] = typer.Option(None, help="Preferences: food, art, nightlife, etc."),
    query: Optional[str] = typer.Option(None, help="Specific question about the trip"),
):
    """Generate a RAG-grounded travel plan for a city."""
    console.print(f"\n[bold blue]🌍 Planning {days}-day trip to {city}...[/bold blue]\n")
    
    agent = build_travel_agent()
    result = _run(agent.plan_trip(city, days, preferences, query))
    
    # Display results with rich formatting
    console.print(Panel(Markdown(result.summary), title="📋 Your Travel Plan", border_style="green"))
    
    console.print(f"\n[bold]🏛️  Points of Interest:[/bold] {', '.join(result.pois) or 'None found'}")
    console.print(f"[bold]🌤️  Weather:[/bold] {result.weather}")
    console.print(f"[bold]📚 Sources:[/bold] {', '.join(result.sources) or 'General knowledge'}")
    console.print(f"[bold]🎯 Confidence:[/bold] {result.confidence:.0%}\n")


@cli.command()
def build_kb(force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if exists")):
    """Build or rebuild the travel knowledge base vector store."""
    console.print("[bold blue]📚 Building knowledge base...[/bold blue]\n")
    build_knowledge_base(force_recreate=force)
    console.print("\n[bold green]✅ Knowledge base ready![/bold green]\n")


if __name__ == "__main__":
    cli()
