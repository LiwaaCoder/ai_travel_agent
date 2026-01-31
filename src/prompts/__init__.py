"""
Prompt management module for the Travel Agent.

Provides structured prompts for different stages of the RAG pipeline.
"""

from .templates import (
    PromptLibrary,
    PromptType,
    get_prompt,
    get_all_prompts,
    INTENT_CLASSIFIER,
    RAG_RETRIEVAL,
    RAG_SYNTHESIS,
    ITINERARY_PLANNER,
    INFO_RESPONDER,
    SAFETY_LAYER,
)

__all__ = [
    "PromptLibrary",
    "PromptType",
    "get_prompt",
    "get_all_prompts",
    "INTENT_CLASSIFIER",
    "RAG_RETRIEVAL", 
    "RAG_SYNTHESIS",
    "ITINERARY_PLANNER",
    "INFO_RESPONDER",
    "SAFETY_LAYER",
]
