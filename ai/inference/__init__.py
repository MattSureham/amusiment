"""
Model inference and context management utilities.

Provides:
- ContextWindow: Manages musical context across multi-turn generation
- PromptEngine: Converts natural language prompts to structured parameters
"""

from .context_window import ContextWindow
from .prompt_engine import PromptEngine

__all__ = ["ContextWindow", "PromptEngine"]
