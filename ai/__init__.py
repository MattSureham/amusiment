"""
AI composition and analysis package.

Provides:
- models/: Concrete AI generator and analyzer plugin implementations
  * ChordGeneratorPlugin — chord progressions in 8 styles
  * MelodyGeneratorPlugin — melodies with 6 contour shapes
  * DrumGeneratorPlugin — drum patterns in 7 styles
  * BasicAnalyzerPlugin — key/chord/rhythm analysis

- inference/: Context management and prompt parsing
  * ContextWindow — multi-turn generation context
  * PromptEngine — natural language → structured parameters

- theory.py — Music theory foundation (scales, chords, progressions)

All generators implement AIGeneratorPlugin; all analyzers implement
AIAnalyzerPlugin. They produce standard MidiClip objects that integrate
directly with the project model, sequencer, and export pipeline.
"""

from .models import (
    ChordGeneratorPlugin,
    MelodyGeneratorPlugin,
    DrumGeneratorPlugin,
    BasicAnalyzerPlugin,
)
from .inference import ContextWindow, PromptEngine

__all__ = [
    "ChordGeneratorPlugin",
    "MelodyGeneratorPlugin",
    "DrumGeneratorPlugin",
    "BasicAnalyzerPlugin",
    "ContextWindow",
    "PromptEngine",
]
