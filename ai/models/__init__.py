"""
AI model implementations.

This package contains concrete AI generator and analyzer plugins
that implement the framework's plugin interfaces. Each generator
produces real MIDI content that integrates directly with the
project model, sequencer, and export pipeline.

Generators:
- ChordGeneratorPlugin: Chord progressions in 8 styles
- MelodyGeneratorPlugin: Melodies with contour shaping
- DrumGeneratorPlugin: Drum patterns in 7 styles

Analyzers:
- BasicAnalyzerPlugin: Key detection, chord ID, rhythm analysis

Infrastructure:
- theory: Music theory foundation (scales, chords, progressions)
"""

from .chord_generator import ChordGeneratorPlugin
from .melody_generator import MelodyGeneratorPlugin
from .drum_generator import DrumGeneratorPlugin
from .basic_analyzer import BasicAnalyzerPlugin

__all__ = [
    "ChordGeneratorPlugin",
    "MelodyGeneratorPlugin",
    "DrumGeneratorPlugin",
    "BasicAnalyzerPlugin",
]
