"""
Plugin system for the amusiment framework.

The plugin system is the primary extensibility mechanism. Seven plugin types
are supported, each with a well-defined ABC interface:

- InstrumentPlugin      — Sound generators (synths, samplers, VST bridges)
- EffectPlugin          — Signal processors (reverb, delay, EQ, compressor)
- AIGeneratorPlugin     — AI content generators (melody, chords, drums, arrangement)
- AIAnalyzerPlugin      — AI analysis tools (key detection, chord analysis, mixing advice)
- UIWidgetPlugin        — UI panel/widget extensions
- ExporterPlugin        — Export format handlers (WAV, MP3, MIDI, MusicXML)
- ImporterPlugin        — Import format handlers (MIDI, MusicXML, stems, loops)

All plugins are discovered, loaded, and sandboxed by the PluginRegistry.
Plugins declare their capabilities via a manifest system.
"""

from .registry import PluginRegistry, PluginManifest, PluginState
from .interfaces import (
    PluginBase,
    InstrumentPlugin,
    EffectPlugin,
    AIGeneratorPlugin,
    AIAnalyzerPlugin,
    UIWidgetPlugin,
    ExporterPlugin,
    ImporterPlugin,
    ImportFormat,
    ImportRequest,
    ImportResult,
    GeneratorCapabilities,
    AnalyzerCapabilities,
    GenerationPrompt,
    MusicalContext,
    GenerationConstraints,
    GeneratedContent,
)

__all__ = [
    "PluginRegistry", "PluginManifest", "PluginState",
    "PluginBase",
    "InstrumentPlugin", "EffectPlugin",
    "AIGeneratorPlugin", "AIAnalyzerPlugin",
    "UIWidgetPlugin", "ExporterPlugin", "ImporterPlugin",
    "ImportFormat", "ImportRequest", "ImportResult",
    "GeneratorCapabilities", "AnalyzerCapabilities",
    "GenerationPrompt", "MusicalContext",
    "GenerationConstraints", "GeneratedContent",
]
