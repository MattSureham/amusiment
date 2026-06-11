"""
Plugin interface definitions (Abstract Base Classes).

All plugin types must inherit from one of these ABCs and implement
the required abstract methods.
"""

from .base import PluginBase
from .instrument import InstrumentPlugin
from .effect import EffectPlugin
from .ai_generator import (
    AIGeneratorPlugin,
    GeneratorCapabilities,
    GenerationPrompt,
    MusicalContext,
    GenerationConstraints,
    GeneratedContent,
    ContentType,
)
from .ai_analyzer import (
    AIAnalyzerPlugin,
    AnalyzerCapabilities,
    AnalysisRequest,
    AnalysisResult,
)
from .ui_widget import UIWidgetPlugin, WidgetType
from .exporter import ExporterPlugin, ExportFormat, ExportRequest, ExportResult
from .importer import ImporterPlugin, ImportFormat, ImportRequest, ImportResult

__all__ = [
    # Base
    "PluginBase",
    # Instrument
    "InstrumentPlugin",
    # Effect
    "EffectPlugin",
    # AI Generator
    "AIGeneratorPlugin", "GeneratorCapabilities",
    "GenerationPrompt", "MusicalContext",
    "GenerationConstraints", "GeneratedContent", "ContentType",
    # AI Analyzer
    "AIAnalyzerPlugin", "AnalyzerCapabilities",
    "AnalysisRequest", "AnalysisResult",
    # UI Widget
    "UIWidgetPlugin", "WidgetType",
    # Exporter
    "ExporterPlugin", "ExportFormat", "ExportRequest", "ExportResult",
    # Importer
    "ImporterPlugin", "ImportFormat", "ImportRequest", "ImportResult",
]
