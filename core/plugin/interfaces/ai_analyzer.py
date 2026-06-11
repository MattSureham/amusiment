"""
AI Analyzer plugin interface — musical analysis and suggestions.

Analyzers examine existing project content and produce insights:
key detection, chord analysis, rhythm analysis, mixing suggestions,
arrangement balance, and more.

Unlike generators, analyzers don't create new musical content — they
help users understand and improve what already exists.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .base import PluginBase, PluginManifest, PluginCategory
from ...model.note import NoteEvent
from ...model.clip import MidiClip


class AnalyzerCapabilities(Enum):
    """Types of analysis an AI analyzer can perform."""
    KEY_DETECTION = auto()           # Detect the musical key
    CHORD_ANALYSIS = auto()          # Identify chords from notes
    RHYTHM_ANALYSIS = auto()         # Analyze rhythmic density and patterns
    ARRANGEMENT_BALANCE = auto()     # Check frequency/instrument balance
    MIXING_ADVICE = auto()           # Suggest mixing improvements
    MELODY_ASSESSMENT = auto()       # Evaluate melodic quality
    STRUCTURE_DETECTION = auto()     # Detect verse/chorus/bridge boundaries
    GROOVE_ANALYSIS = auto()         # Analyze timing/groove/swing
    CONFLICT_DETECTION = auto()      # Find clashing notes/frequencies
    STYLE_CLASSIFICATION = auto()    # Classify the genre/style
    MOOD_DETECTION = auto()          # Detect emotional character


@dataclass
class AnalysisRequest:
    """
    A request for musical analysis.
    
    Attributes:
        capabilities_requested: Which analyses to perform.
        target_track_ids: Specific tracks to analyze (empty = all).
        clip_range_start: Start tick of analysis region (0 = entire project).
        clip_range_end: End tick of analysis region (0 = entire project).
        context: Optional additional context (key hint, tempo, etc.).
    """
    capabilities_requested: list[AnalyzerCapabilities] = field(default_factory=list)
    target_track_ids: list[str] = field(default_factory=list)
    clip_range_start: int = 0
    clip_range_end: int = 0
    context: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """
    The output of an analysis operation.
    
    Contains structured findings plus a human-readable summary.
    Results are keyed by capability for easy access.
    
    Attributes:
        findings: Dict mapping AnalyzerCapabilities to their results.
        summary: Human-readable summary of all findings.
        confidence: Overall confidence score 0.0-1.0.
        suggestions: Actionable suggestions derived from analysis.
        details: Additional structured data (charts, scores, etc.).
    """
    findings: dict[AnalyzerCapabilities, dict] = field(default_factory=dict)
    summary: str = ""
    confidence: float = 1.0
    suggestions: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    
    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0
    
    def get_finding(self, capability: AnalyzerCapabilities) -> Optional[dict]:
        """Get the result for a specific analysis capability."""
        return self.findings.get(capability)


class AIAnalyzerPlugin(PluginBase):
    """
    Abstract AI analyzer plugin — examines music and provides insights.
    
    Analyzers are typically called on-demand by the user (e.g., "analyze my mix")
    but can also be configured for realtime feedback (e.g., showing current chord
    name while playing).
    
    Lifecycle:
        1. initialize() — load analysis models
        2. get_capabilities() — declare what this analyzer can detect
        3. analyze() — analyze the given clips/notes (may be called many times)
        4. shutdown() — unload models
    """
    
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="",
            name="Unnamed AI Analyzer",
            version="0.1.0",
            category=PluginCategory.AI_ANALYZER,
            description="An AI analysis plugin",
            capabilities=["analyze.music"],
        )
    
    @abstractmethod
    def get_capabilities(self) -> list[AnalyzerCapabilities]:
        """
        Return which analysis types this plugin supports.
        
        The UI uses this to determine which analysis options to show.
        """
        ...
    
    @abstractmethod
    def analyze(self, clips: dict[str, list[MidiClip]],
                request: AnalysisRequest) -> AnalysisResult:
        """
        Analyze the given musical content.
        
        Args:
            clips: MIDI clips keyed by track_id.
            request: Which analyses to perform and on what range.
        
        Returns:
            AnalysisResult with findings, summary, and suggestions.
        """
        ...
    
    def analyze_realtime(self, notes: list[NoteEvent],
                         capability: AnalyzerCapabilities) -> Optional[dict]:
        """
        Optional realtime analysis — called on each audio buffer.
        
        Override for low-latency analysis like live chord display.
        Must be fast (< 5ms typically). Default returns None.
        
        Args:
            notes: Currently playing notes.
            capability: Which analysis to perform.
        
        Returns:
            Analysis finding dict, or None if not supported.
        """
        return None
    
    def get_confidence_threshold(self) -> float:
        """
        Minimum confidence threshold for reporting findings.
        
        Findings below this threshold will be suppressed or flagged
        as uncertain.
        """
        return 0.5
