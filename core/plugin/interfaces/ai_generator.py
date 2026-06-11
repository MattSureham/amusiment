"""
AI Generator plugin interface — the primary AI integration point.

AI Generator plugins produce musical content: melodies, chord progressions,
drum patterns, bass lines, arrangements, and more. They take structured
context and constraints as input, and return GeneratedContent containing
MIDI clips, chord tracks, automation data, and explanatory metadata.

This is the flagship plugin type — it's how any AI model, from a simple
Markov chain to a large transformer, integrates into the framework.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from .base import PluginBase, PluginManifest, PluginCategory
from ...model.note import NoteEvent, NotePitch
from ...model.clip import MidiClip
from ...model.time_model import Ticks


class ContentType(Enum):
    """Types of musical content an AI generator can produce."""
    MELODY = auto()
    CHORDS = auto()
    BASS = auto()
    DRUMS = auto()
    ARPEGGIO = auto()
    PAD = auto()            # Sustained textures
    LEAD = auto()           # Lead/solo lines
    RHYTHM = auto()         # Rhythmic patterns (non-pitched)
    ARRANGEMENT = auto()    # Section structure
    AUTOMATION = auto()     # Parameter automation curves
    FILL = auto()           # Transitions/fills
    VARIATION = auto()      # Variation of existing material


@dataclass
class GeneratorCapabilities:
    """
    What an AI generator plugin can produce.
    
    This is the contract between the plugin and the UI/framework.
    The UI uses this to decide which generators to show for which tasks.
    
    Attributes:
        content_types: What types of content this generator can create.
        supports_continuation: Whether it can extend existing material.
        supports_variation: Whether it can create variations of existing material.
        supports_style_transfer: Whether it can apply a style to existing material.
        max_bars: Maximum number of bars it can generate in one call.
        min_bars: Minimum number of bars for meaningful output.
        supports_audio_input: Whether it can take audio as input (e.g., humming).
        supports_text_prompt: Whether it accepts free-text prompts.
        style_tags: Predefined style tags this generator understands.
        model_name: The underlying model architecture (for info display).
    """
    content_types: list[ContentType] = field(default_factory=list)
    supports_continuation: bool = False
    supports_variation: bool = False
    supports_style_transfer: bool = False
    max_bars: int = 32
    min_bars: int = 1
    supports_audio_input: bool = False
    supports_text_prompt: bool = True
    style_tags: list[str] = field(default_factory=list)
    model_name: str = ""


@dataclass
class MusicalContext:
    """
    Structured musical context passed to AI generators.
    
    This is NOT a free-text prompt — it's a structured snapshot of the
    current project state that AI models can use as conditioning input.
    
    Attributes:
        bpm: Current tempo.
        key_sharps_flats: Key signature (positive=sharps, negative=flats).
        key_mode: "major" or "minor".
        time_signature_numerator: e.g., 4 in 4/4.
        time_signature_denominator: e.g., 4 in 4/4.
        current_section: Section label (verse, chorus, bridge, etc.).
        bar_start: First bar number for generation.
        bar_count: Number of bars to generate.
        existing_tracks: Summary of existing tracks (instrument, note density, range).
        existing_notes: Existing notes in the generation region (for continuation/variation).
        chord_progression: Optional explicit chord progression.
        style_tags: Style/genre tags applied.
        energy_target: Desired energy level 0.0-1.0.
        density_target: Desired note density 0.0-1.0.
    """
    bpm: float = 120.0
    key_sharps_flats: int = 0
    key_mode: str = "major"
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4
    current_section: str = ""
    bar_start: int = 0
    bar_count: int = 8
    existing_tracks: list[dict] = field(default_factory=list)
    existing_notes: list[NoteEvent] = field(default_factory=list)
    chord_progression: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    energy_target: float = 0.5
    density_target: float = 0.5
    
    @property
    def ticks_per_bar(self) -> int:
        """Calculate ticks per bar based on time signature."""
        from ...model.time_model import PPQ
        beats_per_bar = self.time_signature_numerator * (4 / self.time_signature_denominator)
        return int(beats_per_bar * PPQ)


@dataclass
class GenerationConstraints:
    """
    Constraints that guide AI content generation.
    
    These are NOT hard limits — they're "suggestions" that guide the model.
    AI generators should respect them as much as possible but may adjust
    for musicality.
    
    Attributes:
        bar_start: First bar to generate content for.
        bar_count: Number of bars to generate.
        min_pitch: Lowest allowed MIDI pitch.
        max_pitch: Highest allowed MIDI pitch.
        min_velocity: Minimum note velocity.
        max_velocity: Maximum note velocity.
        max_note_density: Maximum notes per bar (0 = unlimited).
        avoid_notes: Pitches to avoid (e.g., non-scale tones).
        prefer_notes: Pitches to prefer (e.g., chord tones).
        must_include_notes: Pitches that must appear at least once.
        temperature: Creativity/temperature parameter (0=deterministic, 1=creative).
        seed: Random seed for reproducibility.
    """
    bar_start: int = 0
    bar_count: int = 8
    min_pitch: int = 21   # A0
    max_pitch: int = 108  # C8
    min_velocity: int = 30
    max_velocity: int = 127
    max_note_density: int = 0  # 0 = unlimited
    avoid_notes: list[int] = field(default_factory=list)
    prefer_notes: list[int] = field(default_factory=list)
    must_include_notes: list[int] = field(default_factory=list)
    temperature: float = 0.7
    seed: Optional[int] = None


@dataclass
class GeneratedContent:
    """
    The output of an AI generation call.
    
    Contains the generated musical data plus metadata for the UI to display.
    Supports multiple alternatives so users can choose or iterate.
    
    Attributes:
        clips: Generated MIDI clips keyed by content type.
        chord_progression: Generated chord progression (e.g., ["Cmaj7", "Dm7", "G7", "Cmaj7"]).
        explanation: Human-readable explanation of what was generated and why.
        confidence: Model confidence score 0.0-1.0.
        alternatives: Alternative GeneratedContent instances.
        warnings: Any warnings about the generation (e.g., "key conflict").
        parameters_used: The actual parameters used (may differ from constraints).
    """
    clips: dict[ContentType, MidiClip] = field(default_factory=dict)
    chord_progression: list[str] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 1.0
    alternatives: list["GeneratedContent"] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameters_used: dict[str, float] = field(default_factory=dict)
    
    @property
    def total_notes(self) -> int:
        """Total number of notes across all generated clips."""
        return sum(len(clip.notes) for clip in self.clips.values())
    
    @property
    def content_types_generated(self) -> list[ContentType]:
        """Which types of content were actually generated."""
        return list(self.clips.keys())


@dataclass
class GenerationPrompt:
    """
    User intent for AI generation — the input to a generation call.
    
    Supports three input modes:
    1. Free-text prompt (natural language)
    2. Structured parameters (explicit settings)
    3. Reference input (humming, existing clip, audio file)
    
    At least one input mode must be provided.
    """
    text: str = ""
    context: MusicalContext = field(default_factory=MusicalContext)
    constraints: GenerationConstraints = field(default_factory=GenerationConstraints)
    reference_clip: Optional[MidiClip] = None
    reference_audio_path: Optional[str] = None
    content_types_requested: list[ContentType] = field(default_factory=list)
    
    def has_text_prompt(self) -> bool:
        return bool(self.text.strip())
    
    def has_reference(self) -> bool:
        return self.reference_clip is not None or self.reference_audio_path is not None


class AIGeneratorPlugin(PluginBase):
    """
    Abstract AI generator plugin — produces musical content from prompts/context.
    
    This is the primary interface for integrating AI models into the framework.
    Any AI model — from a simple rule-based generator to a large transformer —
    can be wrapped as an AIGeneratorPlugin.
    
    Key design decisions:
    - Input is structured (MusicalContext + GenerationConstraints), not raw text.
      The intent parser in the UI layer converts natural language to these structures.
    - Output is GeneratedContent containing Standard MIDI clips that integrate
      directly into the project.
    - Multiple alternatives are supported so users can browse and choose.
    - Every generated output includes an explanation for transparency.
    
    Lifecycle:
        1. initialize() — load model, warm up inference
        2. get_capabilities() — declare what this generator can do
        3. generate() — produce content (may be called many times)
        4. shutdown() — unload model
    """
    
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="",
            name="Unnamed AI Generator",
            version="0.1.0",
            category=PluginCategory.AI_GENERATOR,
            description="An AI content generator plugin",
            capabilities=["generate.midi"],
        )
    
    @abstractmethod
    def get_capabilities(self) -> GeneratorCapabilities:
        """
        Declare what types of content this generator can produce.
        
        The framework uses this to:
        - Show/hide this generator in the UI based on user's task
        - Validate generation requests before calling generate()
        - Display model info to the user
        
        Returns:
            A GeneratorCapabilities object describing this generator.
        """
        ...
    
    @abstractmethod
    def generate(self, prompt: GenerationPrompt) -> GeneratedContent:
        """
        Generate musical content based on the given prompt.
        
        This is the main entry point. The method may be called from a
        background thread — long-running inference should not block the UI.
        
        Args:
            prompt: The generation request containing context, constraints,
                    and optionally a text prompt or reference material.
        
        Returns:
            GeneratedContent with clips, chord progressions, and metadata.
        
        Raises:
            ValueError: If the prompt requests unsupported content types.
            RuntimeError: If generation fails (model error, OOM, etc.).
        """
        ...
    
    def generate_async(self, prompt: GenerationPrompt,
                       callback: "Callable[[GeneratedContent], None]") -> None:
        """
        Optional async generation — override if your model supports streaming.
        
        Default implementation calls generate() synchronously and invokes
        the callback with the result. Override for progressive generation
        (e.g., generating bar by bar and updating the UI incrementally).
        
        Args:
            prompt: The generation request.
            callback: Called when generation completes (may be called multiple
                      times for progressive results).
        """
        result = self.generate(prompt)
        callback(result)
    
    def cancel(self) -> None:
        """
        Cancel any in-progress generation.
        
        Default is no-op. Override if your model supports cancellation.
        """
        pass
    
    def get_parameters(self) -> dict[str, dict]:
        """
        Return the generator's adjustable parameters for UI display.
        
        Returns:
            Dict mapping parameter_id to {name, min, max, default, description}.
        
        Example:
            {
                "temperature": {
                    "name": "Creativity",
                    "min": 0.0, "max": 1.0, "default": 0.7,
                    "description": "Higher values produce more varied output"
                },
                "note_density": {
                    "name": "Note Density",
                    "min": 0.1, "max": 1.0, "default": 0.5,
                    "description": "How many notes to generate per bar"
                },
            }
        """
        return {}
    
    def validate_prompt(self, prompt: GenerationPrompt) -> list[str]:
        """
        Validate that a GenerationPrompt is compatible with this generator.
        
        Override to add generator-specific validation. Default checks
        capabilities match.
        
        Returns:
            List of validation error messages (empty = valid).
        """
        errors = []
        caps = self.get_capabilities()
        
        for ct in prompt.content_types_requested:
            if ct not in caps.content_types:
                errors.append(f"Generator does not support content type: {ct}")
        
        if prompt.has_text_prompt() and not caps.supports_text_prompt:
            errors.append("Generator does not support text prompts")
        
        return errors
