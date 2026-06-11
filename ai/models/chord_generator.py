"""
Chord progression generator — an AIGeneratorPlugin implementation.

Generates chord progressions in any key, with style-specific patterns,
voicings, and rhythmic variations. Supports multiple complexity levels
and provides alternatives with explanations.

Algorithm overview:
1. Determine key/scale from MusicalContext
2. Select progression pattern based on style
3. Apply chord qualities (triads or sevenths based on complexity)
4. Voice each chord as MIDI notes
5. Apply rhythmic placement
6. Package into a MidiClip with explanatory metadata
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass, field
from typing import Optional, Callable
from random import Random

from core.plugin.interfaces.base import PluginBase, PluginManifest, PluginCategory, PluginState
from core.plugin.interfaces.ai_generator import (
    AIGeneratorPlugin, ContentType, GeneratorCapabilities,
    MusicalContext, GenerationConstraints, GenerationPrompt, GeneratedContent,
)
from core.model.note import NoteEvent, NotePitch, NoteVelocity
from core.model.clip import MidiClip
from core.model.time_model import Ticks, PPQ, beats_to_ticks

from .theory import (
    diatonic_chords, get_style_progression, chord_tones,
    chord_name, voice_lead, rhythm_pattern, scale_degrees,
    pc_to_key_name, CHORD_TYPES, PITCH_CLASSES,
)

# ── Voicing presets ─────────────────────────────────────────────────

@dataclass
class VoicingPreset:
    """How chords are spread across multiple octaves."""
    name: str
    octave_spread: int    # Number of octaves to spread voices across
    base_octave: int      # Starting octave for lowest voice
    double_root: bool     # Whether to double the root
    close_voicing: bool   # Close (narrow) vs open (wide) voicing


VOICING_PRESETS = {
    "simple": VoicingPreset("Simple", 1, 3, True, True),
    "close": VoicingPreset("Close", 1, 3, False, True),
    "open": VoicingPreset("Open", 2, 3, True, False),
    "wide": VoicingPreset("Wide", 3, 2, True, False),
    "cluster": VoicingPreset("Cluster", 1, 4, False, True),
}


# ── The plugin ──────────────────────────────────────────────────────

class ChordGeneratorPlugin(AIGeneratorPlugin):
    """
    AI chord progression generator.

    Generates chord progressions with:
    - 8 musical styles (pop, jazz, classical, lofi, edm, rnb, rock, blues)
    - Multiple voicing presets per style
    - Configurable complexity (triads through extended chords)
    - Rhythmic variation (steady, syncopated, sparse, dense, swing)
    - Voice leading between successive chords
    - 3 alternatives per generation for user choice
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = Random(seed)
        self._initialized = False
        self._state = PluginState.DISCOVERED

    # ── PluginBase interface ─────────────────────────────────────

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="amusiment.chord-generator",
            name="Chord Progression Generator",
            version="1.0.0",
            category=PluginCategory.AI_GENERATOR,
            author="amusiment",
            description="Generates chord progressions in multiple styles with "
                        "voice leading and voicing presets. Supports pop, jazz, "
                        "classical, lofi, EDM, R&B, rock, and blues.",
            capabilities=[
                "generate.chords",
                "generate.progression",
                "generate.harmony",
            ],
        )

    def initialize(self) -> None:
        self._initialized = True
        self._state = PluginState.INITIALIZED

    def shutdown(self) -> None:
        self._initialized = False
        self._state = PluginState.UNLOADED

    # ── AIGeneratorPlugin interface ───────────────────────────────

    def get_capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            content_types=[
                ContentType.CHORDS,
                ContentType.PAD,
            ],
            supports_continuation=True,
            supports_variation=True,
            supports_style_transfer=False,
            max_bars=32,
            min_bars=2,
            supports_audio_input=False,
            supports_text_prompt=True,
            style_tags=[
                "pop", "jazz", "classical", "lofi", "edm",
                "rnb", "rock", "blues",
                "happy", "sad", "energetic", "chill",
                "simple", "complex",
            ],
            model_name="ChordEngine v1.0 (deterministic music theory)",
        )

    def generate(self, prompt: GenerationPrompt) -> GeneratedContent:
        """Generate a chord progression."""
        ctx = prompt.context
        constraints = prompt.constraints
        rng = Random(constraints.seed) if constraints.seed else self._rng

        # ── Determine key ─────────────────────────────────────────
        tonic_pc = ctx.key_sharps_flats % 12
        mode = ctx.key_mode if ctx.key_mode in ("major", "minor") else "major"

        # Map sharps/flats count to actual tonic
        if ctx.key_sharps_flats >= 0:
            # Sharps: cycle of fifths forward
            tonic_pc = (ctx.key_sharps_flats * 7) % 12
        else:
            # Flats: cycle of fourths
            tonic_pc = (abs(ctx.key_sharps_flats) * 5) % 12

        key_name = pc_to_key_name(tonic_pc, mode)

        # ── Determine style ────────────────────────────────────────
        style = "pop"
        for tag in ctx.style_tags:
            tag_lower = tag.lower()
            if tag_lower in COMMON_PROGRESSIONS:
                style = tag_lower
                break

        # ── Determine complexity ───────────────────────────────────
        density = getattr(ctx, 'density_target', 0.5)
        use_sevenths = density > 0.4 or style in ("jazz", "lofi", "rnb")
        use_extended = density > 0.7 and style in ("jazz", "rnb")

        voicing_preset = "simple"
        if density < 0.3:
            voicing_preset = "simple"
        elif density < 0.6:
            voicing_preset = "close" if style in ("jazz", "lofi") else "simple"
        elif density < 0.8:
            voicing_preset = "open"
        else:
            voicing_preset = "wide"

        # ── Determine bars ─────────────────────────────────────────
        bar_count = ctx.bar_count or 8
        bar_count = max(constraints.bar_start // 4 + 2, bar_count)

        # ── Build progression ─────────────────────────────────────
        progression = get_style_progression(style, rng)

        # ── Generate content ──────────────────────────────────────
        content = self._build_progression_content(
            tonic_pc=tonic_pc,
            mode=mode,
            progression=progression,
            bar_count=bar_count,
            use_sevenths=use_sevenths,
            use_extended=use_extended,
            voicing_preset=voicing_preset,
            style=style,
            constraints=constraints,
            rng=rng,
            key_name=key_name,
            existing_notes=list(ctx.existing_notes),
        )

        # ── Generate alternatives ──────────────────────────────────
        alternatives = []
        for i in range(3):
            alt_prog = get_style_progression(style, rng)
            alt_voicing = rng.choice(list(VOICING_PRESETS.keys()))
            alt = self._build_progression_content(
                tonic_pc=tonic_pc,
                mode=mode,
                progression=alt_prog,
                bar_count=bar_count,
                use_sevenths=use_sevenths,
                use_extended=use_extended,
                voicing_preset=alt_voicing,
                style=style,
                constraints=constraints,
                rng=rng,
                key_name=key_name,
            )
            alternatives.append(alt)

        content.alternatives = alternatives
        return content

    def get_parameters(self) -> dict[str, dict]:
        return {
            "density": {
                "name": "Harmonic Density",
                "min": 0.1, "max": 1.0, "default": 0.5,
                "description": "Simple triads (low) to extended jazz chords (high)",
            },
            "style": {
                "name": "Style",
                "min": 0, "max": 1, "default": 0,
                "description": "Musical style: pop, jazz, classical, lofi, edm, rnb, rock, blues",
                "type": "choice",
                "options": ["pop", "jazz", "classical", "lofi", "edm", "rnb", "rock", "blues"],
            },
        }

    # ── Internal generation ───────────────────────────────────────

    def _build_progression_content(
        self,
        tonic_pc: int,
        mode: str,
        progression: list,
        bar_count: int,
        use_sevenths: bool,
        use_extended: bool,
        voicing_preset: str,
        style: str,
        constraints: GenerationConstraints,
        rng: Random,
        key_name: str,
        existing_notes: Optional[list] = None,
    ) -> GeneratedContent:
        """Build a GeneratedContent from a progression specification."""

        scale_name = "major" if mode == "major" else "natural_minor"
        diatonic = diatonic_chords(tonic_pc, scale_name, use_sevenths)

        # Expand progression to fill bar_count by repeating
        beats_per_cycle = sum(item[2] for item in progression)
        total_beats = bar_count * 4  # assume 4/4
        cycles = max(1, total_beats // max(1, beats_per_cycle) + 1)

        # Determine rhythmic pattern
        if style in ("edm", "rock"):
            rhythm_style = "steady"
        elif style in ("jazz", "lofi"):
            rhythm_style = "swing"
        elif style == "classical":
            rhythm_style = "sparse"
        else:
            rhythm_style = "steady"

        # Apply rhythm
        beat_positions = rhythm_pattern(rhythm_style, total_beats, rng)

        # Build chord sequence
        chord_sequence = []
        for _ in range(cycles):
            chord_sequence.extend(progression)

        # Generate MIDI notes
        voicing_info = VOICING_PRESETS[voicing_preset]
        notes: list[NoteEvent] = []
        chord_symbols: list[str] = []
        prev_chord_tones = None

        chord_idx = 0
        for chord_pos in beat_positions:
            if chord_idx >= len(chord_sequence):
                break

            degree, quality_override, beats_held = chord_sequence[chord_idx]

            # Get chord from diatonic set
            dt_chord = diatonic[degree - 1]
            quality = quality_override or dt_chord.quality

            # Adjust for extended chords
            if use_extended and quality in ("maj7", "min7", "7"):
                if rng.random() < 0.3:
                    if quality == "maj7":
                        quality = "maj9"
                    elif quality == "min7":
                        quality = "min9"
                    elif quality == "7":
                        quality = "9"

            root_pc = dt_chord.root_pc
            tones = chord_tones(root_pc, quality)

            # Apply voicing
            chord_midi = self._voice_chord(
                tones, root_pc,
                base_octave=voicing_info.base_octave,
                spread=voicing_info.octave_spread,
                double_root=voicing_info.double_root,
                close=voicing_info.close_voicing,
                prev_tones=prev_chord_tones,
            )

            # Convert beat position to ticks
            start_tick = beats_to_ticks(chord_pos)
            dur_ticks = beats_to_ticks(0.8 * beats_held)  # slight gap between chords

            for midi_note in chord_midi:
                notes.append(NoteEvent(
                    pitch=NotePitch(midi_note),
                    velocity=NoteVelocity(rng.randint(60, 100)),
                    start_tick=start_tick,
                    duration_ticks=dur_ticks,
                    channel=0,
                ))

            chord_symbols.append(chord_name(root_pc, quality))
            prev_chord_tones = chord_midi

            # Advance to next chord chord_positions
            chord_idx += 1

        # Build the clip
        clip = MidiClip(
            name=f"Chords - {key_name} ({style})",
            start_tick=Ticks(0),
            length_ticks=beats_to_ticks(total_beats),
            notes=tuple(notes),
        )

        # Build explanation
        roman_numerals = []
        for degree, q_override, _ in progression:
            dt = diatonic[degree - 1]
            roman_numerals.append(dt.roman)

        explanation = (
            f"Generated a {key_name} chord progression in {style} style. "
            f"Progression: {' - '.join(roman_numerals)} "
            f"({', '.join(chord_symbols[:len(progression)])}). "
            f"Voicing: {VOICING_PRESETS[voicing_preset].name.lower()}. "
            f"Rhythm: {rhythm_style}."
        )

        return GeneratedContent(
            clips={ContentType.CHORDS: clip},
            chord_progression=chord_symbols,
            explanation=explanation,
            confidence=0.95,
            warnings=[],
            parameters_used={
                "density": constraints.temperature,
                "style": style,
                "voicing": voicing_preset,
                "complexity": "sevenths" if use_sevenths else "triads",
            },
        )

    def _voice_chord(
        self,
        chord_pcs: list[int],
        root_pc: int,
        base_octave: int = 3,
        spread: int = 1,
        double_root: bool = True,
        close: bool = True,
        prev_tones: Optional[list[int]] = None,
    ) -> list[int]:
        """Voice a chord as MIDI notes with specified parameters."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from ai.models.theory import pc_to_midi

        if close:
            # Close voicing: all notes stacked within 1-2 octaves
            midi_notes = []
            current_octave = base_octave
            for i, pc in enumerate(chord_pcs):
                if i == 0:
                    midi_notes.append(pc_to_midi(pc, current_octave))
                else:
                    prev_midi = midi_notes[-1]
                    candidate = pc_to_midi(pc, current_octave)
                    while candidate <= prev_midi:
                        candidate += 12
                    midi_notes.append(candidate)
            # Add doubled root if requested
            if double_root and len(midi_notes) > 0:
                root_midi = pc_to_midi(chord_pcs[0], base_octave + 1)
                if root_midi not in midi_notes:
                    midi_notes.append(root_midi)
        else:
            # Open voicing: spread across multiple octaves
            midi_notes = []
            for i, pc in enumerate(chord_pcs):
                octave = base_octave + (i * spread) // len(chord_pcs)
                midi_notes.append(pc_to_midi(pc, octave))

        # Voice leading: if we have previous chord, move minimally
        if prev_tones is not None:
            voiced = []
            used = set()
            for prev_midi in prev_tones:
                best, best_dist = None, 999
                for candidate in midi_notes:
                    if candidate in used:
                        continue
                    dist = abs(candidate - prev_midi)
                    if dist < best_dist:
                        best_dist, best = dist, candidate
                if best is not None:
                    voiced.append(best)
                    used.add(best)
            # Add any remaining notes
            for note in midi_notes:
                if note not in used:
                    voiced.append(note)
            midi_notes = sorted(voiced)

        return midi_notes


# ── Alias for backward compatibility ──────────────────────────────

# Map style names used in theory module
COMMON_PROGRESSIONS = {
    "pop": [(1, None, 4), (5, None, 4), (6, None, 4), (4, None, 4),
            (6, None, 4), (4, None, 4), (1, None, 4), (5, None, 4),
            (1, None, 4), (6, None, 4), (4, None, 4), (5, None, 4),
            (4, None, 4), (1, None, 4), (5, None, 4), (6, None, 4)],
    "jazz": [(2, "min7", 4), (5, "7", 4), (1, "maj7", 8),
             (1, "maj7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4),
             (3, "min7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4)],
    "classical": [(1, None, 8), (4, None, 8), (5, None, 8), (1, None, 8),
                  (1, None, 4), (2, None, 4), (5, None, 4), (1, None, 4)],
    "lofi": [(1, "maj7", 4), (6, "min7", 4), (4, "maj7", 4), (5, "7", 4),
             (3, "min7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4)],
    "edm": [(1, "min", 4), (6, "maj", 4), (3, "maj", 4), (7, "maj", 4),
            (1, "min", 4), (6, "maj", 4), (4, "min", 4), (5, "maj", 4)],
    "rnb": [(1, "maj7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4),
            (1, "maj7", 4), (4, "maj7", 4), (3, "min7", 4), (6, "min7", 4)],
    "rock": [(1, None, 8), (4, None, 8), (5, None, 8),
             (1, "min", 4), (6, "maj", 4), (7, "maj", 4)],
    "blues": [(1, "7", 16), (4, "7", 8), (1, "7", 8), (5, "7", 4), (4, "7", 4), (1, "7", 8)],
}
