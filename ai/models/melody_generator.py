"""
Melody generator — an AIGeneratorPlugin implementation.

Generates melodies over chord progressions using music-theory-aware
algorithms. Considers chord tones, passing tones, neighbor tones,
rhythmic patterns, and melodic contour.

Algorithm overview:
1. Analyze the chord progression and identify chord tones per beat
2. Generate a rhythmic skeleton based on style and density
3. Fill pitches: chord tones on strong beats, passing tones elsewhere
4. Shape melodic contour (arch, rising, falling, wave)
5. Apply articulation, velocity shaping, and phrasing
6. Ensure the melody stays within the specified range
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional
from random import Random

from core.plugin.interfaces.base import PluginManifest, PluginCategory, PluginState
from core.plugin.interfaces.ai_generator import (
    AIGeneratorPlugin, ContentType, GeneratorCapabilities,
    MusicalContext, GenerationConstraints, GenerationPrompt, GeneratedContent,
)
from core.model.note import NoteEvent, NotePitch, NoteVelocity
from core.model.clip import MidiClip
from core.model.time_model import Ticks, PPQ, beats_to_ticks

from .theory import (
    scale_degrees, chord_tones, chord_name, parse_chord_symbol,
    note_in_scale, closest_scale_note, note_is_chord_tone,
    rhythm_pattern, pc_to_midi, midi_to_pc, PITCH_CLASSES,
)

# ── Melodic contour shapes ──────────────────────────────────────────

CONTOUR_SHAPES = {
    "arch":     "Rise then fall (peak in middle)",
    "rising":   "Gradually ascending",
    "falling":  "Gradually descending",
    "wave":     "Multiple small arcs",
    "flat":     "Stay within a narrow range",
    "spike":    "Sharp short peaks with low base",
}


def _contour_weight(position: float, total_length: float, shape: str) -> float:
    """Get a pitch height weight (0-1) based on contour shape and position.

    Args:
        position: Current position (e.g., beat number).
        total_length: Total length of the melody.
        shape: Contour shape name.

    Returns:
        Weight 0.0-1.0 indicating relative pitch height preference.
    """
    frac = position / max(total_length, 1)
    frac = max(0.0, min(1.0, frac))

    if shape == "arch":
        return 1.0 - 4.0 * (frac - 0.5) ** 2
    elif shape == "rising":
        return frac
    elif shape == "falling":
        return 1.0 - frac
    elif shape == "wave":
        import math
        return 0.5 + 0.5 * math.sin(frac * math.pi * 3)
    elif shape == "flat":
        return 0.5 + 0.1 * (0.5 - frac)
    elif shape == "spike":
        import math
        base = abs(math.sin(frac * math.pi * 2))
        return base * base
    else:
        return 0.5


# ── The plugin ──────────────────────────────────────────────────────

class MelodyGeneratorPlugin(AIGeneratorPlugin):
    """
    AI melody generator.

    Generates melodies with:
    - Chord-tone-aware note selection
    - 6 contour shapes (arch, rising, falling, wave, flat, spike)
    - Multiple rhythmic styles
    - Configurable note density and range
    - Passing and neighbor tone decoration
    - Velocity shaping for phrasing
    - 3 alternatives per generation
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = Random(seed)
        self._initialized = False

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="amusiment.melody-generator",
            name="Melody Generator",
            version="1.0.0",
            category=PluginCategory.AI_GENERATOR,
            author="amusiment",
            description="Generates melodies over chord progressions with "
                        "contour shaping, chord-tone awareness, and rhythmic "
                        "variation. Supports 6 contour shapes and multiple "
                        "musical styles.",
            capabilities=[
                "generate.melody",
                "generate.lead",
                "generate.variation",
            ],
        )

    def initialize(self) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def get_capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            content_types=[
                ContentType.MELODY,
                ContentType.LEAD,
                ContentType.ARPEGGIO,
                ContentType.VARIATION,
            ],
            supports_continuation=True,
            supports_variation=True,
            supports_style_transfer=False,
            max_bars=64,
            min_bars=1,
            supports_audio_input=False,
            supports_text_prompt=True,
            style_tags=[
                "pop", "jazz", "classical", "lofi", "edm", "rnb", "rock",
                "happy", "sad", "energetic", "chill",
                "simple", "complex",
                "arch", "rising", "falling", "wave", "flat", "spike",
            ],
            model_name="MelodyEngine v1.0 (theory-driven)",
        )

    def generate(self, prompt: GenerationPrompt) -> GeneratedContent:
        ctx = prompt.context
        constraints = prompt.constraints
        rng = Random(constraints.seed) if constraints.seed else self._rng

        # ── Determine key ─────────────────────────────────────────
        tonic_pc = (ctx.key_sharps_flats * 7) % 12 if ctx.key_sharps_flats >= 0 \
                   else (abs(ctx.key_sharps_flats) * 5) % 12
        mode = ctx.key_mode if ctx.key_mode in ("major", "minor") else "major"
        scale_name = "major" if mode == "major" else "natural_minor"
        scale_pcs = scale_degrees(tonic_pc, scale_name)

        # ── Parse chord progression ───────────────────────────────
        chords = []
        if ctx.chord_progression:
            for sym in ctx.chord_progression:
                try:
                    root_pc, ctype = parse_chord_symbol(sym)
                    chords.append((root_pc, ctype, chord_name(root_pc, ctype)))
                except ValueError:
                    continue

        if not chords:
            # Default: I-IV-V-I
            from .theory import diatonic_chords
            dt = diatonic_chords(tonic_pc, scale_name, use_sevenths=False)
            chords = [(dt[0].root_pc, dt[0].quality, dt[0].name),
                      (dt[3].root_pc, dt[3].quality, dt[3].name),
                      (dt[4].root_pc, dt[4].quality, dt[4].name),
                      (dt[0].root_pc, dt[0].quality, dt[0].name)]

        # ── Determine parameters ──────────────────────────────────
        bar_count = ctx.bar_count or 8
        total_beats = bar_count * 4  # assume 4/4

        density = getattr(ctx, 'density_target', 0.5)
        energy = getattr(ctx, 'energy_target', 0.5)

        # Rhythm style from density
        if density < 0.3:
            rhythm_style = "sparse"
        elif density < 0.6:
            rhythm_style = "swing"
        elif density < 0.8:
            rhythm_style = "syncopated"
        else:
            rhythm_style = "dense"

        # Contour from style tags
        contour = "arch"  # default
        for tag in ctx.style_tags:
            if tag.lower() in CONTOUR_SHAPES:
                contour = tag.lower()
                break

        # Octave range
        min_pitch = constraints.min_pitch or 60
        max_pitch = constraints.max_pitch or 84

        # ── Generate ───────────────────────────────────────────────
        content = self._build_melody(
            tonic_pc=tonic_pc,
            scale_name=scale_name,
            scale_pcs=scale_pcs,
            chords=chords,
            bar_count=bar_count,
            total_beats=total_beats,
            rhythm_style=rhythm_style,
            contour=contour,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            energy=energy,
            density=density,
            constraints=constraints,
            rng=rng,
        )

        # ── Alternatives ──────────────────────────────────────────
        alt_shapes = [s for s in CONTOUR_SHAPES if s != contour]
        if len(alt_shapes) > 3:
            alt_shapes = [alt_shapes[i] for i in rng.sample(range(len(alt_shapes)), 3)]

        alternatives = []
        for alt_contour in alt_shapes:
            alt = self._build_melody(
                tonic_pc=tonic_pc,
                scale_name=scale_name,
                scale_pcs=scale_pcs,
                chords=chords,
                bar_count=bar_count,
                total_beats=total_beats,
                rhythm_style=rhythm_style,
                contour=alt_contour,
                min_pitch=min_pitch,
                max_pitch=max_pitch,
                energy=energy,
                density=density,
                constraints=constraints,
                rng=rng,
            )
            alternatives.append(alt)

        content.alternatives = alternatives
        return content

    def get_parameters(self) -> dict[str, dict]:
        return {
            "density": {
                "name": "Note Density",
                "min": 0.1, "max": 1.0, "default": 0.5,
                "description": "How many notes per bar",
            },
            "contour": {
                "name": "Melodic Contour",
                "min": 0, "max": 1, "default": 0,
                "description": "Shape of the melody",
                "type": "choice",
                "options": list(CONTOUR_SHAPES.keys()),
            },
            "temperature": {
                "name": "Creativity",
                "min": 0.0, "max": 1.0, "default": 0.7,
                "description": "How much to deviate from chord tones",
            },
        }

    def _build_melody(
        self,
        tonic_pc: int,
        scale_name: str,
        scale_pcs: list[int],
        chords: list[tuple[int, str, str]],
        bar_count: int,
        total_beats: float,
        rhythm_style: str,
        contour: str,
        min_pitch: int,
        max_pitch: int,
        energy: float,
        density: float,
        constraints: GenerationConstraints,
        rng: Random,
    ) -> GeneratedContent:
        """Build a melody as GeneratedContent."""

        # Get beat positions
        beat_positions = rhythm_pattern(rhythm_style, int(total_beats), rng)
        if not beat_positions:
            beat_positions = list(range(int(total_beats)))

        # Build pitch range
        pitch_range = max_pitch - min_pitch
        center_pitch = (min_pitch + max_pitch) // 2

        # Assign chord to each beat
        chord_duration_beats = total_beats / len(chords) if chords else 4.0
        chord_duration_beats = max(2.0, chord_duration_beats)  # at least 2 beats per chord

        notes: list[NoteEvent] = []
        prev_pitch = center_pitch  # start near center

        for i, pos in enumerate(beat_positions):
            if i >= len(beat_positions):
                break

            # Which chord is active?
            chord_idx = min(int(pos / chord_duration_beats), len(chords) - 1)
            root_pc, ctype, cname = chords[chord_idx] if chords else (0, "maj", "C")
            ctones = chord_tones(root_pc, ctype)

            # Determine if this is a strong or weak beat
            is_strong = (pos % 1.0) < 0.01  # exact beat = strong

            # Contour weight at this position
            w = _contour_weight(pos, total_beats, contour)
            target_pitch = int(min_pitch + w * pitch_range)

            # Select pitch
            temperature = constraints.temperature if constraints else 0.7

            if is_strong or rng.random() < (1.0 - temperature * 0.5):
                # Prefer chord tones on strong beats
                chord_midis = [
                    pc_to_midi(ct, 4) if pc_to_midi(ct, 4) >= min_pitch
                    else pc_to_midi(ct, 5)
                    for ct in ctones
                ]
                # Adjust to range
                candidates = []
                for cm in chord_midis:
                    for oct_offset in range(-1, 3):
                        candidate = cm + oct_offset * 12
                        if min_pitch <= candidate <= max_pitch:
                            candidates.append(candidate)
                if not candidates:
                    candidates = [center_pitch]
            else:
                # Scale tones (passing/neighbor)
                scale_midis = []
                for pc in scale_pcs:
                    base = pc_to_midi(pc, 4)
                    for oct_offset in range(-1, 3):
                        candidate = base + oct_offset * 12
                        if min_pitch <= candidate <= max_pitch:
                            scale_midis.append(candidate)
                candidates = list(set(scale_midis))
                if not candidates:
                    candidates = [center_pitch]

            # Select from candidates, preferring smooth voice leading
            # Weight: closer to target_pitch + smooth from previous
            if candidates:
                weights = []
                for c in candidates:
                    contour_weight = 1.0 / (1.0 + abs(c - target_pitch) / pitch_range * 3)
                    leading_weight = 1.0 / (1.0 + abs(c - prev_pitch) / 12)
                    total_weight = contour_weight * 0.5 + leading_weight * 0.5
                    weights.append(total_weight)

                total = sum(weights)
                probs = [w / total for w in weights]

                # Weighted random selection
                r = rng.random()
                cumulative = 0.0
                chosen = candidates[0]
                for j, prob in enumerate(probs):
                    cumulative += prob
                    if r <= cumulative:
                        chosen = candidates[j]
                        break

                pitch = chosen
            else:
                pitch = center_pitch

            # Avoid big jumps (more than an octave)
            if abs(pitch - prev_pitch) > 12:
                if pitch > prev_pitch:
                    pitch = min(pitch, prev_pitch + 12)
                else:
                    pitch = max(pitch, prev_pitch - 12)

            # Duration: based on position spacing
            if i + 1 < len(beat_positions):
                gap = beat_positions[i + 1] - pos
            else:
                gap = 1.0

            dur_beats = min(gap * 0.85, 2.0)  # leave gap, cap at 2 beats
            dur_ticks = beats_to_ticks(max(0.125, dur_beats))

            start_tick = beats_to_ticks(pos)

            # Velocity: stronger on strong beats, swell with contour
            base_vel = 90 if is_strong else 70
            energy_boost = int(energy * 30)
            vel = min(127, max(30, base_vel + energy_boost + rng.randint(-10, 10)))

            notes.append(NoteEvent(
                pitch=NotePitch(max(0, min(127, pitch))),
                velocity=NoteVelocity(vel),
                start_tick=start_tick,
                duration_ticks=dur_ticks,
                channel=0,
            ))

            prev_pitch = pitch

        # Build clip
        clip = MidiClip(
            name=f"Melody - {contour}",
            start_tick=Ticks(0),
            length_ticks=beats_to_ticks(total_beats),
            notes=tuple(notes),
        )

        explanation = (
            f"Generated a {bar_count}-bar melody with {contour} contour "
            f"over {len(chords)} chords. "
            f"{len(notes)} notes, density={density:.1f}, "
            f"range={min_pitch}-{max_pitch}."
        )

        return GeneratedContent(
            clips={ContentType.MELODY: clip},
            explanation=explanation,
            confidence=0.9,
            warnings=[],
            parameters_used={
                "density": density,
                "contour": contour,
                "rhythm": rhythm_style,
                "bar_count": bar_count,
            },
        )
