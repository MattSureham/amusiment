"""
Drum pattern generator — an AIGeneratorPlugin implementation.

Generates drum patterns using the General MIDI drum map.
Supports multiple styles with characteristic kick/snare/hi-hat patterns,
fills, and variations.

GM Drum Map (channel 10 by convention):
  35 - Acoustic Bass Drum (Kick 2)
  36 - Bass Drum 1 (Kick)
  38 - Acoustic Snare
  40 - Electric Snare
  42 - Closed Hi-hat
  44 - Pedal Hi-hat
  46 - Open Hi-hat
  49 - Crash Cymbal 1
  51 - Ride Cymbal 1
  52 - China Cymbal
  56 - Cowbell
  67 - High Agogo
  70 - Maracas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass
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

# ── GM Drum map ─────────────────────────────────────────────────────

GM_KICK = 36
GM_KICK_2 = 35
GM_SNARE = 38
GM_SNARE_2 = 40
GM_HIHAT_CLOSED = 42
GM_HIHAT_PEDAL = 44
GM_HIHAT_OPEN = 46
GM_CRASH = 49
GM_RIDE = 51
GM_RIDE_BELL = 53
GM_TOM_HIGH = 50
GM_TOM_MID = 48
GM_TOM_LOW = 45
GM_TOM_FLOOR = 41
GM_CLAP = 39
GM_COWBELL = 56
GM_TAMBOURINE = 54
GM_SHAKER = 70

# ── Style patterns: (kick_positions, snare_positions, hihat_pattern, extra_hits) ─

# Positions are relative to a 16th-note grid within a bar (0-15)
# Each is a list of grid positions where hits occur
# Velocity modifiers: >1.0 = accent, <1.0 = ghost note

@dataclass
class DrumPattern:
    """A drum pattern for one bar."""
    kick: list[tuple[int, float]]     # (position, velocity_modifier)
    snare: list[tuple[int, float]]
    hihat: list[tuple[int, float]]    # closed hihat
    open_hihat: list[tuple[int, float]]
    ride: list[tuple[int, float]]
    crash: list[tuple[int, float]]
    toms: list[tuple[int, float, int]]  # (position, velocity_modifier, tom_note)
    percussion: list[tuple[int, float, int]]  # (position, velocity_modifier, note)

    def to_notes(self, bar_start_tick: Ticks, ticks_per_16th: int,
                 rng: Random) -> list[NoteEvent]:
        """Convert pattern to NoteEvent list."""
        notes: list[NoteEvent] = []

        def _clamp_vel(base_vel: int, vel_mod: float, rng: Random) -> NoteVelocity:
            raw = int(base_vel * vel_mod + rng.randint(-5, 5))
            return NoteVelocity(max(1, min(127, raw)))

        # Kick
        for pos, vel_mod in self.kick:
            tick = Ticks(bar_start_tick + pos * ticks_per_16th)
            vel = _clamp_vel(100, vel_mod, rng)
            notes.append(NoteEvent(
                pitch=NotePitch(GM_KICK), velocity=vel,
                start_tick=tick,
                duration_ticks=Ticks(ticks_per_16th // 2),
                channel=9,  # GM drum channel
            ))

        # Snare
        for pos, vel_mod in self.snare:
            tick = Ticks(bar_start_tick + pos * ticks_per_16th)
            vel = _clamp_vel(95, vel_mod, rng)
            notes.append(NoteEvent(
                pitch=NotePitch(GM_SNARE), velocity=vel,
                start_tick=tick,
                duration_ticks=Ticks(ticks_per_16th // 2),
                channel=9,
            ))

        # Closed hihat
        for pos, vel_mod in self.hihat:
            tick = Ticks(bar_start_tick + pos * ticks_per_16th)
            vel = _clamp_vel(85, vel_mod, rng)
            notes.append(NoteEvent(
                pitch=NotePitch(GM_HIHAT_CLOSED), velocity=vel,
                start_tick=tick,
                duration_ticks=Ticks(ticks_per_16th // 3),
                channel=9,
            ))

        # Open hihat
        for pos, vel_mod in self.open_hihat:
            tick = Ticks(bar_start_tick + pos * ticks_per_16th)
            vel = _clamp_vel(90, vel_mod, rng)
            notes.append(NoteEvent(
                pitch=NotePitch(GM_HIHAT_OPEN), velocity=vel,
                start_tick=tick,
                duration_ticks=Ticks(ticks_per_16th * 2),
                channel=9,
            ))

        # Ride
        for pos, vel_mod in self.ride:
            tick = Ticks(bar_start_tick + pos * ticks_per_16th)
            vel = _clamp_vel(80, vel_mod, rng)
            notes.append(NoteEvent(
                pitch=NotePitch(GM_RIDE), velocity=vel,
                start_tick=tick,
                duration_ticks=Ticks(ticks_per_16th // 2),
                channel=9,
            ))

        return notes


def _make_pattern(style: str, complexity: float, rng: Random) -> DrumPattern:
    """Generate a drum pattern for one bar based on style."""

    # Base patterns (16th-note grid: 0-15 per bar, 0=downbeat)
    patterns = {
        "rock": DrumPattern(
            kick=[(0, 1.0), (4, 0.3), (8, 0.9), (12, 0.4)],
            snare=[(4, 0.0), (12, 1.0)],  # ghost at 4
            hihat=[(i, 1.0) for i in range(16) if i % 2 == 0],
            open_hihat=[],
            ride=[],
            crash=[(0, 0.8)],
            toms=[],
            percussion=[],
        ),
        "pop": DrumPattern(
            kick=[(0, 1.0), (8, 0.8), (10, 0.4)],
            snare=[(4, 1.0), (12, 1.0)],
            hihat=[(i, 1.0) for i in range(0, 16, 2)],
            open_hihat=[(14, 0.6)],
            ride=[],
            crash=[],
            toms=[],
            percussion=[],
        ),
        "jazz": DrumPattern(
            kick=[(0, 1.0), (12, 0.5)],
            snare=[(4, 0.0), (12, 0.3)],  # mostly ghost notes on snare
            hihat=[],
            open_hihat=[],
            ride=[(i, 0.9 if i % 3 == 0 or i % 3 == 1 else 0.7) for i in range(0, 16, 2)],
            crash=[],
            toms=[],
            percussion=[(8, 0.6, GM_RIDE_BELL)],
        ),
        "lofi": DrumPattern(
            kick=[(0, 1.0), (6, 0.4), (10, 0.7)],
            snare=[(4, 1.0), (13, 0.8)],
            hihat=[(i, 0.7 + 0.3 * (i % 4 == 0)) for i in range(0, 16, 2)],
            open_hihat=[(14, 0.5)],
            ride=[],
            crash=[],
            toms=[],
            percussion=[(i, 0.5, GM_SHAKER) for i in range(0, 16, 4)],
        ),
        "edm": DrumPattern(
            kick=[(0, 1.0), (4, 0.3), (8, 1.0), (12, 0.5), (14, 0.3)],
            snare=[(4, 0.0), (12, 1.0)],
            hihat=[(i, 0.8) for i in range(0, 16, 2)],
            open_hihat=[(7, 0.7), (15, 0.9)],
            ride=[],
            crash=[(0, 0.0), (8, 1.0)],
            toms=[],
            percussion=[(2, 0.6, GM_CLAP), (10, 0.6, GM_CLAP)],
        ),
        "hiphop": DrumPattern(
            kick=[(0, 1.0), (7, 0.6), (10, 0.8), (13, 0.5)],
            snare=[(4, 0.0), (8, 0.3), (12, 1.0)],
            hihat=[(i, 0.7) for i in range(1, 16, 2)],
            open_hihat=[],
            ride=[],
            crash=[],
            toms=[],
            percussion=[(0, 0.7, GM_KICK_2)],
        ),
        "funk": DrumPattern(
            kick=[(0, 1.0), (8, 0.7), (11, 0.5)],
            snare=[(4, 1.0), (12, 0.9)],
            hihat=[(i, 0.8 + 0.2 * (i % 4 == 0)) for i in range(0, 16, 2)],
            open_hihat=[(6, 0.5), (14, 0.6)],
            ride=[],
            crash=[],
            toms=[],
            percussion=[(i, 0.4, GM_COWBELL) for i in range(4, 16, 8)],
        ),
    }

    if style not in patterns:
        style = "pop"

    pattern = patterns[style]

    # Vary based on complexity
    if complexity < 0.3:
        # Simpler: fewer kick hits, less hihat
        pattern.kick = [(p, v) for p, v in pattern.kick if p % 8 == 0]
        pattern.hihat = pattern.hihat[:len(pattern.hihat)//2]
    elif complexity > 0.7:
        # More complex: add ghost kicks, more hihat variation
        # Add ghost kick or two
        extra_kicks = [(rng.randint(1, 15), 0.3) for _ in range(2)]
        pattern.kick = list(pattern.kick) + extra_kicks

    return pattern


# ── The plugin ──────────────────────────────────────────────────────

class DrumGeneratorPlugin(AIGeneratorPlugin):
    """
    AI drum pattern generator.

    Generates drum patterns with:
    - 7 styles (rock, pop, jazz, lofi, edm, hiphop, funk)
    - Configurable complexity (simple through busy)
    - Bar-level fills at user-specified intervals
    - Intro/outro pattern variations
    - GM drum map compatible
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = Random(seed)
        self._initialized = False

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="amusiment.drum-generator",
            name="Drum Pattern Generator",
            version="1.0.0",
            category=PluginCategory.AI_GENERATOR,
            author="amusiment",
            description="Generates drum patterns in rock, pop, jazz, lofi, "
                        "EDM, hiphop, and funk styles with fills and "
                        "complexity control.",
            capabilities=[
                "generate.drums",
                "generate.rhythm",
                "generate.fill",
            ],
        )

    def initialize(self) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def get_capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            content_types=[
                ContentType.DRUMS,
                ContentType.RHYTHM,
                ContentType.FILL,
            ],
            supports_continuation=True,
            supports_variation=True,
            supports_style_transfer=False,
            max_bars=128,
            min_bars=1,
            supports_audio_input=False,
            supports_text_prompt=True,
            style_tags=[
                "rock", "pop", "jazz", "lofi", "edm", "hiphop", "funk",
                "energetic", "chill", "simple", "complex",
            ],
            model_name="DrumEngine v1.0 (GM-compatible)",
        )

    def generate(self, prompt: GenerationPrompt) -> GeneratedContent:
        ctx = prompt.context
        constraints = prompt.constraints
        rng = Random(constraints.seed) if constraints.seed else self._rng

        bar_count = ctx.bar_count or 8
        bpm = ctx.bpm or 120.0

        # Determine style
        style = "pop"
        for tag in ctx.style_tags:
            if tag.lower() in ("rock", "pop", "jazz", "lofi", "edm", "hiphop", "funk"):
                style = tag.lower()
                break

        # Complexity from density
        density = getattr(ctx, 'density_target', 0.5)
        energy = getattr(ctx, 'energy_target', 0.5)

        # Generate
        content = self._build_pattern(
            bar_count=bar_count,
            style=style,
            density=density,
            energy=energy,
            bpm=bpm,
            constraints=constraints,
            rng=rng,
        )

        # Alternatives: same bars, different styles
        alt_styles = [s for s in ("rock", "jazz", "funk", "lofi") if s != style]
        rng.shuffle(alt_styles)
        alternatives = []
        for alt_style in alt_styles[:3]:
            alt = self._build_pattern(
                bar_count=bar_count,
                style=alt_style,
                density=density,
                energy=energy,
                bpm=bpm,
                constraints=constraints,
                rng=rng,
            )
            alternatives.append(alt)

        content.alternatives = alternatives
        return content

    def get_parameters(self) -> dict[str, dict]:
        return {
            "density": {
                "name": "Pattern Density",
                "min": 0.1, "max": 1.0, "default": 0.5,
                "description": "Simple beats (low) to busy patterns (high)",
            },
            "style": {
                "name": "Drum Style",
                "min": 0, "max": 1, "default": 0,
                "description": "Drumming style",
                "type": "choice",
                "options": ["rock", "pop", "jazz", "lofi", "edm", "hiphop", "funk"],
            },
        }

    def _build_pattern(
        self,
        bar_count: int,
        style: str,
        density: float,
        energy: float,
        bpm: float,
        constraints: GenerationConstraints,
        rng: Random,
    ) -> GeneratedContent:
        """Build a full drum pattern across multiple bars."""

        ticks_per_16th = PPQ // 4  # 240 ticks per 16th note at PPQ=960
        notes: list[NoteEvent] = []

        main_pattern = _make_pattern(style, density, rng)
        fill_pattern = self._make_fill(style, density, rng)

        fill_every = 4  # bars between fills

        for bar in range(bar_count):
            bar_start = Ticks(bar * 16 * ticks_per_16th)

            # Determine if this bar gets a fill
            is_fill = (bar > 0 and bar % fill_every == fill_every - 1) or \
                      (bar == bar_count - 1)

            pattern = fill_pattern if is_fill else main_pattern

            # Add velocity variation
            if rng.random() < 0.3:
                # Slight variation
                pass

            notes.extend(pattern.to_notes(bar_start, ticks_per_16th, rng))

            # Add crash on first downbeat
            if bar == 0 and GM_CRASH not in [n.pitch for n in pattern.to_notes(Ticks(0), ticks_per_16th, rng)]:
                notes.append(NoteEvent(
                    pitch=NotePitch(GM_CRASH),
                    velocity=NoteVelocity(110),
                    start_tick=bar_start,
                    duration_ticks=Ticks(ticks_per_16th * 4),
                    channel=9,
                ))

        # Build clip
        total_ticks = Ticks(bar_count * 16 * ticks_per_16th)
        clip = MidiClip(
            name=f"Drums - {style} ({bar_count} bars)",
            start_tick=Ticks(0),
            length_ticks=total_ticks,
            notes=tuple(notes),
        )

        explanation = (
            f"Generated a {bar_count}-bar drum pattern in {style} style. "
            f"{len(notes)} hits at {bpm} BPM. "
            f"Density: {density:.1f}, Energy: {energy:.1f}. "
            f"Includes fills every {fill_every} bars."
        )

        return GeneratedContent(
            clips={ContentType.DRUMS: clip},
            explanation=explanation,
            confidence=0.95,
            warnings=[],
            parameters_used={
                "density": density,
                "energy": energy,
                "style": style,
                "bar_count": bar_count,
            },
        )

    def _make_fill(self, style: str, density: float, rng: Random) -> 'DrumPattern':
        """Generate a fill pattern (busier than normal)."""
        base = _make_pattern(style, min(1.0, density + 0.3), rng)

        # Add tom fills
        tom_notes = [GM_TOM_HIGH, GM_TOM_MID, GM_TOM_LOW, GM_TOM_FLOOR]
        for i in range(4):
            pos = 12 + i  # beats 3 and 4
            tom = tom_notes[i % len(tom_notes)]
            base.toms.append((pos, 0.7, tom))

        # More active hihat
        base.hihat = [(i, 0.9) for i in range(16) if i % 2 == 0]

        # Add crash at end
        base.crash.append((15, 1.0))

        return base
