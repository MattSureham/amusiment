"""
Musical context window management for multi-turn AI generation.

Maintains a sliding window of musical context across multiple
generation calls, enabling coherent multi-turn interactions
(e.g., "make it more energetic" → "now add a bridge").
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass, field
from typing import Optional, Any

from core.plugin.interfaces.ai_generator import MusicalContext, GeneratedContent


@dataclass
class ContextWindow:
    """
    Maintains musical context across multiple generation turns.

    Stores previous generation parameters and results, enabling
    the system to understand what was already generated and make
    coherent follow-up suggestions.

    Usage:
        window = ContextWindow(max_history=10)
        # First generation
        ctx = window.build_context(bpm=120, key="C major")
        # ... after generation ...
        window.record_generation(prompt, result)
        # Next turn
        ctx = window.build_context()  # includes history
    """

    max_history: int = 10
    _turn_history: list[dict[str, Any]] = field(default_factory=list)
    _last_bpm: float = 120.0
    _last_key_sharps_flats: int = 0
    _last_key_mode: str = "major"
    _last_style_tags: list[str] = field(default_factory=list)
    _generated_clips: dict[str, list] = field(default_factory=dict)

    def build_context(
        self,
        bpm: Optional[float] = None,
        key_sharps_flats: Optional[int] = None,
        key_mode: Optional[str] = None,
        bar_start: int = 0,
        bar_count: int = 8,
        style_tags: Optional[list[str]] = None,
        energy_target: float = 0.5,
        density_target: float = 0.5,
        chord_progression: Optional[list[str]] = None,
    ) -> MusicalContext:
        """
        Build a MusicalContext incorporating history.

        Args:
            bpm: Tempo (uses last value if None).
            key_sharps_flats: Key signature (uses last value if None).
            key_mode: Mode (uses last value if None).
            bar_start: Starting bar for generation.
            bar_count: Number of bars to generate.
            style_tags: Style/genre tags.
            energy_target: Desired energy level 0.0-1.0.
            density_target: Desired note density 0.0-1.0.
            chord_progression: Optional explicit chord progression.

        Returns:
            MusicalContext object with prior context included.
        """
        # Update stored values
        if bpm is not None:
            self._last_bpm = bpm
        if key_sharps_flats is not None:
            self._last_key_sharps_flats = key_sharps_flats
        if key_mode is not None:
            self._last_key_mode = key_mode
        if style_tags is not None:
            self._last_style_tags = list(style_tags)

        # Collect existing notes from previous generations
        existing_notes = []
        existing_tracks = []
        for content_type, clips in self._generated_clips.items():
            existing_tracks.append({
                "content_type": content_type,
                "clip_count": len(clips),
            })
            for clip in clips:
                if hasattr(clip, 'notes'):
                    existing_notes.extend(clip.notes)

        # Build history summary
        context = MusicalContext(
            bpm=self._last_bpm,
            key_sharps_flats=self._last_key_sharps_flats,
            key_mode=self._last_key_mode,
            bar_start=bar_start,
            bar_count=bar_count,
            existing_tracks=existing_tracks,
            existing_notes=existing_notes,
            chord_progression=chord_progression or [],
            style_tags=list(self._last_style_tags),
            energy_target=energy_target,
            density_target=density_target,
        )

        return context

    def record_generation(
        self,
        content_type: str,
        result: GeneratedContent,
    ) -> None:
        """
        Record a generation result into the context history.

        Args:
            content_type: What was generated (e.g., "chords", "melody").
            result: The GeneratedContent from the AI generator.
        """
        self._turn_history.append({
            "content_type": content_type,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "parameters": result.parameters_used,
        })

        # Store clips by content type for future context
        for ct, clip in result.clips.items():
            ct_name = ct.name.lower()
            if ct_name not in self._generated_clips:
                self._generated_clips[ct_name] = []
            self._generated_clips[ct_name].append(clip)

        # Trim history
        if len(self._turn_history) > self.max_history:
            self._turn_history = self._turn_history[-self.max_history:]

        # Also trim clip storage
        total_clips = sum(len(v) for v in self._generated_clips.values())
        if total_clips > self.max_history * 3:
            # Remove oldest clips
            for ct_name in list(self._generated_clips.keys()):
                if len(self._generated_clips[ct_name]) > 3:
                    self._generated_clips[ct_name] = \
                        self._generated_clips[ct_name][-3:]

    def get_turn_summary(self) -> str:
        """Get a human-readable summary of all generation turns."""
        if not self._turn_history:
            return "No generations recorded yet."

        lines = []
        for i, turn in enumerate(self._turn_history):
            lines.append(
                f"  Turn {i+1}: Generated {turn['content_type']} "
                f"(confidence: {turn['confidence']:.0%})"
            )

        return (
            f"Context: {len(self._turn_history)} generation turns, "
            f"key={pc_to_key_name(self._last_key_sharps_flats, self._last_key_mode)}, "
            f"bpm={self._last_bpm}\n" + "\n".join(lines)
        )

    def clear(self) -> None:
        """Reset the context window."""
        self._turn_history.clear()
        self._generated_clips.clear()
        self._last_bpm = 120.0
        self._last_key_sharps_flats = 0
        self._last_key_mode = "major"
        self._last_style_tags = []

    @property
    def turn_count(self) -> int:
        return len(self._turn_history)

    @property
    def has_context(self) -> bool:
        return len(self._turn_history) > 0


def pc_to_key_name(sharps_flats: int, mode: str) -> str:
    """Convert sharps/flats count to key name."""
    tonic_pc = (sharps_flats * 7) % 12 if sharps_flats >= 0 \
               else (abs(sharps_flats) * 5) % 12
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from ai.models.theory import PITCH_CLASSES, PITCH_CLASSES_FLAT
    if mode in ("minor", "min"):
        return f"{PITCH_CLASSES_FLAT[tonic_pc]} minor"
    return f"{PITCH_CLASSES[tonic_pc]} major"
