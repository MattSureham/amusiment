"""
Unified time model for the amusiment framework.

All musical timing is stored internally as ticks at a fixed PPQ resolution.
Conversions to/from beats, seconds, and samples are provided as utility functions.

Design decisions:
- PPQ = 960 (pulses per quarter note) — industry standard, matches MIDI 2.0 recommendation
- All conversions are exact (no floating-point accumulation in sequencer)
- Tempo map supports arbitrary tempo changes at any tick position
- Time signature map supports changes (e.g., 4/4 → 3/4 → 6/8)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NewType

# ── Primitive time types ──────────────────────────────────────────

# PPQ: Pulses (ticks) Per Quarter note
PPQ: int = 960

# NewType wrappers for type safety (zero-cost at runtime, catches bugs at type-check time)
Ticks = NewType("Ticks", int)        # Internal clock: integer tick count
Beats = NewType("Beats", float)      # Musical beats (quarter notes)
Seconds = NewType("Seconds", float)  # Wall-clock seconds
Samples = NewType("Samples", int)    # Audio samples (integer count)


# ── Conversion functions ───────────────────────────────────────────

def ticks_to_beats(ticks: Ticks) -> Beats:
    """Convert ticks to beats (quarter notes)."""
    return Beats(ticks / PPQ)


def beats_to_ticks(beats: Beats) -> Ticks:
    """Convert beats to ticks. Rounds to nearest integer tick."""
    return Ticks(round(beats * PPQ))


def ticks_to_seconds(
    ticks: Ticks,
    tempo_map: "TempoMap",
    start_bpm: float = 120.0,
) -> Seconds:
    """
    Convert absolute ticks to wall-clock seconds, accounting for the tempo map.
    
    Walks through tempo changes to accumulate elapsed time accurately.
    
    Args:
        ticks: Absolute tick position to convert.
        tempo_map: The project's tempo map.
        start_bpm: Fallback BPM if tempo_map has no changes.
    
    Returns:
        Elapsed seconds from tick 0 to the given tick.
    """
    if ticks <= 0:
        return Seconds(0.0)
    
    elapsed = 0.0
    prev_tick = Ticks(0)
    prev_bpm = start_bpm
    
    for change in sorted(tempo_map.changes, key=lambda c: c.tick):
        if change.tick >= ticks:
            break
        # Time from prev_tick to this change at prev_bpm
        delta_ticks = change.tick - prev_tick
        elapsed += (delta_ticks / PPQ) * (60.0 / prev_bpm)
        prev_tick = change.tick
        prev_bpm = change.bpm
    
    # Remaining ticks after last tempo change
    delta_ticks = ticks - prev_tick
    elapsed += (delta_ticks / PPQ) * (60.0 / prev_bpm)
    
    return Seconds(elapsed)


def seconds_to_ticks(
    seconds: Seconds,
    tempo_map: "TempoMap",
    start_bpm: float = 120.0,
) -> Ticks:
    """
    Convert wall-clock seconds to absolute ticks, accounting for the tempo map.
    
    Args:
        seconds: Elapsed seconds from tick 0.
        tempo_map: The project's tempo map.
        start_bpm: Fallback BPM if tempo_map has no changes.
    
    Returns:
        Absolute tick position.
    """
    if seconds <= 0:
        return Ticks(0)
    
    remaining = seconds
    prev_tick = Ticks(0)
    prev_bpm = start_bpm
    
    for change in sorted(tempo_map.changes, key=lambda c: c.tick):
        # Time from prev_tick to this change at prev_bpm
        delta_ticks = change.tick - prev_tick
        delta_time = (delta_ticks / PPQ) * (60.0 / prev_bpm)
        
        if remaining <= delta_time:
            # We land within this segment
            frac = remaining / delta_time if delta_time > 0 else 0.0
            return Ticks(prev_tick + round(frac * delta_ticks))
        
        remaining -= delta_time
        prev_tick = change.tick
        prev_bpm = change.bpm
    
    # Remaining time after last tempo change
    result_ticks = prev_tick + round(remaining * prev_bpm * PPQ / 60.0)
    return Ticks(result_ticks)


def ticks_to_samples(
    ticks: Ticks,
    sample_rate: int,
    tempo_map: "TempoMap",
    start_bpm: float = 120.0,
) -> Samples:
    """Convert ticks to audio samples at the given sample rate."""
    secs = ticks_to_seconds(ticks, tempo_map, start_bpm)
    return Samples(round(secs * sample_rate))


def samples_to_ticks(
    samples: Samples,
    sample_rate: int,
    tempo_map: "TempoMap",
    start_bpm: float = 120.0,
) -> Ticks:
    """Convert audio samples to ticks at the given sample rate."""
    secs = Seconds(samples / sample_rate)
    return seconds_to_ticks(secs, tempo_map, start_bpm)


# ── Time signature ─────────────────────────────────────────────────

class BeatValue(IntEnum):
    """Denominator of a time signature (beat unit)."""
    WHOLE = 1
    HALF = 2
    QUARTER = 4
    EIGHTH = 8
    SIXTEENTH = 16
    THIRTY_SECOND = 32


@dataclass(frozen=True)
class TimeSignature:
    """
    A musical time signature.
    
    Attributes:
        numerator: Beats per measure (e.g., 4 in 4/4).
        denominator: Beat unit (e.g., QUARTER in 4/4).
    """
    numerator: int = 4
    denominator: "BeatValue" = BeatValue.QUARTER
    
    @property
    def ticks_per_beat(self) -> int:
        """How many ticks per beat in this signature."""
        # PPQ is always per quarter note, so adjust for denominator
        ratio = BeatValue.QUARTER / self.denominator
        return int(PPQ * ratio)
    
    @property
    def ticks_per_measure(self) -> int:
        """How many ticks per full measure in this signature."""
        return self.ticks_per_beat * self.numerator
    
    @classmethod
    def from_str(cls, s: str) -> "TimeSignature":
        """Parse from string like '4/4', '3/4', '6/8'."""
        num, den = s.split("/")
        return cls(numerator=int(num), denominator=BeatValue(int(den)))
    
    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator.value}"


# ── Tempo ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TempoChange:
    """
    A tempo change event at a specific tick position.
    
    Attributes:
        tick: Absolute tick position of this tempo change.
        bpm: Beats per minute from this point forward.
    """
    tick: Ticks
    bpm: float  # BPM (beats per minute)
    
    def __post_init__(self):
        if self.bpm <= 0:
            raise ValueError(f"BPM must be positive, got {self.bpm}")


@dataclass(frozen=True)
class TempoMap:
    """
    Ordered collection of tempo changes throughout a project.
    
    The initial tempo is 120 BPM unless overridden by the first TempoChange at tick 0.
    """
    changes: tuple[TempoChange, ...] = field(default_factory=tuple)
    
    def bpm_at(self, tick: Ticks) -> float:
        """Return the effective BPM at the given tick position."""
        effective = 120.0
        for change in sorted(self.changes, key=lambda c: c.tick):
            if change.tick > tick:
                break
            effective = change.bpm
        return effective
    
    def with_change(self, change: TempoChange) -> "TempoMap":
        """Return a new TempoMap with the given change added (replaces at same tick)."""
        others = tuple(c for c in self.changes if c.tick != change.tick)
        return TempoMap(changes=tuple(sorted(
            others + (change,), key=lambda c: c.tick
        )))
    
    def without_tick(self, tick: Ticks) -> "TempoMap":
        """Return a new TempoMap with the change at the given tick removed."""
        return TempoMap(changes=tuple(c for c in self.changes if c.tick != tick))
    
    def to_dict(self) -> list[dict]:
        return [{"tick": c.tick, "bpm": c.bpm} for c in self.changes]
    
    @classmethod
    def from_dict(cls, data: list[dict]) -> "TempoMap":
        return cls(changes=tuple(
            TempoChange(tick=Ticks(d["tick"]), bpm=d["bpm"]) for d in data
        ))
