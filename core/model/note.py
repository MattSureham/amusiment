"""
Musical note events — the fundamental building block of MIDI content.

Notes are immutable value objects. They represent a single note-on/note-off
pair with pitch, velocity, timing, and channel information.
"""

from dataclasses import dataclass, field
from typing import NewType, Optional
from .time_model import Ticks, beats_to_ticks, Beats

# ── Type aliases for clarity ───────────────────────────────────────

NotePitch = NewType("NotePitch", int)       # MIDI note number 0-127 (60 = C4)
NoteVelocity = NewType("NoteVelocity", int)  # MIDI velocity 0-127
NoteDuration = NewType("NoteDuration", int)  # Duration in ticks


# ── MIDI constants ──────────────────────────────────────────────────

MIDI_PITCH_MIN = 0
MIDI_PITCH_MAX = 127
MIDI_VELOCITY_MIN = 0
MIDI_VELOCITY_MAX = 127
MIDI_CHANNEL_MIN = 0
MIDI_CHANNEL_MAX = 15

# Common note names (C4 = middle C = MIDI 60)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_to_name(pitch: NotePitch) -> str:
    """Convert MIDI pitch to note name with octave (e.g., 60 -> 'C4')."""
    octave = (pitch // 12) - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def name_to_pitch(name: str) -> NotePitch:
    """Convert note name to MIDI pitch (e.g., 'C4' -> 60)."""
    # Parse: note letter + optional sharp + octave
    if len(name) < 2:
        raise ValueError(f"Invalid note name: {name}")
    
    # Handle sharp/flat
    if name[1] in ("#", "b"):
        note_part = name[:2]
        octave_part = name[2:]
    else:
        note_part = name[0]
        octave_part = name[1:]
    
    if note_part not in NOTE_NAMES:
        raise ValueError(f"Unknown note: {note_part}")
    
    octave = int(octave_part)
    semitone = NOTE_NAMES.index(note_part)
    return NotePitch((octave + 1) * 12 + semitone)


# ── Note event ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class NoteEvent:
    """
    A single MIDI note event with immutable attributes.
    
    Attributes:
        pitch: MIDI note number (0-127).
        velocity: Note-on velocity (0-127). 0 = rest/silence.
        start_tick: Tick position of note-on.
        duration_ticks: Duration from note-on to note-off, in ticks.
        channel: MIDI channel (0-15).
        articulation: Optional articulation tag (staccato, legato, accent, etc.).
    """
    pitch: NotePitch
    velocity: NoteVelocity
    start_tick: Ticks
    duration_ticks: Ticks
    channel: int = 0
    articulation: Optional[str] = None
    
    def __post_init__(self):
        if not MIDI_PITCH_MIN <= self.pitch <= MIDI_PITCH_MAX:
            raise ValueError(f"Pitch {self.pitch} out of range [{MIDI_PITCH_MIN}, {MIDI_PITCH_MAX}]")
        if not MIDI_VELOCITY_MIN <= self.velocity <= MIDI_VELOCITY_MAX:
            raise ValueError(f"Velocity {self.velocity} out of range [{MIDI_VELOCITY_MIN}, {MIDI_VELOCITY_MAX}]")
        if not MIDI_CHANNEL_MIN <= self.channel <= MIDI_CHANNEL_MAX:
            raise ValueError(f"Channel {self.channel} out of range [{MIDI_CHANNEL_MIN}, {MIDI_CHANNEL_MAX}]")
        if self.duration_ticks < 0:
            raise ValueError(f"Duration must be non-negative, got {self.duration_ticks}")
    
    @property
    def end_tick(self) -> Ticks:
        """Tick position of note-off."""
        return Ticks(self.start_tick + self.duration_ticks)
    
    @property
    def pitch_name(self) -> str:
        """Human-readable note name (e.g., 'C4')."""
        return pitch_to_name(self.pitch)
    
    def overlaps(self, other: "NoteEvent") -> bool:
        """Check if this note temporally overlaps with another."""
        return self.start_tick < other.end_tick and other.start_tick < self.end_tick
    
    def contains_tick(self, tick: Ticks) -> bool:
        """Check if this note is sounding at the given tick."""
        return self.start_tick <= tick < self.end_tick
    
    def transposed(self, semitones: int) -> "NoteEvent":
        """Return a new NoteEvent transposed by the given number of semitones."""
        new_pitch = NotePitch(
            max(MIDI_PITCH_MIN, min(MIDI_PITCH_MAX, self.pitch + semitones))
        )
        return NoteEvent(
            pitch=new_pitch,
            velocity=self.velocity,
            start_tick=self.start_tick,
            duration_ticks=self.duration_ticks,
            channel=self.channel,
            articulation=self.articulation,
        )
    
    def with_velocity(self, velocity: NoteVelocity) -> "NoteEvent":
        """Return a copy with a different velocity."""
        return NoteEvent(
            pitch=self.pitch,
            velocity=velocity,
            start_tick=self.start_tick,
            duration_ticks=self.duration_ticks,
            channel=self.channel,
            articulation=self.articulation,
        )
    
    def moved(self, delta_ticks: int) -> "NoteEvent":
        """Return a copy shifted in time by delta_ticks."""
        new_start = Ticks(max(0, self.start_tick + delta_ticks))
        return NoteEvent(
            pitch=self.pitch,
            velocity=self.velocity,
            start_tick=new_start,
            duration_ticks=self.duration_ticks,
            channel=self.channel,
            articulation=self.articulation,
        )
    
    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return {
            "pitch": self.pitch,
            "velocity": self.velocity,
            "start_tick": self.start_tick,
            "duration_ticks": self.duration_ticks,
            "channel": self.channel,
            "articulation": self.articulation,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NoteEvent":
        """Deserialize from a plain dict."""
        return cls(
            pitch=NotePitch(data["pitch"]),
            velocity=NoteVelocity(data["velocity"]),
            start_tick=Ticks(data["start_tick"]),
            duration_ticks=Ticks(data["duration_ticks"]),
            channel=data.get("channel", 0),
            articulation=data.get("articulation"),
        )
    
    def __repr__(self) -> str:
        art = f" [{self.articulation}]" if self.articulation else ""
        return (
            f"Note({self.pitch_name}, vel={self.velocity}, "
            f"start={self.start_tick}, dur={self.duration_ticks}, "
            f"ch={self.channel}{art})"
        )
