"""
Clip models — the containers for musical content on a timeline.

Clips are the primary editable units in the arrangement view.
They live on tracks and contain either MIDI notes, audio references,
or automation data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from uuid import uuid4

from .time_model import Ticks, PPQ
from .note import NoteEvent


class ClipType(Enum):
    """Types of clips that can exist on the timeline."""
    MIDI = auto()
    AUDIO = auto()
    AUTOMATION = auto()


@dataclass(frozen=True)
class Clip:
    """
    Base clip — a positioned container on a track's timeline.
    
    All clip types share: an ID, a position (start_tick), a length, a name,
    and a muted flag.
    """
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = "Clip"
    start_tick: Ticks = Ticks(0)
    length_ticks: Ticks = Ticks(PPQ * 4)  # Default: 1 bar of 4/4
    muted: bool = False
    color: Optional[str] = None  # Hex color for UI display
    
    @property
    def end_tick(self) -> Ticks:
        return Ticks(self.start_tick + self.length_ticks)


@dataclass(frozen=True)
class MidiClip(Clip):
    """
    A clip containing MIDI note events.
    
    Notes are stored as an immutable tuple. The clip itself can span
    a longer range than its contained notes (silence at either end).
    
    Attributes:
        notes: Tuple of NoteEvent objects.
        loop_enabled: Whether this clip loops when playback extends past its end.
    """
    clip_type: ClipType = ClipType.MIDI
    notes: tuple[NoteEvent, ...] = field(default_factory=tuple)
    loop_enabled: bool = False
    
    @property
    def note_count(self) -> int:
        return len(self.notes)
    
    @property
    def pitch_range(self) -> tuple[int, int]:
        """Return (min_pitch, max_pitch) of contained notes, or (60, 60) if empty."""
        if not self.notes:
            return (60, 60)
        pitches = [n.pitch for n in self.notes]
        return (min(pitches), max(pitches))
    
    def notes_in_range(self, start_tick: Ticks, end_tick: Ticks) -> tuple[NoteEvent, ...]:
        """Return notes that overlap with the given tick range."""
        return tuple(
            n for n in self.notes
            if n.start_tick < end_tick and n.end_tick > start_tick
        )
    
    def with_notes(self, notes: tuple[NoteEvent, ...]) -> "MidiClip":
        """Return a new clip with replaced notes."""
        return MidiClip(
            id=self.id,
            name=self.name,
            start_tick=self.start_tick,
            length_ticks=self.length_ticks,
            muted=self.muted,
            color=self.color,
            loop_enabled=self.loop_enabled,
            notes=notes,
        )
    
    def with_added_note(self, note: NoteEvent) -> "MidiClip":
        """Return a new clip with an additional note."""
        return self.with_notes(self.notes + (note,))
    
    def with_removed_note(self, note: NoteEvent) -> "MidiClip":
        """Return a new clip with a specific note removed."""
        return self.with_notes(tuple(n for n in self.notes if n != note))
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_tick": self.start_tick,
            "length_ticks": self.length_ticks,
            "muted": self.muted,
            "color": self.color,
            "clip_type": "midi",
            "loop_enabled": self.loop_enabled,
            "notes": [n.to_dict() for n in self.notes],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MidiClip":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "Clip"),
            start_tick=Ticks(data.get("start_tick", 0)),
            length_ticks=Ticks(data.get("length_ticks", PPQ * 4)),
            muted=data.get("muted", False),
            color=data.get("color"),
            loop_enabled=data.get("loop_enabled", False),
            notes=tuple(NoteEvent.from_dict(n) for n in data.get("notes", [])),
        )


@dataclass(frozen=True)
class AudioClip(Clip):
    """
    A clip referencing an external audio file.
    
    Attributes:
        source_path: Relative path to the audio file within the project.
        source_start_sample: Start offset into the source file (for slip editing).
        gain_db: Clip gain in decibels.
        warp_enabled: Whether time-stretching is active for this clip.
    """
    clip_type: ClipType = ClipType.AUDIO
    source_path: str = ""
    source_start_sample: int = 0
    gain_db: float = 0.0
    warp_enabled: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_tick": self.start_tick,
            "length_ticks": self.length_ticks,
            "muted": self.muted,
            "color": self.color,
            "clip_type": "audio",
            "source_path": self.source_path,
            "source_start_sample": self.source_start_sample,
            "gain_db": self.gain_db,
            "warp_enabled": self.warp_enabled,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AudioClip":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "Audio Clip"),
            start_tick=Ticks(data.get("start_tick", 0)),
            length_ticks=Ticks(data.get("length_ticks", PPQ * 4)),
            muted=data.get("muted", False),
            color=data.get("color"),
            source_path=data.get("source_path", ""),
            source_start_sample=data.get("source_start_sample", 0),
            gain_db=data.get("gain_db", 0.0),
            warp_enabled=data.get("warp_enabled", False),
        )


@dataclass(frozen=True)
class AutomationClip(Clip):
    """
    A clip containing automation envelope data for a specific parameter.
    
    Attributes:
        parameter_id: Which parameter this automation targets.
        points: Automation data points (time, value).
    """
    clip_type: ClipType = ClipType.AUTOMATION
    parameter_id: str = ""
    points: tuple[tuple[Ticks, float], ...] = field(default_factory=tuple)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_tick": self.start_tick,
            "length_ticks": self.length_ticks,
            "muted": self.muted,
            "color": self.color,
            "clip_type": "automation",
            "parameter_id": self.parameter_id,
            "points": [[p[0], p[1]] for p in self.points],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AutomationClip":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "Automation"),
            start_tick=Ticks(data.get("start_tick", 0)),
            length_ticks=Ticks(data.get("length_ticks", PPQ * 4)),
            muted=data.get("muted", False),
            color=data.get("color"),
            parameter_id=data.get("parameter_id", ""),
            points=tuple((Ticks(p[0]), p[1]) for p in data.get("points", [])),
        )
