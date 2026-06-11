"""
Scheduled event types for the sequencer engine.

The sequencer pre-processes Project state into a flat, time-sorted
list of ScheduledEvent objects. Each event carries a tick position
and enough context to be dispatched to an audio/MIDI backend.

Design:
- Events are pure data (no behavior)
- Events are sortable by tick
- Note-on/off pairs share the same note reference for correlation
- Loop/transport meta-events are included for engine consumption
"""

import functools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from ..model.time_model import Ticks
from ..model.note import NoteEvent


class ScheduledEventType(Enum):
    """All event types the sequencer can emit."""
    NOTE_ON = auto()
    NOTE_OFF = auto()
    TEMPO_CHANGE = auto()
    KEY_SIGNATURE_CHANGE = auto()
    TIME_SIGNATURE_CHANGE = auto()
    MARKER = auto()
    LOOP_START = auto()
    LOOP_END = auto()
    ALL_NOTES_OFF = auto()      # Panic — silence all voices
    TRANSPORT_STOP = auto()     # Sent when the transport stops


@functools.total_ordering
@dataclass(frozen=True)
class ScheduledEvent:
    """
    A single event on the sequencer timeline.

    Comparison (ordering) is by tick, then by priority within the same tick:
    NOTE_OFF fires before NOTE_ON at the same tick to avoid voice stealing.

    Attributes:
        tick: Absolute tick position when this event should fire.
        event_type: What kind of event (note on, tempo change, etc.).
        note: The NoteEvent (for NOTE_ON / NOTE_OFF only).
        track_id: Which track this event belongs to.
        clip_id: Which clip this event belongs to.
        channel: MIDI channel (0-15).
        bpm: New tempo (for TEMPO_CHANGE only).
        data: Opaque extra data for extensibility.
    """
    tick: Ticks
    event_type: ScheduledEventType
    note: Optional[NoteEvent] = None
    track_id: str = ""
    clip_id: str = ""
    channel: int = 0
    bpm: float = 0.0
    data: Optional[dict] = None

    # Custom ordering: sort by tick, then NOTE_OFF before NOTE_ON
    def __lt__(self, other: "ScheduledEvent") -> bool:
        if self.tick != other.tick:
            return self.tick < other.tick
        return self._priority < other._priority

    @property
    def _priority(self) -> int:
        """Lower priority fires first within the same tick."""
        _order = {
            ScheduledEventType.ALL_NOTES_OFF: 0,
            ScheduledEventType.TRANSPORT_STOP: 1,
            ScheduledEventType.NOTE_OFF: 2,
            ScheduledEventType.TEMPO_CHANGE: 3,
            ScheduledEventType.TIME_SIGNATURE_CHANGE: 4,
            ScheduledEventType.KEY_SIGNATURE_CHANGE: 5,
            ScheduledEventType.MARKER: 6,
            ScheduledEventType.LOOP_START: 7,
            ScheduledEventType.LOOP_END: 8,
            ScheduledEventType.NOTE_ON: 9,
        }
        return _order.get(self.event_type, 50)


# ── Factory helpers ───────────────────────────────────────────────────

def make_note_on(tick: Ticks, note: NoteEvent, track_id: str = "",
                 clip_id: str = "") -> ScheduledEvent:
    """Create a NOTE_ON event."""
    return ScheduledEvent(
        tick=tick,
        event_type=ScheduledEventType.NOTE_ON,
        note=note,
        track_id=track_id,
        clip_id=clip_id,
        channel=note.channel,
    )


def make_note_off(tick: Ticks, note: NoteEvent, track_id: str = "",
                  clip_id: str = "") -> ScheduledEvent:
    """Create a NOTE_OFF event."""
    return ScheduledEvent(
        tick=tick,
        event_type=ScheduledEventType.NOTE_OFF,
        note=note,
        track_id=track_id,
        clip_id=clip_id,
        channel=note.channel,
    )


def make_tempo_change(tick: Ticks, bpm: float) -> ScheduledEvent:
    """Create a TEMPO_CHANGE event."""
    return ScheduledEvent(
        tick=tick,
        event_type=ScheduledEventType.TEMPO_CHANGE,
        bpm=bpm,
    )


def make_all_notes_off(tick: Ticks) -> ScheduledEvent:
    """Create an ALL_NOTES_OFF (panic) event."""
    return ScheduledEvent(
        tick=tick,
        event_type=ScheduledEventType.ALL_NOTES_OFF,
    )


def make_transport_stop(tick: Ticks) -> ScheduledEvent:
    """Create a TRANSPORT_STOP event."""
    return ScheduledEvent(
        tick=tick,
        event_type=ScheduledEventType.TRANSPORT_STOP,
    )


def make_loop_start(tick: Ticks) -> ScheduledEvent:
    """Create a LOOP_START boundary event."""
    return ScheduledEvent(
        tick=tick,
        event_type=ScheduledEventType.LOOP_START,
    )


def make_loop_end(tick: Ticks) -> ScheduledEvent:
    """Create a LOOP_END boundary event."""
    return ScheduledEvent(
        tick=tick,
        event_type=ScheduledEventType.LOOP_END,
    )
