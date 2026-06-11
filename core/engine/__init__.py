"""
Audio and MIDI engine package.

Core playback infrastructure:
- events.py   — Scheduled event types (NoteOn, NoteOff, TempoChange, ...)
- clock.py    — Tick-based playback clock with tempo map awareness
- transport.py — Play/stop/pause/seek/loop transport controller
- sequencer.py — Compiles Project state into a sorted event timeline

Usage:
    from core.engine import Sequencer, Transport, PlaybackClock

    clock = PlaybackClock(project.tempo_map, project.metadata.bpm)
    seq = Sequencer()
    seq.compile(project)

    transport = Transport(clock=clock)
    transport.bind_sequencer(seq)
    transport.play()

    # In audio callback:
    events = transport.process_block(sample_count=256, sample_rate=44100)
    for event in events:
        midi_out.send(event)
"""

from .events import (
    ScheduledEventType,
    ScheduledEvent,
    make_note_on,
    make_note_off,
    make_tempo_change,
    make_all_notes_off,
    make_transport_stop,
    make_loop_start,
    make_loop_end,
)
from .clock import PlaybackClock, ClockPosition
from .transport import (
    Transport,
    TransportState,
    TransportConfig,
    LoopMode,
    EventCallback,
    StateChangeCallback,
    LoopCallback,
)
from .sequencer import (
    Sequencer,
    SequencerState,
    RenderResult,
    render_project,
)

__all__ = [
    # Events
    "ScheduledEventType",
    "ScheduledEvent",
    "make_note_on",
    "make_note_off",
    "make_tempo_change",
    "make_all_notes_off",
    "make_transport_stop",
    "make_loop_start",
    "make_loop_end",
    # Clock
    "PlaybackClock",
    "ClockPosition",
    # Transport
    "Transport",
    "TransportState",
    "TransportConfig",
    "LoopMode",
    "EventCallback",
    "StateChangeCallback",
    "LoopCallback",
    # Sequencer
    "Sequencer",
    "SequencerState",
    "RenderResult",
    "render_project",
]
