"""
Transport controller — play, stop, pause, seek, loop, record.

The Transport is the user-facing playback controller. It manages:
- Play state (stopped, playing, paused, recording)
- Playhead position
- Loop region (start/end, enabled/disabled)
- Punch-in/out regions for recording

It delegates time-keeping to the PlaybackClock and event scheduling
to the Sequencer. The Transport itself is a pure state machine —
it does not touch audio buffers or MIDI ports directly.

Integration pattern:
    transport = Transport(clock, sequencer)
    transport.play()
    # In audio callback:
    events = transport.tick(sample_count, sample_rate)
    for event in events:
        dispatch(event)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from ..model.time_model import Ticks
from .clock import PlaybackClock, ClockPosition
from .events import (
    ScheduledEvent, ScheduledEventType,
    make_all_notes_off, make_transport_stop,
    make_loop_start, make_loop_end,
)


class TransportState(Enum):
    """Playback states of the transport."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    RECORDING = auto()


class LoopMode(Enum):
    """Loop behavior during playback."""
    DISABLED = auto()     # Play straight through, stop at end
    LOOP_REGION = auto()  # Loop between loop_start and loop_end
    LOOP_CLIP = auto()    # Loop the selected clip(s)
    LOOP_SECTION = auto() # Loop the current section (between markers)


@dataclass
class TransportConfig:
    """
    Configuration for transport behavior.

    Attributes:
        loop_start: Loop region start tick (0 = disabled).
        loop_end: Loop region end tick (0 = disabled).
        loop_mode: How looping behaves.
        punch_in: Punch-in recording start tick (0 = disabled).
        punch_out: Punch-out recording end tick (0 = disabled).
        auto_rewind: Whether stop rewinds to the start position.
        metronome_enabled: Whether the click track is active.
        preroll_bars: Bars of preroll before punch-in recording starts.
    """
    loop_start: Ticks = Ticks(0)
    loop_end: Ticks = Ticks(0)
    loop_mode: LoopMode = LoopMode.DISABLED
    punch_in: Ticks = Ticks(0)
    punch_out: Ticks = Ticks(0)
    auto_rewind: bool = True
    metronome_enabled: bool = False
    preroll_bars: int = 2


# ── Event callback types ─────────────────────────────────────────────

# Called with each scheduled event as it fires
EventCallback = Callable[[ScheduledEvent], None]

# Called when transport state changes
StateChangeCallback = Callable[[TransportState, TransportState], None]

# Called when loop wraps
LoopCallback = Callable[[], None]


@dataclass
class Transport:
    """
    Playback transport controller.

    Manages the play/stop/pause/record state machine and coordinates
    the clock with the sequencer to produce events for audio rendering.

    Usage:
        clock = PlaybackClock(project.tempo_map, project.metadata.bpm)
        seq = Sequencer(project)
        transport = Transport(clock, seq)

        transport.on_event = lambda e: midi_out.send(e)
        transport.play()

        # In audio callback (called at buffer rate):
        sample_count = 256
        transport.process_block(sample_count, sample_rate=44100)
    """

    clock: PlaybackClock = field(default_factory=PlaybackClock)
    config: TransportConfig = field(default_factory=TransportConfig)
    state: TransportState = TransportState.STOPPED
    _play_start_tick: Ticks = Ticks(0)  # Where playback was started
    _pending_events: list[ScheduledEvent] = field(default_factory=list)

    # Callbacks
    on_event: Optional[EventCallback] = None
    on_state_change: Optional[StateChangeCallback] = None
    on_loop: Optional[LoopCallback] = None
    on_position_change: Optional[Callable[[ClockPosition], None]] = None

    # Sequencer integration (set after construction)
    _sequencer: Optional[object] = None  # Set by Sequencer.bind_transport()

    @property
    def is_playing(self) -> bool:
        return self.state == TransportState.PLAYING

    @property
    def is_stopped(self) -> bool:
        return self.state == TransportState.STOPPED

    @property
    def is_paused(self) -> bool:
        return self.state == TransportState.PAUSED

    @property
    def is_recording(self) -> bool:
        return self.state == TransportState.RECORDING

    @property
    def current_tick(self) -> Ticks:
        """Current playhead position in ticks."""
        return self.clock.tick

    @property
    def position(self) -> ClockPosition:
        """Current playhead position with beat/bar info."""
        return self.clock.position

    @property
    def loop_active(self) -> bool:
        """Whether a loop region is active."""
        return (
            self.config.loop_mode != LoopMode.DISABLED
            and self.config.loop_start > Ticks(0)
            and self.config.loop_end > self.config.loop_start
        )

    # ── Transport controls ──────────────────────────────────────────

    def play(self, from_tick: Optional[Ticks] = None) -> None:
        """
        Start or resume playback.

        Args:
            from_tick: Position to start from. If None, starts from
                       the current position (for resuming after pause).
        """
        if from_tick is not None:
            self.clock.seek(from_tick)

        self._play_start_tick = self.clock.tick
        self._set_state(TransportState.PLAYING)

    def stop(self) -> None:
        """
        Stop playback.

        Sends ALL_NOTES_OFF to silence any hanging notes.
        If auto_rewind is enabled, moves playhead to the start position.
        """
        # Send panic to silence hanging notes
        self._emit(make_all_notes_off(self.clock.tick))
        self._emit(make_transport_stop(self.clock.tick))

        if self.config.auto_rewind:
            self.clock.reset()

        self._set_state(TransportState.STOPPED)

    def pause(self) -> None:
        """Pause playback without resetting position."""
        if self.state == TransportState.PLAYING:
            # Send note-offs for currently playing notes
            self._emit(make_all_notes_off(self.clock.tick))
            self._set_state(TransportState.PAUSED)
        elif self.state == TransportState.RECORDING:
            self._emit(make_all_notes_off(self.clock.tick))
            self._set_state(TransportState.PAUSED)

    def record(self, from_tick: Optional[Ticks] = None) -> None:
        """
        Start recording (play + arm for recording).

        Args:
            from_tick: Position to start from.
        """
        if from_tick is not None:
            self.clock.seek(from_tick)
        self._play_start_tick = self.clock.tick
        self._set_state(TransportState.RECORDING)

    def toggle_play(self) -> None:
        """Toggle between play and stop/pause."""
        if self.state == TransportState.PLAYING:
            self.pause()
        elif self.state == TransportState.RECORDING:
            self.pause()
        else:
            self.play()

    def seek(self, tick: Ticks) -> None:
        """
        Move the playhead to an absolute tick position.

        Works during playback (scrubbing) or while stopped.
        If seeking during playback, sends note-offs for currently
        playing notes to avoid hanging voices.
        """
        was_playing = self.is_playing or self.is_recording

        if was_playing:
            self._emit(make_all_notes_off(self.clock.tick))

        self.clock.seek(tick)

        if self.on_position_change:
            self.on_position_change(self.clock.position)

    def seek_bar(self, bar: int, beat: float = 1.0) -> None:
        """Seek to a specific bar/beat."""
        pos = self.clock.seek_bar(bar, beat)
        if self.on_position_change:
            self.on_position_change(pos)

    def seek_forward(self, delta_ticks: Ticks = Ticks(960)) -> None:
        """Seek forward by a given tick amount."""
        self.seek(Ticks(self.clock.tick + delta_ticks))

    def seek_backward(self, delta_ticks: Ticks = Ticks(960)) -> None:
        """Seek backward by a given tick amount."""
        self.seek(Ticks(max(0, self.clock.tick - delta_ticks)))

    def go_to_start(self) -> None:
        """Seek to the very beginning of the project."""
        self.seek(Ticks(0))

    def go_to_end(self) -> None:
        """Seek to the end of the project."""
        if self._sequencer:
            max_tick = getattr(self._sequencer, 'max_tick', Ticks(0))
            self.seek(max_tick)

    # ── Loop controls ────────────────────────────────────────────────

    def set_loop(self, start: Ticks, end: Ticks) -> None:
        """
        Set the loop region.

        Args:
            start: Loop start tick.
            end: Loop end tick (must be > start).
        """
        if end <= start:
            raise ValueError(f"Loop end ({end}) must be > start ({start})")
        self.config.loop_start = start
        self.config.loop_end = end
        self.config.loop_mode = LoopMode.LOOP_REGION

    def clear_loop(self) -> None:
        """Disable the loop region."""
        self.config.loop_mode = LoopMode.DISABLED
        self.config.loop_start = Ticks(0)
        self.config.loop_end = Ticks(0)

    def toggle_loop(self) -> None:
        """Toggle loop on/off."""
        if self.config.loop_mode == LoopMode.DISABLED:
            if self.config.loop_end > self.config.loop_start:
                self.config.loop_mode = LoopMode.LOOP_REGION
        else:
            self.config.loop_mode = LoopMode.DISABLED

    # ── Audio callback integration ───────────────────────────────────

    def process_block(
        self, sample_count: int, sample_rate: int
    ) -> list[ScheduledEvent]:
        """
        Process one block of audio samples.

        This is the main entry point for the audio callback thread.
        It advances the clock, queries the sequencer for events in
        the time range, handles looping, and returns all events that
        should be dispatched during this block.

        Args:
            sample_count: Number of audio samples in this block.
            sample_rate: Audio sample rate.

        Returns:
            List of ScheduledEvent that should be dispatched, in order.
        """
        if not self.is_playing and not self.is_recording:
            return []

        start_tick, end_tick = self.clock.advance_by_samples(
            sample_count, sample_rate
        )

        # Check for loop wrap
        if self.loop_active and end_tick >= self.config.loop_end:
            # Get events up to the loop end
            events = self._get_events_in_range(start_tick, self.config.loop_end)
            # Wrap the playhead back to loop start
            self.clock.seek(self.config.loop_start)
            if self.on_loop:
                self.on_loop()
            return events

        events = self._get_events_in_range(start_tick, end_tick)
        return events

    def process_tick(self) -> list[ScheduledEvent]:
        """
        Advance by one tick and return events at that tick.

        Used for offline/headless rendering and testing.

        Returns:
            Events at the current tick position.
        """
        if not self.is_playing and not self.is_recording:
            return []

        current = self.clock.tick
        self.clock.advance_by_ticks(Ticks(1))

        # Loop wrap
        if self.loop_active and self.clock.tick >= self.config.loop_end:
            events = self._get_events_at_tick(current)
            self.clock.seek(self.config.loop_start)
            if self.on_loop:
                self.on_loop()
            return events

        return self._get_events_at_tick(current)

    # ── Internal ─────────────────────────────────────────────────────

    def _set_state(self, new_state: TransportState) -> None:
        """Change transport state and notify listeners."""
        if new_state == self.state:
            return
        old_state = self.state
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def _get_events_in_range(
        self, start: Ticks, end: Ticks
    ) -> list[ScheduledEvent]:
        """Query sequencer for events in [start, end)."""
        if self._sequencer is None:
            return []
        get_range = getattr(self._sequencer, 'get_events_in_range', None)
        if get_range is None:
            return []
        return get_range(start, end)

    def _get_events_at_tick(self, tick: Ticks) -> list[ScheduledEvent]:
        """Query sequencer for events at a specific tick."""
        return self._get_events_in_range(tick, Ticks(tick + 1))

    def _emit(self, event: ScheduledEvent) -> None:
        """Send an event through the callback."""
        if self.on_event:
            self.on_event(event)

    def bind_sequencer(self, sequencer: object) -> None:
        """
        Bind a sequencer for event queries.

        The sequencer must provide get_events_in_range(start, end)
        and have a max_tick property.
        """
        self._sequencer = sequencer
