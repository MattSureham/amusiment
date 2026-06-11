"""
Sequencer — compiles Project state into a scheduled event timeline.

The sequencer is the bridge between the immutable project state tree
and the real-time playback engine. It takes a Project and produces a
sorted list of ScheduledEvent objects that the Transport consumes
during playback.

Architecture:
    1. Project state → compile() → sorted event list (lazy, cached)
    2. Transport queries get_events_in_range(start, end) during playback
    3. Cache is invalidated when the underlying project changes

Features:
- MIDI note-on/note-off scheduling from all MIDI clips
- Tempo change events from the tempo map
- Key/time signature change events
- Marker events for UI synchronization
- Mute/solo-aware event filtering
- Loop-aware clip handling (unrolls looped clips)
- Efficient binary-search range queries on the sorted timeline
- Lazy compilation with explicit invalidation
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Optional

from ..model.project import Project
from ..model.track import Track, MidiTrack, TrackType
from ..model.clip import MidiClip
from ..model.note import NoteEvent
from ..model.time_model import Ticks, TempoMap, TempoChange, PPQ
from .events import (
    ScheduledEvent, ScheduledEventType,
    make_note_on, make_note_off, make_tempo_change,
    make_all_notes_off, make_transport_stop,
    make_loop_start, make_loop_end,
)


@dataclass
class SequencerState:
    """
    Compiled sequencer state — a sorted event timeline.

    Events are sorted by tick, then by priority within the same tick.
    This list is the single source of truth for what the transport
    dispatches during playback.

    The _tick_index maps tick positions to list indices for O(log n)
    range queries.
    """
    events: list[ScheduledEvent] = field(default_factory=list)
    max_tick: Ticks = Ticks(0)
    _tick_index: list[Ticks] = field(default_factory=list)

    def get_events_in_range(self, start: Ticks, end: Ticks) -> list[ScheduledEvent]:
        """
        Return all events with tick in [start, end).

        Uses binary search on the tick index for O(log n) lookup,
        then slices the event list for the matching range.

        Args:
            start: Inclusive start tick.
            end: Exclusive end tick.

        Returns:
            List of ScheduledEvent in order.
        """
        if not self.events or not self._tick_index:
            return []

        left = bisect_left(self._tick_index, start)
        right = bisect_left(self._tick_index, end)

        return self.events[left:right]

    def get_events_at_tick(self, tick: Ticks) -> list[ScheduledEvent]:
        """Return all events exactly at the given tick."""
        return self.get_events_in_range(tick, Ticks(tick + 1))

    def events_up_to(self, tick: Ticks) -> list[ScheduledEvent]:
        """Return all events from the start up to (but not including) the given tick."""
        return self.get_events_in_range(Ticks(0), tick)

    def __len__(self) -> int:
        return len(self.events)

    def __bool__(self) -> bool:
        return len(self.events) > 0


@dataclass
class Sequencer:
    """
    Compiles Project state into a playable event timeline.

    The sequencer holds a reference to a project and lazily compiles
    it into a sorted event list. Re-compilation only happens when
    invalidate() is called (e.g., after a state mutation).

    Usage:
        project = Project.create_new()
        seq = Sequencer(project)
        seq.compile()  # Build the event timeline

        # Attach to transport
        transport.bind_sequencer(seq)

        # During playback, transport calls:
        events = seq.get_events_in_range(start_tick, end_tick)

        # After project mutation:
        seq.invalidate()
        seq.compile(project)  # Rebuild with new state

    Track solo logic:
        If any track is soloed, only soloed tracks produce events.
        If no track is soloed, all non-muted tracks produce events.
    """

    _state: Optional[SequencerState] = None
    _project: Optional[Project] = None
    _valid: bool = False
    _solo_active: bool = False
    _tracks_with_solo: set[str] = field(default_factory=set)

    def compile(self, project: Optional[Project] = None) -> SequencerState:
        """
        Compile the project into a sorted event timeline.

        This is the main entry point. It extracts all MIDI notes from
        all tracks, converts them to scheduled note-on/off pairs, adds
        tempo/marker/meta events, sorts everything, and builds the
        tick index for fast range queries.

        Args:
            project: The project to compile. If None, uses the last
                     project passed to compile() or set_project().

        Returns:
            The compiled SequencerState.
        """
        if project is not None:
            self._project = project

        if self._project is None:
            self._state = SequencerState()
            self._valid = True
            return self._state

        events: list[ScheduledEvent] = []
        project = self._project

        # ── Step 1: Determine solo state ─────────────────────────
        self._tracks_with_solo = set()
        for track in project.tracks:
            if track.soloed:
                self._tracks_with_solo.add(track.id)
        self._solo_active = len(self._tracks_with_solo) > 0

        # ── Step 2: Compile MIDI note events ─────────────────────
        for track in project.tracks:
            if not self._track_is_audible(track):
                continue

            if isinstance(track, MidiTrack):
                events.extend(self._compile_midi_track(track, project.tempo_map))

        # ── Step 3: Compile meta events ──────────────────────────
        meta_events = self._compile_meta_events(project)
        events.extend(meta_events)

        # ── Step 4: Sort and build tick index ────────────────────
        events.sort()

        tick_index = [e.tick for e in events]
        max_tick = events[-1].tick if events else Ticks(0)

        self._state = SequencerState(
            events=events,
            max_tick=max_tick,
            _tick_index=tick_index,
        )
        self._valid = True

        return self._state

    def invalidate(self) -> None:
        """Mark the compiled state as stale. Next compile() will rebuild."""
        self._valid = False

    def is_valid(self) -> bool:
        """Whether the compiled state is up to date."""
        return self._valid and self._state is not None

    @property
    def state(self) -> SequencerState:
        """Get the current compiled state (compiles if needed)."""
        if not self._valid or self._state is None:
            self.compile()
        return self._state

    @property
    def max_tick(self) -> Ticks:
        """The latest tick in the compiled event timeline."""
        return self.state.max_tick

    def get_events_in_range(self, start: Ticks, end: Ticks) -> list[ScheduledEvent]:
        """
        Get all scheduled events in the tick range [start, end).

        Used by the Transport during playback.

        Args:
            start: Inclusive start tick.
            end: Exclusive end tick.

        Returns:
            Sorted list of ScheduledEvent.
        """
        return self.state.get_events_in_range(start, end)

    def get_events_at_tick(self, tick: Ticks) -> list[ScheduledEvent]:
        """Get all events exactly at the given tick."""
        return self.state.get_events_at_tick(tick)

    # ── Internal compilation methods ────────────────────────────────

    def _track_is_audible(self, track: Track) -> bool:
        """Check whether a track should produce events (mute/solo logic)."""
        if track.muted:
            return False
        if self._solo_active:
            return track.id in self._tracks_with_solo
        return True

    def _compile_midi_track(
        self, track: MidiTrack, tempo_map: TempoMap
    ) -> list[ScheduledEvent]:
        """Compile all MIDI clips on a track into scheduled events."""
        events: list[ScheduledEvent] = []

        for clip in track.clips:
            if clip.muted:
                continue

            clip_events = self._compile_midi_clip(clip, track, tempo_map)
            events.extend(clip_events)

        return events

    def _compile_midi_clip(
        self, clip: MidiClip, track: MidiTrack, tempo_map: TempoMap,
    ) -> list[ScheduledEvent]:
        """
        Compile a single MIDI clip into note-on/off events.

        Handles looped clips by unrolling the loop to cover the
        project length (up to a safety limit to prevent infinite loops).

        Each note is offset by clip.start_tick so that clip positions
        are preserved in the absolute timeline.
        """
        events: list[ScheduledEvent] = []
        offset = clip.start_tick
        clip_length = clip.length_ticks

        if clip_length <= Ticks(0):
            return events

        # Determine how many loop iterations to unroll
        if clip.loop_enabled and self._project is not None:
            # Unroll to cover project length, but cap at 256 iterations
            project_length = max(
                self._project.total_duration_ticks,
                self._project.project_length_ticks,
            )
            if project_length > Ticks(0):
                # Ceiling division: ensure we cover the full project length
                iterations = min(
                    (project_length + clip_length - 1) // clip_length,
                    256,
                )
            else:
                iterations = 1
        else:
            iterations = 1

        for loop_idx in range(iterations):
            loop_offset = Ticks(offset + loop_idx * clip_length)

            for note in clip.notes:
                # Absolute tick position in the project timeline
                abs_start = Ticks(loop_offset + note.start_tick)
                abs_end = Ticks(abs_start + note.duration_ticks)

                # Note-on
                events.append(make_note_on(
                    tick=abs_start,
                    note=note,
                    track_id=track.id,
                    clip_id=clip.id,
                ))

                # Note-off
                events.append(make_note_off(
                    tick=abs_end,
                    note=note,
                    track_id=track.id,
                    clip_id=clip.id,
                ))

        return events

    def _compile_meta_events(self, project: Project) -> list[ScheduledEvent]:
        """Compile tempo changes, markers, loop boundaries, and key changes."""
        events: list[ScheduledEvent] = []

        # Tempo changes
        for change in project.tempo_map.changes:
            events.append(make_tempo_change(change.tick, change.bpm))

        # Key signature changes
        for ks in project.key_signatures:
            events.append(ScheduledEvent(
                tick=ks.tick,
                event_type=ScheduledEventType.KEY_SIGNATURE_CHANGE,
                data={"sharps_flats": ks.sharps_flats, "mode": ks.mode},
            ))

        # Markers
        for marker in project.markers:
            events.append(ScheduledEvent(
                tick=marker.tick,
                event_type=ScheduledEventType.MARKER,
                data={"name": marker.name, "color": marker.color},
            ))

        # Loop boundaries
        if project.loop_start > Ticks(0):
            events.append(make_loop_start(project.loop_start))
        if project.loop_end > Ticks(0):
            events.append(make_loop_end(project.loop_end))

        return events


# ── Offline rendering helper ──────────────────────────────────────────

@dataclass
class RenderResult:
    """
    Result of an offline render pass.

    Contains all events in the project, ordered by tick, suitable
    for export to MIDI files or feeding to an offline audio renderer.
    """
    events: list[ScheduledEvent]
    total_ticks: Ticks
    total_seconds: float
    note_count: int

    @property
    def note_on_events(self) -> list[ScheduledEvent]:
        return [e for e in self.events if e.event_type == ScheduledEventType.NOTE_ON]

    @property
    def note_off_events(self) -> list[ScheduledEvent]:
        return [e for e in self.events if e.event_type == ScheduledEventType.NOTE_OFF]

    @property
    def tempo_events(self) -> list[ScheduledEvent]:
        return [e for e in self.events if e.event_type == ScheduledEventType.TEMPO_CHANGE]


def render_project(project: Project) -> RenderResult:
    """
    Render an entire project to a flat event list (offline).

    This is a convenience function that compiles the project and
    returns all events plus timing metadata. Useful for:
    - MIDI export (pass events to a MIDI file writer)
    - Audio bounce (render events through instrument plugins)
    - Analysis (count notes, compute density, etc.)

    Args:
        project: The project to render.

    Returns:
        RenderResult with all events and timing metadata.
    """
    from ..model.time_model import ticks_to_seconds

    sequencer = Sequencer()
    state = sequencer.compile(project)

    total_seconds = ticks_to_seconds(
        state.max_tick or project.total_duration_ticks or Ticks(0),
        project.tempo_map,
        project.metadata.bpm,
    )

    note_count = sum(
        1 for e in state.events
        if e.event_type == ScheduledEventType.NOTE_ON
    )

    return RenderResult(
        events=list(state.events),
        total_ticks=state.max_tick,
        total_seconds=total_seconds,
        note_count=note_count,
    )
