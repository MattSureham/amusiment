"""
Tests for the sequencer engine: events, clock, transport, and sequencer.

Verifies:
- Event creation, ordering, and priority within same ticks
- Clock tick↔second conversion with tempo maps
- Transport state machine (play/stop/pause/seek/loop)
- Sequencer compilation of Project → event timeline
- Range queries on compiled events
- Mute/solo-aware event filtering
- Looped clip unrolling
- Offline render_project() helper
- Full integration: project→sequencer→transport→events during playback
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine.events import (
    ScheduledEvent, ScheduledEventType,
    make_note_on, make_note_off, make_tempo_change,
    make_all_notes_off, make_transport_stop,
    make_loop_start, make_loop_end,
)
from core.engine.clock import PlaybackClock, ClockPosition
from core.engine.transport import (
    Transport, TransportState, TransportConfig, LoopMode,
)
from core.engine.sequencer import (
    Sequencer, SequencerState, RenderResult, render_project,
)
from core.model.time_model import (
    Ticks, Seconds, PPQ, TempoMap, TempoChange, TimeSignature, BeatValue,
)
from core.model.note import NoteEvent, NotePitch, NoteVelocity, pitch_to_name
from core.model.clip import MidiClip
from core.model.track import MidiTrack
from core.model.project import Project, ProjectMetadata


# ═══════════════════════════════════════════════════════════════════════
# Event Tests
# ═══════════════════════════════════════════════════════════════════════

class TestScheduledEvents:
    """Test event creation, ordering, and priority."""

    def test_note_on_creation(self):
        note = NoteEvent(
            pitch=NotePitch(60), velocity=NoteVelocity(100),
            start_tick=Ticks(0), duration_ticks=Ticks(480),
            channel=0,
        )
        event = make_note_on(Ticks(960), note, track_id="t1", clip_id="c1")
        assert event.event_type == ScheduledEventType.NOTE_ON
        assert event.tick == Ticks(960)
        assert event.note is note
        assert event.track_id == "t1"
        assert event.clip_id == "c1"

    def test_note_off_creation(self):
        note = NoteEvent(
            pitch=NotePitch(72), velocity=NoteVelocity(80),
            start_tick=Ticks(480), duration_ticks=Ticks(960),
        )
        event = make_note_off(Ticks(1440), note, track_id="t2")
        assert event.event_type == ScheduledEventType.NOTE_OFF
        assert event.tick == Ticks(1440)

    def test_tempo_change_creation(self):
        event = make_tempo_change(Ticks(0), 140.0)
        assert event.event_type == ScheduledEventType.TEMPO_CHANGE
        assert event.bpm == 140.0

    def test_all_notes_off_creation(self):
        event = make_all_notes_off(Ticks(5000))
        assert event.event_type == ScheduledEventType.ALL_NOTES_OFF

    def test_loop_boundary_creation(self):
        start = make_loop_start(Ticks(960))
        end = make_loop_end(Ticks(9600))
        assert start.event_type == ScheduledEventType.LOOP_START
        assert end.event_type == ScheduledEventType.LOOP_END

    def test_ordering_by_tick(self):
        """Events should sort by tick first."""
        early = make_note_on(Ticks(100), NoteEvent(
            pitch=NotePitch(60), velocity=NoteVelocity(100),
            start_tick=Ticks(0), duration_ticks=Ticks(480),
        ))
        late = make_note_on(Ticks(200), NoteEvent(
            pitch=NotePitch(62), velocity=NoteVelocity(100),
            start_tick=Ticks(0), duration_ticks=Ticks(480),
        ))
        assert early < late
        assert sorted([late, early]) == [early, late]

    def test_ordering_same_tick_note_off_before_note_on(self):
        """At the same tick, NOTE_OFF fires before NOTE_ON."""
        note = NoteEvent(
            pitch=NotePitch(60), velocity=NoteVelocity(100),
            start_tick=Ticks(0), duration_ticks=Ticks(480),
        )
        note_off = make_note_off(Ticks(500), note)
        note_on = make_note_on(Ticks(500), note)
        assert note_off < note_on

    def test_ordering_same_tick_all_notes_off_first(self):
        """ALL_NOTES_OFF should fire before anything else."""
        note = NoteEvent(
            pitch=NotePitch(60), velocity=NoteVelocity(100),
            start_tick=Ticks(0), duration_ticks=Ticks(480),
        )
        panic = make_all_notes_off(Ticks(300))
        note_on = make_note_on(Ticks(300), note)
        assert panic < note_on


# ═══════════════════════════════════════════════════════════════════════
# Clock Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlaybackClock:
    """Test clock positioning, seeking, and tempo-aware advancement."""

    def test_initial_position(self):
        clock = PlaybackClock(start_bpm=120.0)
        assert clock.tick == Ticks(0)
        assert clock.elapsed_seconds == Seconds(0.0)
        assert clock.current_bpm == 120.0

    def test_seek_to_tick(self):
        clock = PlaybackClock(start_bpm=120.0)
        pos = clock.seek(Ticks(PPQ * 4))  # 4 beats at 120 BPM = 2 seconds
        assert clock.tick == Ticks(PPQ * 4)
        assert pos.bar == 2  # bar 2 (beats 1-4 are bar 1)
        assert pos.beat_in_bar == 1.0

    def test_seek_to_bar(self):
        clock = PlaybackClock(start_bpm=120.0, beats_per_bar=4)
        pos = clock.seek_bar(3, 3.0)  # bar 3, beat 3
        # total_beats = (3-1)*4 + (3-1) = 8 + 2 = 10
        assert clock.tick == Ticks(10 * PPQ)
        assert pos.bar == 3
        assert pos.beat_in_bar == 3.0

    def test_advance_by_ticks(self):
        clock = PlaybackClock(start_bpm=120.0)
        clock.advance_by_ticks(Ticks(PPQ * 2))  # 2 beats
        assert clock.tick == Ticks(PPQ * 2)

    def test_advance_by_samples_at_120bpm(self):
        """At 120 BPM, 1 beat = 0.5 sec. 44100 samples = 1 sec = 2 beats."""
        clock = PlaybackClock(start_bpm=120.0)
        start, end = clock.advance_by_samples(44100, 44100)
        # 1 second at 120 BPM = 2 beats = 1920 ticks
        expected_ticks = 2 * PPQ
        # Allow small rounding delta
        assert abs(end - expected_ticks) <= Ticks(2)
        assert start == Ticks(0)

    def test_advance_by_samples_at_60bpm(self):
        """At 60 BPM, 1 beat = 1 sec. 22050 samples = 0.5 sec = 0.5 beats."""
        clock = PlaybackClock(start_bpm=60.0)
        start, end = clock.advance_by_samples(22050, 44100)
        # 0.5 seconds at 60 BPM = 0.5 beats = 480 ticks
        expected_ticks = PPQ // 2
        assert abs(end - expected_ticks) <= Ticks(2)

    def test_clock_with_tempo_changes(self):
        """Clock should respect a tempo map during advance."""
        tempo_map = TempoMap(changes=(
            TempoChange(tick=Ticks(0), bpm=120.0),
            TempoChange(tick=Ticks(PPQ * 4), bpm=60.0),
        ))
        clock = PlaybackClock(tempo_map=tempo_map, start_bpm=120.0)

        # First 4 beats at 120 BPM = 2 seconds
        clock.seek(Ticks(PPQ * 4))
        assert abs(clock.elapsed_seconds - 2.0) < 0.01

    def test_position_snapshot(self):
        clock = PlaybackClock(start_bpm=120.0, beats_per_bar=4)
        clock.seek(Ticks(PPQ * 7))  # beat 8 (7 beats in, 0-indexed)
        pos = clock.position
        assert pos.tick == Ticks(PPQ * 7)
        assert pos.bar == 2  # beats 0-3 = bar 1, beats 4-7 = bar 2
        assert pos.current_bpm == 120.0

    def test_reset(self):
        clock = PlaybackClock(start_bpm=120.0)
        clock.seek(Ticks(PPQ * 100))
        clock.reset()
        assert clock.tick == Ticks(0)
        assert clock.elapsed_seconds == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Transport Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTransport:
    """Test transport state machine and controls."""

    def test_initial_state_stopped(self):
        transport = Transport()
        assert transport.state == TransportState.STOPPED
        assert transport.is_stopped
        assert not transport.is_playing

    def test_play(self):
        transport = Transport()
        transport.play()
        assert transport.state == TransportState.PLAYING
        assert transport.is_playing

    def test_stop(self):
        transport = Transport()
        transport.play()
        transport.stop()
        assert transport.state == TransportState.STOPPED
        assert transport.clock.tick == Ticks(0)  # auto-rewind by default

    def test_stop_without_auto_rewind(self):
        clock = PlaybackClock(start_bpm=120.0)
        config = TransportConfig(auto_rewind=False)
        transport = Transport(clock=clock, config=config)
        transport.play()
        clock.advance_by_ticks(Ticks(PPQ * 4))
        transport.stop()
        assert transport.state == TransportState.STOPPED
        assert transport.clock.tick == Ticks(PPQ * 4)  # Not rewound

    def test_pause_resume(self):
        transport = Transport()
        transport.play()
        transport.pause()
        assert transport.state == TransportState.PAUSED
        transport.play()  # Resume
        assert transport.state == TransportState.PLAYING

    def test_toggle_play(self):
        transport = Transport()
        transport.toggle_play()
        assert transport.is_playing
        transport.toggle_play()
        assert transport.is_paused

    def test_seek_during_playback(self):
        transport = Transport()
        transport.play()
        transport.seek(Ticks(PPQ * 8))
        assert transport.clock.tick == Ticks(PPQ * 8)

    def test_seek_forward_backward(self):
        transport = Transport()
        transport.seek_forward(Ticks(PPQ * 4))
        assert transport.clock.tick == Ticks(PPQ * 4)
        transport.seek_backward(Ticks(PPQ * 2))
        assert transport.clock.tick == Ticks(PPQ * 2)
        # Can't go below 0
        transport.seek_backward(Ticks(PPQ * 10))
        assert transport.clock.tick == Ticks(0)

    def test_set_loop(self):
        transport = Transport()
        transport.set_loop(Ticks(PPQ * 4), Ticks(PPQ * 16))
        assert transport.config.loop_start == Ticks(PPQ * 4)
        assert transport.config.loop_end == Ticks(PPQ * 16)
        assert transport.loop_active

    def test_clear_loop(self):
        transport = Transport()
        transport.set_loop(Ticks(1000), Ticks(5000))
        transport.clear_loop()
        assert not transport.loop_active

    def test_toggle_loop(self):
        transport = Transport()
        transport.config.loop_start = Ticks(1000)
        transport.config.loop_end = Ticks(5000)
        transport.toggle_loop()
        assert transport.loop_active
        transport.toggle_loop()
        assert not transport.loop_active

    def test_state_change_callback(self):
        state_changes = []
        transport = Transport()
        transport.on_state_change = lambda old, new: state_changes.append((old, new))
        transport.play()
        transport.stop()
        assert len(state_changes) == 2
        assert state_changes[0] == (TransportState.STOPPED, TransportState.PLAYING)
        assert state_changes[1] == (TransportState.PLAYING, TransportState.STOPPED)

    def test_process_block_returns_empty_when_stopped(self):
        transport = Transport()
        events = transport.process_block(256, 44100)
        assert events == []

    def test_bind_sequencer(self):
        transport = Transport()
        seq = Sequencer()
        transport.bind_sequencer(seq)
        assert transport._sequencer is seq


# ═══════════════════════════════════════════════════════════════════════
# Sequencer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSequencerCompilation:
    """Test that the sequencer correctly compiles projects into events."""

    def _make_test_note(self, pitch=60, start=0, duration=480):
        return NoteEvent(
            pitch=NotePitch(pitch),
            velocity=NoteVelocity(100),
            start_tick=Ticks(start),
            duration_ticks=Ticks(duration),
            channel=0,
        )

    def _make_test_project(self) -> Project:
        """Create a simple project with one MIDI track and two notes."""
        notes = (
            self._make_test_note(pitch=60, start=0, duration=480),
            self._make_test_note(pitch=64, start=480, duration=480),
        )
        clip = MidiClip(
            name="Test Clip",
            start_tick=Ticks(0),
            length_ticks=Ticks(PPQ * 2),
            notes=notes,
        )
        track = MidiTrack(name="Piano", clips=(clip,))
        return Project.create_new(name="Test").with_track(track)

    def test_compile_empty_project(self):
        seq = Sequencer()
        project = Project.create_new()
        state = seq.compile(project)
        assert len(state.events) == 0

    def test_compile_notes_become_on_off_pairs(self):
        seq = Sequencer()
        project = self._make_test_project()
        state = seq.compile(project)
        # 2 notes → 4 events (2 on + 2 off)
        assert len(state.events) == 4
        note_ons = [e for e in state.events if e.event_type == ScheduledEventType.NOTE_ON]
        note_offs = [e for e in state.events if e.event_type == ScheduledEventType.NOTE_OFF]
        assert len(note_ons) == 2
        assert len(note_offs) == 2

    def test_compile_events_are_sorted(self):
        seq = Sequencer()
        project = self._make_test_project()
        state = seq.compile(project)
        ticks = [e.tick for e in state.events]
        assert ticks == sorted(ticks), f"Events not sorted: {ticks}"

    def test_compile_clip_offset_is_preserved(self):
        """Notes in a clip at start_tick=960 should appear at tick 960+note.start."""
        note = self._make_test_note(pitch=60, start=100, duration=480)
        clip = MidiClip(
            start_tick=Ticks(PPQ),  # clip starts at beat 2
            length_ticks=Ticks(PPQ * 2),
            notes=(note,),
        )
        track = MidiTrack(name="Track", clips=(clip,))
        project = Project.create_new().with_track(track)

        seq = Sequencer()
        state = seq.compile(project)
        note_on = [e for e in state.events if e.event_type == ScheduledEventType.NOTE_ON][0]
        # Absolute tick = clip.start (960) + note.start (100) = 1060
        assert note_on.tick == Ticks(PPQ + 100)

    def test_compile_returns_same_on_subsequent_call(self):
        seq = Sequencer()
        project = self._make_test_project()
        state1 = seq.compile(project)
        state2 = seq.compile()  # Re-use cached project
        assert len(state1.events) == len(state2.events)

    def test_compile_after_invalidation_rebuilds(self):
        seq = Sequencer()
        project = self._make_test_project()
        seq.compile(project)
        seq.invalidate()
        assert not seq.is_valid()
        seq.compile()
        assert seq.is_valid()

    def test_compile_muted_track_excluded(self):
        note = self._make_test_note()
        clip = MidiClip(notes=(note,))
        track = MidiTrack(name="Muted", clips=(clip,), muted=True)
        project = Project.create_new().with_track(track)

        seq = Sequencer()
        state = seq.compile(project)
        assert len(state.events) == 0, "Muted track should produce no events"

    def test_compile_soloed_track_only(self):
        """When one track is soloed, only that track's notes appear."""
        note1 = self._make_test_note(pitch=60)
        note2 = self._make_test_note(pitch=72)
        clip1 = MidiClip(notes=(note1,))
        clip2 = MidiClip(notes=(note2,))

        track1 = MidiTrack(name="Soloed", clips=(clip1,), soloed=True)
        track2 = MidiTrack(name="Normal", clips=(clip2,))

        project = Project.create_new().with_track(track1).with_track(track2)

        seq = Sequencer()
        state = seq.compile(project)
        assert len(state.events) == 2  # Only soloed track (1 note → 2 events)

    def test_compile_muted_clip_excluded(self):
        note = self._make_test_note()
        clip = MidiClip(notes=(note,), muted=True)
        track = MidiTrack(name="Track", clips=(clip,))
        project = Project.create_new().with_track(track)

        seq = Sequencer()
        state = seq.compile(project)
        assert len(state.events) == 0

    def test_compile_tempo_changes(self):
        project = Project(
            metadata=ProjectMetadata(bpm=100.0),
            tempo_map=TempoMap(changes=(
                TempoChange(tick=Ticks(0), bpm=100.0),
                TempoChange(tick=Ticks(PPQ * 4), bpm=140.0),
            )),
        )
        seq = Sequencer()
        state = seq.compile(project)
        tempo_events = [e for e in state.events if e.event_type == ScheduledEventType.TEMPO_CHANGE]
        assert len(tempo_events) == 2

    def test_compile_markers(self):
        from core.model.project import Marker
        project = Project(
            markers=(
                Marker(tick=Ticks(PPQ * 8), name="Chorus"),
                Marker(tick=Ticks(PPQ * 16), name="Bridge"),
            ),
        )
        seq = Sequencer()
        state = seq.compile(project)
        marker_events = [e for e in state.events if e.event_type == ScheduledEventType.MARKER]
        assert len(marker_events) == 2

    def test_compile_key_signatures(self):
        from core.model.project import KeySignature
        project = Project(
            key_signatures=(
                KeySignature(tick=Ticks(0), sharps_flats=2, mode="major"),
            ),
        )
        seq = Sequencer()
        state = seq.compile(project)
        key_events = [e for e in state.events if e.event_type == ScheduledEventType.KEY_SIGNATURE_CHANGE]
        assert len(key_events) == 1


class TestSequencerQueries:
    """Test range queries on compiled events."""

    def _make_test_project(self) -> Project:
        """Project with notes at ticks 0, 960, 1920."""
        notes = (
            NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(480)),
            NoteEvent(NotePitch(64), NoteVelocity(100), Ticks(960), Ticks(480)),
            NoteEvent(NotePitch(67), NoteVelocity(100), Ticks(1920), Ticks(480)),
        )
        clip = MidiClip(start_tick=Ticks(0), length_ticks=Ticks(PPQ * 8), notes=notes)
        track = MidiTrack(name="Piano", clips=(clip,))
        return Project.create_new().with_track(track)

    def test_range_query_partial(self):
        seq = Sequencer()
        project = self._make_test_project()
        seq.compile(project)

        # Get events in [0, 960) — should contain first note on at 0,
        # first note off at 480, second note on at 960? No, 960 is exclusive
        events = seq.get_events_in_range(Ticks(0), Ticks(960))
        # Events: note_on@0, note_off@480 — that's 2 events
        ticks = [e.tick for e in events]
        assert Ticks(0) in ticks
        assert Ticks(480) in ticks
        assert Ticks(960) not in ticks  # Excluded (range end is exclusive)

    def test_range_query_inclusive_start(self):
        seq = Sequencer()
        project = self._make_test_project()
        seq.compile(project)

        events = seq.get_events_in_range(Ticks(960), Ticks(1921))
        # Should include note_on@960, note_off@1440, note_on@1920
        ticks = {e.tick for e in events}
        assert Ticks(960) in ticks
        assert Ticks(1440) in ticks
        assert Ticks(1920) in ticks

    def test_range_query_empty(self):
        seq = Sequencer()
        project = self._make_test_project()
        seq.compile(project)

        events = seq.get_events_in_range(Ticks(10000), Ticks(20000))
        assert len(events) == 0

    def test_get_events_at_tick(self):
        seq = Sequencer()
        project = self._make_test_project()
        seq.compile(project)

        events = seq.get_events_at_tick(Ticks(0))
        assert len(events) == 1
        assert events[0].event_type == ScheduledEventType.NOTE_ON
        assert events[0].note.pitch == 60


class TestLoopedClips:
    """Test that looped clips are properly unrolled by the sequencer."""

    def test_looped_clip_unrolls(self):
        """A 1-bar clip looped 4x should produce note events repeated."""
        note = NoteEvent(
            NotePitch(60), NoteVelocity(100),
            Ticks(0), Ticks(PPQ // 2),  # Eighth note
        )
        clip = MidiClip(
            start_tick=Ticks(0),
            length_ticks=Ticks(PPQ * 4),  # 1 bar of 4/4
            notes=(note,),
            loop_enabled=True,
        )
        track = MidiTrack(name="Loop", clips=(clip,))

        # Project is 8 bars long
        project = Project(
            metadata=ProjectMetadata(),
            project_length_ticks=Ticks(PPQ * 4 * 8),
        )
        project = project.with_track(track)

        seq = Sequencer()
        state = seq.compile(project)

        note_ons = [e for e in state.events if e.event_type == ScheduledEventType.NOTE_ON]
        # The note should repeat across the project length (8 bars / 1 bar clip = 8 iterations)
        # 1 note per iteration → 8 note-ons
        assert len(note_ons) == 8, f"Expected 8 note-ons, got {len(note_ons)}"

    def test_non_looped_clip_does_not_repeat(self):
        note = NoteEvent(
            NotePitch(60), NoteVelocity(100),
            Ticks(0), Ticks(PPQ),
        )
        clip = MidiClip(
            start_tick=Ticks(0),
            length_ticks=Ticks(PPQ * 4),
            notes=(note,),
            loop_enabled=False,
        )
        track = MidiTrack(name="NoLoop", clips=(clip,))
        project = Project(
            metadata=ProjectMetadata(),
            project_length_ticks=Ticks(PPQ * 4 * 16),
        )
        project = project.with_track(track)

        seq = Sequencer()
        state = seq.compile(project)

        note_ons = [e for e in state.events if e.event_type == ScheduledEventType.NOTE_ON]
        assert len(note_ons) == 1  # Only one instance, no loop


class TestRenderProject:
    """Test the offline render_project() helper."""

    def test_render_empty_project(self):
        project = Project.create_new()
        result = render_project(project)
        assert len(result.events) == 0
        assert result.total_ticks == Ticks(0)

    def test_render_with_notes(self):
        notes = (
            NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(480)),
            NoteEvent(NotePitch(64), NoteVelocity(80), Ticks(480), Ticks(480)),
        )
        clip = MidiClip(start_tick=Ticks(0), length_ticks=Ticks(PPQ * 4), notes=notes)
        track = MidiTrack(name="Test", clips=(clip,))
        project = Project.create_new().with_track(track)

        result = render_project(project)
        assert result.note_count == 2  # 2 note-on events
        assert result.total_ticks > Ticks(0)
        assert result.total_seconds > 0

    def test_render_respects_muted_tracks(self):
        notes = (NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(480)),)
        clip = MidiClip(notes=notes, muted=True)
        track = MidiTrack(name="Muted", clips=(clip,))
        project = Project.create_new().with_track(track)

        result = render_project(project)
        assert result.note_count == 0


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTransportSequencerIntegration:
    """Test the full pipeline: project → sequencer → transport → events."""

    def _make_project_with_notes(self) -> Project:
        notes = (
            NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(PPQ // 2)),
            NoteEvent(NotePitch(64), NoteVelocity(100), Ticks(PPQ), Ticks(PPQ // 2)),
            NoteEvent(NotePitch(67), NoteVelocity(100), Ticks(PPQ * 2), Ticks(PPQ // 2)),
        )
        clip = MidiClip(
            start_tick=Ticks(0),
            length_ticks=Ticks(PPQ * 4),
            notes=notes,
        )
        track = MidiTrack(name="Piano", clips=(clip,))
        return Project.create_new(name="Integration", bpm=120.0).with_track(track)

    def test_full_pipeline_playback_generates_events(self):
        """Transport + Sequencer should generate events during process_block."""
        project = self._make_project_with_notes()

        seq = Sequencer()
        seq.compile(project)

        clock = PlaybackClock(start_bpm=project.metadata.bpm)
        transport = Transport(clock=clock)
        transport.bind_sequencer(seq)
        transport.play()

        # Process a few blocks
        all_events = []
        for _ in range(20):
            events = transport.process_block(256, 44100)
            all_events.extend(events)
            if not transport.is_playing:
                break

        # Should have captured at least some note-on events
        note_ons = [e for e in all_events if e.event_type == ScheduledEventType.NOTE_ON]
        assert len(note_ons) >= 1, "Expected at least one note-on event during playback"

    def test_loop_wraps_playhead(self):
        """When playback reaches loop_end, it should wrap to loop_start."""
        project = self._make_project_with_notes()

        seq = Sequencer()
        seq.compile(project)

        clock = PlaybackClock(start_bpm=120.0)
        config = TransportConfig(
            loop_start=Ticks(PPQ * 2),
            loop_end=Ticks(PPQ * 4),
            loop_mode=LoopMode.LOOP_REGION,
        )
        transport = Transport(clock=clock, config=config)
        transport.bind_sequencer(seq)

        # Seek to just before loop end
        clock.seek(Ticks(PPQ * 4 - 10))
        transport.play()

        loop_callbacks = []
        transport.on_loop = lambda: loop_callbacks.append(True)

        # Process enough to trigger loop wrap
        transport.process_block(1024, 44100)

        # After wrap, playhead should be near loop_start
        assert transport.clock.tick < Ticks(PPQ * 4)
        assert len(loop_callbacks) >= 1

    def test_tick_by_tick_playback(self):
        """process_tick should advance one tick at a time."""
        project = self._make_project_with_notes()

        seq = Sequencer()
        seq.compile(project)

        clock = PlaybackClock(start_bpm=120.0)
        transport = Transport(clock=clock)
        transport.bind_sequencer(seq)
        transport.play()

        # Tick through the first beat
        all_events = []
        for _ in range(PPQ):  # 960 ticks = 1 beat
            events = transport.process_tick()
            all_events.extend(events)

        # The first note (at tick 0) should have fired
        note_ons = [e for e in all_events if e.event_type == ScheduledEventType.NOTE_ON]
        assert len(note_ons) >= 1

    def test_stop_sends_all_notes_off(self):
        project = self._make_project_with_notes()

        seq = Sequencer()
        seq.compile(project)

        captured_events = []
        transport = Transport()
        transport.bind_sequencer(seq)
        transport.on_event = lambda e: captured_events.append(e)

        transport.play()
        transport.stop()

        panic_events = [e for e in captured_events if e.event_type == ScheduledEventType.ALL_NOTES_OFF]
        assert len(panic_events) == 1
        stop_events = [e for e in captured_events if e.event_type == ScheduledEventType.TRANSPORT_STOP]
        assert len(stop_events) == 1

    def test_multiple_tracks_compile_correctly(self):
        """Multiple MIDI tracks should produce interleaved events."""
        note_a = NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(PPQ))
        note_b = NoteEvent(NotePitch(72), NoteVelocity(100), Ticks(PPQ // 2), Ticks(PPQ))

        clip_a = MidiClip(notes=(note_a,))
        clip_b = MidiClip(notes=(note_b,))

        track_a = MidiTrack(name="A", clips=(clip_a,))
        track_b = MidiTrack(name="B", clips=(clip_b,))

        project = Project.create_new().with_track(track_a).with_track(track_b)

        seq = Sequencer()
        state = seq.compile(project)

        # 2 notes × 2 events each = 4 events
        assert len(state.events) == 4

        # Track IDs should be preserved
        track_ids = {e.track_id for e in state.events}
        assert len(track_ids) == 2


# ═══════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test sequencer/transport edge cases and error handling."""

    def test_compile_zero_duration_clip(self):
        clip = MidiClip(start_tick=Ticks(0), length_ticks=Ticks(0), notes=())
        track = MidiTrack(name="Empty", clips=(clip,))
        project = Project.create_new().with_track(track)

        seq = Sequencer()
        state = seq.compile(project)
        assert len(state.events) == 0

    def test_range_query_on_empty_sequencer(self):
        seq = Sequencer()
        seq.compile(Project.create_new())
        events = seq.get_events_in_range(Ticks(0), Ticks(1000))
        assert events == []

    def test_max_tick_with_no_events(self):
        seq = Sequencer()
        seq.compile(Project.create_new())
        assert seq.max_tick == Ticks(0)

    def test_process_block_during_pause_returns_empty(self):
        transport = Transport()
        transport.play()
        transport.pause()
        events = transport.process_block(256, 44100)
        assert events == []

    def test_seek_backward_bounded_at_zero(self):
        transport = Transport()
        transport.seek(Ticks(100))
        transport.seek_backward(Ticks(500))
        assert transport.clock.tick == Ticks(0)

    def test_record_mode(self):
        transport = Transport()
        transport.record()
        assert transport.state == TransportState.RECORDING
        transport.stop()
        assert transport.state == TransportState.STOPPED

    def test_render_respects_loop_boundaries_in_meta(self):
        project = Project(
            loop_start=Ticks(PPQ * 4),
            loop_end=Ticks(PPQ * 16),
        )
        result = render_project(project)
        loop_starts = [e for e in result.events if e.event_type == ScheduledEventType.LOOP_START]
        loop_ends = [e for e in result.events if e.event_type == ScheduledEventType.LOOP_END]
        assert len(loop_starts) == 1
        assert len(loop_ends) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
