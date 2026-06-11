"""
Test suite for the core data models.

Verifies:
- Time model conversions
- Note event creation and manipulation
- Clip creation and serialization
- Track management
- Project serialization roundtrip
- Plugin interface contracts
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.time_model import (
    Ticks, Beats, Seconds, Samples, PPQ,
    TimeSignature, BeatValue, TempoChange, TempoMap,
    ticks_to_beats, beats_to_ticks,
    ticks_to_seconds, seconds_to_ticks,
)
from core.model.note import (
    NoteEvent, NotePitch, NoteVelocity,
    pitch_to_name, name_to_pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX,
)
from core.model.clip import MidiClip, AudioClip, AutomationClip, ClipType
from core.model.track import MidiTrack, AudioTrack, GroupTrack, FxTrack, TrackType
from core.model.device import (
    Device, InstrumentDevice, EffectDevice,
    DeviceChain, DeviceParameter, DeviceType,
)
from core.model.automation import (
    AutomationPoint, AutomationLane, AutomationEnvelope,
    InterpolationMode,
)
from core.model.mixer import MixerChannel, Mixer, SendConfig
from core.model.project import Project, ProjectMetadata, Marker, KeySignature


# ── Time Model Tests ───────────────────────────────────────────────

def test_ticks_to_beats():
    """Basic tick-to-beat conversion."""
    assert ticks_to_beats(Ticks(PPQ)) == Beats(1.0), "1 quarter note = PPQ ticks"
    assert ticks_to_beats(Ticks(PPQ * 4)) == Beats(4.0), "4 beats = 1 bar in 4/4"
    assert ticks_to_beats(Ticks(PPQ // 2)) == Beats(0.5), "eighth note"
    print("  ✓ ticks_to_beats")


def test_beats_to_ticks():
    """Basic beat-to-tick conversion."""
    assert beats_to_ticks(Beats(1.0)) == Ticks(PPQ)
    assert beats_to_ticks(Beats(0.25)) == Ticks(PPQ // 4), "sixteenth note"
    print("  ✓ beats_to_ticks")


def test_tempo_map():
    """Tempo map with changes."""
    tm = TempoMap()
    assert tm.bpm_at(Ticks(0)) == 120.0
    
    tm = tm.with_change(TempoChange(tick=Ticks(0), bpm=140.0))
    assert tm.bpm_at(Ticks(0)) == 140.0
    
    tm = tm.with_change(TempoChange(tick=Ticks(PPQ * 4), bpm=80.0))
    assert tm.bpm_at(Ticks(PPQ * 2)) == 140.0  # Before change
    assert tm.bpm_at(Ticks(PPQ * 6)) == 80.0   # After change
    print("  ✓ tempo_map")


def test_ticks_to_seconds_conversion():
    """Ticks to seconds with constant tempo."""
    tm = TempoMap()
    # At 120 BPM, 1 beat = 0.5 seconds
    assert abs(ticks_to_seconds(Ticks(PPQ), tm) - 0.5) < 0.001
    # 1 bar of 4/4 = 4 beats = 2 seconds
    assert abs(ticks_to_seconds(Ticks(PPQ * 4), tm) - 2.0) < 0.001
    print("  ✓ ticks_to_seconds")


def test_time_signature():
    """Time signature creation and properties."""
    ts = TimeSignature(numerator=4, denominator=BeatValue.QUARTER)
    assert ts.ticks_per_measure == PPQ * 4
    assert str(ts) == "4/4"
    
    ts_68 = TimeSignature(numerator=6, denominator=BeatValue.EIGHTH)
    assert ts_68.ticks_per_beat == PPQ // 2  # Eighth note = half quarter note ticks
    assert str(ts_68) == "6/8"
    
    ts_parsed = TimeSignature.from_str("3/4")
    assert ts_parsed.numerator == 3
    assert ts_parsed.denominator == BeatValue.QUARTER
    print("  ✓ time_signature")


# ── Note Model Tests ───────────────────────────────────────────────

def test_note_creation():
    """NoteEvent creation and validation."""
    n = NoteEvent(
        pitch=NotePitch(60),
        velocity=NoteVelocity(100),
        start_tick=Ticks(0),
        duration_ticks=Ticks(PPQ),
    )
    assert n.pitch == 60
    assert n.velocity == 100
    assert n.pitch_name == "C4"
    assert n.end_tick == Ticks(PPQ)
    print("  ✓ note_creation")


def test_note_validation():
    """Note validation rejects invalid values."""
    try:
        NoteEvent(pitch=NotePitch(128), velocity=NoteVelocity(100),
                  start_tick=Ticks(0), duration_ticks=Ticks(PPQ))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        NoteEvent(pitch=NotePitch(60), velocity=NoteVelocity(200),
                  start_tick=Ticks(0), duration_ticks=Ticks(PPQ))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✓ note_validation")


def test_note_transformation():
    """Note transposition and velocity changes."""
    n = NoteEvent(pitch=NotePitch(60), velocity=NoteVelocity(80),
                  start_tick=Ticks(PPQ), duration_ticks=Ticks(PPQ))
    
    transposed = n.transposed(7)  # Up a perfect fifth
    assert transposed.pitch == 67
    assert transposed.pitch_name == "G4"
    assert transposed.start_tick == n.start_tick  # Timing unchanged
    
    softer = n.with_velocity(NoteVelocity(40))
    assert softer.velocity == 40
    
    moved = n.moved(PPQ)  # Shift by 1 beat
    assert moved.start_tick == Ticks(PPQ * 2)
    print("  ✓ note_transformation")


def test_pitch_conversion():
    """Pitch name ↔ MIDI number."""
    assert pitch_to_name(NotePitch(60)) == "C4"
    assert pitch_to_name(NotePitch(69)) == "A4"  # A440
    assert pitch_to_name(NotePitch(0)) == "C-1"
    assert pitch_to_name(NotePitch(127)) == "G9"
    
    assert name_to_pitch("C4") == 60
    assert name_to_pitch("A4") == 69
    assert name_to_pitch("F#3") == 54
    print("  ✓ pitch_conversion")


# ── Clip Model Tests ───────────────────────────────────────────────

def test_midi_clip():
    """MIDI clip creation and manipulation."""
    notes = (
        NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(PPQ // 2)),
        NoteEvent(NotePitch(64), NoteVelocity(100), Ticks(PPQ // 2), Ticks(PPQ // 2)),
        NoteEvent(NotePitch(67), NoteVelocity(100), Ticks(PPQ), Ticks(PPQ // 2)),
    )
    clip = MidiClip(name="Test", notes=notes, length_ticks=Ticks(PPQ * 4))
    
    assert clip.note_count == 3
    assert clip.pitch_range == (60, 67)
    assert clip.end_tick == Ticks(PPQ * 4)
    
    # Add a note
    new_note = NoteEvent(NotePitch(72), NoteVelocity(80), Ticks(PPQ * 2), Ticks(PPQ))
    clip2 = clip.with_added_note(new_note)
    assert clip2.note_count == 4
    
    # Serialization roundtrip
    d = clip.to_dict()
    restored = MidiClip.from_dict(d)
    assert restored.name == "Test"
    assert restored.note_count == 3
    assert restored.notes[0].pitch == 60
    print("  ✓ midi_clip")


def test_clip_serialization():
    """Clip JSON serialization roundtrip."""
    notes = (
        NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(PPQ)),
    )
    clip = MidiClip(
        id="test123",
        name="My Clip",
        start_tick=Ticks(PPQ * 2),
        length_ticks=Ticks(PPQ * 8),
        muted=False,
        color="#FF5733",
        notes=notes,
    )
    
    d = clip.to_dict()
    assert d["id"] == "test123"
    assert d["clip_type"] == "midi"
    assert d["color"] == "#FF5733"
    assert len(d["notes"]) == 1
    assert d["notes"][0]["pitch"] == 60
    
    restored = MidiClip.from_dict(d)
    assert restored.id == "test123"
    assert restored.name == "My Clip"
    assert restored.color == "#FF5733"
    assert restored.note_count == 1
    print("  ✓ clip_serialization")


# ── Track Tests ────────────────────────────────────────────────────

def test_midi_track():
    """MIDI track with clips."""
    track = MidiTrack(name="Piano", channel=0)
    assert track.track_type == TrackType.MIDI
    assert len(track.clips) == 0
    
    clip = MidiClip(name="Chords", notes=(
        NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(PPQ * 4)),
    ))
    track2 = track.with_clip(clip)
    assert len(track2.clips) == 1
    
    track3 = track2.with_clip_removed(clip.id)
    assert len(track3.clips) == 0
    print("  ✓ midi_track")


def test_track_serialization():
    """Track serialization roundtrip."""
    track = MidiTrack(
        id="track1",
        name="Synth Lead",
        channel=1,
        instrument_id="inst_vst_synth",
        volume_db=-3.0,
        pan=0.2,
    )
    
    d = track.to_dict()
    assert d["track_type"] == "midi"
    assert d["volume_db"] == -3.0
    
    restored = MidiTrack.from_dict(d)
    assert restored.name == "Synth Lead"
    assert restored.channel == 1
    print("  ✓ track_serialization")


# ── Mixer Tests ────────────────────────────────────────────────────

def test_mixer_channel():
    """Mixer channel creation and modification."""
    ch = MixerChannel(track_id="track1", volume_db=-6.0, pan=0.5)
    assert ch.gain_multiplier < 1.0
    
    ch2 = ch.with_volume(0.0)
    assert abs(ch2.gain_multiplier - 1.0) < 0.01
    
    # Add a send
    send = SendConfig(target_track_id="fx_reverb", amount_db=-12.0)
    ch3 = ch2.with_send(send)
    assert len(ch3.sends) == 1
    print("  ✓ mixer_channel")


def test_mixer():
    """Mixer with multiple channels."""
    mixer = Mixer()
    
    ch1 = MixerChannel(track_id="track1", volume_db=-3.0)
    ch2 = MixerChannel(track_id="track2", volume_db=0.0, soloed=True)
    
    mixer = mixer.with_channel(ch1).with_channel(ch2)
    assert len(mixer.channels) == 2
    
    # Solo logic
    audible = mixer.audible_channels
    assert len(audible) == 1
    assert audible[0].track_id == "track2"
    print("  ✓ mixer")


# ── Project Tests ──────────────────────────────────────────────────

def test_project_creation():
    """Create a new project with defaults."""
    p = Project.create_new(name="My Song", bpm=128.0)
    assert p.metadata.name == "My Song"
    assert p.metadata.bpm == 128.0
    assert len(p.tracks) == 0
    assert str(p.time_signature) == "4/4"
    print("  ✓ project_creation")


def test_project_with_track():
    """Add a track to a project."""
    p = Project.create_new()
    track = MidiTrack(name="Bass", channel=2)
    p2 = p.with_track(track)
    assert len(p2.tracks) == 1
    assert p2.midi_tracks[0].name == "Bass"
    assert len(p.midi_tracks) == 0  # Original unchanged (immutability)
    print("  ✓ project_with_track")


def test_project_total_duration():
    """Calculate project duration from clips."""
    p = Project.create_new()
    clip = MidiClip(
        start_tick=Ticks(PPQ * 4),
        length_ticks=Ticks(PPQ * 8),
        notes=(),
    )
    track = MidiTrack(name="Test").with_clip(clip)
    p = p.with_track(track)
    # End tick = 4*PPQ + 8*PPQ = 12*PPQ
    assert p.total_duration_ticks == Ticks(PPQ * 12)
    print("  ✓ project_total_duration")


def test_project_serialization_roundtrip():
    """Full project serialization roundtrip."""
    p = Project.create_new(name="Test Project", bpm=140.0)
    
    # Add a track with a clip with notes
    notes = (
        NoteEvent(NotePitch(60), NoteVelocity(100), Ticks(0), Ticks(PPQ // 2)),
        NoteEvent(NotePitch(64), NoteVelocity(90), Ticks(PPQ // 2), Ticks(PPQ // 2)),
        NoteEvent(NotePitch(67), NoteVelocity(80), Ticks(PPQ), Ticks(PPQ // 2)),
        NoteEvent(NotePitch(72), NoteVelocity(70), Ticks(PPQ * 3 // 2), Ticks(PPQ // 2)),
    )
    clip = MidiClip(name="Melody", notes=notes, length_ticks=Ticks(PPQ * 4))
    track = MidiTrack(name="Piano", channel=0).with_clip(clip)
    p = p.with_track(track)
    
    # Add tempo change
    p = Project(
        id=p.id, metadata=p.metadata,
        tempo_map=TempoMap(changes=(
            TempoChange(tick=Ticks(0), bpm=140.0),
            TempoChange(tick=Ticks(PPQ * 4), bpm=160.0),
        )),
        time_signature=p.time_signature,
        key_signatures=(
            KeySignature(tick=Ticks(0), sharps_flats=0, mode="major"),
        ),
        tracks=p.tracks, mixer=p.mixer,
        automation=p.automation, markers=p.markers,
        loop_start=p.loop_start, loop_end=p.loop_end,
        project_length_ticks=p.project_length_ticks,
    )
    
    # Serialize
    d = p.to_dict()
    assert d["schema_version"] == "1.0.0"
    assert d["metadata"]["name"] == "Test Project"
    assert len(d["tracks"]) == 1
    assert len(d["tracks"][0]["clips"]) == 1
    assert len(d["tracks"][0]["clips"][0]["notes"]) == 4
    assert len(d["tempo_map"]) == 2
    
    # Deserialize
    restored = Project.from_dict(d)
    assert restored.metadata.name == "Test Project"
    assert restored.metadata.bpm == 140.0
    assert len(restored.tracks) == 1
    assert isinstance(restored.tracks[0], MidiTrack)
    assert restored.tracks[0].name == "Piano"
    assert len(restored.midi_tracks[0].clips) == 1
    assert restored.midi_tracks[0].clips[0].note_count == 4
    assert restored.midi_tracks[0].clips[0].notes[2].pitch == 67
    assert len(restored.tempo_map.changes) == 2
    assert restored.tempo_map.bpm_at(Ticks(PPQ * 2)) == 140.0
    assert restored.tempo_map.bpm_at(Ticks(PPQ * 8)) == 160.0
    
    # Verify notes survived
    notes_restored = restored.midi_tracks[0].clips[0].notes
    assert notes_restored[0].pitch_name == "C4"
    assert notes_restored[3].pitch_name == "C5"
    
    print("  ✓ project_serialization_roundtrip")


# ── Automation Tests ───────────────────────────────────────────────

def test_automation_envelope():
    """Automation envelope evaluation."""
    lane = AutomationLane(device_id="dev1", parameter_id="volume")
    points = (
        AutomationPoint(tick=Ticks(0), value=0.0, interpolation=InterpolationMode.LINEAR),
        AutomationPoint(tick=Ticks(PPQ), value=1.0, interpolation=InterpolationMode.LINEAR),
    )
    env = AutomationEnvelope(lane=lane, points=points)
    
    # Before first point
    assert env.value_at(Ticks(-10)) == 0.0
    
    # Midpoint (linear interpolation)
    mid = env.value_at(Ticks(PPQ // 2))
    assert abs(mid - 0.5) < 0.01
    
    # At second point
    assert env.value_at(Ticks(PPQ)) == 1.0
    
    # After last point
    assert env.value_at(Ticks(PPQ * 2)) == 1.0
    print("  ✓ automation_envelope")


def test_automation_smooth():
    """Smooth interpolation."""
    points = (
        AutomationPoint(tick=Ticks(0), value=0.0, interpolation=InterpolationMode.SMOOTH),
        AutomationPoint(tick=Ticks(PPQ), value=1.0),
    )
    env = AutomationEnvelope(points=points)
    
    mid = env.value_at(Ticks(PPQ // 2))
    # Smoothstep should be ~0.5 at midpoint but with easing
    assert 0.45 < mid < 0.55
    print("  ✓ automation_smooth")


# ── Device Tests ───────────────────────────────────────────────────

def test_device_chain():
    """Device chain management."""
    synth = InstrumentDevice(name="Synth", plugin_id="builtin.synth")
    reverb = EffectDevice(name="Reverb", plugin_id="builtin.reverb", wet_dry=0.3)
    delay = EffectDevice(name="Delay", plugin_id="builtin.delay", wet_dry=0.5)
    
    chain = DeviceChain(devices=(synth, reverb, delay))
    
    assert chain.instrument is not None
    assert chain.instrument.name == "Synth"
    assert len(chain.effects) == 2
    
    # Remove an effect
    chain2 = chain.without_device(delay.id)
    assert len(chain2.devices) == 2
    print("  ✓ device_chain")


def test_device_parameters():
    """Device parameter manipulation."""
    param = DeviceParameter(
        id="cutoff", name="Filter Cutoff",
        value=500.0, min_value=20.0, max_value=20000.0,
        unit="Hz",
    )
    
    p2 = param.with_value(1000.0)
    assert p2.value == 1000.0
    
    # Clamping
    p3 = param.with_value(30000.0)
    assert p3.value == 20000.0
    
    # Normalization
    # (1000 - 20) / (20000 - 20) ≈ 0.049
    norm = p2.normalized
    assert 0.04 < norm < 0.06
    print("  ✓ device_parameters")


# ── Plugin Interface Tests ─────────────────────────────────────────

def test_plugin_interface_imports():
    """Verify all plugin interfaces are importable."""
    from core.plugin.interfaces.base import PluginBase, PluginManifest, PluginCategory, PluginState
    from core.plugin.interfaces.instrument import InstrumentPlugin
    from core.plugin.interfaces.effect import EffectPlugin
    from core.plugin.interfaces.ai_generator import (
        AIGeneratorPlugin, GeneratorCapabilities, GenerationPrompt,
        MusicalContext, GenerationConstraints, GeneratedContent, ContentType,
    )
    from core.plugin.interfaces.ai_analyzer import (
        AIAnalyzerPlugin, AnalyzerCapabilities, AnalysisRequest, AnalysisResult,
    )
    from core.plugin.interfaces.ui_widget import UIWidgetPlugin, WidgetType
    from core.plugin.interfaces.exporter import ExporterPlugin, ExportFormat, ExportRequest, ExportResult
    
    # Verify basic construction
    caps = GeneratorCapabilities(
        content_types=[ContentType.MELODY, ContentType.CHORDS],
        max_bars=16,
        supports_text_prompt=True,
        style_tags=["jazz", "pop", "lofi"],
    )
    assert ContentType.MELODY in caps.content_types
    assert caps.max_bars == 16
    
    ctx = MusicalContext(bpm=120.0, key_sharps_flats=0, key_mode="major", bar_count=8)
    assert ctx.bpm == 120.0
    assert ctx.ticks_per_bar > 0
    
    # GeneratedContent
    gc = GeneratedContent(
        explanation="Generated a simple melody in C major",
        confidence=0.85,
    )
    assert gc.total_notes == 0
    assert gc.explanation != ""
    
    # Analysis result
    ar = AnalysisResult(
        summary="Key detected: C major with 95% confidence",
        confidence=0.95,
        suggestions=["The bass line conflicts with the chords in bar 5"],
    )
    assert ar.has_findings == False  # No specific findings, just summary
    
    print("  ✓ plugin_interface_imports")


# ── State Management Tests ─────────────────────────────────────────

def test_store_and_actions():
    """Store dispatch and action application."""
    from core.state.store import Store
    from core.state.actions import AddTrackAction, SetTempoAction, AddClipAction
    
    store = Store()
    initial = store.get_state()
    assert initial.metadata.name == "Untitled"
    
    # Add a track
    track = MidiTrack(name="Test Track")
    action = AddTrackAction(track=track)
    new_state = store.dispatch(action)
    assert len(new_state.tracks) == 1
    
    # Set tempo
    tempo_action = SetTempoAction(tick=Ticks(0), bpm=140.0)
    new_state = store.dispatch(tempo_action)
    assert new_state.tempo_map.bpm_at(Ticks(0)) == 140.0
    
    # State from store is updated
    assert store.get_state().tempo_map.bpm_at(Ticks(0)) == 140.0
    print("  ✓ store_and_actions")


def test_history_undo_redo():
    """Undo/redo history manager."""
    from core.state.store import Store
    from core.state.actions import SetTempoAction
    from core.state.history import HistoryManager
    
    store = Store()
    history = HistoryManager()
    
    # Record initial state
    initial_state = store.get_state()
    
    # Apply and record a tempo change
    action = SetTempoAction(tick=Ticks(0), bpm=160.0)
    state_before = store.get_state()
    new_state = store.dispatch(action)
    history.record(action, state_before)
    
    assert store.get_state().tempo_map.bpm_at(Ticks(0)) == 160.0
    assert history.can_undo
    assert not history.can_redo
    
    # Undo
    restored = history.undo(store)
    assert restored is not None
    assert store.get_state().tempo_map.bpm_at(Ticks(0)) == 120.0  # Default
    assert history.can_redo
    
    # Redo
    redone = history.redo(store)
    assert redone is not None
    assert store.get_state().tempo_map.bpm_at(Ticks(0)) == 160.0
    assert not history.can_redo
    
    print("  ✓ history_undo_redo")


# ── Plugin Registry Tests ──────────────────────────────────────────

def test_registry_basic():
    """Plugin registry creation and basic operations."""
    from core.plugin.registry import PluginRegistry, PluginCategory
    
    registry = PluginRegistry()
    assert registry.plugin_count == 0
    
    # Register a programmatic instance
    from core.plugin.interfaces.ai_generator import AIGeneratorPlugin
    
    class MockGenerator(AIGeneratorPlugin):
        """Simple mock generator for testing."""
        def initialize(self):
            from core.plugin.interfaces.ai_generator import (
                GeneratorCapabilities, ContentType,
            )
            self._caps = GeneratorCapabilities(
                content_types=[ContentType.MELODY],
                max_bars=8,
            )

        def shutdown(self):
            pass

        def get_capabilities(self):
            return self._caps

        def get_manifest(self):
            from core.plugin.interfaces.ai_generator import (
                PluginManifest,
            )
            from core.plugin.interfaces.base import PluginCategory
            return PluginManifest(
                plugin_id="test.mock_generator",
                name="Mock Generator",
                version="1.0.0",
                category=PluginCategory.AI_GENERATOR,
                capabilities=["generate.melody"],
            )

        def generate(self, prompt):
            from core.plugin.interfaces.ai_generator import GeneratedContent
            return GeneratedContent(explanation="Mock generation")
    
    gen = MockGenerator()
    gen.initialize()
    registry.register(gen)
    
    assert registry.plugin_count == 1
    assert len(registry.get_by_category(PluginCategory.AI_GENERATOR)) == 1
    
    instance = registry.get("test.mock_generator")
    assert instance is not None
    assert instance.get_manifest().name == "Mock Generator"
    
    print("  ✓ registry_basic")


def test_registry_discovers_and_loads_plugin_package(tmp_path):
    """Filesystem plugin packages can be discovered and loaded."""
    from core.plugin.registry import PluginRegistry

    plugins_dir = tmp_path / "plugins"
    package_dir = plugins_dir / "relative_plugin"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "helper.py").write_text(
        "PLUGIN_NAME = 'Relative Plugin'\n",
        encoding="utf-8",
    )
    (package_dir / "plugin.py").write_text(
        "\n".join(
            [
                "from .helper import PLUGIN_NAME",
                "from core.plugin.interfaces.base import PluginBase, PluginCategory, PluginManifest",
                "",
                "class RelativePlugin(PluginBase):",
                "    def __init__(self):",
                "        self.initialized = False",
                "",
                "    def get_manifest(self):",
                "        return PluginManifest(",
                "            plugin_id='test.relative_plugin',",
                "            name=PLUGIN_NAME,",
                "            version='1.0.0',",
                "            category=PluginCategory.UTILITY,",
                "        )",
                "",
                "    def initialize(self):",
                "        self.initialized = True",
                "",
                "    def shutdown(self):",
                "        self.initialized = False",
            ]
        ),
        encoding="utf-8",
    )

    registry = PluginRegistry()
    assert registry.discover_plugins([str(plugins_dir)]) == 1

    handle = registry.get_handle("test.relative_plugin")
    assert handle is not None
    assert handle.file_path.endswith("plugin.py")

    plugin = registry.load_plugin("test.relative_plugin")
    assert plugin is not None
    assert plugin.get_manifest().name == "Relative Plugin"
    assert plugin.initialized is True

    print("  ✓ registry_discovers_and_loads_plugin_package")


def test_registry_discovers_single_file_plugin(tmp_path):
    """Single-file plugin modules named plugin_<name>.py can be loaded."""
    from core.plugin.registry import PluginRegistry

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "plugin_loose.py").write_text(
        "\n".join(
            [
                "from core.plugin.interfaces.base import PluginBase, PluginCategory, PluginManifest",
                "",
                "class LoosePlugin(PluginBase):",
                "    def __init__(self):",
                "        self.initialized = False",
                "",
                "    def get_manifest(self):",
                "        return PluginManifest(",
                "            plugin_id='test.loose_plugin',",
                "            name='Loose Plugin',",
                "            version='1.0.0',",
                "            category=PluginCategory.UTILITY,",
                "        )",
                "",
                "    def initialize(self):",
                "        self.initialized = True",
                "",
                "    def shutdown(self):",
                "        self.initialized = False",
            ]
        ),
        encoding="utf-8",
    )

    registry = PluginRegistry()
    assert registry.discover_plugins([str(plugins_dir)]) == 1

    plugin = registry.load_plugin("test.loose_plugin")
    assert plugin is not None
    assert plugin.get_manifest().name == "Loose Plugin"
    assert plugin.initialized is True

    print("  ✓ registry_discovers_single_file_plugin")


# ── Run all tests ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("amusiment Core Model Tests")
    print("=" * 60)
    
    print("\n[Time Model]")
    test_ticks_to_beats()
    test_beats_to_ticks()
    test_tempo_map()
    test_ticks_to_seconds_conversion()
    test_time_signature()
    
    print("\n[Note Model]")
    test_note_creation()
    test_note_validation()
    test_note_transformation()
    test_pitch_conversion()
    
    print("\n[Clip Model]")
    test_midi_clip()
    test_clip_serialization()
    
    print("\n[Track Model]")
    test_midi_track()
    test_track_serialization()
    
    print("\n[Mixer Model]")
    test_mixer_channel()
    test_mixer()
    
    print("\n[Project Model]")
    test_project_creation()
    test_project_with_track()
    test_project_total_duration()
    test_project_serialization_roundtrip()
    
    print("\n[Automation]")
    test_automation_envelope()
    test_automation_smooth()
    
    print("\n[Device Model]")
    test_device_chain()
    test_device_parameters()
    
    print("\n[Plugin Interfaces]")
    test_plugin_interface_imports()
    
    print("\n[State Management]")
    test_store_and_actions()
    test_history_undo_redo()
    
    print("\n[Plugin Registry]")
    test_registry_basic()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
