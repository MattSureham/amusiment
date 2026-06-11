"""
Tests for the built-in Standard MIDI File importer.
"""

import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.export.midi import build_standard_midi_bytes
from core.importer.midi import MidiImporterPlugin, import_standard_midi_bytes
from core.model.clip import MidiClip
from core.model.note import NoteEvent, NotePitch, NoteVelocity
from core.model.project import KeySignature, Project, ProjectMetadata
from core.model.time_model import BeatValue, PPQ, Ticks, TimeSignature
from core.model.track import MidiTrack
from core.plugin.interfaces.importer import ImportFormat, ImportRequest


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack(">I", len(payload)) + payload


def _vlq(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def _meta(meta_type: int, payload: bytes) -> bytes:
    return b"\xFF" + bytes([meta_type]) + _vlq(len(payload)) + payload


def _smf(fmt: int, division: int, tracks: list[bytes]) -> bytes:
    header = _chunk(b"MThd", struct.pack(">HHH", fmt, len(tracks), division))
    return header + b"".join(_chunk(b"MTrk", track) for track in tracks)


def test_midi_importer_reads_exported_project():
    project = Project(
        metadata=ProjectMetadata(name="Roundtrip Source", bpm=128.0),
        time_signature=TimeSignature(numerator=3, denominator=BeatValue.QUARTER),
        key_signatures=(KeySignature(tick=Ticks(0), sharps_flats=2, mode="major"),),
    )
    note = NoteEvent(
        pitch=NotePitch(60),
        velocity=NoteVelocity(100),
        start_tick=Ticks(PPQ),
        duration_ticks=Ticks(PPQ // 2),
        channel=3,
    )
    clip = MidiClip(
        name="Lead Clip",
        start_tick=Ticks(PPQ * 2),
        length_ticks=Ticks(PPQ * 4),
        notes=(note,),
    )
    project = project.with_track(MidiTrack(name="Lead").with_clip(clip))

    data = build_standard_midi_bytes(project)
    imported, warnings = import_standard_midi_bytes(data, project_name="Imported")

    assert warnings == []
    assert imported.metadata.name == "Imported"
    assert abs(imported.metadata.bpm - 128.0) < 0.001
    assert imported.time_signature.numerator == 3
    assert imported.time_signature.denominator == BeatValue.QUARTER
    assert imported.key_signatures[0].sharps_flats == 2
    assert imported.key_signatures[0].mode == "major"
    assert len(imported.midi_tracks) == 1

    track = imported.midi_tracks[0]
    assert track.name == "Lead"
    assert track.channel == 3
    assert len(track.clips) == 1

    imported_note = track.clips[0].notes[0]
    assert imported_note.pitch == 60
    assert imported_note.velocity == 100
    assert imported_note.channel == 3
    assert imported_note.start_tick == Ticks(PPQ * 3)
    assert imported_note.duration_ticks == Ticks(PPQ // 2)


def test_midi_importer_reads_format_zero_running_status_and_scales_ppq():
    # Source division is 480, so imported ticks should double to framework PPQ=960.
    track_payload = b"".join(
        [
            _vlq(0),
            _meta(0x03, b"Piano"),
            _vlq(0),
            _meta(0x51, (500000).to_bytes(3, "big")),
            _vlq(0),
            bytes([0x92, 60, 100]),  # note-on, channel 3 (zero-based channel 2)
            _vlq(240),
            bytes([64, 90]),        # running-status note-on
            _vlq(240),
            bytes([60, 0]),         # running-status note-off via velocity 0
            _vlq(0),
            bytes([64, 0]),         # same tick, second note-off
            _vlq(0),
            _meta(0x2F, b""),
        ]
    )
    data = _smf(fmt=0, division=480, tracks=[track_payload])

    imported, warnings = import_standard_midi_bytes(data)

    assert warnings == []
    assert abs(imported.metadata.bpm - 120.0) < 0.001
    assert len(imported.midi_tracks) == 1

    track = imported.midi_tracks[0]
    assert track.name == "Piano"
    assert track.channel == 2

    notes = track.clips[0].notes
    assert len(notes) == 2
    assert notes[0].pitch == 60
    assert notes[0].start_tick == Ticks(0)
    assert notes[0].duration_ticks == Ticks(PPQ)
    assert notes[1].pitch == 64
    assert notes[1].start_tick == Ticks(PPQ // 2)
    assert notes[1].duration_ticks == Ticks(PPQ // 2)


def test_midi_importer_plugin_imports_file_and_splits_channels():
    track_payload = b"".join(
        [
            _vlq(0),
            _meta(0x03, b"Sketch"),
            _vlq(0),
            bytes([0x90, 60, 100]),
            _vlq(0),
            bytes([0x91, 72, 90]),
            _vlq(480),
            bytes([0x80, 60, 64]),
            _vlq(0),
            bytes([0x81, 72, 64]),
            _vlq(0),
            _meta(0x2F, b""),
        ]
    )
    data = _smf(fmt=0, division=480, tracks=[track_payload])

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "sketch.mid")
        with open(input_path, "wb") as f:
            f.write(data)

        importer = MidiImporterPlugin()
        importer.initialize()
        result = importer.import_file(
            ImportRequest(
                format=ImportFormat.MIDI,
                input_path=input_path,
                project_name="Split Import",
                track_name_prefix="Imported ",
                split_channels=True,
            )
        )

    assert result.success, result.error_message
    assert result.imported_track_count == 2
    assert result.imported_clip_count == 2
    assert result.duration_ticks == PPQ
    assert result.project_state["metadata"]["name"] == "Split Import"

    imported = Project.from_dict(result.project_state)
    assert [track.channel for track in imported.midi_tracks] == [0, 1]
    assert [track.name for track in imported.midi_tracks] == [
        "Imported Sketch Ch 1",
        "Imported Sketch Ch 2",
    ]


if __name__ == "__main__":
    test_midi_importer_reads_exported_project()
    print("  ok: reads_exported_project")
    test_midi_importer_reads_format_zero_running_status_and_scales_ppq()
    print("  ok: reads_format_zero_running_status_and_scales_ppq")
    test_midi_importer_plugin_imports_file_and_splits_channels()
    print("  ok: plugin_imports_file_and_splits_channels")
