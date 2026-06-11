"""
Tests for the built-in Standard MIDI File exporter.
"""

import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.export.midi import MidiExporterPlugin
from core.model.clip import MidiClip
from core.model.note import NoteEvent, NotePitch, NoteVelocity
from core.model.project import Project
from core.model.time_model import PPQ, Ticks
from core.model.track import MidiTrack
from core.plugin.interfaces.exporter import ExportFormat, ExportRequest


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, pos


def _read_chunks(data: bytes) -> tuple[tuple[int, int, int], list[bytes]]:
    assert data[:4] == b"MThd"
    header_length = struct.unpack(">I", data[4:8])[0]
    fmt, track_count, division = struct.unpack(">HHH", data[8:14])
    pos = 8 + header_length

    tracks = []
    while pos < len(data):
        assert data[pos : pos + 4] == b"MTrk"
        length = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        pos += 8
        tracks.append(data[pos : pos + length])
        pos += length

    return (fmt, track_count, division), tracks


def _parse_track_events(track: bytes) -> list[tuple[int, int, bytes]]:
    events = []
    pos = 0
    tick = 0
    running_status = None

    while pos < len(track):
        delta, pos = _read_vlq(track, pos)
        tick += delta
        status = track[pos]
        pos += 1

        if status < 0x80:
            if running_status is None:
                raise AssertionError("running status used before status byte")
            first_data = status
            status = running_status
        else:
            first_data = None
            if status < 0xF0:
                running_status = status

        if status == 0xFF:
            meta_type = track[pos]
            pos += 1
            length, pos = _read_vlq(track, pos)
            payload = track[pos : pos + length]
            pos += length
            events.append((tick, 0xFF00 | meta_type, payload))
            if meta_type == 0x2F:
                break
        elif status in (0xF0, 0xF7):
            length, pos = _read_vlq(track, pos)
            pos += length
        else:
            event_type = status & 0xF0
            data_len = 1 if event_type in (0xC0, 0xD0) else 2
            if first_data is None:
                payload = track[pos : pos + data_len]
                pos += data_len
            else:
                payload = bytes([first_data]) + track[pos : pos + data_len - 1]
                pos += data_len - 1
            events.append((tick, status, payload))

    return events


def test_midi_exporter_writes_standard_midi_file():
    project = Project.create_new(name="MIDI Test", bpm=128.0)
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
    track = MidiTrack(name="Lead").with_clip(clip)
    project = project.with_track(track)

    exporter = MidiExporterPlugin()
    exporter.initialize()

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "song.mid")
        request = ExportRequest(format=ExportFormat.MIDI, output_path=output_path)
        result = exporter.export(request, project.to_dict())

        assert result.success, result.error_message
        assert result.file_size_bytes > 0
        assert os.path.exists(output_path)

        with open(output_path, "rb") as f:
            data = f.read()

    (fmt, track_count, division), tracks = _read_chunks(data)
    assert fmt == 1
    assert track_count == 2
    assert division == PPQ

    meta_events = _parse_track_events(tracks[0])
    tempo_events = [payload for _, event_type, payload in meta_events if event_type == 0xFF51]
    assert tempo_events
    assert int.from_bytes(tempo_events[0], "big") == round(60_000_000 / 128.0)

    note_events = _parse_track_events(tracks[1])
    assert (PPQ * 3, 0x93, bytes([60, 100])) in note_events
    assert (PPQ * 3 + PPQ // 2, 0x83, bytes([60, 64])) in note_events


def test_midi_exporter_filters_tracks_and_crops_range():
    project = Project.create_new(name="Range Test", bpm=120.0)

    ignored_track = MidiTrack(name="Ignored").with_clip(
        MidiClip(
            notes=(
                NoteEvent(NotePitch(72), NoteVelocity(80), Ticks(0), Ticks(PPQ)),
            )
        )
    )

    cropped_note = NoteEvent(
        NotePitch(48),
        NoteVelocity(90),
        Ticks(PPQ // 2),
        Ticks(PPQ * 2),
        channel=1,
    )
    selected_track = MidiTrack(name="Selected").with_clip(
        MidiClip(
            start_tick=Ticks(0),
            length_ticks=Ticks(PPQ * 4),
            notes=(cropped_note,),
        )
    )

    project = project.with_track(ignored_track).with_track(selected_track)
    exporter = MidiExporterPlugin()
    exporter.initialize()

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "range.mid")
        request = ExportRequest(
            format=ExportFormat.MIDI,
            output_path=output_path,
            track_ids=[selected_track.id],
            start_tick=PPQ,
            end_tick=PPQ * 3,
        )
        result = exporter.export(request, project.to_dict())
        assert result.success, result.error_message

        with open(output_path, "rb") as f:
            data = f.read()

    (_, track_count, _), tracks = _read_chunks(data)
    assert track_count == 2

    events = _parse_track_events(tracks[1])
    assert (0, 0x91, bytes([48, 90])) in events
    assert (PPQ + PPQ // 2, 0x81, bytes([48, 64])) in events
    assert all(event[1] != 0x90 or event[2][0] != 72 for event in events)


if __name__ == "__main__":
    test_midi_exporter_writes_standard_midi_file()
    print("  ok: writes_standard_midi_file")
    test_midi_exporter_filters_tracks_and_crops_range()
    print("  ok: filters_tracks_and_crops_range")
