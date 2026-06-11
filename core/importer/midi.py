"""
Built-in Standard MIDI File importer.

This importer reads MIDI Format 0 and Format 1 files using only the Python
standard library. It extracts tempo, time signature, key signature, markers,
track names, and note events, then converts them into a Project containing
MidiTrack/MidiClip data at the framework's PPQ resolution.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.model.clip import MidiClip
from core.model.note import NoteEvent, NotePitch, NoteVelocity
from core.model.project import KeySignature, Marker, Project, ProjectMetadata
from core.model.time_model import (
    PPQ,
    BeatValue,
    TempoChange,
    TempoMap,
    Ticks,
    TimeSignature,
)
from core.model.track import MidiTrack
from core.plugin.interfaces.base import PluginCategory, PluginManifest, PluginState
from core.plugin.interfaces.importer import (
    ImportFormat,
    ImportRequest,
    ImportResult,
    ImporterPlugin,
)


@dataclass(frozen=True)
class _MidiChunk:
    kind: bytes
    payload: bytes


@dataclass
class _ParsedTrack:
    index: int
    name: str = ""
    notes: list[NoteEvent] = field(default_factory=list)
    tempos: list[tuple[int, float]] = field(default_factory=list)
    time_signatures: list[tuple[int, TimeSignature]] = field(default_factory=list)
    key_signatures: list[KeySignature] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    end_tick: int = 0


class MidiImporterPlugin(ImporterPlugin):
    """Built-in importer for Standard MIDI Files (.mid/.midi)."""

    def __init__(self) -> None:
        self._state = PluginState.LOADED

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="builtin.import.midi",
            name="Standard MIDI Importer",
            version="0.1.0",
            category=PluginCategory.IMPORTER,
            author="amusiment",
            description="Imports Standard MIDI File format 0/1 into editable MIDI clips.",
            capabilities=["import.midi", "import.standard-midi-file"],
        )

    def initialize(self) -> None:
        self._state = PluginState.INITIALIZED

    def shutdown(self) -> None:
        self._state = PluginState.UNLOADED

    def get_state(self) -> PluginState:
        return self._state

    def get_supported_formats(self) -> list[ImportFormat]:
        return [ImportFormat.MIDI]

    def import_file(
        self,
        request: ImportRequest,
        progress_callback: "Optional[Callable[[float, str], None]]" = None,
    ) -> ImportResult:
        def progress(value: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(value, message)

        progress(0.0, "Validating MIDI import request")
        errors = self.validate_request(request)
        if errors:
            return ImportResult(
                success=False,
                input_path=request.input_path,
                format=request.format,
                error_message="; ".join(errors),
            )

        try:
            progress(0.2, "Reading MIDI file")
            with open(request.input_path, "rb") as f:
                data = f.read()

            project_name = request.project_name or os.path.splitext(
                os.path.basename(request.input_path)
            )[0]

            progress(0.7, "Parsing MIDI events")
            project, warnings = import_standard_midi_bytes(
                data,
                project_name=project_name,
                track_name_prefix=request.track_name_prefix,
                split_channels=request.split_channels,
            )

            progress(1.0, "MIDI import complete")
            return ImportResult(
                success=True,
                input_path=request.input_path,
                format=ImportFormat.MIDI,
                project_state=project.to_dict(),
                imported_track_count=len(project.midi_tracks),
                imported_clip_count=sum(len(track.clips) for track in project.midi_tracks),
                duration_ticks=int(project.project_length_ticks),
                warnings=warnings,
            )
        except Exception as exc:
            return ImportResult(
                success=False,
                input_path=request.input_path,
                format=ImportFormat.MIDI,
                error_message=str(exc),
            )


def import_standard_midi_bytes(
    data: bytes,
    project_name: str = "Imported MIDI",
    track_name_prefix: str = "",
    split_channels: bool = False,
) -> tuple[Project, list[str]]:
    """
    Parse a Standard MIDI File byte string into a Project.

    Args:
        data: Complete .mid file contents.
        project_name: Name for the created project.
        track_name_prefix: Optional prefix applied to created track names.
        split_channels: If True, split each MIDI track into channel tracks.

    Returns:
        A tuple of (Project, warnings).
    """
    header, chunks = _read_smf_chunks(data)
    fmt, declared_track_count, division = header
    warnings: list[str] = []

    if fmt not in (0, 1):
        raise ValueError(f"Unsupported MIDI format {fmt}; only format 0 and 1 are supported")
    if division & 0x8000:
        raise ValueError("SMPTE-time MIDI files are not supported")
    if division <= 0:
        raise ValueError(f"Invalid MIDI ticks-per-quarter division: {division}")

    track_chunks = [chunk for chunk in chunks if chunk.kind == b"MTrk"]
    if len(track_chunks) < declared_track_count:
        warnings.append(
            f"MIDI header declares {declared_track_count} tracks but file contains {len(track_chunks)}"
        )
    track_chunks = track_chunks[:declared_track_count]

    parsed_tracks = [
        _parse_track(chunk.payload, index, division)
        for index, chunk in enumerate(track_chunks)
    ]
    for parsed in parsed_tracks:
        warnings.extend(parsed.warnings)

    tempo_events = _dedupe_tempos(
        event
        for parsed in parsed_tracks
        for event in parsed.tempos
    )
    time_signature = _first_time_signature(parsed_tracks, warnings)
    key_signatures = _dedupe_key_signatures(
        key
        for parsed in parsed_tracks
        for key in parsed.key_signatures
    )
    markers = _dedupe_markers(
        marker
        for parsed in parsed_tracks
        for marker in parsed.markers
    )

    initial_bpm = _bpm_at_zero(tempo_events)
    tempo_map = TempoMap(
        changes=tuple(
            TempoChange(tick=Ticks(tick), bpm=bpm)
            for tick, bpm in tempo_events
        )
    )

    midi_tracks = _build_tracks(
        parsed_tracks,
        track_name_prefix=track_name_prefix,
        split_channels=split_channels,
        time_signature=time_signature,
        warnings=warnings,
    )
    project_length_ticks = Ticks(
        max(
            [0]
            + [int(note.end_tick) for track in midi_tracks for clip in track.clips for note in clip.notes]
            + [parsed.end_tick for parsed in parsed_tracks]
        )
    )

    metadata = ProjectMetadata(
        name=project_name or "Imported MIDI",
        bpm=initial_bpm,
        description="Imported from Standard MIDI File",
    )

    project = Project(
        metadata=metadata,
        tempo_map=tempo_map,
        time_signature=time_signature,
        key_signatures=tuple(key_signatures),
        tracks=tuple(midi_tracks),
        markers=tuple(markers),
        project_length_ticks=project_length_ticks,
    )
    return project, warnings


def _read_smf_chunks(data: bytes) -> tuple[tuple[int, int, int], list[_MidiChunk]]:
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File: missing MThd header")

    header_length = struct.unpack(">I", data[4:8])[0]
    if header_length < 6:
        raise ValueError(f"Invalid MIDI header length: {header_length}")
    if len(data) < 8 + header_length:
        raise ValueError("Truncated MIDI header")

    fmt, track_count, division = struct.unpack(">HHH", data[8:14])
    pos = 8 + header_length
    chunks: list[_MidiChunk] = []

    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("Truncated MIDI chunk header")
        kind = data[pos : pos + 4]
        length = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        pos += 8
        if pos + length > len(data):
            raise ValueError(f"Truncated MIDI chunk {kind!r}")
        chunks.append(_MidiChunk(kind=kind, payload=data[pos : pos + length]))
        pos += length

    return (fmt, track_count, division), chunks


def _parse_track(payload: bytes, index: int, source_ppq: int) -> _ParsedTrack:
    parsed = _ParsedTrack(index=index)
    active_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    pos = 0
    source_tick = 0
    running_status: Optional[int] = None

    while pos < len(payload):
        delta, pos = _read_vlq(payload, pos)
        source_tick += delta
        parsed.end_tick = max(parsed.end_tick, _scale_tick(source_tick, source_ppq))

        if pos >= len(payload):
            raise ValueError(f"Track {index + 1} ends after delta-time without an event")

        status_byte = payload[pos]
        first_data: Optional[int] = None

        if status_byte < 0x80:
            if running_status is None:
                raise ValueError(f"Track {index + 1} uses running status before a status byte")
            status = running_status
            first_data = status_byte
            pos += 1
        else:
            status = status_byte
            pos += 1
            if status < 0xF0:
                running_status = status
            elif status in (0xF0, 0xF7, 0xFF):
                running_status = None

        if status == 0xFF:
            if pos >= len(payload):
                raise ValueError(f"Track {index + 1} has a truncated meta event")
            meta_type = payload[pos]
            pos += 1
            length, pos = _read_vlq(payload, pos)
            meta_data = payload[pos : pos + length]
            if len(meta_data) != length:
                raise ValueError(f"Track {index + 1} has a truncated meta payload")
            pos += length
            _handle_meta_event(parsed, meta_type, meta_data, source_tick, source_ppq)
            if meta_type == 0x2F:
                break
        elif status in (0xF0, 0xF7):
            length, pos = _read_vlq(payload, pos)
            pos += length
            if pos > len(payload):
                raise ValueError(f"Track {index + 1} has a truncated SysEx event")
        elif status < 0xF0:
            event_type = status & 0xF0
            channel = status & 0x0F
            data_len = 1 if event_type in (0xC0, 0xD0) else 2
            event_data = _read_channel_data(payload, pos, data_len, first_data)
            pos += data_len if first_data is None else data_len - 1
            _handle_channel_event(
                parsed,
                event_type,
                channel,
                event_data,
                source_tick,
                source_ppq,
                active_notes,
            )
        else:
            # System common/realtime events are rare in files; skip their fixed data.
            data_len = _system_event_data_length(status)
            pos += data_len
            if pos > len(payload):
                raise ValueError(f"Track {index + 1} has a truncated system event")

    if any(active_notes.values()):
        dangling_count = sum(len(items) for items in active_notes.values())
        parsed.warnings.append(
            f"Track {index + 1} has {dangling_count} note-on event(s) without note-off; skipped"
        )

    parsed.notes.sort(key=lambda note: (note.start_tick, note.pitch, note.channel))
    return parsed


def _handle_meta_event(
    parsed: _ParsedTrack,
    meta_type: int,
    data: bytes,
    source_tick: int,
    source_ppq: int,
) -> None:
    tick = _scale_tick(source_tick, source_ppq)

    if meta_type == 0x03 and data:
        parsed.name = _decode_text(data)
    elif meta_type == 0x06 and data:
        parsed.markers.append(Marker(tick=Ticks(tick), name=_decode_text(data)))
    elif meta_type == 0x51:
        if len(data) != 3:
            parsed.warnings.append(f"Track {parsed.index + 1} has invalid tempo meta length")
            return
        mpqn = int.from_bytes(data, "big")
        if mpqn <= 0:
            parsed.warnings.append(f"Track {parsed.index + 1} has invalid tempo value")
            return
        parsed.tempos.append((tick, 60_000_000 / mpqn))
    elif meta_type == 0x58:
        if len(data) < 2:
            parsed.warnings.append(f"Track {parsed.index + 1} has invalid time signature")
            return
        denominator = 2 ** data[1]
        try:
            beat_value = BeatValue(denominator)
        except ValueError:
            parsed.warnings.append(
                f"Track {parsed.index + 1} uses unsupported time signature denominator {denominator}"
            )
            return
        parsed.time_signatures.append(
            (tick, TimeSignature(numerator=data[0], denominator=beat_value))
        )
    elif meta_type == 0x59:
        if len(data) < 2:
            parsed.warnings.append(f"Track {parsed.index + 1} has invalid key signature")
            return
        sharps_flats = data[0] if data[0] < 0x80 else data[0] - 0x100
        mode = "minor" if data[1] == 1 else "major"
        parsed.key_signatures.append(
            KeySignature(
                tick=Ticks(tick),
                sharps_flats=max(-7, min(7, sharps_flats)),
                mode=mode,
            )
        )


def _handle_channel_event(
    parsed: _ParsedTrack,
    event_type: int,
    channel: int,
    data: bytes,
    source_tick: int,
    source_ppq: int,
    active_notes: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    if event_type not in (0x80, 0x90):
        return
    if len(data) < 2:
        raise ValueError(f"Track {parsed.index + 1} has truncated note event data")

    pitch = data[0]
    velocity = data[1]
    key = (channel, pitch)

    if event_type == 0x90 and velocity > 0:
        active_notes.setdefault(key, []).append((source_tick, velocity))
        return

    starts = active_notes.get(key)
    if not starts:
        parsed.warnings.append(
            f"Track {parsed.index + 1} has note-off without note-on for pitch {pitch}"
        )
        return

    start_source_tick, start_velocity = starts.pop(0)
    if not starts:
        active_notes.pop(key, None)

    start_tick = _scale_tick(start_source_tick, source_ppq)
    end_tick = _scale_tick(source_tick, source_ppq)
    duration = end_tick - start_tick
    if duration <= 0:
        return

    parsed.notes.append(
        NoteEvent(
            pitch=NotePitch(pitch),
            velocity=NoteVelocity(start_velocity),
            start_tick=Ticks(start_tick),
            duration_ticks=Ticks(duration),
            channel=channel,
        )
    )


def _read_channel_data(
    payload: bytes,
    pos: int,
    data_len: int,
    first_data: Optional[int],
) -> bytes:
    if first_data is None:
        data = payload[pos : pos + data_len]
        if len(data) != data_len:
            raise ValueError("Truncated MIDI channel event")
        return data

    remaining_len = data_len - 1
    data = bytes([first_data]) + payload[pos : pos + remaining_len]
    if len(data) != data_len:
        raise ValueError("Truncated MIDI running-status channel event")
    return data


def _build_tracks(
    parsed_tracks: list[_ParsedTrack],
    track_name_prefix: str,
    split_channels: bool,
    time_signature: TimeSignature,
    warnings: list[str],
) -> list[MidiTrack]:
    tracks: list[MidiTrack] = []
    for parsed in parsed_tracks:
        if not parsed.notes:
            continue
        if split_channels:
            channels = sorted({note.channel for note in parsed.notes})
            for channel in channels:
                channel_notes = [note for note in parsed.notes if note.channel == channel]
                tracks.append(
                    _make_track(
                        parsed,
                        channel_notes,
                        default_name=f"MIDI Track {parsed.index + 1} Ch {channel + 1}",
                        suffix=f" Ch {channel + 1}",
                        track_name_prefix=track_name_prefix,
                        time_signature=time_signature,
                    )
                )
        else:
            channels = sorted({note.channel for note in parsed.notes})
            if len(channels) > 1:
                warnings.append(
                    f"Track {parsed.index + 1} contains multiple MIDI channels; imported as one track"
                )
            tracks.append(
                _make_track(
                    parsed,
                    parsed.notes,
                    default_name=f"MIDI Track {parsed.index + 1}",
                    suffix="",
                    track_name_prefix=track_name_prefix,
                    time_signature=time_signature,
                )
            )
    return tracks


def _make_track(
    parsed: _ParsedTrack,
    notes: list[NoteEvent],
    default_name: str,
    suffix: str,
    track_name_prefix: str,
    time_signature: TimeSignature,
) -> MidiTrack:
    base_name = parsed.name or default_name
    track_name = f"{track_name_prefix}{base_name}{suffix}"
    channel = notes[0].channel if notes and all(note.channel == notes[0].channel for note in notes) else 0
    length_ticks = _clip_length(notes, time_signature)
    clip = MidiClip(
        name=f"{track_name} Clip",
        start_tick=Ticks(0),
        length_ticks=Ticks(length_ticks),
        notes=tuple(notes),
    )
    return MidiTrack(name=track_name, channel=channel).with_clip(clip)


def _clip_length(notes: list[NoteEvent], time_signature: TimeSignature) -> int:
    if not notes:
        return time_signature.ticks_per_measure
    max_end = max(int(note.end_tick) for note in notes)
    measure = max(1, time_signature.ticks_per_measure)
    return ((max_end + measure - 1) // measure) * measure


def _first_time_signature(
    parsed_tracks: list[_ParsedTrack],
    warnings: list[str],
) -> TimeSignature:
    signatures = sorted(
        (
            (tick, signature)
            for parsed in parsed_tracks
            for tick, signature in parsed.time_signatures
        ),
        key=lambda item: item[0],
    )
    if not signatures:
        return TimeSignature()

    first_tick, first_signature = signatures[0]
    later = [
        (tick, signature)
        for tick, signature in signatures[1:]
        if tick != first_tick or signature != first_signature
    ]
    if later:
        warnings.append(
            "MIDI contains time signature changes; only the first signature is stored in Project"
        )
    return first_signature


def _dedupe_tempos(events) -> list[tuple[int, float]]:
    deduped: dict[int, float] = {}
    for tick, bpm in events:
        deduped[int(tick)] = float(bpm)
    return sorted(deduped.items(), key=lambda item: item[0])


def _dedupe_key_signatures(events) -> list[KeySignature]:
    deduped: dict[int, KeySignature] = {}
    for key in events:
        deduped[int(key.tick)] = key
    return [deduped[tick] for tick in sorted(deduped)]


def _dedupe_markers(events) -> list[Marker]:
    deduped: dict[tuple[int, str], Marker] = {}
    for marker in events:
        deduped[(int(marker.tick), marker.name)] = marker
    return [deduped[key] for key in sorted(deduped)]


def _bpm_at_zero(tempo_events: list[tuple[int, float]]) -> float:
    bpm = 120.0
    for tick, event_bpm in tempo_events:
        if tick > 0:
            break
        bpm = event_bpm
    return bpm


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if pos >= len(data):
            raise ValueError("Truncated MIDI variable-length quantity")
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, pos
    raise ValueError("Invalid MIDI variable-length quantity")


def _scale_tick(source_tick: int, source_ppq: int) -> int:
    return round(source_tick * PPQ / source_ppq)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").strip()


def _system_event_data_length(status: int) -> int:
    if status in (0xF1, 0xF3):
        return 1
    if status == 0xF2:
        return 2
    if status in (0xF6, 0xF8, 0xFA, 0xFB, 0xFC, 0xFE):
        return 0
    return 0
