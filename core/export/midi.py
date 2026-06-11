"""
Built-in Standard MIDI File exporter.

This exporter writes MIDI Format 1 files using only the Python standard
library. It preserves the framework's PPQ resolution, emits a dedicated
tempo/meta track, and writes one MIDI track per Project MIDI track.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from typing import Callable, Optional

from core.model.clip import MidiClip
from core.model.project import KeySignature, Project
from core.model.time_model import PPQ, Ticks
from core.model.track import MidiTrack
from core.plugin.interfaces.base import PluginCategory, PluginManifest, PluginState
from core.plugin.interfaces.exporter import (
    ExportFormat,
    ExportRequest,
    ExportResult,
    ExporterPlugin,
)


_DEFAULT_NOTE_OFF_VELOCITY = 64


@dataclass(frozen=True)
class _MidiEvent:
    tick: int
    priority: int
    data: bytes


class MidiExporterPlugin(ExporterPlugin):
    """Built-in exporter for Standard MIDI Files (.mid)."""

    def __init__(self) -> None:
        self._state = PluginState.LOADED

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="builtin.export.midi",
            name="Standard MIDI Exporter",
            version="0.1.0",
            category=PluginCategory.EXPORTER,
            author="amusiment",
            description="Exports MIDI clips to Standard MIDI File format 1.",
            capabilities=["export.midi", "export.standard-midi-file"],
        )

    def initialize(self) -> None:
        self._state = PluginState.INITIALIZED

    def shutdown(self) -> None:
        self._state = PluginState.UNLOADED

    def get_state(self) -> PluginState:
        return self._state

    def get_supported_formats(self) -> list[ExportFormat]:
        return [ExportFormat.MIDI]

    def validate_request(self, request: ExportRequest) -> list[str]:
        errors = super().validate_request(request)

        if request.start_tick < 0:
            errors.append("start_tick must be non-negative")

        if request.end_tick < 0:
            errors.append("end_tick must be non-negative")

        if request.end_tick and request.end_tick <= request.start_tick:
            errors.append("end_tick must be greater than start_tick")

        return errors

    def export(
        self,
        request: ExportRequest,
        project_state: dict,
        progress_callback: "Optional[Callable[[float, str], None]]" = None,
    ) -> ExportResult:
        def progress(value: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(value, message)

        progress(0.0, "Validating MIDI export request")
        errors = self.validate_request(request)
        if errors:
            return ExportResult(
                success=False,
                output_path=request.output_path,
                format=request.format,
                error_message="; ".join(errors),
            )

        try:
            progress(0.2, "Preparing project data")
            project = Project.from_dict(project_state)
            warnings: list[str] = []

            progress(0.6, "Building MIDI tracks")
            midi_bytes = build_standard_midi_bytes(project, request, warnings)

            parent = os.path.dirname(os.path.abspath(request.output_path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)

            with open(request.output_path, "wb") as f:
                f.write(midi_bytes)

            file_size = os.path.getsize(request.output_path)
            start_tick, end_tick = _export_range(project, request)
            duration_seconds = max(
                0.0,
                _seconds_at(project, end_tick) - _seconds_at(project, start_tick),
            )

            progress(1.0, "MIDI export complete")
            return ExportResult(
                success=True,
                output_path=request.output_path,
                format=ExportFormat.MIDI,
                duration_seconds=duration_seconds,
                file_size_bytes=file_size,
                warnings=warnings,
            )
        except Exception as exc:
            return ExportResult(
                success=False,
                output_path=request.output_path,
                format=ExportFormat.MIDI,
                error_message=str(exc),
            )


def build_standard_midi_bytes(
    project: Project,
    request: ExportRequest | None = None,
    warnings: list[str] | None = None,
) -> bytes:
    """
    Build a Standard MIDI File byte string from a project.

    Args:
        project: Project to export.
        request: Optional export request. When omitted, exports all MIDI tracks.
        warnings: Optional list that receives non-fatal export warnings.

    Returns:
        Complete .mid file contents.
    """
    if request is None:
        request = ExportRequest(format=ExportFormat.MIDI)
    if warnings is None:
        warnings = []

    if request.format is not ExportFormat.MIDI:
        raise ValueError("MidiExporterPlugin only supports ExportFormat.MIDI")

    start_tick, end_tick = _export_range(project, request)
    selected_ids = set(request.track_ids)

    midi_tracks = [
        track
        for track in project.midi_tracks
        if not selected_ids or track.id in selected_ids
    ]

    ignored_ids = selected_ids - {track.id for track in project.tracks}
    if ignored_ids:
        warnings.append(f"Ignored unknown track ids: {', '.join(sorted(ignored_ids))}")

    chunks = [_build_meta_track(project, start_tick, end_tick)]
    for track in midi_tracks:
        chunks.append(_build_midi_track(track, start_tick, end_tick, warnings))

    header = _chunk(
        b"MThd",
        struct.pack(">HHH", 1, len(chunks), PPQ),
    )
    return header + b"".join(chunks)


def _build_meta_track(project: Project, start_tick: int, end_tick: int) -> bytes:
    events: list[_MidiEvent] = []

    events.append(_MidiEvent(0, 0, _meta_event_bytes(0x03, b"Amusiment Tempo Map")))
    events.append(
        _MidiEvent(
            0,
            1,
            _meta_event_bytes(
                0x58,
                bytes(
                    [
                        project.time_signature.numerator & 0xFF,
                        _time_signature_denominator_power(project.time_signature.denominator.value),
                        24,
                        8,
                    ]
                ),
            ),
        )
    )

    for tick, bpm in _tempo_events(project, start_tick, end_tick):
        mpqn = round(60_000_000 / bpm)
        events.append(_MidiEvent(tick, 2, _meta_event_bytes(0x51, mpqn.to_bytes(3, "big"))))

    for tick, key in _key_events(project, start_tick, end_tick):
        mode = 1 if key.mode == "minor" else 0
        sharps_flats = _signed_byte(max(-7, min(7, key.sharps_flats)))
        events.append(_MidiEvent(tick, 3, _meta_event_bytes(0x59, bytes([sharps_flats, mode]))))

    return _events_to_track(events)


def _build_midi_track(
    track: MidiTrack,
    start_tick: int,
    end_tick: int,
    warnings: list[str],
) -> bytes:
    events: list[_MidiEvent] = [
        _MidiEvent(0, 0, _meta_event_bytes(0x03, _safe_text(track.name)))
    ]

    if track.muted:
        warnings.append(f"Track '{track.name}' is muted; exported as an empty MIDI track")
        return _events_to_track(events)

    for clip in track.clips:
        if clip.muted:
            continue
        if clip.loop_enabled:
            warnings.append(f"Clip '{clip.name}' has looping enabled; MIDI export writes one clip pass")
        events.extend(_note_events_for_clip(track, clip, start_tick, end_tick))

    return _events_to_track(events)


def _note_events_for_clip(
    track: MidiTrack,
    clip: MidiClip,
    start_tick: int,
    end_tick: int,
) -> list[_MidiEvent]:
    events: list[_MidiEvent] = []

    for note in clip.notes:
        if note.duration_ticks <= 0:
            continue

        local_start = max(0, int(note.start_tick))
        local_end = min(int(note.end_tick), int(clip.length_ticks))
        if local_end <= local_start:
            continue

        absolute_start = int(clip.start_tick) + local_start
        absolute_end = int(clip.start_tick) + local_end
        if absolute_end <= start_tick or absolute_start >= end_tick:
            continue

        visible_start = max(absolute_start, start_tick)
        visible_end = min(absolute_end, end_tick)
        if visible_end <= visible_start:
            continue

        channel = _clamp_channel(note.channel if note.channel is not None else track.channel)
        pitch = max(0, min(127, int(note.pitch)))
        velocity = max(0, min(127, int(note.velocity)))
        note_on_tick = visible_start - start_tick
        note_off_tick = visible_end - start_tick

        events.append(
            _MidiEvent(
                note_on_tick,
                2,
                bytes([0x90 | channel, pitch, velocity]),
            )
        )
        events.append(
            _MidiEvent(
                note_off_tick,
                1,
                bytes([0x80 | channel, pitch, _DEFAULT_NOTE_OFF_VELOCITY]),
            )
        )

    return events


def _events_to_track(events: list[_MidiEvent]) -> bytes:
    sorted_events = sorted(events, key=lambda event: (event.tick, event.priority, event.data))
    payload = bytearray()
    last_tick = 0

    for event in sorted_events:
        delta = event.tick - last_tick
        if delta < 0:
            raise ValueError("MIDI events must be sorted by non-negative tick")
        payload.extend(_vlq(delta))
        payload.extend(event.data)
        last_tick = event.tick

    payload.extend(_vlq(0))
    payload.extend(b"\xFF\x2F\x00")
    return _chunk(b"MTrk", bytes(payload))


def _meta_event_bytes(meta_type: int, data: bytes) -> bytes:
    return bytes([0xFF, meta_type & 0x7F]) + _vlq(len(data)) + data


def _chunk(kind: bytes, payload: bytes) -> bytes:
    if len(kind) != 4:
        raise ValueError("MIDI chunk kind must be exactly 4 bytes")
    return kind + struct.pack(">I", len(payload)) + payload


def _vlq(value: int) -> bytes:
    if value < 0:
        raise ValueError("Variable-length quantity cannot be negative")

    out = [value & 0x7F]
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def _export_range(project: Project, request: ExportRequest) -> tuple[int, int]:
    start_tick = int(request.start_tick)
    if request.end_tick:
        end_tick = int(request.end_tick)
    else:
        end_tick = max(int(project.project_length_ticks), int(project.total_duration_ticks))

    if end_tick <= start_tick:
        end_tick = start_tick

    return start_tick, end_tick


def _tempo_events(project: Project, start_tick: int, end_tick: int) -> list[tuple[int, float]]:
    events: list[tuple[int, float]] = [(0, _bpm_at(project, start_tick))]
    for change in sorted(project.tempo_map.changes, key=lambda item: item.tick):
        change_tick = int(change.tick)
        if change_tick <= start_tick:
            continue
        if change_tick >= end_tick:
            break
        events.append((change_tick - start_tick, change.bpm))
    return events


def _key_events(project: Project, start_tick: int, end_tick: int) -> list[tuple[int, KeySignature]]:
    events: list[tuple[int, KeySignature]] = [(0, project.key_at(Ticks(start_tick)))]
    for key in sorted(project.key_signatures, key=lambda item: item.tick):
        key_tick = int(key.tick)
        if key_tick <= start_tick:
            continue
        if key_tick >= end_tick:
            break
        events.append((key_tick - start_tick, key))
    return events


def _bpm_at(project: Project, tick: int) -> float:
    bpm = project.metadata.bpm
    for change in sorted(project.tempo_map.changes, key=lambda item: item.tick):
        if int(change.tick) > tick:
            break
        bpm = change.bpm
    return bpm


def _seconds_at(project: Project, tick: int) -> float:
    if tick <= 0:
        return 0.0

    elapsed = 0.0
    previous_tick = 0
    previous_bpm = project.metadata.bpm

    for change in sorted(project.tempo_map.changes, key=lambda item: item.tick):
        change_tick = int(change.tick)
        if change_tick >= tick:
            break
        elapsed += ((change_tick - previous_tick) / PPQ) * (60.0 / previous_bpm)
        previous_tick = change_tick
        previous_bpm = change.bpm

    elapsed += ((tick - previous_tick) / PPQ) * (60.0 / previous_bpm)
    return elapsed


def _time_signature_denominator_power(denominator: int) -> int:
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError(f"Time signature denominator must be a power of two: {denominator}")
    return int(math.log2(denominator))


def _signed_byte(value: int) -> int:
    return value & 0xFF


def _safe_text(value: str) -> bytes:
    return value.encode("utf-8", errors="replace")


def _clamp_channel(value: int) -> int:
    return max(0, min(15, int(value)))
