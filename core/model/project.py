"""
Project model — the top-level container for all musical content.

A Project is the root of the document model. It holds tracks, the mixer,
tempo/timing info, key signatures, markers, and metadata.

The project is designed as an immutable value tree — modifying any part
produces a new Project instance (structural sharing where possible).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .time_model import Ticks, TempoMap, TimeSignature, BeatValue
from .track import Track, MidiTrack, AudioTrack, GroupTrack, FxTrack, TrackType
from .mixer import Mixer
from .automation import AutomationEnvelope
from .device import DeviceChain


@dataclass(frozen=True)
class ProjectMetadata:
    """Descriptive metadata for a project."""
    name: str = "Untitled"
    author: str = ""
    genre: str = ""
    bpm: float = 120.0
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "author": self.author,
            "genre": self.genre,
            "bpm": self.bpm,
            "description": self.description,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMetadata":
        return cls(
            name=data.get("name", "Untitled"),
            author=data.get("author", ""),
            genre=data.get("genre", ""),
            bpm=data.get("bpm", 120.0),
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            version=data.get("version", "1.0.0"),
        )


@dataclass(frozen=True)
class KeySignature:
    """
    A key signature at a given tick position.
    
    Uses standard notation: positive = sharps, negative = flats.
    E.g., 0 = C major, 1 = G major, -1 = F major.
    """
    tick: Ticks
    sharps_flats: int = 0   # Positive = sharps, negative = flats
    mode: str = "major"     # "major" or "minor"
    
    @property
    def key_name(self) -> str:
        """Human-readable key name, e.g., 'C major', 'Eb minor'."""
        if self.mode == "minor":
            minor_keys = [
                "A minor", "E minor", "B minor", "F# minor", "C# minor",
                "G# minor", "D# minor", "A# minor",
                "D minor", "G minor", "C minor", "F minor",
                "Bb minor", "Eb minor", "Ab minor",
            ]
            idx = self.sharps_flats + 7
            if 0 <= idx < len(minor_keys):
                return minor_keys[idx]
        else:
            major_keys = [
                "C major", "G major", "D major", "A major", "E major",
                "B major", "F# major", "C# major",
                "F major", "Bb major", "Eb major", "Ab major",
                "Db major", "Gb major", "Cb major",
            ]
            idx = self.sharps_flats
            if 0 <= idx < len(major_keys):
                return major_keys[idx]
        return f"{self.sharps_flats} {'sharps' if self.sharps_flats > 0 else 'flats'} {self.mode}"
    
    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "sharps_flats": self.sharps_flats,
            "mode": self.mode,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "KeySignature":
        return cls(
            tick=Ticks(data["tick"]),
            sharps_flats=data.get("sharps_flats", 0),
            mode=data.get("mode", "major"),
        )


@dataclass(frozen=True)
class Marker:
    """
    A labelled position marker on the timeline.
    
    Markers help navigate large projects — verse, chorus, bridge, etc.
    """
    tick: Ticks
    name: str = "Marker"
    color: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "name": self.name,
            "color": self.color,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Marker":
        return cls(
            tick=Ticks(data["tick"]),
            name=data.get("name", "Marker"),
            color=data.get("color"),
        )


@dataclass(frozen=True)
class Project:
    """
    The root document — an immutable project state tree.
    
    Modifications always produce a new Project. This enables:
    - Simple undo/redo (store previous states)
    - Safe concurrent access
    - Deterministic AI processing
    
    Attributes:
        id: Unique project identifier.
        metadata: Project name, author, genre, etc.
        tempo_map: Tempo changes throughout the project.
        time_signature: Default time signature (can be overridden per section).
        key_signatures: Key signature changes.
        tracks: All tracks in the project (MIDI, audio, group, FX).
        mixer: Mixer console state.
        automation: Automation envelopes for device parameters.
        markers: Timeline markers (verse, chorus, etc.).
        loop_start: Loop region start tick (0 = disabled).
        loop_end: Loop region end tick (0 = disabled).
        project_length_ticks: Total project length (auto-calculated or manual).
    """
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    tempo_map: TempoMap = field(default_factory=TempoMap)
    time_signature: TimeSignature = field(default_factory=TimeSignature)
    key_signatures: tuple[KeySignature, ...] = field(default_factory=tuple)
    tracks: tuple[Track, ...] = field(default_factory=tuple)
    mixer: Mixer = field(default_factory=Mixer)
    automation: tuple[AutomationEnvelope, ...] = field(default_factory=tuple)
    markers: tuple[Marker, ...] = field(default_factory=tuple)
    loop_start: Ticks = Ticks(0)
    loop_end: Ticks = Ticks(0)
    project_length_ticks: Ticks = Ticks(0)
    
    # ── Track management ──────────────────────────────────────────
    
    def with_track(self, track: Track) -> "Project":
        """Add a new track or replace an existing one by ID."""
        others = tuple(t for t in self.tracks if t.id != track.id)
        return Project(
            id=self.id, metadata=self.metadata,
            tempo_map=self.tempo_map, time_signature=self.time_signature,
            key_signatures=self.key_signatures,
            tracks=others + (track,), mixer=self.mixer,
            automation=self.automation, markers=self.markers,
            loop_start=self.loop_start, loop_end=self.loop_end,
            project_length_ticks=self.project_length_ticks,
        )
    
    def without_track(self, track_id: str) -> "Project":
        """Remove a track by ID."""
        return Project(
            id=self.id, metadata=self.metadata,
            tempo_map=self.tempo_map, time_signature=self.time_signature,
            key_signatures=self.key_signatures,
            tracks=tuple(t for t in self.tracks if t.id != track_id),
            mixer=self.mixer,
            automation=tuple(
                e for e in self.automation
                if e.lane.device_id not in self._device_ids_for_track(track_id)
            ),
            markers=self.markers,
            loop_start=self.loop_start, loop_end=self.loop_end,
            project_length_ticks=self.project_length_ticks,
        )
    
    def get_track(self, track_id: str) -> Optional[Track]:
        """Get a track by ID."""
        for t in self.tracks:
            if t.id == track_id:
                return t
        return None
    
    def _device_ids_for_track(self, track_id: str) -> set[str]:
        """Helper: get device IDs belonging to a track (for cleanup)."""
        return set()  # Simplified — actual implementation would query device chains
    
    # ── Serialization ─────────────────────────────────────────────
    
    def to_dict(self) -> dict:
        """Serialize the entire project to a plain dict (JSON-compatible)."""
        return {
            "schema_version": "1.0.0",
            "id": self.id,
            "metadata": self.metadata.to_dict(),
            "tempo_map": self.tempo_map.to_dict(),
            "time_signature": str(self.time_signature),
            "key_signatures": [ks.to_dict() for ks in self.key_signatures],
            "tracks": [t.to_dict() for t in self.tracks],
            "mixer": self.mixer.to_dict(),
            "automation": [a.to_dict() for a in self.automation],
            "markers": [m.to_dict() for m in self.markers],
            "loop_start": self.loop_start,
            "loop_end": self.loop_end,
            "project_length_ticks": self.project_length_ticks,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Deserialize a project from a plain dict."""
        return cls(
            id=data.get("id", uuid4().hex[:16]),
            metadata=ProjectMetadata.from_dict(data.get("metadata", {})),
            tempo_map=TempoMap.from_dict(data.get("tempo_map", [])),
            time_signature=TimeSignature.from_str(data.get("time_signature", "4/4")),
            key_signatures=tuple(
                KeySignature.from_dict(ks) for ks in data.get("key_signatures", [])
            ),
            tracks=tuple(_deserialize_track(t) for t in data.get("tracks", [])),
            mixer=Mixer.from_dict(data.get("mixer", {})),
            automation=tuple(
                AutomationEnvelope.from_dict(a) for a in data.get("automation", [])
            ),
            markers=tuple(
                Marker.from_dict(m) for m in data.get("markers", [])
            ),
            loop_start=Ticks(data.get("loop_start", 0)),
            loop_end=Ticks(data.get("loop_end", 0)),
            project_length_ticks=Ticks(data.get("project_length_ticks", 0)),
        )
    
    # ── Factory methods ───────────────────────────────────────────
    
    @classmethod
    def create_new(cls, name: str = "Untitled", bpm: float = 120.0) -> "Project":
        """Create a new empty project with sensible defaults."""
        return cls(
            metadata=ProjectMetadata(name=name, bpm=bpm),
            time_signature=TimeSignature(numerator=4, denominator=BeatValue.QUARTER),
        )
    
    # ── Convenience queries ───────────────────────────────────────
    
    @property
    def midi_tracks(self) -> tuple[MidiTrack, ...]:
        return tuple(t for t in self.tracks if isinstance(t, MidiTrack))
    
    @property
    def audio_tracks(self) -> tuple[AudioTrack, ...]:
        return tuple(t for t in self.tracks if isinstance(t, AudioTrack))
    
    @property
    def group_tracks(self) -> tuple[GroupTrack, ...]:
        return tuple(t for t in self.tracks if isinstance(t, GroupTrack))
    
    @property
    def fx_tracks(self) -> tuple[FxTrack, ...]:
        return tuple(t for t in self.tracks if isinstance(t, FxTrack))
    
    @property
    def total_duration_ticks(self) -> Ticks:
        """Calculate the total duration from the furthest clip end."""
        max_end = Ticks(0)
        for track in self.tracks:
            if isinstance(track, (MidiTrack, AudioTrack)):
                for clip in track.clips:
                    max_end = Ticks(max(max_end, clip.end_tick))
        return max_end
    
    def key_at(self, tick: Ticks) -> KeySignature:
        """Return the effective key signature at a given tick."""
        effective = KeySignature(tick=Ticks(0), sharps_flats=0, mode="major")
        for ks in sorted(self.key_signatures, key=lambda k: k.tick):
            if ks.tick > tick:
                break
            effective = ks
        return effective


def _deserialize_track(data: dict) -> Track:
    """Deserialize a track dict into the appropriate subclass."""
    track_type = data.get("track_type", "midi")
    deserializers = {
        "midi": MidiTrack.from_dict,
        "audio": AudioTrack.from_dict,
        "group": GroupTrack.from_dict,
        "fx": FxTrack.from_dict,
    }
    if track_type in deserializers:
        return deserializers[track_type](data)
    raise ValueError(f"Unknown track type: {track_type}")
