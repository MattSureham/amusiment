"""
Track models — the horizontal lanes in the arrangement view.

Tracks organize clips and route audio/MIDI through device chains.
Each track type has specific capabilities and routing behavior.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from uuid import uuid4

from .clip import Clip, MidiClip, AudioClip, AutomationClip


class TrackType(Enum):
    """Types of tracks supported by the framework."""
    MIDI = auto()       # Holds MIDI clips → routes to an instrument
    AUDIO = auto()      # Holds audio clips → routes through effects
    GROUP = auto()      # Groups other tracks for collective processing
    FX = auto()         # Return/FX track (receives sends from other tracks)
    MASTER = auto()     # Master output bus (exactly one per project)


@dataclass(frozen=True)
class Track:
    """
    Base track — an immutable lane in the arrangement.
    
    All tracks have an ID, name, type, color, volume/pan, mute/solo state,
    and a list of clips.
    
    Subclasses specialize for different track types.
    """
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = "Track"
    track_type: TrackType = TrackType.MIDI
    color: Optional[str] = None
    volume_db: float = 0.0       # -inf to +12 dB
    pan: float = 0.0             # -1.0 (L) to 1.0 (R)
    muted: bool = False
    soloed: bool = False
    armed: bool = False          # Record-arm state
    
    @property
    def is_audible(self) -> bool:
        """Whether this track should produce sound (accounts for solo logic)."""
        return not self.muted


@dataclass(frozen=True)
class MidiTrack(Track):
    """
    A track that holds MIDI clips and routes them through an instrument device.
    
    Attributes:
        clips: MidiClip instances on this track.
        instrument_id: Reference to the instrument device in the device chain.
        channel: Default MIDI output channel (0-15).
        input_device: Optional MIDI input device name (for recording).
    """
    track_type: TrackType = TrackType.MIDI
    clips: tuple[MidiClip, ...] = field(default_factory=tuple)
    instrument_id: Optional[str] = None
    channel: int = 0
    input_device: Optional[str] = None
    
    def with_clip(self, clip: MidiClip) -> "MidiTrack":
        """Return a new track with the given clip added."""
        return MidiTrack(
            id=self.id, name=self.name, color=self.color,
            volume_db=self.volume_db, pan=self.pan,
            muted=self.muted, soloed=self.soloed, armed=self.armed,
            instrument_id=self.instrument_id, channel=self.channel,
            input_device=self.input_device,
            clips=self.clips + (clip,),
        )
    
    def with_clip_removed(self, clip_id: str) -> "MidiTrack":
        """Return a new track with the given clip removed."""
        return MidiTrack(
            id=self.id, name=self.name, color=self.color,
            volume_db=self.volume_db, pan=self.pan,
            muted=self.muted, soloed=self.soloed, armed=self.armed,
            instrument_id=self.instrument_id, channel=self.channel,
            input_device=self.input_device,
            clips=tuple(c for c in self.clips if c.id != clip_id),
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "track_type": "midi",
            "color": self.color,
            "volume_db": self.volume_db,
            "pan": self.pan,
            "muted": self.muted,
            "soloed": self.soloed,
            "armed": self.armed,
            "instrument_id": self.instrument_id,
            "channel": self.channel,
            "input_device": self.input_device,
            "clips": [c.to_dict() for c in self.clips],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MidiTrack":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "MIDI Track"),
            color=data.get("color"),
            volume_db=data.get("volume_db", 0.0),
            pan=data.get("pan", 0.0),
            muted=data.get("muted", False),
            soloed=data.get("soloed", False),
            armed=data.get("armed", False),
            instrument_id=data.get("instrument_id"),
            channel=data.get("channel", 0),
            input_device=data.get("input_device"),
            clips=tuple(MidiClip.from_dict(c) for c in data.get("clips", [])),
        )


@dataclass(frozen=True)
class AudioTrack(Track):
    """
    A track that holds audio clips.
    
    Attributes:
        clips: AudioClip instances on this track.
        input_channel: Audio input channel(s) for recording (1-based, or "1/2" for stereo).
    """
    track_type: TrackType = TrackType.AUDIO
    clips: tuple[AudioClip, ...] = field(default_factory=tuple)
    input_channel: str = "1/2"
    
    def with_clip(self, clip: AudioClip) -> "AudioTrack":
        return AudioTrack(
            id=self.id, name=self.name, color=self.color,
            volume_db=self.volume_db, pan=self.pan,
            muted=self.muted, soloed=self.soloed, armed=self.armed,
            input_channel=self.input_channel,
            clips=self.clips + (clip,),
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "track_type": "audio",
            "color": self.color,
            "volume_db": self.volume_db,
            "pan": self.pan,
            "muted": self.muted,
            "soloed": self.soloed,
            "armed": self.armed,
            "input_channel": self.input_channel,
            "clips": [c.to_dict() for c in self.clips],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AudioTrack":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "Audio Track"),
            color=data.get("color"),
            volume_db=data.get("volume_db", 0.0),
            pan=data.get("pan", 0.0),
            muted=data.get("muted", False),
            soloed=data.get("soloed", False),
            armed=data.get("armed", False),
            input_channel=data.get("input_channel", "1/2"),
            clips=tuple(AudioClip.from_dict(c) for c in data.get("clips", [])),
        )


@dataclass(frozen=True)
class GroupTrack(Track):
    """
    A group/bus track that other tracks can route their output through.
    
    Attributes:
        child_track_ids: IDs of tracks that route into this group.
    """
    track_type: TrackType = TrackType.GROUP
    child_track_ids: tuple[str, ...] = field(default_factory=tuple)
    
    def with_child(self, track_id: str) -> "GroupTrack":
        if track_id in self.child_track_ids:
            return self
        return GroupTrack(
            id=self.id, name=self.name, color=self.color,
            volume_db=self.volume_db, pan=self.pan,
            muted=self.muted, soloed=self.soloed, armed=self.armed,
            child_track_ids=self.child_track_ids + (track_id,),
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "track_type": "group",
            "color": self.color,
            "volume_db": self.volume_db,
            "pan": self.pan,
            "muted": self.muted,
            "soloed": self.soloed,
            "armed": self.armed,
            "child_track_ids": list(self.child_track_ids),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GroupTrack":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "Group"),
            color=data.get("color"),
            volume_db=data.get("volume_db", 0.0),
            pan=data.get("pan", 0.0),
            muted=data.get("muted", False),
            soloed=data.get("soloed", False),
            armed=data.get("armed", False),
            child_track_ids=tuple(data.get("child_track_ids", [])),
        )


@dataclass(frozen=True)
class FxTrack(Track):
    """
    A return/FX track that receives audio via sends from other tracks.
    
    Typically used for reverb, delay, and other shared time-based effects.
    """
    track_type: TrackType = TrackType.FX
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "track_type": "fx",
            "color": self.color,
            "volume_db": self.volume_db,
            "pan": self.pan,
            "muted": self.muted,
            "soloed": self.soloed,
            "armed": self.armed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FxTrack":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "FX Track"),
            color=data.get("color"),
            volume_db=data.get("volume_db", 0.0),
            pan=data.get("pan", 0.0),
            muted=data.get("muted", False),
            soloed=data.get("soloed", False),
            armed=data.get("armed", False),
        )
