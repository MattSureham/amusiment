"""
Mixer model — the audio routing and mixing console.

The mixer manages all audio channels, their levels, routings,
and send/return configurations.
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class SendConfig:
    """
    A send routing from one channel to an FX return track.
    
    Attributes:
        target_track_id: ID of the FX return track to send to.
        amount_db: Send level in dB (-inf = no send, 0 = unity).
        pre_fader: If True, send is taken before the channel fader.
    """
    target_track_id: str
    amount_db: float = -70.0  # -70 dB ≈ -inf (effectively off)
    pre_fader: bool = False
    
    def with_amount(self, db: float) -> "SendConfig":
        return SendConfig(
            target_track_id=self.target_track_id,
            amount_db=db,
            pre_fader=self.pre_fader,
        )
    
    def to_dict(self) -> dict:
        return {
            "target_track_id": self.target_track_id,
            "amount_db": self.amount_db,
            "pre_fader": self.pre_fader,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SendConfig":
        return cls(
            target_track_id=data["target_track_id"],
            amount_db=data.get("amount_db", -70.0),
            pre_fader=data.get("pre_fader", False),
        )


@dataclass(frozen=True)
class MixerChannel:
    """
    A single mixer channel strip.
    
    Every track (MIDI, audio, group, FX, master) has a corresponding
    mixer channel that controls its level, pan, mute, solo, and sends.
    
    Attributes:
        track_id: Which track this channel corresponds to.
        volume_db: Channel fader level (-inf to +12 dB).
        pan: Stereo balance (-1.0 L to 1.0 R).
        muted: Channel mute state.
        soloed: Channel solo state.
        sends: Aux sends to FX return tracks.
        output_target: ID of the bus/group this channel routes to (None = master).
    """
    track_id: str = ""
    volume_db: float = 0.0
    pan: float = 0.0
    muted: bool = False
    soloed: bool = False
    sends: tuple[SendConfig, ...] = field(default_factory=tuple)
    output_target: Optional[str] = None  # None means route to master
    
    @property
    def gain_multiplier(self) -> float:
        """Convert dB to linear gain multiplier."""
        if self.volume_db <= -70.0:
            return 0.0
        import math
        return math.pow(10.0, self.volume_db / 20.0)
    
    def with_volume(self, db: float) -> "MixerChannel":
        return MixerChannel(
            track_id=self.track_id, volume_db=db, pan=self.pan,
            muted=self.muted, soloed=self.soloed,
            sends=self.sends, output_target=self.output_target,
        )
    
    def with_send(self, send: SendConfig) -> "MixerChannel":
        """Add or update a send to an FX track."""
        others = tuple(s for s in self.sends if s.target_track_id != send.target_track_id)
        return MixerChannel(
            track_id=self.track_id, volume_db=self.volume_db, pan=self.pan,
            muted=self.muted, soloed=self.soloed,
            sends=others + (send,),
            output_target=self.output_target,
        )
    
    def without_send(self, target_track_id: str) -> "MixerChannel":
        """Remove a send to the given FX track."""
        return MixerChannel(
            track_id=self.track_id, volume_db=self.volume_db, pan=self.pan,
            muted=self.muted, soloed=self.soloed,
            sends=tuple(s for s in self.sends if s.target_track_id != target_track_id),
            output_target=self.output_target,
        )
    
    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "volume_db": self.volume_db,
            "pan": self.pan,
            "muted": self.muted,
            "soloed": self.soloed,
            "sends": [s.to_dict() for s in self.sends],
            "output_target": self.output_target,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MixerChannel":
        return cls(
            track_id=data.get("track_id", ""),
            volume_db=data.get("volume_db", 0.0),
            pan=data.get("pan", 0.0),
            muted=data.get("muted", False),
            soloed=data.get("soloed", False),
            sends=tuple(SendConfig.from_dict(s) for s in data.get("sends", [])),
            output_target=data.get("output_target"),
        )


@dataclass(frozen=True)
class Mixer:
    """
    The full mixing console — all channels plus master bus.
    
    Attributes:
        channels: All mixer channels keyed by track_id.
        master_volume_db: Master fader level.
        master_muted: Master mute.
    """
    channels: tuple[MixerChannel, ...] = field(default_factory=tuple)
    master_volume_db: float = 0.0
    master_muted: bool = False
    
    def get_channel(self, track_id: str) -> Optional[MixerChannel]:
        """Get a channel by track ID."""
        for ch in self.channels:
            if ch.track_id == track_id:
                return ch
        return None
    
    def with_channel(self, channel: MixerChannel) -> "Mixer":
        """Add or update a mixer channel."""
        others = tuple(ch for ch in self.channels if ch.track_id != channel.track_id)
        return Mixer(
            channels=others + (channel,),
            master_volume_db=self.master_volume_db,
            master_muted=self.master_muted,
        )
    
    def without_channel(self, track_id: str) -> "Mixer":
        """Remove a channel by track ID."""
        return Mixer(
            channels=tuple(ch for ch in self.channels if ch.track_id != track_id),
            master_volume_db=self.master_volume_db,
            master_muted=self.master_muted,
        )
    
    def with_volume(self, track_id: str, db: float) -> "Mixer":
        """Set the volume of a specific channel."""
        return Mixer(
            channels=tuple(
                ch.with_volume(db) if ch.track_id == track_id else ch
                for ch in self.channels
            ),
            master_volume_db=self.master_volume_db,
            master_muted=self.master_muted,
        )
    
    @property
    def audible_channels(self) -> tuple[MixerChannel, ...]:
        """Return channels that should produce sound (accounting for solo)."""
        any_soloed = any(ch.soloed for ch in self.channels)
        if any_soloed:
            return tuple(ch for ch in self.channels if ch.soloed and not ch.muted)
        return tuple(ch for ch in self.channels if not ch.muted)
    
    def to_dict(self) -> dict:
        return {
            "master_volume_db": self.master_volume_db,
            "master_muted": self.master_muted,
            "channels": [ch.to_dict() for ch in self.channels],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Mixer":
        return cls(
            master_volume_db=data.get("master_volume_db", 0.0),
            master_muted=data.get("master_muted", False),
            channels=tuple(MixerChannel.from_dict(c) for c in data.get("channels", [])),
        )
