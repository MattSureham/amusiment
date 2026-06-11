"""
Action definitions for the amusiment state store.

Actions are the ONLY way to modify the project state. Every mutation
— whether from the UI, AI generation, scripting, or undo/redo — goes
through an Action.

Each Action implements a `reduce` method that takes the current state
and returns the new state (or the same state if no change needed).

This design ensures:
- All state changes are traceable
- Undo/redo is trivial (store inverse actions)
- AI and UI edits use the same mechanism
- Replay and collaboration are possible
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from uuid import uuid4

from ..model.project import Project, ProjectMetadata
from ..model.track import Track, MidiTrack, AudioTrack, GroupTrack, FxTrack, TrackType
from ..model.clip import Clip, MidiClip, AudioClip, AutomationClip
from ..model.note import NoteEvent, NotePitch, NoteVelocity
from ..model.time_model import Ticks, TempoChange, TempoMap, TimeSignature
from ..model.mixer import MixerChannel, Mixer
from ..model.device import Device, InstrumentDevice, EffectDevice, DeviceChain, DeviceType


class ActionType(Enum):
    """All possible action types for serialization and routing."""
    ADD_TRACK = auto()
    REMOVE_TRACK = auto()
    ADD_CLIP = auto()
    REMOVE_CLIP = auto()
    ADD_NOTE = auto()
    REMOVE_NOTE = auto()
    MOVE_CLIP = auto()
    RESIZE_CLIP = auto()
    SET_MIXER_VOLUME = auto()
    SET_MIXER_PAN = auto()
    SET_TEMPO = auto()
    SET_KEY_SIGNATURE = auto()
    ADD_DEVICE = auto()
    REMOVE_DEVICE = auto()
    SET_DEVICE_PARAMETER = auto()
    SET_PROJECT_METADATA = auto()
    ADD_AUTOMATION_POINT = auto()
    REMOVE_AUTOMATION_POINT = auto()
    GENERATE_AI_CONTENT = auto()  # AI-generated content insertion
    BATCH = auto()                # Multiple actions applied as one (for undo grouping)


@dataclass
class Action(ABC):
    """
    Abstract base for all state-modifying actions.
    
    Subclasses must implement `reduce` and `inverse`.
    """
    action_type: ActionType = field(default=ActionType.BATCH)
    action_id: str = field(default_factory=lambda: uuid4().hex[:12])
    description: str = ""
    
    @abstractmethod
    def reduce(self, state: Project) -> Project:
        """Apply this action to the state, returning the new state."""
        ...
    
    @abstractmethod
    def inverse(self) -> Optional["Action"]:
        """
        Return the action that would undo this one.
        
        Returns None if this action is not undoable.
        """
        ...
    
    def to_dict(self) -> dict:
        """Serialize action to a dict (for history persistence)."""
        return {
            "action_type": self.action_type.name,
            "action_id": self.action_id,
            "description": self.description,
        }
    
    @staticmethod
    def from_dict(data: dict) -> Optional["Action"]:
        """Deserialize an action from a dict."""
        # Dispatch to specific action deserializers
        action_type = data.get("action_type", "")
        # Simplified — real implementation would have a registry
        return None


# ── Track actions ──────────────────────────────────────────────────

@dataclass
class AddTrackAction(Action):
    """Add a track to the project."""
    
    track: Track = field(default_factory=lambda: MidiTrack(name="New Track"))
    
    def __post_init__(self):
        self.action_type = ActionType.ADD_TRACK
        if not self.description:
            self.description = f"Add track '{self.track.name}'"
    
    def reduce(self, state: Project) -> Project:
        return state.with_track(self.track)
    
    def inverse(self) -> Optional[Action]:
        return RemoveTrackAction(track_id=self.track.id)


@dataclass
class RemoveTrackAction(Action):
    """Remove a track from the project."""
    
    track_id: str = ""
    
    def __post_init__(self):
        self.action_type = ActionType.REMOVE_TRACK
        if not self.description:
            self.description = f"Remove track {self.track_id}"
    
    def reduce(self, state: Project) -> Project:
        return state.without_track(self.track_id)
    
    def inverse(self) -> Optional[Action]:
        # To properly undo, we need the original track data
        # This is stored separately by the HistoryManager
        return None  # Inversion requires original data — handled by history snapshot


# ── Clip actions ───────────────────────────────────────────────────

@dataclass
class AddClipAction(Action):
    """Add a clip to a specific track."""
    
    track_id: str = ""
    clip: Clip = field(default_factory=MidiClip)
    
    def __post_init__(self):
        self.action_type = ActionType.ADD_CLIP
        if not self.description:
            self.description = f"Add clip '{self.clip.name}' to track {self.track_id}"
    
    def reduce(self, state: Project) -> Project:
        track = state.get_track(self.track_id)
        if track is None:
            return state
        
        if isinstance(track, MidiTrack) and isinstance(self.clip, MidiClip):
            updated = track.with_clip(self.clip)
        elif isinstance(track, AudioTrack) and isinstance(self.clip, AudioClip):
            updated = track.with_clip(self.clip)
        else:
            return state  # Type mismatch — no change
        
        return state.with_track(updated)
    
    def inverse(self) -> Optional[Action]:
        return RemoveClipAction(track_id=self.track_id, clip_id=self.clip.id)


@dataclass
class RemoveClipAction(Action):
    """Remove a clip from a track."""
    
    track_id: str = ""
    clip_id: str = ""
    
    def __post_init__(self):
        self.action_type = ActionType.REMOVE_CLIP
        if not self.description:
            self.description = f"Remove clip {self.clip_id} from track {self.track_id}"
    
    def reduce(self, state: Project) -> Project:
        track = state.get_track(self.track_id)
        if track is None:
            return state
        
        if isinstance(track, MidiTrack):
            updated = track.with_clip_removed(self.clip_id)
        elif isinstance(track, AudioTrack):
            # AudioTrack has with_clip_removed method
            updated = track.__class__(
                id=track.id, name=track.name, color=track.color,
                volume_db=track.volume_db, pan=track.pan,
                muted=track.muted, soloed=track.soloed, armed=track.armed,
                clips=tuple(c for c in track.clips if c.id != self.clip_id),
            )
        else:
            return state
        
        return state.with_track(updated)
    
    def inverse(self) -> Optional[Action]:
        return None  # Requires original clip data from history


# ── Note actions ───────────────────────────────────────────────────

@dataclass
class AddNoteAction(Action):
    """Add a note event to a MIDI clip."""
    
    track_id: str = ""
    clip_id: str = ""
    note: NoteEvent = field(default_factory=lambda: NoteEvent(
        pitch=NotePitch(60), velocity=NoteVelocity(100),
        start_tick=Ticks(0), duration_ticks=Ticks(480),
    ))
    
    def __post_init__(self):
        self.action_type = ActionType.ADD_NOTE
        if not self.description:
            self.description = f"Add note {self.note.pitch_name} to clip {self.clip_id}"
    
    def reduce(self, state: Project) -> Project:
        track = state.get_track(self.track_id)
        if track is None or not isinstance(track, MidiTrack):
            return state
        
        for clip in track.clips:
            if clip.id == self.clip_id:
                updated_clip = clip.with_added_note(self.note)
                updated_track = track.with_clip_removed(self.clip_id)
                # Re-add the updated clip
                result = state
                result = result.with_track(
                    updated_track.__class__(
                        id=updated_track.id, name=updated_track.name,
                        color=updated_track.color,
                        volume_db=updated_track.volume_db, pan=updated_track.pan,
                        muted=updated_track.muted, soloed=updated_track.soloed,
                        armed=updated_track.armed,
                        clips=tuple(c for c in updated_track.clips),
                    )
                )
                # Now add the updated clip
                final_track = state.get_track(self.track_id)
                if isinstance(final_track, MidiTrack):
                    return state.with_track(final_track.with_clip(updated_clip))
                return state
        
        return state
    
    def inverse(self) -> Optional[Action]:
        return RemoveNoteAction(
            track_id=self.track_id,
            clip_id=self.clip_id,
            note=self.note,
        )


@dataclass
class RemoveNoteAction(Action):
    """Remove a note event from a MIDI clip."""
    
    track_id: str = ""
    clip_id: str = ""
    note: NoteEvent = field(default_factory=lambda: NoteEvent(
        pitch=NotePitch(60), velocity=NoteVelocity(100),
        start_tick=Ticks(0), duration_ticks=Ticks(480),
    ))
    
    def __post_init__(self):
        self.action_type = ActionType.REMOVE_NOTE
        if not self.description:
            self.description = f"Remove note {self.note.pitch_name} from clip {self.clip_id}"
    
    def reduce(self, state: Project) -> Project:
        track = state.get_track(self.track_id)
        if track is None or not isinstance(track, MidiTrack):
            return state
        
        for clip in track.clips:
            if clip.id == self.clip_id:
                updated_clip = clip.with_removed_note(self.note)
                return state.with_track(track.with_clip(updated_clip))
        
        return state
    
    def inverse(self) -> Optional[Action]:
        return AddNoteAction(
            track_id=self.track_id,
            clip_id=self.clip_id,
            note=self.note,
        )


# ── Mixer actions ─────────────────────────────────────────────────

@dataclass
class SetMixerVolumeAction(Action):
    """Set the volume of a mixer channel."""
    
    track_id: str = ""
    volume_db: float = 0.0
    _previous_volume: Optional[float] = None  # Set by reducer for undo
    
    def __post_init__(self):
        self.action_type = ActionType.SET_MIXER_VOLUME
        if not self.description:
            self.description = f"Set volume of {self.track_id} to {self.volume_db:.1f} dB"
    
    def reduce(self, state: Project) -> Project:
        channel = state.mixer.get_channel(self.track_id)
        if channel:
            object.__setattr__(self, '_previous_volume', channel.volume_db)
        else:
            # Create a new channel for this track
            channel = MixerChannel(track_id=self.track_id)
            object.__setattr__(self, '_previous_volume', 0.0)
        
        updated_mixer = state.mixer.with_volume(self.track_id, self.volume_db)
        
        return Project(
            id=state.id, metadata=state.metadata,
            tempo_map=state.tempo_map, time_signature=state.time_signature,
            key_signatures=state.key_signatures,
            tracks=state.tracks, mixer=updated_mixer,
            automation=state.automation, markers=state.markers,
            loop_start=state.loop_start, loop_end=state.loop_end,
            project_length_ticks=state.project_length_ticks,
        )
    
    def inverse(self) -> Optional[Action]:
        if self._previous_volume is not None:
            return SetMixerVolumeAction(
                track_id=self.track_id,
                volume_db=self._previous_volume,
            )
        return None


# ── Tempo actions ─────────────────────────────────────────────────

@dataclass
class SetTempoAction(Action):
    """Add or modify a tempo change."""
    
    tick: Ticks = Ticks(0)
    bpm: float = 120.0
    
    def __post_init__(self):
        self.action_type = ActionType.SET_TEMPO
        if not self.description:
            self.description = f"Set tempo to {self.bpm} BPM at tick {self.tick}"
    
    def reduce(self, state: Project) -> Project:
        change = TempoChange(tick=self.tick, bpm=self.bpm)
        new_tempo_map = state.tempo_map.with_change(change)
        
        return Project(
            id=state.id, metadata=state.metadata,
            tempo_map=new_tempo_map, time_signature=state.time_signature,
            key_signatures=state.key_signatures,
            tracks=state.tracks, mixer=state.mixer,
            automation=state.automation, markers=state.markers,
            loop_start=state.loop_start, loop_end=state.loop_end,
            project_length_ticks=state.project_length_ticks,
        )
    
    def inverse(self) -> Optional[Action]:
        # Undo by removing this tempo change
        # A more complete implementation would restore the previous value
        return None


# ── Key signature actions ─────────────────────────────────────────

@dataclass
class SetKeySignatureAction(Action):
    """Set the key signature at a given position."""
    
    tick: Ticks = Ticks(0)
    sharps_flats: int = 0
    mode: str = "major"
    
    def __post_init__(self):
        self.action_type = ActionType.SET_KEY_SIGNATURE
        keys = ["C", "G", "D", "A", "E", "B", "F#", "C#",
                "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]
        key_name = keys[self.sharps_flats % len(keys)]
        if not self.description:
            self.description = f"Set key to {key_name} {self.mode}"
    
    def reduce(self, state: Project) -> Project:
        from ..model.project import KeySignature
        new_ks = KeySignature(tick=self.tick, sharps_flats=self.sharps_flats, mode=self.mode)
        others = tuple(ks for ks in state.key_signatures if ks.tick != self.tick)
        
        return Project(
            id=state.id, metadata=state.metadata,
            tempo_map=state.tempo_map, time_signature=state.time_signature,
            key_signatures=others + (new_ks,),
            tracks=state.tracks, mixer=state.mixer,
            automation=state.automation, markers=state.markers,
            loop_start=state.loop_start, loop_end=state.loop_end,
            project_length_ticks=state.project_length_ticks,
        )
    
    def inverse(self) -> Optional[Action]:
        return None  # Would need to know the previous key signature
