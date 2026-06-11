"""
Core domain models for the amusiment framework.

Models follow these principles:
- Immutable where practical (frozen dataclasses)
- Serializable to/from JSON for .amus project files
- Time values are always stored internally as ticks (PPQ-based)
"""

from .time_model import (
    Ticks, Beats, Seconds, Samples, PPQ,
    TimeSignature, TempoChange, TempoMap,
    ticks_to_beats, beats_to_ticks,
    ticks_to_seconds, seconds_to_ticks,
    ticks_to_samples, samples_to_ticks,
)
from .note import NoteEvent, NoteVelocity, NotePitch, NoteDuration
from .clip import Clip, MidiClip, AudioClip, AutomationClip, ClipType
from .track import Track, MidiTrack, AudioTrack, GroupTrack, FxTrack, TrackType
from .device import Device, InstrumentDevice, EffectDevice, DeviceChain
from .automation import AutomationPoint, AutomationLane, AutomationEnvelope, InterpolationMode
from .mixer import MixerChannel, Mixer, SendConfig
from .project import Project, ProjectMetadata, Marker, KeySignature

__all__ = [
    "Ticks", "Beats", "Seconds", "Samples", "PPQ",
    "TimeSignature", "TempoChange", "TempoMap",
    "ticks_to_beats", "beats_to_ticks",
    "ticks_to_seconds", "seconds_to_ticks",
    "ticks_to_samples", "samples_to_ticks",
    "NoteEvent", "NoteVelocity", "NotePitch", "NoteDuration",
    "Clip", "MidiClip", "AudioClip", "AutomationClip", "ClipType",
    "Track", "MidiTrack", "AudioTrack", "GroupTrack", "FxTrack", "TrackType",
    "Device", "InstrumentDevice", "EffectDevice", "DeviceChain",
    "AutomationPoint", "AutomationLane", "AutomationEnvelope", "InterpolationMode",
    "MixerChannel", "Mixer", "SendConfig",
    "Project", "ProjectMetadata", "Marker", "KeySignature",
]
