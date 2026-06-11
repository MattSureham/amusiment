"""
State management for the amusiment framework.

Implements an immutable state tree pattern with command-based undo/redo.
All mutations flow through the Store, producing new Project states.
"""

from .store import Store, StoreListener
from .actions import (
    Action,
    ActionType,
    AddTrackAction,
    RemoveTrackAction,
    AddClipAction,
    RemoveClipAction,
    AddNoteAction,
    RemoveNoteAction,
    SetMixerVolumeAction,
    SetTempoAction,
    SetKeySignatureAction,
)
from .history import HistoryManager, HistoryEntry

__all__ = [
    "Store", "StoreListener",
    "Action", "ActionType",
    "AddTrackAction", "RemoveTrackAction",
    "AddClipAction", "RemoveClipAction",
    "AddNoteAction", "RemoveNoteAction",
    "SetMixerVolumeAction", "SetTempoAction",
    "SetKeySignatureAction",
    "HistoryManager", "HistoryEntry",
]
