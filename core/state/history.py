"""
Undo/redo history manager for the amusiment framework.

Uses a double-ended command pattern:
- Actions are pushed onto the undo stack as they're executed.
- Undo pops from undo stack, inverts the action, pushes to redo stack.
- Redo pops from redo stack, re-applies the action, pushes back to undo.

For actions that cannot produce an inverse (e.g., complex deletions),
the HistoryManager falls back to full state snapshots.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

from ..model.project import Project
from .actions import Action

logger = logging.getLogger(__name__)

# Maximum undo history depth
MAX_HISTORY = 256


@dataclass
class HistoryEntry:
    """
    A single entry in the undo/redo history.
    
    Each entry stores both the action that was performed and
    the full project state BEFORE the action, enabling reliable undo
    even when actions cannot produce a clean inverse.
    """
    action: Optional[Action]  # The action that was applied (None for snapshot-only)
    state_before: Project     # Full state snapshot before the action
    description: str = ""


@dataclass
class HistoryManager:
    """
    Manages undo and redo stacks for a project.
    
    Integrates with the Store to automatically record actions.
    Provides undo() and redo() that restore project state.
    
    Usage:
        history = HistoryManager()
        store.subscribe(history.on_state_changed)
        # ... user makes edits via store.dispatch() ...
        history.undo(store)  # Undo last action
        history.redo(store)  # Redo
    """
    
    _undo_stack: list[HistoryEntry] = field(default_factory=list)
    _redo_stack: list[HistoryEntry] = field(default_factory=list)
    _last_state: Optional[Project] = None
    _batch_active: bool = False
    _batch_entries: list[HistoryEntry] = field(default_factory=list)
    _max_history: int = MAX_HISTORY
    
    @property
    def can_undo(self) -> bool:
        """Whether there are actions to undo."""
        return len(self._undo_stack) > 0
    
    @property
    def can_redo(self) -> bool:
        """Whether there are actions to redo."""
        return len(self._redo_stack) > 0
    
    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)
    
    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)
    
    @property
    def undo_descriptions(self) -> list[str]:
        """Human-readable descriptions of undoable actions."""
        return [e.description or (e.action.description if e.action else "Unknown")
                for e in reversed(self._undo_stack)]
    
    @property
    def redo_descriptions(self) -> list[str]:
        """Human-readable descriptions of redoable actions."""
        return [e.description or (e.action.description if e.action else "Unknown")
                for e in reversed(self._redo_stack)]
    
    def record(self, action: Action, state_before: Project) -> None:
        """
        Record an action that was applied to the state.
        
        Called automatically by the Store when dispatching actions.
        This clears the redo stack (standard behavior).
        
        Args:
            action: The action that was just applied.
            state_before: The state before the action was applied.
        """
        entry = HistoryEntry(
            action=action,
            state_before=state_before,
            description=action.description,
        )
        
        if self._batch_active:
            self._batch_entries.append(entry)
        else:
            self._push_undo(entry)
    
    def record_snapshot(self, state_before: Project, description: str = "") -> None:
        """
        Record a state snapshot (no action) for undo.
        
        Useful when the state was modified externally or via batch operations.
        """
        entry = HistoryEntry(
            action=None,
            state_before=state_before,
            description=description,
        )
        self._push_undo(entry)
    
    def undo(self, store: "Store") -> Optional[Project]:
        """
        Undo the most recent action.
        
        Args:
            store: The state store to restore state into.
        
        Returns:
            The restored state, or None if nothing to undo.
        """
        if not self._undo_stack:
            return None
        
        entry = self._undo_stack.pop()
        
        # Push current state to redo before restoring
        current_state = store.get_state()
        self._redo_stack.append(HistoryEntry(
            action=entry.action,
            state_before=current_state,
            description=entry.description,
        ))
        
        # Restore the state from before the action
        store.replace_state(entry.state_before)
        
        # Trim redo stack
        if len(self._redo_stack) > self._max_history:
            self._redo_stack = self._redo_stack[-self._max_history:]
        
        logger.debug(f"Undo: {entry.description}")
        return entry.state_before
    
    def redo(self, store: "Store") -> Optional[Project]:
        """
        Redo the most recently undone action.
        
        Args:
            store: The state store to apply the action to.
        
        Returns:
            The new state, or None if nothing to redo.
        """
        if not self._redo_stack:
            return None
        
        entry = self._redo_stack.pop()
        
        # Record current state for undo
        current_state = store.get_state()
        redo_entry = HistoryEntry(
            action=entry.action,
            state_before=current_state,
            description=entry.description,
        )
        
        if entry.action:
            # Re-apply the action
            new_state = entry.action.reduce(current_state)
            store.replace_state(new_state)
            self._push_undo(redo_entry)
            logger.debug(f"Redo: {entry.description}")
            return new_state
        else:
            # Snapshot-only — this shouldn't happen in normal operation
            logger.warning("Attempted to redo a snapshot-only entry")
            return None
    
    def clear(self) -> None:
        """Clear all undo/redo history (e.g., when opening a new project)."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._batch_entries.clear()
        self._batch_active = False
    
    # ── Batch operations ─────────────────────────────────────────
    
    def begin_batch(self) -> None:
        """
        Begin a batch of actions that should be undone/redone together.
        
        While batching is active, recorded actions are held rather than
        pushed directly to the undo stack.
        """
        self._batch_active = True
        self._batch_entries = []
    
    def end_batch(self, description: str = "Batch operation") -> None:
        """
        End the current batch and commit it as a single undo entry.
        
        The undo entry stores the state before the first batched action.
        """
        if not self._batch_active:
            return
        
        self._batch_active = False
        
        if self._batch_entries:
            # Combine all batched entries into one
            first_state = self._batch_entries[0].state_before
            entry = HistoryEntry(
                action=None,  # Batches don't have a single invertible action
                state_before=first_state,
                description=description,
            )
            self._push_undo(entry)
        
        self._batch_entries = []
    
    # ── Internal ──────────────────────────────────────────────────
    
    def _push_undo(self, entry: HistoryEntry) -> None:
        """Push an entry onto the undo stack, clearing redo."""
        self._undo_stack.append(entry)
        self._redo_stack.clear()  # New action invalidates redo
        
        # Trim undo stack if needed
        if len(self._undo_stack) > self._max_history:
            self._undo_stack = self._undo_stack[-self._max_history:]
