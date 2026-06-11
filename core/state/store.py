"""
Immutable state store for the amusiment framework.

The Store holds the current Project state and dispatches Actions
through reducers to produce new states. Listeners are notified
on every state change.

Pattern: Redux-like unidirectional data flow.
- Actions describe what happened
- Reducers produce the new state
- Listeners react to state changes
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import logging

from ..model.project import Project

logger = logging.getLogger(__name__)

# Type alias for listener callbacks
StoreListener = Callable[[Project], None]


@dataclass
class Store:
    """
    Central state store for a project.
    
    Usage:
        store = Store(project)
        store.subscribe(my_listener)
        store.dispatch(some_action)
    
    The store is NOT thread-safe by design — it expects single-threaded
    access from the main/UI thread. Audio thread access should go through
    a lock-free queue.
    """
    
    _state: Project = field(default_factory=Project.create_new)
    _listeners: list[StoreListener] = field(default_factory=list)
    _dispatching: bool = False
    _pending: list["Action"] = field(default_factory=list)
    
    @property
    def state(self) -> Project:
        """Get the current immutable project state."""
        return self._state
    
    def subscribe(self, listener: StoreListener) -> Callable[[], None]:
        """
        Register a listener to be called on every state change.
        
        Returns an unsubscribe function.
        """
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)
    
    def dispatch(self, action: "Action") -> Project:
        """
        Dispatch an action to mutate the state.
        
        Actions are processed synchronously. If a dispatch is already
        in progress (re-entrant), the action is queued.
        
        Returns the new state after applying the action.
        
        Args:
            action: The action to apply.
        
        Returns:
            The new Project state.
        """
        if self._dispatching:
            self._pending.append(action)
            return self._state
        
        self._dispatching = True
        try:
            new_state = action.reduce(self._state)
            if new_state is not self._state:
                self._state = new_state
                self._notify_listeners(new_state)
            
            # Process any queued actions
            while self._pending:
                pending = self._pending.pop(0)
                new_state = pending.reduce(self._state)
                if new_state is not self._state:
                    self._state = new_state
                    self._notify_listeners(new_state)
            
            return self._state
        finally:
            self._dispatching = False
    
    def _notify_listeners(self, state: Project) -> None:
        """Call all registered listeners with the new state."""
        for listener in self._listeners:
            try:
                listener(state)
            except Exception:
                logger.exception("Store listener raised an exception")
    
    def get_state(self) -> Project:
        """Get the current state (alias for .state)."""
        return self._state
    
    def replace_state(self, project: Project) -> None:
        """
        Replace the entire state (used for undo/redo and project loading).
        
        This does NOT go through the action/reducer pipeline.
        Listeners ARE notified.
        """
        self._state = project
        self._notify_listeners(project)
