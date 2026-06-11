"""
Base plugin interface — the root ABC that all plugins inherit.

Every plugin must provide an id, name, version, and basic lifecycle methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class PluginCategory(Enum):
    """Top-level category for plugin classification."""
    INSTRUMENT = auto()
    EFFECT = auto()
    AI_GENERATOR = auto()
    AI_ANALYZER = auto()
    UI_WIDGET = auto()
    EXPORTER = auto()
    IMPORTER = auto()
    UTILITY = auto()


@dataclass
class PluginManifest:
    """
    Declarative manifest for a plugin.
    
    Plugins register their identity, capabilities, and requirements
    through this manifest. The registry uses it for discovery and validation.
    
    Attributes:
        plugin_id: Unique identifier (reverse domain notation recommended).
        name: Human-readable plugin name.
        version: Semver version string.
        category: Plugin category.
        author: Plugin author name.
        description: Short description of what this plugin does.
        capabilities: List of capability tags (e.g., ["generate.melody", "generate.chords"]).
        dependencies: Other plugin IDs this plugin requires.
        min_framework_version: Minimum amusiment framework version required.
    """
    plugin_id: str = ""
    name: str = "Unnamed Plugin"
    version: str = "0.1.0"
    category: PluginCategory = PluginCategory.UTILITY
    author: str = ""
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    min_framework_version: str = "0.1.0"


class PluginState(Enum):
    """Runtime lifecycle states for a plugin."""
    DISCOVERED = auto()    # Found but not loaded
    LOADED = auto()        # Code loaded, not yet initialized
    INITIALIZED = auto()   # Ready for use
    ACTIVE = auto()        # Currently processing/running
    PAUSED = auto()        # Temporarily inactive
    ERROR = auto()         # Failed state
    UNLOADED = auto()      # Cleaned up and removed


class PluginBase(ABC):
    """
    Abstract base class for all amusiment plugins.
    
    Every plugin must implement:
    - get_manifest(): Return identity and capabilities.
    - initialize(): Set up resources.
    - shutdown(): Release resources.
    
    Plugins run in their own process space for isolation.
    Communication with the core is via well-defined interfaces only.
    """
    
    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """Return the plugin's identity and capability declaration."""
        ...
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the plugin — allocate resources, load models, etc.
        
        Called once after the plugin code is loaded. Must be idempotent.
        """
        ...
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Shut down the plugin — release all resources.
        
        Called when the plugin is being unloaded. Must be safe to call
        multiple times.
        """
        ...
    
    def get_state(self) -> PluginState:
        """Return the current runtime state (overridable for custom state tracking)."""
        return PluginState.INITIALIZED
    
    def on_project_load(self, project_state: dict[str, Any]) -> None:
        """
        Called when a project is loaded, giving the plugin a chance to
        restore its saved state from the project file.
        
        Args:
            project_state: The plugin-specific state from the project file.
        """
        pass
    
    def on_project_save(self) -> dict[str, Any]:
        """
        Called when a project is being saved. The plugin should return
        any state it wants persisted.
        
        Returns:
            A JSON-serializable dict of plugin state.
        """
        return {}
    
    def on_activate(self) -> None:
        """Called when the plugin becomes active (e.g., track armed)."""
        pass
    
    def on_deactivate(self) -> None:
        """Called when the plugin becomes inactive."""
        pass
