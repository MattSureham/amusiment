"""
Plugin registry — central discovery, loading, and lifecycle manager.

The PluginRegistry is the single source of truth for all plugins
in the framework. It handles:
- Discovery: Scanning plugin directories for manifests
- Loading: Importing plugin modules and instantiating classes
- Lifecycle: initialize(), activate/deactivate, shutdown()
- Querying: Finding plugins by category, capability, or ID
- Sandboxing: Process isolation for unstable plugins

Plugins can be:
- Built-in (shipped with the framework)
- User-installed (in the user's plugin directory)
- Remote (loaded from a package registry)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Type
import importlib
import logging
import os
import sys
from pathlib import Path

from .interfaces.base import PluginBase, PluginManifest, PluginCategory, PluginState
from .interfaces.instrument import InstrumentPlugin
from .interfaces.effect import EffectPlugin
from .interfaces.ai_generator import AIGeneratorPlugin
from .interfaces.ai_analyzer import AIAnalyzerPlugin
from .interfaces.ui_widget import UIWidgetPlugin
from .interfaces.exporter import ExporterPlugin
from .interfaces.importer import ImporterPlugin

logger = logging.getLogger(__name__)


# Map PluginCategory to ABC class for validation
CATEGORY_BASE_CLASS: dict[PluginCategory, Type[PluginBase]] = {
    PluginCategory.INSTRUMENT: InstrumentPlugin,
    PluginCategory.EFFECT: EffectPlugin,
    PluginCategory.AI_GENERATOR: AIGeneratorPlugin,
    PluginCategory.AI_ANALYZER: AIAnalyzerPlugin,
    PluginCategory.UI_WIDGET: UIWidgetPlugin,
    PluginCategory.EXPORTER: ExporterPlugin,
    PluginCategory.IMPORTER: ImporterPlugin,
    # UTILITY uses PluginBase directly (generic)
}


@dataclass
class PluginHandle:
    """
    Runtime handle for a loaded plugin.
    
    The registry manages these handles — external code should not
    create them directly.
    """
    manifest: PluginManifest
    instance: Optional[PluginBase] = None
    state: PluginState = PluginState.DISCOVERED
    module_path: str = ""      # Python module path (e.g., "plugins.builtin.chord_gen")
    file_path: Optional[str] = None  # Absolute path to the plugin file/directory
    load_error: Optional[str] = None


@dataclass
class PluginRegistry:
    """
    Central plugin registry and lifecycle manager.
    
    Usage:
        registry = PluginRegistry()
        registry.discover_plugins(["/path/to/plugins", "/path/to/builtin"])
        registry.load_all()
        
        # Query plugins
        generators = registry.get_by_category(PluginCategory.AI_GENERATOR)
        
        # Get a specific plugin
        plugin = registry.get("my.chord_generator")
        result = plugin.generate(prompt)
    """
    
    _plugins: dict[str, PluginHandle] = field(default_factory=dict)
    _search_paths: list[str] = field(default_factory=list)
    
    # ── Discovery ─────────────────────────────────────────────────
    
    def discover_plugins(self, search_paths: list[str]) -> int:
        """
        Scan directories for plugin modules.
        
        Each plugin is expected to be a Python package with:
        - plugin.py (containing a class that inherits from a PluginBase subclass)
        - manifest.json (optional, merged with get_manifest() output)
        
        Args:
            search_paths: List of directory paths to scan.
        
        Returns:
            Number of newly discovered plugins.
        """
        self._search_paths = search_paths
        count = 0
        
        for search_path in search_paths:
            path = Path(search_path).expanduser().resolve()
            if not path.is_dir():
                logger.warning(f"Plugin search path not found: {path}")
                continue
            
            # Add the search directory to Python path so plugin packages can
            # resolve package-relative imports while being loaded.
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            
            # Scan for plugin packages (directories with __init__.py or plugin.py)
            for item in path.iterdir():
                if item.is_dir():
                    plugin_file = item / "plugin.py"
                    if plugin_file.exists():
                        if self._discover_plugin_package(item):
                            count += 1
                elif item.is_file() and item.suffix == ".py" and item.stem.startswith("plugin_"):
                    # Single-file plugin: plugin_<name>.py
                    if self._discover_plugin_file(item):
                        count += 1
        
        logger.info(f"Discovered {count} plugins across {len(search_paths)} paths")
        return count
    
    def _discover_plugin_package(self, package_path: Path) -> bool:
        """Discover a plugin from a package directory."""
        try:
            plugin_file = package_path / "plugin.py"
            module_path = f"{package_path.name}.plugin"
            module = self._load_module(module_path, plugin_file)
            
            # Find the plugin class in the module
            plugin_class = self._find_plugin_class(module, package_path)
            if plugin_class is None:
                return False
            
            # Create a temporary instance to get the manifest
            temp_instance = plugin_class()
            manifest = temp_instance.get_manifest()
            
            if manifest.plugin_id in self._plugins:
                logger.warning(f"Duplicate plugin ID: {manifest.plugin_id}")
                return False
            
            handle = PluginHandle(
                manifest=manifest,
                module_path=module_path,
                file_path=str(plugin_file),
            )
            self._plugins[manifest.plugin_id] = handle
            logger.debug(f"Discovered plugin: {manifest.plugin_id} ({manifest.name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to discover plugin at {package_path}: {e}")
            return False
    
    def _discover_plugin_file(self, file_path: Path) -> bool:
        """Discover a plugin from a single .py file."""
        try:
            module_path = f"_amusiment_plugin_{file_path.stem}"
            module = self._load_module(module_path, file_path)

            plugin_class = self._find_plugin_class(module, file_path)
            if plugin_class is None:
                return False

            temp_instance = plugin_class()
            manifest = temp_instance.get_manifest()

            if manifest.plugin_id in self._plugins:
                logger.warning(f"Duplicate plugin ID: {manifest.plugin_id}")
                return False

            handle = PluginHandle(
                manifest=manifest,
                module_path=module_path,
                file_path=str(file_path),
            )
            self._plugins[manifest.plugin_id] = handle
            logger.debug(f"Discovered plugin: {manifest.plugin_id} ({manifest.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to discover plugin at {file_path}: {e}")
            return False

    def _load_module(self, module_path: str, file_path: Path):
        """Load a plugin module from a concrete Python file."""
        if module_path in sys.modules:
            return sys.modules[module_path]

        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_path, None)
            raise
        return module
    
    def _find_plugin_class(self, module, path) -> Optional[Type[PluginBase]]:
        """Find a PluginBase subclass in a module (basic discovery)."""
        import inspect
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is PluginBase:
                continue
            if issubclass(obj, PluginBase) and not inspect.isabstract(obj):
                return obj
        
        return None
    
    # ── Loading ───────────────────────────────────────────────────
    
    def load_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """
        Load and initialize a specific plugin.
        
        Args:
            plugin_id: The plugin's unique identifier.
        
        Returns:
            The initialized plugin instance, or None on failure.
        """
        handle = self._plugins.get(plugin_id)
        if handle is None:
            logger.error(f"Plugin not found: {plugin_id}")
            return None
        
        if handle.state in (PluginState.LOADED, PluginState.INITIALIZED, PluginState.ACTIVE):
            return handle.instance
        
        try:
            if handle.file_path is None:
                raise ImportError(f"Plugin has no file path: {plugin_id}")

            module = self._load_module(handle.module_path, Path(handle.file_path))
            
            # Find and instantiate the plugin class
            plugin_class = self._find_plugin_class(module, handle.file_path)
            if plugin_class is None:
                raise ValueError(f"No PluginBase subclass found in {handle.file_path}")
            
            instance = plugin_class()
            handle.instance = instance
            handle.state = PluginState.LOADED
            
            # Initialize
            instance.initialize()
            handle.state = PluginState.INITIALIZED
            
            logger.info(f"Loaded plugin: {plugin_id} v{handle.manifest.version}")
            return instance
            
        except Exception as e:
            handle.state = PluginState.ERROR
            handle.load_error = str(e)
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return None
    
    def load_all(self) -> dict[str, PluginBase]:
        """
        Load and initialize all discovered plugins.
        
        Returns:
            Dict mapping plugin_id to initialized instance.
        """
        loaded = {}
        for plugin_id in list(self._plugins.keys()):
            instance = self.load_plugin(plugin_id)
            if instance is not None:
                loaded[plugin_id] = instance
        return loaded
    
    def unload_plugin(self, plugin_id: str) -> None:
        """Shut down and unload a specific plugin."""
        handle = self._plugins.get(plugin_id)
        if handle is None or handle.instance is None:
            return
        
        try:
            handle.instance.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down plugin {plugin_id}: {e}")
        
        handle.instance = None
        handle.state = PluginState.UNLOADED
    
    def unload_all(self) -> None:
        """Shut down all loaded plugins."""
        for plugin_id in list(self._plugins.keys()):
            self.unload_plugin(plugin_id)
    
    # ── Querying ──────────────────────────────────────────────────
    
    def get(self, plugin_id: str) -> Optional[PluginBase]:
        """Get a loaded plugin instance by ID."""
        handle = self._plugins.get(plugin_id)
        if handle and handle.instance:
            return handle.instance
        return None
    
    def get_handle(self, plugin_id: str) -> Optional[PluginHandle]:
        """Get a plugin handle (including unloaded plugins)."""
        return self._plugins.get(plugin_id)
    
    def get_by_category(self, category: PluginCategory) -> list[PluginBase]:
        """Get all loaded plugins of a given category."""
        return [
            h.instance for h in self._plugins.values()
            if h.manifest.category == category and h.instance is not None
        ]
    
    def get_by_capability(self, capability: str) -> list[PluginBase]:
        """Get all loaded plugins that declare a specific capability."""
        return [
            h.instance for h in self._plugins.values()
            if capability in h.manifest.capabilities and h.instance is not None
        ]
    
    def get_generators_for_type(self, content_type: "ContentType") -> list["AIGeneratorPlugin"]:
        """Get all AI generators that support a specific content type."""
        from .interfaces.ai_generator import AIGeneratorPlugin, ContentType
        generators = self.get_by_category(PluginCategory.AI_GENERATOR)
        return [
            g for g in generators
            if isinstance(g, AIGeneratorPlugin)
            and content_type in g.get_capabilities().content_types
        ]
    
    def list_plugins(self, category: Optional[PluginCategory] = None) -> list[PluginHandle]:
        """List all plugins, optionally filtered by category."""
        if category:
            return [h for h in self._plugins.values() if h.manifest.category == category]
        return list(self._plugins.values())
    
    @property
    def plugin_count(self) -> int:
        return len(self._plugins)
    
    @property
    def loaded_count(self) -> int:
        return sum(1 for h in self._plugins.values() if h.instance is not None)
    
    # ── Registration (for programmatic plugin registration) ───────
    
    def register(self, instance: PluginBase) -> str:
        """
        Register a programmatically-created plugin instance.
        
        Useful for built-in plugins created in code rather than
        discovered from the filesystem.
        
        Args:
            instance: An initialized plugin instance.
        
        Returns:
            The plugin's ID.
        """
        manifest = instance.get_manifest()
        
        if manifest.plugin_id in self._plugins:
            logger.warning(f"Re-registering plugin: {manifest.plugin_id}")
        
        handle = PluginHandle(
            manifest=manifest,
            instance=instance,
            state=PluginState.INITIALIZED,
        )
        self._plugins[manifest.plugin_id] = handle
        return manifest.plugin_id
    
    def unregister(self, plugin_id: str) -> None:
        """Unregister and unload a plugin."""
        self.unload_plugin(plugin_id)
        self._plugins.pop(plugin_id, None)
