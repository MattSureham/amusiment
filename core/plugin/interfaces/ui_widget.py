"""
UI Widget plugin interface — extensible UI panels and components.

UI Widget plugins allow third-party developers to add custom panels,
tool windows, and interactive components to the amusiment UI.

Widgets are rendered by the UI shell in their assigned dock areas.
They have access to the project state store for reading data and
dispatching actions.
"""

from abc import abstractmethod
from enum import Enum, auto
from typing import Any, Optional

from .base import PluginBase, PluginManifest, PluginCategory


class WidgetType(Enum):
    """Types of UI widgets."""
    PANEL = auto()           # Full panel (piano roll, mixer, AI chat)
    TOOLBAR_BUTTON = auto()  # A button in the toolbar
    STATUS_BAR_ITEM = auto() # An item in the status bar
    INSPECTOR_SECTION = auto()  # A section in the inspector panel
    POPOVER = auto()         # A popover/dialog
    CONTEXT_MENU = auto()    # A context menu contribution


class WidgetDockArea(Enum):
    """Where a panel widget can be docked in the UI."""
    LEFT = auto()
    RIGHT = auto()
    BOTTOM = auto()
    CENTER = auto()
    FLOATING = auto()


class UIWidgetPlugin(PluginBase):
    """
    Abstract UI widget plugin — adds a custom UI component to the workspace.
    
    Widgets can be simple (e.g., a toolbar button that triggers an action)
    or complex (e.g., a full editor panel with custom rendering).
    
    The UI shell calls lifecycle methods to create, update, and destroy
    the widget's DOM/component tree.
    
    Lifecycle:
        1. initialize() — set up UI resources
        2. create_widget() — create the widget's UI elements
        3. update() — called when project state changes
        4. destroy_widget() — tear down UI elements
        5. shutdown() — release resources
    
    Communication with the core:
    - Read state: Via the state object passed to update()
    - Write state: Via dispatch() callback passed to create_widget()
    """
    
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="",
            name="Unnamed Widget",
            version="0.1.0",
            category=PluginCategory.UI_WIDGET,
            description="A UI widget plugin",
            capabilities=["ui.widget"],
        )
    
    @property
    @abstractmethod
    def widget_type(self) -> WidgetType:
        """What kind of widget this is."""
        ...
    
    @property
    @abstractmethod
    def widget_title(self) -> str:
        """Human-readable title for the widget."""
        ...
    
    @property
    def dock_area(self) -> WidgetDockArea:
        """Default dock area for panel widgets."""
        return WidgetDockArea.CENTER
    
    @property
    def default_visible(self) -> bool:
        """Whether this widget is visible by default when first loaded."""
        return True
    
    @property
    def complexity_level(self) -> str:
        """
        Which user skill level this widget is appropriate for.
        
        Returns: "beginner", "intermediate", "pro", or "all"
        """
        return "all"
    
    @abstractmethod
    def create_widget(self, parent: Any, dispatch: callable) -> Any:
        """
        Create and return the widget's UI element.
        
        Called once when the widget is first shown.
        
        Args:
            parent: The parent UI element to attach to.
            dispatch: Callback for dispatching actions: dispatch(action) -> None.
        
        Returns:
            The widget's root UI element (framework-specific).
        """
        ...
    
    @abstractmethod
    def update(self, project_state: Any) -> None:
        """
        Update the widget with the latest project state.
        
        Called whenever the project state changes. The widget should
        efficiently update only what's needed.
        
        Args:
            project_state: The current immutable project state.
        """
        ...
    
    @abstractmethod
    def destroy_widget(self) -> None:
        """
        Destroy the widget's UI elements and clean up.
        
        Called when the widget is hidden or the plugin is unloaded.
        """
        ...
    
    def on_resize(self, width: int, height: int) -> None:
        """Called when the widget's container is resized."""
        pass
    
    def on_focus(self) -> None:
        """Called when the widget gains focus."""
        pass
    
    def on_blur(self) -> None:
        """Called when the widget loses focus."""
        pass
    
    def get_toolbar_icon(self) -> Optional[str]:
        """Return an icon identifier for toolbar buttons (if WidgetType.TOOLBAR_BUTTON)."""
        return None
    
    def get_keyboard_shortcuts(self) -> dict[str, callable]:
        """
        Return keyboard shortcuts that this widget handles.
        
        Format: {"Ctrl+N": handler_function, "Delete": handler_function, ...}
        """
        return {}
