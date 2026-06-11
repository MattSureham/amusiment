"""
Importer plugin interface — bring external musical assets into projects.

Importer plugins convert external file formats such as MIDI, MusicXML, loop
packs, stems, and interchange JSON into the framework's project state tree.
Like exporters, importers operate on serialized dictionaries at the plugin
boundary so UI, scripting, and sandboxed plugin runtimes can use the same API.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from .base import PluginBase, PluginManifest, PluginCategory


class ImportFormat(Enum):
    """Supported import formats."""
    MIDI = auto()
    MUSICXML = auto()
    AUDIO = auto()
    STEMS = auto()
    JSON = auto()
    ABC = auto()
    LILYPOND = auto()


@dataclass
class ImportRequest:
    """
    A request to import an external file.

    Attributes:
        format: Source file format.
        input_path: Absolute or relative path to the source file.
        project_name: Optional name for the created project.
        track_name_prefix: Optional prefix applied to imported track names.
        split_channels: Whether channel-based formats should become one track
            per MIDI channel where possible.
        metadata: Format-specific options.
    """
    format: ImportFormat = ImportFormat.MIDI
    input_path: str = ""
    project_name: str = ""
    track_name_prefix: str = ""
    split_channels: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class ImportResult:
    """
    The result of an import operation.

    Attributes:
        success: Whether the import completed successfully.
        input_path: Source file path.
        format: The format that was imported.
        project_state: Serialized Project produced by the importer.
        imported_track_count: Number of tracks created.
        imported_clip_count: Number of clips created.
        duration_ticks: Imported timeline duration in framework ticks.
        warnings: Non-fatal issues found during import.
        error_message: Error description if import failed.
    """
    success: bool = False
    input_path: str = ""
    format: ImportFormat = ImportFormat.MIDI
    project_state: dict = field(default_factory=dict)
    imported_track_count: int = 0
    imported_clip_count: int = 0
    duration_ticks: int = 0
    warnings: list[str] = field(default_factory=list)
    error_message: str = ""


class ImporterPlugin(PluginBase):
    """
    Abstract importer plugin — converts external files into project state.

    Importers should be deterministic and side-effect free apart from reading
    the requested input file. The returned Project dict can then be merged,
    inserted, archived as .amus, or passed through the command system.
    """

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="",
            name="Unnamed Importer",
            version="0.1.0",
            category=PluginCategory.IMPORTER,
            description="An import plugin",
            capabilities=["import.file"],
        )

    @abstractmethod
    def get_supported_formats(self) -> list[ImportFormat]:
        """Return which import formats this plugin handles."""
        ...

    @abstractmethod
    def import_file(
        self,
        request: ImportRequest,
        progress_callback: "Optional[Callable[[float, str], None]]" = None,
    ) -> ImportResult:
        """
        Import external content into a serialized Project state.

        Args:
            request: Import configuration.
            progress_callback: Optional callback for progress reporting.
                               Signature: (progress_float, status_string) -> None

        Returns:
            ImportResult indicating success/failure and imported project data.
        """
        ...

    def validate_request(self, request: ImportRequest) -> list[str]:
        """
        Validate an import request before executing.

        Returns:
            List of validation error messages (empty = valid).
        """
        import os

        errors = []

        if request.format not in self.get_supported_formats():
            errors.append(f"Format {request.format.name} not supported by this importer")

        if not request.input_path:
            errors.append("Input path is required")
        elif not os.path.isfile(request.input_path):
            errors.append(f"Input file not found: {request.input_path}")

        return errors

    def get_default_extensions(self, fmt: ImportFormat) -> list[str]:
        """Return common file extensions for a format."""
        extensions = {
            ImportFormat.MIDI: [".mid", ".midi"],
            ImportFormat.MUSICXML: [".musicxml", ".xml", ".mxl"],
            ImportFormat.AUDIO: [".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"],
            ImportFormat.STEMS: [".zip"],
            ImportFormat.JSON: [".json", ".amus.json"],
            ImportFormat.ABC: [".abc"],
            ImportFormat.LILYPOND: [".ly"],
        }
        return extensions.get(fmt, [])

    def get_mime_types(self, fmt: ImportFormat) -> list[str]:
        """Return common MIME types for a format."""
        mimes = {
            ImportFormat.MIDI: ["audio/midi", "audio/x-midi"],
            ImportFormat.MUSICXML: ["application/vnd.recordare.musicxml+xml", "application/xml"],
            ImportFormat.AUDIO: ["audio/wav", "audio/flac", "audio/mpeg", "audio/ogg"],
            ImportFormat.STEMS: ["application/zip"],
            ImportFormat.JSON: ["application/json"],
            ImportFormat.ABC: ["text/vnd.abc"],
            ImportFormat.LILYPOND: ["text/x-lilypond"],
        }
        return mimes.get(fmt, ["application/octet-stream"])
