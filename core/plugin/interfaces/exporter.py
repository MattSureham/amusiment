"""
Exporter plugin interface — project export to various formats.

Exporter plugins convert project data into standard file formats:
WAV, MP3, FLAC, MIDI, MusicXML, stems, and more.

Each exporter handles one or more output formats and can be configured
with format-specific settings (bitrate, sample rate, dithering, etc.).
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from .base import PluginBase, PluginManifest, PluginCategory


class ExportFormat(Enum):
    """Supported export formats."""
    WAV = auto()
    MP3 = auto()
    FLAC = auto()
    OGG = auto()
    MIDI = auto()
    MUSICXML = auto()
    STEMS = auto()     # Individual track stems
    JSON = auto()      # Raw project data (for backup/interchange)
    ABC = auto()       # ABC notation
    LILYPOND = auto()  # LilyPond sheet music


@dataclass
class ExportRequest:
    """
    A request to export project content.
    
    Attributes:
        format: Target export format.
        output_path: Absolute path for the output file.
        track_ids: Specific tracks to export (empty = all).
        start_tick: Export start position (0 = from beginning).
        end_tick: Export end position (0 = entire project length).
        sample_rate: Audio sample rate (for audio formats).
        bit_depth: Audio bit depth (16, 24, 32).
        mp3_bitrate_kbps: MP3 bitrate in kbps (if applicable).
        dither_enabled: Whether to apply dithering.
        normalize_enabled: Whether to normalize to peak level.
        normalize_target_db: Target peak for normalization.
        include_effects: Whether to render through effects chain.
        include_automation: Whether to render automation.
        metadata: Format-specific metadata (tags, album art, etc.).
    """
    format: ExportFormat = ExportFormat.WAV
    output_path: str = ""
    track_ids: list[str] = field(default_factory=list)
    start_tick: int = 0
    end_tick: int = 0
    sample_rate: int = 44100
    bit_depth: int = 24
    mp3_bitrate_kbps: int = 320
    dither_enabled: bool = True
    normalize_enabled: bool = False
    normalize_target_db: float = -0.3
    include_effects: bool = True
    include_automation: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class ExportResult:
    """
    The result of an export operation.
    
    Attributes:
        success: Whether the export completed successfully.
        output_path: Path to the exported file.
        format: The format that was exported.
        duration_seconds: Duration of the exported content.
        file_size_bytes: Size of the output file.
        warnings: Any warnings generated during export.
        error_message: Error description if export failed.
    """
    success: bool = False
    output_path: str = ""
    format: ExportFormat = ExportFormat.WAV
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    warnings: list[str] = field(default_factory=list)
    error_message: str = ""


class ExporterPlugin(PluginBase):
    """
    Abstract exporter plugin — converts project data to a file format.
    
    Exporters can be synchronous (simple formats) or asynchronous
    (long renders). For long renders, progress should be reported
    via the progress callback.
    
    Lifecycle:
        1. initialize() — set up encoder resources
        2. get_supported_formats() — declare what this exporter can output
        3. export() — perform the export (may be called many times)
        4. shutdown() — release resources
    
    Export happens on a background thread so the UI stays responsive.
    """
    
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="",
            name="Unnamed Exporter",
            version="0.1.0",
            category=PluginCategory.EXPORTER,
            description="An export plugin",
            capabilities=["export.file"],
        )
    
    @abstractmethod
    def get_supported_formats(self) -> list[ExportFormat]:
        """
        Return which export formats this plugin handles.
        
        A single exporter can support multiple formats (e.g., WAV + FLAC).
        """
        ...
    
    @abstractmethod
    def export(self, request: ExportRequest,
               project_state: dict,
               progress_callback: "Optional[Callable[[float, str], None]]" = None) -> ExportResult:
        """
        Export project content to the requested format.
        
        This is called from a background thread. Long operations should
        periodically call progress_callback(0.0-1.0, status_message).
        
        Args:
            request: Export configuration.
            project_state: Serialized project state (dict from Project.to_dict()).
            progress_callback: Optional callback for progress reporting.
                               Signature: (progress_float, status_string) -> None
        
        Returns:
            ExportResult indicating success/failure and output details.
        """
        ...
    
    def validate_request(self, request: ExportRequest) -> list[str]:
        """
        Validate an export request before executing.
        
        Override to add format-specific validation.
        
        Returns:
            List of validation error messages (empty = valid).
        """
        errors = []
        
        if request.format not in self.get_supported_formats():
            errors.append(f"Format {request.format.name} not supported by this exporter")
        
        if not request.output_path:
            errors.append("Output path is required")
        
        if request.sample_rate not in (22050, 44100, 48000, 88200, 96000, 192000):
            errors.append(f"Unsupported sample rate: {request.sample_rate}")
        
        return errors
    
    def get_default_extension(self, fmt: ExportFormat) -> str:
        """Return the default file extension for a format."""
        extensions = {
            ExportFormat.WAV: ".wav",
            ExportFormat.MP3: ".mp3",
            ExportFormat.FLAC: ".flac",
            ExportFormat.OGG: ".ogg",
            ExportFormat.MIDI: ".mid",
            ExportFormat.MUSICXML: ".musicxml",
            ExportFormat.JSON: ".amus.json",
            ExportFormat.ABC: ".abc",
            ExportFormat.LILYPOND: ".ly",
        }
        return extensions.get(fmt, ".dat")
    
    def get_mime_type(self, fmt: ExportFormat) -> str:
        """Return the MIME type for a format."""
        mimes = {
            ExportFormat.WAV: "audio/wav",
            ExportFormat.MP3: "audio/mpeg",
            ExportFormat.FLAC: "audio/flac",
            ExportFormat.OGG: "audio/ogg",
            ExportFormat.MIDI: "audio/midi",
            ExportFormat.MUSICXML: "application/vnd.recordare.musicxml+xml",
            ExportFormat.JSON: "application/json",
        }
        return mimes.get(fmt, "application/octet-stream")
