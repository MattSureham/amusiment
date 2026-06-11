"""
Project file IO for amusiment.

The .amus project format is implemented as a ZIP package with project JSON,
manifest metadata, optional assets, and reserved extension areas.
"""

from .project_archive import (
    ARCHIVE_FORMAT,
    ARCHIVE_FORMAT_VERSION,
    ArchiveAsset,
    ProjectArchive,
    ProjectArchiveError,
    inspect_project_archive,
    load_project_archive,
    save_project_archive,
    validate_project_archive,
)

__all__ = [
    "ARCHIVE_FORMAT",
    "ARCHIVE_FORMAT_VERSION",
    "ArchiveAsset",
    "ProjectArchive",
    "ProjectArchiveError",
    "inspect_project_archive",
    "load_project_archive",
    "save_project_archive",
    "validate_project_archive",
]
