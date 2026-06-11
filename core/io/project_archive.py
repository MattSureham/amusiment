"""
Project archive reader/writer for .amus files.

The .amus format is a ZIP package with a small, stable core layout:

    manifest.json
    project.json
    assets/...
    plugin-state/state.json       (optional, reserved for plugin data)

Keeping the project JSON separate from binary assets makes the format easy to
inspect, version, sync, and extend without changing the domain model.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import posixpath
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional

from core.model.project import Project


ARCHIVE_FORMAT = "amusiment.project.archive"
ARCHIVE_FORMAT_VERSION = "1.0.0"
PROJECT_JSON_PATH = "project.json"
MANIFEST_JSON_PATH = "manifest.json"
PLUGIN_STATE_JSON_PATH = "plugin-state/state.json"
ASSETS_PREFIX = "assets/"


class ProjectArchiveError(Exception):
    """Raised when a .amus archive cannot be safely read or written."""


@dataclass(frozen=True)
class ArchiveAsset:
    """
    A binary asset bundled into a .amus archive.

    source_path points to a local file while saving or to an extracted file after
    loading with extract_assets_to. archive_path is always a safe package path,
    normally under assets/.
    """

    source_path: str = ""
    archive_path: str = ""
    role: str = "asset"
    media_type: str = ""
    size_bytes: int = 0
    sha256: str = ""

    @classmethod
    def from_file(
        cls,
        source_path: str,
        archive_path: Optional[str] = None,
        role: str = "asset",
        media_type: str = "",
    ) -> "ArchiveAsset":
        """Create an asset descriptor for a local file."""
        if archive_path is None:
            archive_path = ASSETS_PREFIX + Path(source_path).name
        return cls(
            source_path=str(source_path),
            archive_path=archive_path,
            role=role,
            media_type=media_type,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize this asset record for manifest.json."""
        return {
            "path": self.archive_path,
            "source_path": self.source_path,
            "role": self.role,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_manifest_dict(cls, data: dict[str, Any]) -> "ArchiveAsset":
        """Deserialize an asset record from manifest.json."""
        return cls(
            source_path=data.get("source_path", ""),
            archive_path=data.get("path", data.get("archive_path", "")),
            role=data.get("role", "asset"),
            media_type=data.get("media_type", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=data.get("sha256", ""),
        )


@dataclass(frozen=True)
class ProjectArchive:
    """A loaded .amus archive."""

    project: Project
    manifest: dict[str, Any]
    assets: tuple[ArchiveAsset, ...] = field(default_factory=tuple)
    plugin_state: dict[str, Any] = field(default_factory=dict)


def save_project_archive(
    project: Project,
    output_path: str,
    assets: Optional[Iterable[ArchiveAsset]] = None,
    plugin_state: Optional[dict[str, Any]] = None,
    extra_manifest: Optional[dict[str, Any]] = None,
    compress: bool = True,
) -> dict[str, Any]:
    """
    Save a Project to a .amus archive.

    Args:
        project: Project state tree to save.
        output_path: Destination .amus path.
        assets: Optional binary assets to package under assets/.
        plugin_state: Optional plugin-specific JSON state.
        extra_manifest: Optional caller-owned manifest extension data.
        compress: Whether to use ZIP_DEFLATED instead of ZIP_STORED.

    Returns:
        The manifest dict written to the archive.
    """
    output = Path(output_path)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)

    prepared_assets = _prepare_assets(tuple(assets or ()))
    project_data = project.to_dict()
    has_plugin_state = bool(plugin_state)
    manifest = _build_manifest(
        project=project,
        project_data=project_data,
        assets=prepared_assets,
        has_plugin_state=has_plugin_state,
        plugin_state=plugin_state or {},
        extra_manifest=extra_manifest or {},
    )

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    try:
        with zipfile.ZipFile(output, "w", compression=compression) as archive:
            _writestr_json(archive, MANIFEST_JSON_PATH, manifest)
            _writestr_json(archive, PROJECT_JSON_PATH, project_data)
            if has_plugin_state:
                _writestr_json(archive, PLUGIN_STATE_JSON_PATH, plugin_state or {})
            for asset in prepared_assets:
                archive.write(asset.source_path, asset.archive_path)
    except OSError as exc:
        raise ProjectArchiveError(f"Failed to write archive '{output_path}': {exc}") from exc
    except zipfile.BadZipFile as exc:
        raise ProjectArchiveError(f"Failed to write archive '{output_path}': {exc}") from exc

    return manifest


def load_project_archive(
    archive_path: str,
    extract_assets_to: Optional[str] = None,
    verify_hashes: bool = True,
) -> ProjectArchive:
    """
    Load a .amus archive.

    Args:
        archive_path: Path to a .amus archive.
        extract_assets_to: Optional directory where manifest-listed assets are
            extracted. Only assets declared in manifest.json are extracted.
        verify_hashes: Validate asset size and SHA-256 against manifest records.

    Returns:
        A ProjectArchive with deserialized project and manifest data.
    """
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            _validate_member_names(archive.namelist())
            manifest = _read_json_member(archive, MANIFEST_JSON_PATH)
            _validate_manifest(manifest)

            project_member = _manifest_entry_path(
                manifest,
                ("entries", "project"),
                PROJECT_JSON_PATH,
            )
            project_data = _read_json_member(archive, project_member)
            project = Project.from_dict(project_data)

            assets = tuple(
                ArchiveAsset.from_manifest_dict(asset)
                for asset in manifest.get("assets", [])
            )
            assets = _validate_manifest_assets(archive, assets, verify_hashes)

            plugin_state = {}
            plugin_state_path = manifest.get("entries", {}).get("plugin_state")
            if plugin_state_path:
                plugin_state = _read_json_member(
                    archive,
                    _safe_archive_path(plugin_state_path, must_be_asset=False),
                )

            if extract_assets_to:
                assets = _extract_manifest_assets(archive, assets, extract_assets_to)

            return ProjectArchive(
                project=project,
                manifest=manifest,
                assets=assets,
                plugin_state=plugin_state,
            )
    except ProjectArchiveError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ProjectArchiveError(f"Failed to load archive '{archive_path}': {exc}") from exc


def inspect_project_archive(archive_path: str) -> dict[str, Any]:
    """Read and validate only manifest.json from a .amus archive."""
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            _validate_member_names(archive.namelist())
            manifest = _read_json_member(archive, MANIFEST_JSON_PATH)
            _validate_manifest(manifest)
            return manifest
    except ProjectArchiveError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ProjectArchiveError(f"Failed to inspect archive '{archive_path}': {exc}") from exc


def validate_project_archive(archive_path: str) -> list[str]:
    """
    Validate a .amus archive.

    Returns an empty list when valid, otherwise a list of human-readable errors.
    """
    try:
        load_project_archive(archive_path, verify_hashes=True)
        return []
    except ProjectArchiveError as exc:
        return [str(exc)]


def _build_manifest(
    project: Project,
    project_data: dict[str, Any],
    assets: tuple[ArchiveAsset, ...],
    has_plugin_state: bool,
    plugin_state: dict[str, Any],
    extra_manifest: dict[str, Any],
) -> dict[str, Any]:
    entries: dict[str, str] = {
        "project": PROJECT_JSON_PATH,
        "assets": ASSETS_PREFIX,
    }
    if has_plugin_state:
        entries["plugin_state"] = PLUGIN_STATE_JSON_PATH

    manifest = {
        "format": ARCHIVE_FORMAT,
        "format_version": ARCHIVE_FORMAT_VERSION,
        "created_by": "amusiment-core",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "project": {
            "id": project.id,
            "name": project.metadata.name,
            "schema_version": project_data.get("schema_version", "1.0.0"),
            "metadata_version": project.metadata.version,
        },
        "assets": [asset.to_manifest_dict() for asset in assets],
        "plugin_state": {
            "count": len(plugin_state),
            "path": PLUGIN_STATE_JSON_PATH if has_plugin_state else "",
        },
        "extensions": extra_manifest,
    }
    return manifest


def _prepare_assets(assets: tuple[ArchiveAsset, ...]) -> tuple[ArchiveAsset, ...]:
    prepared: list[ArchiveAsset] = []
    seen_paths: set[str] = set()

    for asset in assets:
        if not asset.source_path:
            raise ProjectArchiveError("Asset source_path is required when saving")

        source = Path(asset.source_path)
        if not source.is_file():
            raise ProjectArchiveError(f"Asset file does not exist: {asset.source_path}")

        archive_path = _safe_archive_path(
            asset.archive_path or ASSETS_PREFIX + source.name,
            must_be_asset=True,
        )
        if archive_path in seen_paths:
            raise ProjectArchiveError(f"Duplicate asset archive path: {archive_path}")
        seen_paths.add(archive_path)

        size_bytes, digest = _file_size_and_sha256(source)
        media_type = asset.media_type or mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        prepared.append(
            ArchiveAsset(
                source_path=str(source),
                archive_path=archive_path,
                role=asset.role,
                media_type=media_type,
                size_bytes=size_bytes,
                sha256=digest,
            )
        )

    return tuple(prepared)


def _validate_manifest_assets(
    archive: zipfile.ZipFile,
    assets: tuple[ArchiveAsset, ...],
    verify_hashes: bool,
) -> tuple[ArchiveAsset, ...]:
    validated: list[ArchiveAsset] = []
    names = set(archive.namelist())

    for asset in assets:
        archive_path = _safe_archive_path(asset.archive_path, must_be_asset=True)
        if archive_path not in names:
            raise ProjectArchiveError(f"Asset declared in manifest is missing: {archive_path}")

        if verify_hashes:
            size_bytes, digest = _member_size_and_sha256(archive, archive_path)
            if asset.size_bytes and asset.size_bytes != size_bytes:
                raise ProjectArchiveError(
                    f"Asset size mismatch for {archive_path}: "
                    f"manifest={asset.size_bytes}, archive={size_bytes}"
                )
            if asset.sha256 and asset.sha256 != digest:
                raise ProjectArchiveError(f"Asset hash mismatch for {archive_path}")
            asset = ArchiveAsset(
                source_path=asset.source_path,
                archive_path=archive_path,
                role=asset.role,
                media_type=asset.media_type,
                size_bytes=size_bytes,
                sha256=digest,
            )
        else:
            asset = ArchiveAsset(
                source_path=asset.source_path,
                archive_path=archive_path,
                role=asset.role,
                media_type=asset.media_type,
                size_bytes=asset.size_bytes,
                sha256=asset.sha256,
            )
        validated.append(asset)

    return tuple(validated)


def _extract_manifest_assets(
    archive: zipfile.ZipFile,
    assets: tuple[ArchiveAsset, ...],
    output_dir: str,
) -> tuple[ArchiveAsset, ...]:
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    extracted: list[ArchiveAsset] = []

    for asset in assets:
        relative_path = _asset_relative_path(asset.archive_path)
        destination = destination_root / Path(*PurePosixPath(relative_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with archive.open(asset.archive_path, "r") as src, open(destination, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        extracted.append(
            ArchiveAsset(
                source_path=str(destination),
                archive_path=asset.archive_path,
                role=asset.role,
                media_type=asset.media_type,
                size_bytes=asset.size_bytes,
                sha256=asset.sha256,
            )
        )

    return tuple(extracted)


def _manifest_entry_path(
    manifest: dict[str, Any],
    path: tuple[str, str],
    default: str,
) -> str:
    current: Any = manifest
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return _safe_archive_path(str(current), must_be_asset=False)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != ARCHIVE_FORMAT:
        raise ProjectArchiveError("Unsupported or missing .amus archive format")

    if not manifest.get("format_version"):
        raise ProjectArchiveError("Archive manifest is missing format_version")

    project_path = _manifest_entry_path(
        manifest,
        ("entries", "project"),
        PROJECT_JSON_PATH,
    )
    if project_path != PROJECT_JSON_PATH:
        _safe_archive_path(project_path, must_be_asset=False)

    for asset in manifest.get("assets", []):
        if not isinstance(asset, dict):
            raise ProjectArchiveError("Manifest assets must be objects")
        _safe_archive_path(asset.get("path", ""), must_be_asset=True)


def _validate_member_names(names: Iterable[str]) -> None:
    for name in names:
        _safe_archive_path(name, must_be_asset=False, allow_directory=True)


def _safe_archive_path(
    value: str,
    must_be_asset: bool,
    allow_directory: bool = False,
) -> str:
    if not value:
        raise ProjectArchiveError("Archive path cannot be empty")

    raw = value.replace("\\", "/")
    if raw.startswith("/"):
        raise ProjectArchiveError(f"Archive path must be relative: {value}")
    raw_parts = PurePosixPath(raw.rstrip("/")).parts
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ProjectArchiveError(f"Unsafe archive path: {value}")

    normalized = posixpath.normpath(raw)
    if normalized == ".":
        raise ProjectArchiveError("Archive path cannot be current directory")
    if raw.endswith("/") and allow_directory:
        normalized = normalized.rstrip("/") + "/"
    elif raw.endswith("/"):
        raise ProjectArchiveError(f"Archive path cannot be a directory: {value}")

    parts = PurePosixPath(normalized.rstrip("/")).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ProjectArchiveError(f"Unsafe archive path: {value}")
    if normalized.startswith("../") or normalized == "..":
        raise ProjectArchiveError(f"Unsafe archive path: {value}")

    if must_be_asset:
        if normalized in (MANIFEST_JSON_PATH, PROJECT_JSON_PATH, PLUGIN_STATE_JSON_PATH):
            raise ProjectArchiveError(f"Asset path conflicts with reserved file: {value}")
        if not normalized.startswith(ASSETS_PREFIX):
            normalized = ASSETS_PREFIX + normalized
        if normalized == ASSETS_PREFIX or normalized.endswith("/"):
            raise ProjectArchiveError(f"Asset path must name a file: {value}")

    return normalized


def _asset_relative_path(archive_path: str) -> str:
    safe_path = _safe_archive_path(archive_path, must_be_asset=True)
    relative = safe_path[len(ASSETS_PREFIX):]
    if not relative:
        raise ProjectArchiveError(f"Asset path must name a file: {archive_path}")
    return relative


def _read_json_member(archive: zipfile.ZipFile, member_name: str) -> dict[str, Any]:
    safe_name = _safe_archive_path(member_name, must_be_asset=False)
    try:
        with archive.open(safe_name, "r") as file_obj:
            data = json.load(file_obj)
    except KeyError as exc:
        raise ProjectArchiveError(f"Archive is missing required member: {safe_name}") from exc
    if not isinstance(data, dict):
        raise ProjectArchiveError(f"JSON member must contain an object: {safe_name}")
    return data


def _writestr_json(archive: zipfile.ZipFile, member_name: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    archive.writestr(member_name, data)


def _file_size_and_sha256(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with open(path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
    return size, hasher.hexdigest()


def _member_size_and_sha256(archive: zipfile.ZipFile, member_name: str) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with archive.open(member_name, "r") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
    return size, hasher.hexdigest()
