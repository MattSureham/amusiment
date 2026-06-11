"""
Tests for the .amus project archive format.
"""

import hashlib
import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import (
    ARCHIVE_FORMAT,
    ArchiveAsset,
    ProjectArchiveError,
    inspect_project_archive,
    load_project_archive,
    save_project_archive,
    validate_project_archive,
)
from core.model.clip import AudioClip, MidiClip
from core.model.note import NoteEvent, NotePitch, NoteVelocity
from core.model.project import Project
from core.model.time_model import PPQ, Ticks
from core.model.track import AudioTrack, MidiTrack


def _demo_project() -> Project:
    project = Project.create_new(name="Archive Test", bpm=104.0)
    note = NoteEvent(
        pitch=NotePitch(60),
        velocity=NoteVelocity(96),
        start_tick=Ticks(0),
        duration_ticks=Ticks(PPQ),
    )
    clip = MidiClip(
        id="clip-main",
        name="Main Idea",
        start_tick=Ticks(PPQ * 2),
        length_ticks=Ticks(PPQ * 4),
        notes=(note,),
    )
    track = MidiTrack(id="track-midi", name="Keys").with_clip(clip)
    return project.with_track(track)


def test_project_archive_roundtrip_with_manifest():
    project = _demo_project()

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "song.amus")
        manifest = save_project_archive(
            project,
            output_path,
            plugin_state={"builtin.test": {"enabled": True}},
            extra_manifest={"ui": {"workspace": "compose"}},
        )

        assert os.path.exists(output_path)
        assert manifest["format"] == ARCHIVE_FORMAT
        assert manifest["project"]["id"] == project.id
        assert manifest["project"]["name"] == "Archive Test"
        assert manifest["entries"]["project"] == "project.json"
        assert manifest["entries"]["plugin_state"] == "plugin-state/state.json"

        inspected = inspect_project_archive(output_path)
        assert inspected["extensions"]["ui"]["workspace"] == "compose"

        loaded = load_project_archive(output_path)
        assert loaded.project.id == project.id
        assert loaded.project.metadata.name == "Archive Test"
        assert len(loaded.project.midi_tracks) == 1
        assert loaded.project.midi_tracks[0].clips[0].notes[0].pitch == 60
        assert loaded.plugin_state["builtin.test"]["enabled"] is True
        assert validate_project_archive(output_path) == []


def test_project_archive_packages_and_extracts_assets():
    project = Project.create_new(name="Asset Test", bpm=120.0)
    audio_clip = AudioClip(
        name="Reference",
        source_path="assets/audio/reference.wav",
        length_ticks=Ticks(PPQ * 4),
    )
    project = project.with_track(AudioTrack(name="Reference").with_clip(audio_clip))

    with tempfile.TemporaryDirectory() as tmp:
        source_path = os.path.join(tmp, "reference.wav")
        payload = b"RIFF....WAVEfmt fake audio bytes"
        with open(source_path, "wb") as f:
            f.write(payload)

        output_path = os.path.join(tmp, "asset-project.amus")
        save_project_archive(
            project,
            output_path,
            assets=(
                ArchiveAsset.from_file(
                    source_path,
                    archive_path="assets/audio/reference.wav",
                    role="audio",
                    media_type="audio/wav",
                ),
            ),
        )

        manifest = inspect_project_archive(output_path)
        assert len(manifest["assets"]) == 1
        asset_record = manifest["assets"][0]
        assert asset_record["path"] == "assets/audio/reference.wav"
        assert asset_record["size_bytes"] == len(payload)
        assert asset_record["sha256"] == hashlib.sha256(payload).hexdigest()

        extract_dir = os.path.join(tmp, "extracted")
        loaded = load_project_archive(output_path, extract_assets_to=extract_dir)
        assert len(loaded.assets) == 1
        assert loaded.assets[0].source_path.endswith(os.path.join("audio", "reference.wav"))
        with open(loaded.assets[0].source_path, "rb") as f:
            assert f.read() == payload


def test_project_archive_rejects_unsafe_asset_paths_on_save():
    project = Project.create_new(name="Unsafe Save", bpm=120.0)

    with tempfile.TemporaryDirectory() as tmp:
        source_path = os.path.join(tmp, "sample.wav")
        with open(source_path, "wb") as f:
            f.write(b"sample")

        try:
            save_project_archive(
                project,
                os.path.join(tmp, "unsafe.amus"),
                assets=(ArchiveAsset.from_file(source_path, archive_path="../escape.wav"),),
            )
            assert False, "unsafe asset path should have been rejected"
        except ProjectArchiveError:
            pass


def test_project_archive_rejects_unsafe_zip_members_on_load():
    project = Project.create_new(name="Unsafe Load", bpm=120.0)

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = os.path.join(tmp, "unsafe.amus")
        manifest = {
            "format": ARCHIVE_FORMAT,
            "format_version": "1.0.0",
            "entries": {"project": "project.json", "assets": "assets/"},
            "project": {"id": project.id, "name": project.metadata.name},
            "assets": [],
        }
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
            archive.writestr("project.json", json.dumps(project.to_dict()).encode("utf-8"))
            archive.writestr("../escape.txt", b"nope")

        try:
            load_project_archive(archive_path)
            assert False, "unsafe archive member should have been rejected"
        except ProjectArchiveError:
            pass


def test_project_archive_validation_reports_hash_mismatch():
    project = Project.create_new(name="Hash Mismatch", bpm=120.0)

    with tempfile.TemporaryDirectory() as tmp:
        source_path = os.path.join(tmp, "sample.wav")
        with open(source_path, "wb") as f:
            f.write(b"original")

        archive_path = os.path.join(tmp, "hash.amus")
        save_project_archive(
            project,
            archive_path,
            assets=(ArchiveAsset.from_file(source_path, archive_path="assets/sample.wav"),),
        )

        corrupt_path = os.path.join(tmp, "hash-corrupt.amus")
        with zipfile.ZipFile(archive_path, "r") as source_archive:
            with zipfile.ZipFile(corrupt_path, "w") as corrupt_archive:
                for member in source_archive.infolist():
                    payload = source_archive.read(member.filename)
                    if member.filename == "assets/sample.wav":
                        payload = b"tampered"
                    corrupt_archive.writestr(member.filename, payload)

        errors = validate_project_archive(corrupt_path)
        assert errors
        assert "hash mismatch" in errors[0].lower()


if __name__ == "__main__":
    test_project_archive_roundtrip_with_manifest()
    print("  ok: roundtrip_with_manifest")
    test_project_archive_packages_and_extracts_assets()
    print("  ok: packages_and_extracts_assets")
    test_project_archive_rejects_unsafe_asset_paths_on_save()
    print("  ok: rejects_unsafe_asset_paths_on_save")
    test_project_archive_rejects_unsafe_zip_members_on_load()
    print("  ok: rejects_unsafe_zip_members_on_load")
    test_project_archive_validation_reports_hash_mismatch()
    print("  ok: validation_reports_hash_mismatch")
