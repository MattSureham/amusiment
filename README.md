# amusiment

amusiment is an early-stage AI-driven music composition framework. It contains the core project model, state and history system, plugin interfaces, MIDI import/export, a lightweight sequencing engine, and rule-based AI composition/analysis plugins.

## What is included

- Core immutable-ish music data models for projects, tracks, clips, notes, automation, devices, mixer state, tempo maps, and project archives.
- MIDI import and export built with the Python standard library.
- Engine primitives for transport, playback clocking, event scheduling, and rendering MIDI events from projects.
- Plugin interfaces for instruments, effects, AI generators, AI analyzers, UI widgets, importers, and exporters.
- Built-in AI models for chord progressions, melodies, drums, and basic musical analysis.
- A placeholder UI package for future shell, panels, design system, and widgets.

## Repository Status

This codebase currently has no runtime dependencies beyond Python's standard library. Tests use `pytest`.

The UI package is scaffolded but not yet implemented as an application.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest
```

## License

No license has been selected yet. Until a license is added, all rights are reserved by the repository owner.
