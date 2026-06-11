"""
Basic music analyzer — an AIAnalyzerPlugin implementation.

Provides fundamental music analysis capabilities without requiring
external ML models. Uses deterministic music theory rules for:
- Key detection from MIDI notes
- Chord identification
- Basic rhythm analysis
- Style classification hints

These analyzers work as building blocks for more advanced AI analyzers
and provide immediate value even without large models loaded.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from core.plugin.interfaces.base import PluginManifest, PluginCategory, PluginState
from core.plugin.interfaces.ai_analyzer import (
    AIAnalyzerPlugin, AnalyzerCapabilities, AnalysisRequest, AnalysisResult,
)
from core.model.note import NoteEvent, NotePitch
from core.model.clip import MidiClip
from core.model.time_model import Ticks

from .theory import (
    scale_degrees, chord_tones, classify_chord_intervals,
    chord_name, pc_to_key_name, note_in_scale,
    SCALES, PITCH_CLASSES, PITCH_CLASSES_FLAT,
)


def _detect_key(notes: list[NoteEvent]) -> dict:
    """Detect the most likely key from a set of notes.

    Algorithm:
    1. Count pitch class occurrences, weighted by duration
    2. Score each possible key by how many notes fit the scale
    3. Return the best matching key with confidence
    """
    if not notes:
        return {"key": "C major", "tonic_pc": 0, "mode": "major",
                "confidence": 0.0, "explanation": "No notes to analyze"}

    # Weighted pitch class distribution
    pc_counts = Counter()
    for note in notes:
        pc = note.pitch % 12
        # Weight by duration and velocity
        weight = note.duration_ticks * note.velocity / 127.0
        pc_counts[pc] += weight

    # Normalize
    total = sum(pc_counts.values()) or 1
    pc_distribution = {pc: count / total for pc, count in pc_counts.items()}

    # Score each candidate key
    candidates = []
    for tonic_pc in range(12):
        for mode in ("major", "minor"):
            if mode == "major":
                scale_name = "major"
            else:
                scale_name = "natural_minor"

            scale_pcs = set(scale_degrees(tonic_pc, scale_name))

            # Score: percentage of weighted notes that fit the scale
            in_scale = sum(
                weight for pc, weight in pc_distribution.items()
                if pc in scale_pcs
            )
            out_scale = sum(
                weight for pc, weight in pc_distribution.items()
                if pc not in scale_pcs
            )

            # Penalize missing scale tones (if a degree is never played)
            missing_tones = sum(1 for spc in scale_pcs if spc not in pc_counts)
            missing_penalty = missing_tones * 0.02

            score = in_scale - out_scale * 0.5 - missing_penalty
            key_name = pc_to_key_name(tonic_pc, mode)
            candidates.append((score, tonic_pc, mode, key_name))

    candidates.sort(reverse=True)
    best_score, best_pc, best_mode, best_name = candidates[0]

    # Confidence: normalized score
    max_possible = 1.0
    confidence = min(1.0, max(0.0, best_score / max_possible))

    # Find alternatives
    alt_keys = []
    for score, pc, mode, name in candidates[1:6]:
        if score > best_score * 0.7:
            alt_keys.append(name)

    explanation = (
        f"Detected key: {best_name} (confidence: {confidence:.0%}). "
        f"{len(alt_keys)} alternative keys considered: {', '.join(alt_keys[:3])}."
        if alt_keys else
        f"Detected key: {best_name} (confidence: {confidence:.0%})."
    )

    return {
        "key": best_name,
        "tonic_pc": best_pc,
        "mode": best_mode,
        "confidence": confidence,
        "explanation": explanation,
        "alternatives": alt_keys,
        "pitch_class_distribution": {
            PITCH_CLASSES[pc]: round(w, 3)
            for pc, w in sorted(pc_distribution.items())
        },
    }


def _identify_chords(notes: list[NoteEvent],
                     start_tick: Ticks = Ticks(0),
                     end_tick: Ticks = Ticks(0)) -> dict:
    """Identify chords from simultaneous/serial note groups."""
    if not notes:
        return {"chords": [], "explanation": "No notes to analyze"}

    # Group notes by onset time (within a small window)
    WINDOW_TICKS = 120  # ~32nd note at PPQ=960

    # Sort by start tick
    sorted_notes = sorted(notes, key=lambda n: n.start_tick)

    # Group into onset clusters
    clusters = []
    current_cluster = []
    current_onset = None

    for note in sorted_notes:
        if current_onset is None:
            current_onset = note.start_tick
            current_cluster = [note]
        elif note.start_tick - current_onset <= WINDOW_TICKS:
            current_cluster.append(note)
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_onset = note.start_tick
            current_cluster = [note]

    if current_cluster:
        clusters.append(current_cluster)

    # Identify chords from clusters
    identified = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue  # skip single notes

        # Get unique pitch classes
        pcs = sorted(set(n.pitch % 12 for n in cluster))
        if len(pcs) < 2:
            continue

        # Build intervals from lowest note
        root_pc = pcs[0]
        intervals = [(pc - root_pc) % 12 for pc in pcs]

        chord_type = classify_chord_intervals(intervals)
        name = chord_name(root_pc, chord_type)

        if chord_type != "unknown":
            identified.append({
                "name": name,
                "root_pc": root_pc,
                "type": chord_type,
                "tick": cluster[0].start_tick,
                "notes": len(cluster),
            })

    explanation = (
        f"Identified {len(identified)} chords from {len(notes)} notes. "
        if identified else
        f"No clear chords identified from {len(notes)} notes. "
        f"Try longer note durations or more simultaneous notes."
    )

    return {
        "chords": identified,
        "count": len(identified),
        "explanation": explanation,
    }


def _analyze_rhythm(notes: list[NoteEvent]) -> dict:
    """Analyze rhythmic characteristics of a set of notes."""
    if not notes:
        return {"explanation": "No notes to analyze"}

    sorted_notes = sorted(notes, key=lambda n: n.start_tick)

    # Calculate inter-onset intervals
    iois = []
    for i in range(1, len(sorted_notes)):
        ioi = sorted_notes[i].start_tick - sorted_notes[i-1].start_tick
        if ioi > 0:
            iois.append(ioi)

    if not iois:
        return {
            "explanation": "Only one note found — no rhythm to analyze",
            "note_count": len(notes),
        }

    avg_ioi = sum(iois) / len(iois)

    # Classify rhythmic density
    from core.model.time_model import PPQ
    quarter_note_ticks = PPQ  # 960

    if avg_ioi < quarter_note_ticks * 0.25:
        density = "very_dense"
        density_label = "Very dense (32nd note level)"
    elif avg_ioi < quarter_note_ticks * 0.5:
        density = "dense"
        density_label = "Dense (16th note level)"
    elif avg_ioi < quarter_note_ticks * 1.0:
        density = "moderate"
        density_label = "Moderate (8th note level)"
    elif avg_ioi < quarter_note_ticks * 2.0:
        density = "sparse"
        density_label = "Sparse (quarter note level)"
    else:
        density = "very_sparse"
        density_label = "Very sparse (half note or longer)"

    # Detect swing/groove: look for alternating short-long patterns
    has_swing = False
    if len(iois) >= 4:
        # Check for 2:1 ratio pattern (triplet feel)
        triplet_pairs = 0
        total_pairs = 0
        for i in range(0, len(iois) - 1, 2):
            if i + 1 < len(iois):
                total_pairs += 1
                a, b = iois[i], iois[i+1]
                if a > 0 and b > 0:
                    ratio = max(a, b) / max(min(a, b), 1)
                    if 1.5 < ratio < 3.0:
                        triplet_pairs += 1
        has_swing = triplet_pairs > total_pairs * 0.3

    # Detect syncopation: check for notes off the grid
    syncopation_count = 0
    for ioi in iois:
        remainder = ioi % (quarter_note_ticks // 2)  # 8th note grid
        if 10 < remainder < (quarter_note_ticks // 2) - 10:
            syncopation_count += 1

    syncopation_ratio = syncopation_count / len(iois) if iois else 0
    is_syncopated = syncopation_ratio > 0.2

    explanation = (
        f"Rhythm analysis: {len(notes)} notes. "
        f"Density: {density_label}. "
        f"Swing feel: {'Yes' if has_swing else 'No'}. "
        f"Syncopation: {'Yes' if is_syncopated else 'No'} "
        f"({syncopation_ratio:.0%})."
    )

    return {
        "note_count": len(notes),
        "density": density,
        "density_label": density_label,
        "average_ioi_ticks": round(avg_ioi, 1),
        "has_swing": has_swing,
        "is_syncopated": is_syncopated,
        "syncopation_ratio": round(syncopation_ratio, 3),
        "explanation": explanation,
    }


# ── The plugin ──────────────────────────────────────────────────────

class BasicAnalyzerPlugin(AIAnalyzerPlugin):
    """
    Basic music theory analyzer.

    Provides key detection, chord identification, and rhythm analysis
    using deterministic music theory rules. No ML models required.
    """

    def __init__(self):
        self._initialized = False

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="amusiment.basic-analyzer",
            name="Basic Music Analyzer",
            version="1.0.0",
            category=PluginCategory.AI_ANALYZER,
            author="amusiment",
            description="Deterministic music analysis: key detection, "
                        "chord identification, and rhythm analysis "
                        "without external ML dependencies.",
            capabilities=[
                "analyze.key",
                "analyze.chords",
                "analyze.rhythm",
            ],
        )

    def initialize(self) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def get_capabilities(self) -> list[AnalyzerCapabilities]:
        return [
            AnalyzerCapabilities.KEY_DETECTION,
            AnalyzerCapabilities.CHORD_ANALYSIS,
            AnalyzerCapabilities.RHYTHM_ANALYSIS,
            AnalyzerCapabilities.STYLE_CLASSIFICATION,
        ]

    def analyze(self, clips: dict[str, list[MidiClip]],
                request: AnalysisRequest) -> AnalysisResult:
        """Run requested analyses on the given clips."""
        findings = {}
        suggestions = []
        total_confidence = 0.0
        num_analyses = 0

        # Collect all notes
        all_notes = []
        for track_clips in clips.values():
            for clip in track_clips:
                # Filter by range if specified
                if request.clip_range_start > 0 or request.clip_range_end > 0:
                    start = request.clip_range_start or Ticks(0)
                    end = request.clip_range_end or Ticks(99999999)
                    all_notes.extend(clip.notes_in_range(start, end))
                else:
                    all_notes.extend(clip.notes)

        capabilities = request.capabilities_requested or [
            AnalyzerCapabilities.KEY_DETECTION,
            AnalyzerCapabilities.CHORD_ANALYSIS,
            AnalyzerCapabilities.RHYTHM_ANALYSIS,
        ]

        # Key detection
        if AnalyzerCapabilities.KEY_DETECTION in capabilities:
            result = _detect_key(all_notes)
            findings[AnalyzerCapabilities.KEY_DETECTION] = result
            total_confidence += result.get("confidence", 0.0)
            num_analyses += 1
            if result.get("confidence", 0) < 0.5:
                suggestions.append(
                    "Key detection confidence is low. Consider adding more "
                    "melodic content or specifying the key manually."
                )

        # Chord analysis
        if AnalyzerCapabilities.CHORD_ANALYSIS in capabilities:
            result = _identify_chords(
                all_notes,
                request.clip_range_start or Ticks(0),
                request.clip_range_end or Ticks(0),
            )
            findings[AnalyzerCapabilities.CHORD_ANALYSIS] = result
            total_confidence += (0.8 if result.get("chords") else 0.3)
            num_analyses += 1
            if not result.get("chords"):
                suggestions.append(
                    "No chords detected. Try notes with longer durations "
                    "or more simultaneous notes."
                )

        # Rhythm analysis
        if AnalyzerCapabilities.RHYTHM_ANALYSIS in capabilities:
            result = _analyze_rhythm(all_notes)
            findings[AnalyzerCapabilities.RHYTHM_ANALYSIS] = result
            total_confidence += 0.8
            num_analyses += 1

        # Try to classify style
        if AnalyzerCapabilities.STYLE_CLASSIFICATION in capabilities:
            style_hints = self._classify_style(all_notes)
            findings[AnalyzerCapabilities.STYLE_CLASSIFICATION] = style_hints
            total_confidence += style_hints.get("confidence", 0.3)
            num_analyses += 1

        # Build summary
        parts = []
        key_finding = findings.get(AnalyzerCapabilities.KEY_DETECTION, {})
        chord_finding = findings.get(AnalyzerCapabilities.CHORD_ANALYSIS, {})
        rhythm_finding = findings.get(AnalyzerCapabilities.RHYTHM_ANALYSIS, {})

        if key_finding:
            parts.append(f"Key: {key_finding.get('key', 'unknown')}")
        if chord_finding:
            parts.append(f"Chords: {chord_finding.get('count', 0)} identified")
        if rhythm_finding:
            parts.append(f"Rhythm: {rhythm_finding.get('density_label', 'unknown')}")

        summary = " | ".join(parts) if parts else "Analysis complete."

        avg_confidence = total_confidence / max(num_analyses, 1)

        return AnalysisResult(
            findings=findings,
            summary=summary,
            confidence=avg_confidence,
            suggestions=suggestions,
            details={
                "note_count": len(all_notes),
                "analyses_performed": num_analyses,
            },
        )

    def _classify_style(self, notes: list[NoteEvent]) -> dict:
        """Heuristic style classification based on note characteristics."""
        if not notes:
            return {"style": "unknown", "confidence": 0.0}

        sorted_notes = sorted(notes, key=lambda n: n.start_tick)
        velocities = [n.velocity for n in notes]
        durations = [n.duration_ticks for n in notes]

        avg_vel = sum(velocities) / len(velocities) if velocities else 0
        avg_dur = sum(durations) / len(durations) if durations else 0

        from core.model.time_model import PPQ

        styles = {}

        # Check for classical: longer durations, wider velocity range
        if avg_dur > PPQ * 1.5 and max(velocities) - min(velocities) > 40:
            styles["classical"] = 0.6

        # Check for jazz: swung rhythm check, moderate dynamics
        iois = []
        for i in range(1, len(sorted_notes)):
            ioi = sorted_notes[i].start_tick - sorted_notes[i-1].start_tick
            if ioi > 0:
                iois.append(ioi)

        has_triplet_feel = False
        if len(iois) >= 4:
            for i in range(0, len(iois) - 1, 2):
                if i + 1 < len(iois) and iois[i] > 0 and iois[i+1] > 0:
                    ratio = max(iois[i], iois[i+1]) / max(min(iois[i], iois[i+1]), 1)
                    if 1.4 < ratio < 2.5:
                        has_triplet_feel = True
                        break

        if has_triplet_feel:
            styles["jazz"] = 0.55
            styles["lofi"] = 0.5

        # Check for EDM: high velocity consistency, steady rhythm
        vel_std = (sum((v - avg_vel) ** 2 for v in velocities) / len(velocities)) ** 0.5
        if vel_std < 15 and avg_vel > 90:
            styles["edm"] = 0.5

        # Check for pop/rock: moderate everything
        styles["pop"] = 0.4

        best_style = max(styles.items(), key=lambda x: x[1])
        alternatives = [s for s in styles if s != best_style[0]][:3]

        return {
            "style": best_style[0],
            "confidence": best_style[1],
            "alternatives": alternatives,
            "all_scores": styles,
        }
