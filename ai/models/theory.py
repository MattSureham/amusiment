"""
Music theory foundation for AI composition.

Provides scales, chords, progressions, and voice-leading utilities
that all AI generators build upon. No external dependencies — pure
deterministic music theory in Python.

Design principles:
- All pitch classes are 0-11 (C=0, C#=1, ..., B=11).
- MIDI note numbers follow standard: middle C (C4) = 60.
- Octaves increment every 12 semitones.
- Chord symbols use standard notation: "Cmaj7", "Dm7", "G7", etc.
"""

from dataclasses import dataclass, field
from typing import Optional
from random import Random

# ── Pitch & interval constants ──────────────────────────────────────

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_CLASSES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Standard MIDI octave: C4 = 60
MIDI_MIDDLE_C = 60


def pc_to_midi(pitch_class: int, octave: int) -> int:
    """Convert pitch class (0-11) + octave to MIDI note number."""
    return (octave + 1) * 12 + pitch_class


def midi_to_pc(midi_note: int) -> tuple[int, int]:
    """Convert MIDI note number to (pitch_class, octave)."""
    return midi_note % 12, midi_note // 12 - 1


# ── Scale definitions ───────────────────────────────────────────────

# Each scale is defined as a list of semitone offsets from the tonic.
SCALES: dict[str, list[int]] = {
    "major":             [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":      [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":     [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":      [0, 2, 3, 5, 7, 9, 11],
    "dorian":             [0, 2, 3, 5, 7, 9, 10],
    "phrygian":           [0, 1, 3, 5, 7, 8, 10],
    "lydian":             [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":         [0, 2, 4, 5, 7, 9, 10],
    "locrian":            [0, 1, 3, 5, 6, 8, 10],
    "major_pentatonic":   [0, 2, 4, 7, 9],
    "minor_pentatonic":   [0, 3, 5, 7, 10],
    "blues":              [0, 3, 5, 6, 7, 10],
    "whole_tone":         [0, 2, 4, 6, 8, 10],
    "chromatic":          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# Scale degree -> chord quality for diatonic chords
MAJOR_DIATONIC_QUALITIES = ["maj", "min", "min", "maj", "maj", "min", "dim"]
MINOR_DIATONIC_QUALITIES = ["min", "dim", "maj", "min", "min", "maj", "maj"]
HARMONIC_MINOR_QUALITIES = ["min", "dim", "aug", "min", "maj", "maj", "dim"]

# Roman numeral notation
ROMAN_NUMERALS_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "vii\xb0"]
ROMAN_NUMERALS_MINOR = ["i", "ii\xb0", "III", "iv", "v", "VI", "VII"]


# ── Chord type definitions ──────────────────────────────────────────

# Chord formulas: name -> list of semitone offsets from root
CHORD_TYPES: dict[str, list[int]] = {
    "maj":      [0, 4, 7],
    "min":      [0, 3, 7],
    "dim":      [0, 3, 6],
    "aug":      [0, 4, 8],
    "sus2":     [0, 2, 7],
    "sus4":     [0, 5, 7],
    "maj7":     [0, 4, 7, 11],
    "min7":     [0, 3, 7, 10],
    "7":        [0, 4, 7, 10],
    "m7b5":     [0, 3, 6, 10],
    "dim7":     [0, 3, 6, 9],
    "minMaj7":  [0, 3, 7, 11],
    "maj9":     [0, 4, 7, 11, 14],
    "min9":     [0, 3, 7, 10, 14],
    "9":        [0, 4, 7, 10, 14],
    "7b9":      [0, 4, 7, 10, 13],
    "maj6":     [0, 4, 7, 9],
    "min6":     [0, 3, 7, 9],
}

CHORD_SYMBOL_SUFFIX: dict[str, str] = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4",
    "maj7": "maj7", "min7": "m7", "7": "7",
    "m7b5": "m7b5", "dim7": "dim7", "minMaj7": "mMaj7",
    "maj9": "maj9", "min9": "m9", "9": "9",
    "7b9": "7b9", "maj6": "6", "min6": "m6",
}


def scale_degrees(tonic_pc: int, scale_name: str) -> list[int]:
    """Get the pitch classes of a scale given the tonic."""
    intervals = SCALES.get(scale_name, SCALES["major"])
    return [(tonic_pc + i) % 12 for i in intervals]


def scale_notes(tonic_pc: int, scale_name: str, octave: int = 4,
                num_octaves: int = 2) -> list[int]:
    """Get MIDI note numbers for a scale across multiple octaves."""
    intervals = SCALES.get(scale_name, SCALES["major"])
    result = []
    for o in range(octave, octave + num_octaves):
        base = pc_to_midi(tonic_pc, o)
        for i in intervals:
            offset = i - tonic_pc
            if offset < 0:
                offset += 12
            note = base + offset
            if note not in result:
                result.append(note)
    return sorted(result)


def chord_tones(root_pc: int, chord_type: str) -> list[int]:
    """Get pitch classes for a chord type from a root."""
    intervals = CHORD_TYPES.get(chord_type, CHORD_TYPES["maj"])
    return [(root_pc + i) % 12 for i in intervals]


def chord_name(root_pc: int, chord_type: str) -> str:
    """Human-readable chord name, e.g. 'Cmaj7', 'Dm7b5'."""
    root_name = PITCH_CLASSES[root_pc]
    suffix = CHORD_SYMBOL_SUFFIX.get(chord_type, "")
    return f"{root_name}{suffix}"


def parse_chord_symbol(symbol: str) -> tuple[int, str]:
    """Parse a chord symbol like 'Cmaj7' into (root_pc, chord_type)."""
    root_name = symbol[0]
    offset = 1
    if len(symbol) > 1 and symbol[1] in ("#", "b"):
        root_name = symbol[:2]
        offset = 2

    root_map = {name: idx for idx, name in enumerate(PITCH_CLASSES)}
    for idx, name in enumerate(PITCH_CLASSES_FLAT):
        if name not in root_map:
            root_map[name] = idx

    if root_name not in root_map:
        raise ValueError(f"Unknown chord root: {root_name}")

    root_pc = root_map[root_name]
    suffix = symbol[offset:]

    suffix_map = {
        "": "maj", "maj": "maj", "M": "maj",
        "m": "min", "min": "min",
        "dim": "dim", "aug": "aug",
        "sus2": "sus2", "sus4": "sus4",
        "maj7": "maj7", "M7": "maj7", "Maj7": "maj7",
        "m7": "min7", "min7": "min7",
        "7": "7",
        "m7b5": "m7b5", "dim7": "dim7",
        "mMaj7": "minMaj7", "mM7": "minMaj7",
        "maj9": "maj9", "M9": "maj9",
        "m9": "min9",
        "9": "9",
        "7b9": "7b9",
        "6": "maj6", "m6": "min6",
    }

    if suffix not in suffix_map:
        raise ValueError(f"Unknown chord suffix: {suffix}")

    return root_pc, suffix_map[suffix]


# ── Diatonic chords ─────────────────────────────────────────────────

@dataclass
class DiatonicChord:
    """A chord that belongs to a key/scale."""
    degree: int          # 1-7, scale degree
    root_pc: int         # Root pitch class
    quality: str         # Chord type
    roman: str           # Roman numeral
    pitch_classes: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return chord_name(self.root_pc, self.quality)

    @property
    def full_name(self) -> str:
        return f"{self.roman} ({self.name})"


def build_triad_from_scale(root_pc: int, scale_intervals: list[int]) -> list[int]:
    """Build a triad by stacking thirds within the scale."""
    result = [0]
    root_idx = scale_intervals.index(root_pc)
    third_idx = (root_idx + 2) % 7
    fifth_idx = (root_idx + 4) % 7
    third = (scale_intervals[third_idx] - root_pc) % 12
    if third == 0:
        third = 12
    fifth = (scale_intervals[fifth_idx] - root_pc) % 12
    if fifth <= third:
        fifth += 12
    result.append(third)
    result.append(fifth)
    return result


def build_seventh_from_scale(root_pc: int, scale_intervals: list[int]) -> list[int]:
    """Build a seventh chord by stacking thirds within the scale."""
    result = [0]
    root_idx = scale_intervals.index(root_pc)
    prev = 0
    for offset in [2, 4, 6]:
        idx = (root_idx + offset) % 7
        interval = (scale_intervals[idx] - root_pc) % 12
        if interval == 0:
            interval = 12
        if interval <= prev:
            interval += 12
        result.append(interval)
        prev = interval
    return result


def classify_chord_intervals(intervals: list[int]) -> str:
    """Classify a set of chord intervals into a chord type name."""
    norm = sorted(set(i % 12 for i in intervals))
    for ctype, pattern in CHORD_TYPES.items():
        if norm == sorted(pattern):
            return ctype
    # Heuristic fallback
    if len(norm) == 3:
        if norm == [0, 3, 6]: return "dim"
        if norm == [0, 3, 7]: return "min"
        if norm == [0, 4, 7]: return "maj"
        if norm == [0, 4, 8]: return "aug"
    if len(norm) == 4:
        if norm == [0, 4, 7, 10]: return "7"
        if norm == [0, 3, 7, 10]: return "min7"
        if norm == [0, 4, 7, 11]: return "maj7"
        if norm == [0, 3, 6, 10]: return "m7b5"
        if norm == [0, 3, 6, 9]: return "dim7"
    return "unknown"


def diatonic_chords(tonic_pc: int, scale_name: str = "major",
                    use_sevenths: bool = True) -> list[DiatonicChord]:
    """Get all diatonic chords for a given key and scale."""
    intervals = SCALES.get(scale_name, SCALES["major"])

    if scale_name == "major":
        qualities = MAJOR_DIATONIC_QUALITIES
        numerals = ROMAN_NUMERALS_MAJOR
    elif scale_name in ("natural_minor", "minor"):
        qualities = MINOR_DIATONIC_QUALITIES
        numerals = ROMAN_NUMERALS_MINOR
    elif scale_name == "harmonic_minor":
        qualities = HARMONIC_MINOR_QUALITIES
        numerals = ROMAN_NUMERALS_MINOR
    else:
        qualities = MAJOR_DIATONIC_QUALITIES
        numerals = ROMAN_NUMERALS_MAJOR

    chords = []
    for deg in range(7):
        root = intervals[deg]
        quality = qualities[deg]
        roman = numerals[deg]

        if use_sevenths:
            chord_intervals = build_seventh_from_scale(root, intervals)
            chord_type = classify_chord_intervals(chord_intervals)
        else:
            chord_intervals = build_triad_from_scale(root, intervals)
            chord_type = classify_chord_intervals(chord_intervals)

        chords.append(DiatonicChord(
            degree=deg + 1,
            root_pc=root,
            quality=chord_type,
            roman=roman,
            pitch_classes=[(root + i) % 12 for i in chord_intervals],
        ))

    return chords


# ── Common chord progressions ───────────────────────────────────────

StyleProgression = list[tuple[int, Optional[str], int]]

COMMON_PROGRESSIONS: dict[str, list[StyleProgression]] = {
    "pop": [
        [(1, None, 4), (5, None, 4), (6, None, 4), (4, None, 4)],
        [(6, None, 4), (4, None, 4), (1, None, 4), (5, None, 4)],
        [(1, None, 4), (6, None, 4), (4, None, 4), (5, None, 4)],
        [(4, None, 4), (1, None, 4), (5, None, 4), (6, None, 4)],
        [(1, None, 4), (4, None, 4), (6, None, 4), (5, None, 4)],
    ],
    "jazz": [
        [(2, "min7", 4), (5, "7", 4), (1, "maj7", 8)],
        [(1, "maj7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4)],
        [(3, "min7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4)],
        [(2, "min7", 4), (5, "7", 4), (1, "maj7", 4), (6, "min7", 4)],
    ],
    "classical": [
        [(1, None, 8), (4, None, 8), (5, None, 8), (1, None, 8)],
        [(1, None, 4), (2, None, 4), (5, None, 4), (1, None, 4)],
        [(1, None, 4), (4, None, 4), (2, None, 4), (5, None, 4), (1, None, 8)],
    ],
    "lofi": [
        [(1, "maj7", 4), (6, "min7", 4), (4, "maj7", 4), (5, "7", 4)],
        [(3, "min7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4)],
        [(1, "maj7", 4), (2, "min7", 4), (5, "7", 4), (1, "maj7", 4)],
        [(6, "min7", 4), (4, "maj7", 4), (1, "maj7", 4), (5, "7", 4)],
    ],
    "edm": [
        [(1, "min", 4), (6, "maj", 4), (3, "maj", 4), (7, "maj", 4)],
        [(1, "min", 4), (6, "maj", 4), (4, "min", 4), (5, "maj", 4)],
        [(6, "maj", 4), (7, "maj", 4), (1, "min", 8)],
        [(1, "min", 4), (3, "maj", 4), (7, "maj", 4), (6, "maj", 4)],
    ],
    "rnb": [
        [(1, "maj7", 4), (6, "min7", 4), (2, "min7", 4), (5, "7", 4)],
        [(1, "maj7", 4), (4, "maj7", 4), (3, "min7", 4), (6, "min7", 4)],
        [(2, "min7", 8), (5, "7", 8), (1, "maj7", 8)],
    ],
    "rock": [
        [(1, None, 8), (4, None, 8), (5, None, 8)],
        [(1, "min", 4), (6, "maj", 4), (7, "maj", 4)],
        [(1, None, 4), (5, None, 4), (6, None, 4), (4, None, 4)],
    ],
    "blues": [
        [(1, "7", 16), (4, "7", 8), (1, "7", 8), (5, "7", 4), (4, "7", 4), (1, "7", 8)],
    ],
}


def get_style_progression(style: str, rng: Optional[Random] = None) -> StyleProgression:
    """Get a random chord progression for a given style."""
    if rng is None:
        rng = Random()
    progressions = COMMON_PROGRESSIONS.get(style.lower(),
                                           COMMON_PROGRESSIONS["pop"])
    return rng.choice(progressions)


# ── Voice leading ───────────────────────────────────────────────────

def voice_lead(chord1_tones: list[int], chord2_tones: list[int],
               octave: int = 4) -> list[int]:
    """Simple voice leading: move each voice to nearest tone in next chord."""
    midi1 = [pc_to_midi(pc, octave) for pc in chord1_tones]
    result = []
    for note in midi1:
        best, best_dist = None, 999
        for pc2 in chord2_tones:
            for o2 in range(octave - 1, octave + 3):
                candidate = pc_to_midi(pc2, o2)
                dist = abs(candidate - note)
                if dist < best_dist:
                    best_dist, best = dist, candidate
        if best is not None and best not in result:
            result.append(best)
    for pc2 in chord2_tones:
        midi = pc_to_midi(pc2, octave)
        if midi not in result:
            result.append(midi)
    return sorted(result)


# ── Rhythm utilities ────────────────────────────────────────────────

def rhythm_pattern(style: str, beats: int, rng: Optional[Random] = None) -> list[float]:
    """Generate a rhythmic pattern as beat positions."""
    if rng is None:
        rng = Random()

    if style == "steady":
        return [float(b) for b in range(beats)]
    elif style == "syncopated":
        positions = []
        for b in range(beats):
            for eighth in [0.0, 0.5]:
                if rng.random() < 0.6:
                    positions.append(b + eighth)
        return sorted(set(positions)) or [float(b) for b in range(beats)]
    elif style == "sparse":
        positions = []
        for b in range(0, beats, 2):
            positions.append(float(b))
            if rng.random() < 0.3:
                positions.append(b + 1.0)
        return positions
    elif style == "dense":
        positions = []
        for b in range(beats):
            for sixteenth in [0.0, 0.25, 0.5, 0.75]:
                if rng.random() < 0.5:
                    positions.append(b + sixteenth)
        return sorted(set(positions)) or [float(b) for b in range(beats)]
    elif style == "swing":
        positions = []
        for b in range(beats):
            positions.append(float(b))
            if rng.random() < 0.7:
                positions.append(b + 0.66)
        return sorted(positions)
    else:
        return [float(b) for b in range(beats)]


# ── Note utilities ──────────────────────────────────────────────────

def note_in_scale(pitch_class: int, tonic_pc: int, scale_name: str) -> bool:
    """Check if a pitch class belongs to a scale."""
    return pitch_class in scale_degrees(tonic_pc, scale_name)


def closest_scale_note(pitch_class: int, tonic_pc: int, scale_name: str) -> int:
    """Find the closest scale note to a given pitch class."""
    scale_pcs = scale_degrees(tonic_pc, scale_name)
    best, best_dist = scale_pcs[0], 13
    for pc in scale_pcs:
        dist = min((pitch_class - pc) % 12, (pc - pitch_class) % 12)
        if dist < best_dist:
            best_dist, best = dist, pc
    return best


def note_is_chord_tone(pitch_class: int, chord_root_pc: int,
                       chord_type: str) -> bool:
    """Check if a pitch class is a chord tone of the given chord."""
    return pitch_class in chord_tones(chord_root_pc, chord_type)


def key_name_to_pc(key_name: str) -> tuple[int, str]:
    """Parse key name like 'C major' or 'Eb minor' into (tonic_pc, mode)."""
    parts = key_name.strip().split()
    tonic_str = parts[0]
    mode = parts[1] if len(parts) > 1 else "major"
    try:
        pc = PITCH_CLASSES.index(tonic_str)
    except ValueError:
        try:
            pc = PITCH_CLASSES_FLAT.index(tonic_str)
        except ValueError:
            pc = 0
    mode = "minor" if mode.lower() in ("minor", "min") else "major"
    return pc, mode


def pc_to_key_name(tonic_pc: int, mode: str = "major") -> str:
    """Convert tonic PC + mode to human-readable key name."""
    if mode in ("minor", "min", "natural_minor"):
        return f"{PITCH_CLASSES_FLAT[tonic_pc]} minor"
    else:
        return f"{PITCH_CLASSES[tonic_pc]} major"


def interval_semitones(interval_name: str) -> int:
    """Convert interval name to semitones."""
    intervals = {
        "P1": 0, "m2": 1, "M2": 2, "m3": 3, "M3": 4,
        "P4": 5, "TT": 6, "A4": 6, "d5": 6,
        "P5": 7, "m6": 8, "M6": 9, "m7": 10, "M7": 11, "P8": 12,
    }
    return intervals.get(interval_name, 0)
