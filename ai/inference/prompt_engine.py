"""
Prompt engine — converts natural language prompts into structured
MusicalContext and GenerationConstraints.

This bridges the gap between user-facing natural language ("make a
happy pop song in C major") and the structured parameters that AI
generator plugins expect.

The engine uses a rule-based approach (no external ML) with keyword
matching, parameter extraction, and sensible defaults. It's designed
to be extended with LLM-based parsing in the future.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from dataclasses import dataclass, field
from typing import Optional

from core.plugin.interfaces.ai_generator import (
    MusicalContext, GenerationConstraints, GenerationPrompt, ContentType,
)


# ── Keyword mappings ────────────────────────────────────────────────

STYLE_KEYWORDS = {
    "pop": ["pop", "popular", "mainstream", "radio"],
    "jazz": ["jazz", "jazzy", "swing", "bebop", "bossa", "fusion"],
    "classical": ["classical", "orchestral", "symphony", "baroque", "romantic"],
    "lofi": ["lofi", "lo-fi", "chill", "study", "sleep", "relaxing"],
    "edm": ["edm", "electronic", "dance", "house", "techno", "trance", "dubstep"],
    "rnb": ["rnb", "r&b", "soul", "neo-soul"],
    "rock": ["rock", "guitar", "metal", "punk", "alternative"],
    "blues": ["blues", "blue", "12-bar"],
    "hiphop": ["hiphop", "hip-hop", "rap", "trap", "beat"],
}

MOOD_KEYWORDS = {
    "happy": ["happy", "joyful", "upbeat", "cheerful", "bright"],
    "sad": ["sad", "melancholic", "somber", "dark", "emotional", "ballad"],
    "energetic": ["energetic", "powerful", "intense", "driving", "hype"],
    "chill": ["chill", "calm", "peaceful", "mellow", "soft", "gentle"],
    "epic": ["epic", "grand", "cinematic", "dramatic", "heroic"],
    "groovy": ["groovy", "funky", "rhythmic", "catchy"],
}

COMPLEXITY_KEYWORDS = {
    "simple": 0.2, "minimal": 0.2, "basic": 0.3,
    "moderate": 0.5, "medium": 0.5,
    "complex": 0.8, "rich": 0.8, "dense": 0.9, "intricate": 0.9,
}

CONTENT_KEYWORDS = {
    "chords": ContentType.CHORDS,
    "chord": ContentType.CHORDS,
    "harmony": ContentType.CHORDS,
    "melody": ContentType.MELODY,
    "melodic": ContentType.MELODY,
    "lead": ContentType.LEAD,
    "solo": ContentType.LEAD,
    "bass": ContentType.BASS,
    "bassline": ContentType.BASS,
    "drums": ContentType.DRUMS,
    "drum": ContentType.DRUMS,
    "beat": ContentType.DRUMS,
    "percussion": ContentType.DRUMS,
    "arpeggio": ContentType.ARPEGGIO,
    "arp": ContentType.ARPEGGIO,
    "pad": ContentType.PAD,
    "texture": ContentType.PAD,
    "arrangement": ContentType.ARRANGEMENT,
    "structure": ContentType.ARRANGEMENT,
    "fill": ContentType.FILL,
    "transition": ContentType.FILL,
}

# Key name → (sharps_flats, mode)
KEY_MAP = {
    # Major keys
    "c": (0, "major"), "c major": (0, "major"),
    "g": (1, "major"), "g major": (1, "major"),
    "d": (2, "major"), "d major": (2, "major"),
    "a": (3, "major"), "a major": (3, "major"),
    "e": (4, "major"), "e major": (4, "major"),
    "b": (5, "major"), "b major": (5, "major"),
    "f#": (6, "major"), "f# major": (6, "major"),
    "c#": (7, "major"), "c# major": (7, "major"),
    "f": (-1, "major"), "f major": (-1, "major"),
    "bb": (-2, "major"), "bb major": (-2, "major"),
    "eb": (-3, "major"), "eb major": (-3, "major"),
    "ab": (-4, "major"), "ab major": (-4, "major"),
    "db": (-5, "major"), "db major": (-5, "major"),
    "gb": (-6, "major"), "gb major": (-6, "major"),
    "cb": (-7, "major"), "cb major": (-7, "major"),
    # Minor keys
    "a minor": (0, "minor"), "am": (0, "minor"),
    "e minor": (1, "minor"), "em": (1, "minor"),
    "b minor": (2, "minor"), "bm": (2, "minor"),
    "f# minor": (3, "minor"), "f#m": (3, "minor"),
    "c# minor": (4, "minor"), "c#m": (4, "minor"),
    "g# minor": (5, "minor"), "g#m": (5, "minor"),
    "d# minor": (6, "minor"), "d#m": (6, "minor"),
    "d minor": (-1, "minor"), "dm": (-1, "minor"),
    "g minor": (-2, "minor"), "gm": (-2, "minor"),
    "c minor": (-3, "minor"), "cm": (-3, "minor"),
    "f minor": (-4, "minor"), "fm": (-4, "minor"),
    "bb minor": (-5, "minor"), "bbm": (-5, "minor"),
    "eb minor": (-6, "minor"), "ebm": (-6, "minor"),
}


@dataclass
class ParsedPrompt:
    """Result of parsing a natural language prompt."""
    style_tags: list[str] = field(default_factory=list)
    mood_tags: list[str] = field(default_factory=list)
    content_types: list[ContentType] = field(default_factory=list)
    key_sharps_flats: int = 0
    key_mode: str = "major"
    bpm: Optional[float] = None
    bar_count: int = 8
    density: float = 0.5
    energy: float = 0.5
    temperature: float = 0.7
    raw_text: str = ""


class PromptEngine:
    """
    Converts natural language prompts into structured generation parameters.

    Uses keyword matching and regex extraction to parse user intent.
    Designed for extensibility — the parse() method can be replaced
    with an LLM-based parser for more nuanced understanding.

    Usage:
        engine = PromptEngine()
        prompt = engine.parse("generate a happy lofi melody in C major")
        # prompt.style_tags = ["lofi"]
        # prompt.mood_tags = ["happy"]
        # prompt.content_types = [ContentType.MELODY]
        # prompt.key = "C major"
    """

    def __init__(self):
        pass

    def parse(self, text: str) -> ParsedPrompt:
        """
        Parse a natural language prompt into structured parameters.

        Args:
            text: User's natural language input.

        Returns:
            ParsedPrompt with extracted parameters.
        """
        text_lower = text.lower()
        result = ParsedPrompt(raw_text=text)

        # Extract style
        for style, keywords in STYLE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    if style not in result.style_tags:
                        result.style_tags.append(style)
                    break

        # Extract mood
        for mood, keywords in MOOD_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    if mood not in result.mood_tags:
                        result.mood_tags.append(mood)
                    break

        # Extract content type
        for content_kw, ct in CONTENT_KEYWORDS.items():
            if content_kw in text_lower:
                if ct not in result.content_types:
                    result.content_types.append(ct)

        # If no content type specified, default to all melodic
        if not result.content_types:
            result.content_types = [ContentType.CHORDS, ContentType.MELODY]

        # Extract key
        for key_name, (sf, mode) in sorted(KEY_MAP.items(),
                                            key=lambda x: -len(x[0])):
            if key_name in text_lower:
                result.key_sharps_flats = sf
                result.key_mode = mode
                break

        # Extract BPM
        bpm_match = re.search(r'(\d{2,3})\s*bpm', text_lower)
        if bpm_match:
            result.bpm = float(bpm_match.group(1))

        # Extract bar count
        bar_match = re.search(r'(\d+)\s*(bar|bars|measure)', text_lower)
        if bar_match:
            result.bar_count = int(bar_match.group(1))

        # Extract complexity/density
        for kw, density in COMPLEXITY_KEYWORDS.items():
            if kw in text_lower:
                result.density = density
                break

        # Infer energy from mood
        if "energetic" in result.mood_tags or "epic" in result.mood_tags:
            result.energy = 0.8
        elif "chill" in result.mood_tags or "sad" in result.mood_tags:
            result.energy = 0.3
        elif "happy" in result.mood_tags:
            result.energy = 0.6
        else:
            result.energy = 0.5

        return result

    def to_musical_context(self, parsed: ParsedPrompt) -> MusicalContext:
        """
        Convert a ParsedPrompt into a MusicalContext for generation.

        Args:
            parsed: The parsed prompt.

        Returns:
            MusicalContext ready for use with AI generators.
        """
        all_tags = list(parsed.style_tags) + list(parsed.mood_tags)
        return MusicalContext(
            bpm=parsed.bpm or 120.0,
            key_sharps_flats=parsed.key_sharps_flats,
            key_mode=parsed.key_mode,
            bar_count=parsed.bar_count,
            style_tags=all_tags,
            energy_target=parsed.energy,
            density_target=parsed.density,
        )

    def to_generation_prompt(self, parsed: ParsedPrompt) -> GenerationPrompt:
        """
        Convert a ParsedPrompt into a full GenerationPrompt.

        Args:
            parsed: The parsed prompt.

        Returns:
            GenerationPrompt ready for use with AI generator plugins.
        """
        ctx = self.to_musical_context(parsed)
        constraints = GenerationConstraints(
            bar_count=parsed.bar_count,
            temperature=parsed.temperature,
        )
        return GenerationPrompt(
            text=parsed.raw_text,
            context=ctx,
            constraints=constraints,
            content_types_requested=parsed.content_types,
        )

    def explain(self, parsed: ParsedPrompt) -> str:
        """
        Generate a human-readable explanation of the parsed prompt.

        Args:
            parsed: The parsed prompt.

        Returns:
            Explanation string.
        """
        parts = []

        if parsed.key_sharps_flats != 0 or parsed.key_mode != "major":
            from ai.models.theory import pc_to_key_name
            tonic = (parsed.key_sharps_flats * 7) % 12 if parsed.key_sharps_flats >= 0 \
                    else (abs(parsed.key_sharps_flats) * 5) % 12
            key = pc_to_key_name(tonic, parsed.key_mode)
            parts.append(f"Key: {key}")

        if parsed.bpm:
            parts.append(f"Tempo: {parsed.bpm} BPM")

        if parsed.style_tags:
            parts.append(f"Style: {', '.join(parsed.style_tags)}")

        if parsed.mood_tags:
            parts.append(f"Mood: {', '.join(parsed.mood_tags)}")

        parts.append(f"Length: {parsed.bar_count} bars")

        if parsed.content_types:
            type_names = [ct.name.lower() for ct in parsed.content_types]
            parts.append(f"Content: {', '.join(type_names)}")

        parts.append(f"Complexity: {parsed.density:.0%}")

        return "\n".join(f"  • {p}" for p in parts)
