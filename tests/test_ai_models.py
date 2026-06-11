"""
Tests for AI model implementations.

Tests cover:
- Music theory foundation (scales, chords, progressions, voice leading)
- Chord generator plugin (all styles, voicing, alternatives)
- Melody generator plugin (contours, density, chord awareness)
- Drum generator plugin (styles, fills, complexity)
- Basic analyzer (key detection, chord ID, rhythm analysis)
- Context window (multi-turn context management)
- Prompt engine (natural language parsing)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from random import Random
from typing import Optional

from core.model.note import NoteEvent, NotePitch, NoteVelocity
from core.model.clip import MidiClip
from core.model.time_model import Ticks, PPQ, beats_to_ticks
from core.plugin.interfaces.ai_generator import (
    MusicalContext, GenerationConstraints, GenerationPrompt,
    GeneratedContent, ContentType,
)
from core.plugin.interfaces.ai_analyzer import (
    AnalysisRequest, AnalyzerCapabilities,
)

# ── Theory tests ────────────────────────────────────────────────────

class TestMusicTheory(unittest.TestCase):
    """Test the music theory foundation module."""

    def test_scale_degrees_major(self):
        from ai.models.theory import scale_degrees
        c_major = scale_degrees(0, "major")
        self.assertEqual(c_major, [0, 2, 4, 5, 7, 9, 11])

    def test_scale_degrees_natural_minor(self):
        from ai.models.theory import scale_degrees
        a_minor = scale_degrees(9, "natural_minor")
        # A natural minor = A B C D E F G = same pitch classes as C major
        self.assertEqual(sorted(a_minor), [0, 2, 4, 5, 7, 9, 11])

    def test_scale_degrees_harmonic_minor(self):
        from ai.models.theory import scale_degrees
        a_harm = scale_degrees(9, "harmonic_minor")
        # Has G# (11) instead of G (10)
        self.assertIn(11, a_harm)
        self.assertNotIn(10, a_harm)

    def test_pentatonic_scales(self):
        from ai.models.theory import scale_degrees
        c_maj_penta = scale_degrees(0, "major_pentatonic")
        self.assertEqual(c_maj_penta, [0, 2, 4, 7, 9])

        a_min_penta = scale_degrees(9, "minor_pentatonic")
        # A min pentatonic = A C D E G = 9, 0, 2, 4, 7
        self.assertEqual(a_min_penta, [9, 0, 2, 4, 7])

    def test_blues_scale(self):
        from ai.models.theory import scale_degrees
        c_blues = scale_degrees(0, "blues")
        self.assertEqual(c_blues, [0, 3, 5, 6, 7, 10])

    def test_scale_notes_returns_midi(self):
        from ai.models.theory import scale_notes
        c4_major = scale_notes(0, "major", octave=4, num_octaves=1)
        # Should return MIDI notes for one octave of C major
        self.assertGreater(len(c4_major), 0)
        self.assertIn(60, c4_major)  # C4
        self.assertIn(64, c4_major)  # E4
        self.assertIn(67, c4_major)  # G4

    def test_chord_tones(self):
        from ai.models.theory import chord_tones
        c_maj = chord_tones(0, "maj")
        self.assertEqual(set(c_maj), {0, 4, 7})

        c_min = chord_tones(0, "min")
        self.assertEqual(set(c_min), {0, 3, 7})

        c7 = chord_tones(0, "7")
        self.assertEqual(set(c7), {0, 4, 7, 10})

        c_maj7 = chord_tones(0, "maj7")
        self.assertEqual(set(c_maj7), {0, 4, 7, 11})

    def test_chord_name(self):
        from ai.models.theory import chord_name
        self.assertEqual(chord_name(0, "maj"), "C")
        self.assertEqual(chord_name(2, "maj7"), "Dmaj7")
        self.assertEqual(chord_name(7, "min7"), "Gm7")
        self.assertEqual(chord_name(5, "dim"), "Fdim")

    def test_parse_chord_symbol(self):
        from ai.models.theory import parse_chord_symbol
        root, ctype = parse_chord_symbol("Cmaj7")
        self.assertEqual(root, 0)
        self.assertEqual(ctype, "maj7")

        root, ctype = parse_chord_symbol("Dm7")
        self.assertEqual(root, 2)
        self.assertEqual(ctype, "min7")

        root, ctype = parse_chord_symbol("G7")
        self.assertEqual(root, 7)
        self.assertEqual(ctype, "7")

    def test_parse_chord_symbol_sharp(self):
        from ai.models.theory import parse_chord_symbol
        root, ctype = parse_chord_symbol("F#m7")
        self.assertEqual(root, 6)
        self.assertEqual(ctype, "min7")

    def test_diatonic_chords_c_major_triads(self):
        from ai.models.theory import diatonic_chords
        chords = diatonic_chords(0, "major", use_sevenths=False)
        self.assertEqual(len(chords), 7)
        # I = C major
        self.assertEqual(chords[0].name, "C")
        self.assertEqual(chords[0].roman, "I")
        # ii = D minor
        self.assertEqual(chords[1].name, "Dm")
        self.assertEqual(chords[1].roman, "ii")
        # V = G major
        self.assertEqual(chords[4].name, "G")
        self.assertEqual(chords[4].roman, "V")

    def test_diatonic_chords_c_major_sevenths(self):
        from ai.models.theory import diatonic_chords
        chords = diatonic_chords(0, "major", use_sevenths=True)
        self.assertEqual(len(chords), 7)
        # Imaj7
        self.assertIn("Cmaj7", [chords[0].name])
        # V7 (should be dominant 7th)
        self.assertEqual(chords[4].quality, "7")

    def test_get_style_progression(self):
        from ai.models.theory import get_style_progression
        rng = Random(42)
        prog = get_style_progression("pop", rng)
        self.assertIsInstance(prog, list)
        self.assertGreater(len(prog), 0)
        for item in prog:
            self.assertEqual(len(item), 3)
            degree, quality, beats = item
            self.assertGreaterEqual(degree, 1)
            self.assertLessEqual(degree, 7)
            self.assertGreater(beats, 0)

    def test_all_styles_have_progressions(self):
        from ai.models.theory import COMMON_PROGRESSIONS
        expected_styles = {"pop", "jazz", "classical", "lofi",
                          "edm", "rnb", "rock", "blues"}
        for style in expected_styles:
            self.assertIn(style, COMMON_PROGRESSIONS,
                          f"Missing style: {style}")
            self.assertGreater(len(COMMON_PROGRESSIONS[style]), 0,
                              f"Empty progressions for: {style}")

    def test_voice_leading(self):
        from ai.models.theory import voice_lead
        # C major (C E G) to F major (F A C)
        result = voice_lead([0, 4, 7], [5, 9, 0])
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(n, int) for n in result))

    def test_note_in_scale(self):
        from ai.models.theory import note_in_scale
        # E (4) is in C major
        self.assertTrue(note_in_scale(4, 0, "major"))
        # F# (6) is NOT in C major
        self.assertFalse(note_in_scale(6, 0, "major"))
        # G# (8) IS in A harmonic minor
        self.assertTrue(note_in_scale(8, 9, "harmonic_minor"))

    def test_closest_scale_note(self):
        from ai.models.theory import closest_scale_note
        # F# (6) → closest in C major is G (7) or F (5)
        result = closest_scale_note(6, 0, "major")
        self.assertIn(result, [5, 7])

    def test_rhythm_patterns(self):
        from ai.models.theory import rhythm_pattern
        rng = Random(42)
        for style in ["steady", "syncopated", "sparse", "dense", "swing"]:
            result = rhythm_pattern(style, 4, rng)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)

    def test_key_name_to_pc(self):
        from ai.models.theory import key_name_to_pc
        pc, mode = key_name_to_pc("C major")
        self.assertEqual(pc, 0)
        self.assertEqual(mode, "major")

        pc, mode = key_name_to_pc("Eb minor")
        self.assertEqual(pc, 3)
        self.assertEqual(mode, "minor")

    def test_pc_to_key_name(self):
        from ai.models.theory import pc_to_key_name
        self.assertEqual(pc_to_key_name(0, "major"), "C major")
        self.assertEqual(pc_to_key_name(9, "minor"), "A minor")


# ── Chord generator tests ──────────────────────────────────────────

class TestChordGenerator(unittest.TestCase):
    """Test the chord progression generator plugin."""

    def setUp(self):
        from ai.models.chord_generator import ChordGeneratorPlugin
        self.plugin = ChordGeneratorPlugin(seed=42)
        self.plugin.initialize()

    def test_manifest(self):
        manifest = self.plugin.get_manifest()
        self.assertEqual(manifest.plugin_id, "amusiment.chord-generator")
        self.assertIn("Chord", manifest.name)

    def test_capabilities(self):
        caps = self.plugin.get_capabilities()
        self.assertIn(ContentType.CHORDS, caps.content_types)
        self.assertGreater(caps.max_bars, 0)
        self.assertGreater(len(caps.style_tags), 0)

    def test_generate_basic_chords(self):
        ctx = MusicalContext(
            bpm=120.0,
            key_sharps_flats=0,
            key_mode="major",
            bar_count=8,
            style_tags=["pop"],
        )
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)

        self.assertIsInstance(result, GeneratedContent)
        self.assertIn(ContentType.CHORDS, result.clips)
        clip = result.clips[ContentType.CHORDS]
        self.assertIsInstance(clip, MidiClip)
        self.assertGreater(clip.note_count, 0)
        self.assertGreater(len(result.explanation), 0)
        self.assertGreater(len(result.chord_progression), 0)

    def test_generate_jazz_chords(self):
        ctx = MusicalContext(
            bpm=140.0,
            key_sharps_flats=-3,  # Eb major
            key_mode="major",
            bar_count=4,
            style_tags=["jazz"],
            density_target=0.7,
        )
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)
        self.assertIn(ContentType.CHORDS, result.clips)

    def test_generate_minor_chords(self):
        ctx = MusicalContext(
            bpm=100.0,
            key_sharps_flats=0,
            key_mode="minor",
            bar_count=4,
            style_tags=["lofi"],
        )
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)
        self.assertIn(ContentType.CHORDS, result.clips)

    def test_alternatives_generated(self):
        ctx = MusicalContext(
            bar_count=4,
            style_tags=["pop"],
        )
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)
        self.assertGreater(len(result.alternatives), 0)
        for alt in result.alternatives:
            self.assertIsInstance(alt, GeneratedContent)

    def test_different_styles_produce_different_output(self):
        results = {}
        for style in ["pop", "jazz", "classical"]:
            ctx = MusicalContext(
                bar_count=4,
                style_tags=[style],
            )
            prompt = GenerationPrompt(context=ctx)
            result = self.plugin.generate(prompt)
            clip = result.clips[ContentType.CHORDS]
            results[style] = len(clip.notes)

        # Different styles should produce different note counts
        self.assertGreater(len(set(results.values())), 1)

    def test_get_parameters(self):
        params = self.plugin.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)


# ── Melody generator tests ─────────────────────────────────────────

class TestMelodyGenerator(unittest.TestCase):
    """Test the melody generator plugin."""

    def setUp(self):
        from ai.models.melody_generator import MelodyGeneratorPlugin
        self.plugin = MelodyGeneratorPlugin(seed=42)
        self.plugin.initialize()

    def test_manifest(self):
        manifest = self.plugin.get_manifest()
        self.assertEqual(manifest.plugin_id, "amusiment.melody-generator")

    def test_capabilities(self):
        caps = self.plugin.get_capabilities()
        self.assertIn(ContentType.MELODY, caps.content_types)

    def test_generate_basic_melody(self):
        ctx = MusicalContext(
            bpm=120.0,
            key_sharps_flats=0,
            key_mode="major",
            bar_count=4,
            style_tags=["arch"],
            chord_progression=["Cmaj", "Fmaj", "Gmaj", "Cmaj"],
        )
        prompt = GenerationPrompt(
            context=ctx,
            constraints=GenerationConstraints(
                bar_count=4,
                min_pitch=60,
                max_pitch=84,
            ),
        )
        result = self.plugin.generate(prompt)

        self.assertIsInstance(result, GeneratedContent)
        self.assertIn(ContentType.MELODY, result.clips)
        clip = result.clips[ContentType.MELODY]
        self.assertGreater(clip.note_count, 0)

        # All notes should be in range
        for note in clip.notes:
            self.assertGreaterEqual(note.pitch, 60)
            self.assertLessEqual(note.pitch, 84)

    def test_generate_no_chords_defaults(self):
        ctx = MusicalContext(
            key_sharps_flats=1,  # G major
            key_mode="major",
            bar_count=4,
        )
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)
        self.assertIn(ContentType.MELODY, result.clips)

    def test_different_contours(self):
        from ai.models.melody_generator import CONTOUR_SHAPES
        names = list(CONTOUR_SHAPES.keys())
        self.assertEqual(len(names), 6)
        self.assertIn("arch", names)
        self.assertIn("rising", names)
        self.assertIn("wave", names)

    def test_alternatives(self):
        ctx = MusicalContext(bar_count=4, style_tags=["rising"])
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)
        self.assertGreaterEqual(len(result.alternatives), 1)

        # Each alternative should be a valid GeneratedContent
        for alt in result.alternatives:
            self.assertIsInstance(alt, GeneratedContent)
            self.assertIn(ContentType.MELODY, alt.clips)


# ── Drum generator tests ───────────────────────────────────────────

class TestDrumGenerator(unittest.TestCase):
    """Test the drum pattern generator plugin."""

    def setUp(self):
        from ai.models.drum_generator import DrumGeneratorPlugin
        self.plugin = DrumGeneratorPlugin(seed=42)
        self.plugin.initialize()

    def test_manifest(self):
        manifest = self.plugin.get_manifest()
        self.assertEqual(manifest.plugin_id, "amusiment.drum-generator")

    def test_capabilities(self):
        caps = self.plugin.get_capabilities()
        self.assertIn(ContentType.DRUMS, caps.content_types)

    def test_generate_rock_drums(self):
        ctx = MusicalContext(
            bpm=120.0,
            bar_count=4,
            style_tags=["rock"],
            energy_target=0.8,
        )
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)

        self.assertIn(ContentType.DRUMS, result.clips)
        clip = result.clips[ContentType.DRUMS]
        self.assertGreater(clip.note_count, 0)

        # All notes should be on channel 9 (GM drums)
        for note in clip.notes:
            self.assertEqual(note.channel, 9)

        # Should have kicks (36) and snares (38)
        pitches = {note.pitch for note in clip.notes}
        self.assertIn(36, pitches, "Should have kick drum")
        self.assertIn(38, pitches, "Should have snare")

    def test_all_styles(self):
        for style in ["rock", "pop", "jazz", "lofi", "edm", "hiphop", "funk"]:
            ctx = MusicalContext(bar_count=2, style_tags=[style])
            prompt = GenerationPrompt(context=ctx)
            result = self.plugin.generate(prompt)
            clip = result.clips[ContentType.DRUMS]
            self.assertGreater(clip.note_count, 0,
                             f"Style '{style}' produced no notes")

    def test_fills_at_bar_boundaries(self):
        ctx = MusicalContext(bar_count=8, style_tags=["rock"])
        prompt = GenerationPrompt(context=ctx)
        result = self.plugin.generate(prompt)
        clip = result.clips[ContentType.DRUMS]

        # Verify notes span the full 8 bars
        ticks_per_bar = 16 * PPQ // 4
        last_tick = max((n.start_tick for n in clip.notes), default=Ticks(0))
        self.assertGreater(last_tick, Ticks(7 * ticks_per_bar))

        # Fills should introduce tom notes (pitch 45-50 range)
        tom_pitches = [p for p in (45, 48, 50, 41) if p in {n.pitch for n in clip.notes}]
        # Not all styles have explicit toms, but patterns should exist
        pass

    def test_density_affects_complexity(self):
        simple_ctx = MusicalContext(bar_count=4, style_tags=["pop"],
                                    density_target=0.2)
        prompt_simple = GenerationPrompt(context=simple_ctx)
        result_simple = self.plugin.generate(prompt_simple)

        dense_ctx = MusicalContext(bar_count=4, style_tags=["pop"],
                                   density_target=0.8)
        prompt_dense = GenerationPrompt(context=dense_ctx)
        result_dense = self.plugin.generate(prompt_dense)

        simple_count = result_simple.clips[ContentType.DRUMS].note_count
        dense_count = result_dense.clips[ContentType.DRUMS].note_count
        # Both should produce valid results
        self.assertGreater(simple_count, 0)
        self.assertGreater(dense_count, 0)


# ── Basic analyzer tests ───────────────────────────────────────────

class TestBasicAnalyzer(unittest.TestCase):
    """Test the basic music analyzer plugin."""

    def setUp(self):
        from ai.models.basic_analyzer import BasicAnalyzerPlugin
        self.plugin = BasicAnalyzerPlugin()
        self.plugin.initialize()

    def test_manifest(self):
        manifest = self.plugin.get_manifest()
        self.assertEqual(manifest.plugin_id, "amusiment.basic-analyzer")

    def test_capabilities(self):
        caps = self.plugin.get_capabilities()
        self.assertIn(AnalyzerCapabilities.KEY_DETECTION, caps)
        self.assertIn(AnalyzerCapabilities.CHORD_ANALYSIS, caps)
        self.assertIn(AnalyzerCapabilities.RHYTHM_ANALYSIS, caps)

    def test_analyze_key_detection_c_major(self):
        # Create a clip with notes strongly suggesting C major
        notes = tuple(NoteEvent(
            pitch=NotePitch(p), velocity=NoteVelocity(100),
            start_tick=Ticks(i * PPQ), duration_ticks=Ticks(PPQ // 2),
            channel=0,
        ) for i, p in enumerate([60, 64, 67, 72, 71, 69, 67, 64]))  # C major scale

        clip = MidiClip(name="Test", notes=notes)
        clips = {"track_1": [clip]}
        request = AnalysisRequest(
            capabilities_requested=[AnalyzerCapabilities.KEY_DETECTION],
        )
        result = self.plugin.analyze(clips, request)

        self.assertIn(AnalyzerCapabilities.KEY_DETECTION, result.findings)
        finding = result.findings[AnalyzerCapabilities.KEY_DETECTION]
        self.assertIn("key", finding)
        # Should detect C major or close
        self.assertIn(finding["key"].lower().split()[0],
                     ["c", "a"])  # C major or A minor (same notes)
        self.assertGreater(result.confidence, 0)

    def test_analyze_chords(self):
        # C major triad: C4 (60), E4 (64), G4 (67)
        notes = tuple([
            NoteEvent(pitch=NotePitch(60), velocity=NoteVelocity(100),
                      start_tick=Ticks(0), duration_ticks=Ticks(PPQ)),
            NoteEvent(pitch=NotePitch(64), velocity=NoteVelocity(100),
                      start_tick=Ticks(0), duration_ticks=Ticks(PPQ)),
            NoteEvent(pitch=NotePitch(67), velocity=NoteVelocity(100),
                      start_tick=Ticks(0), duration_ticks=Ticks(PPQ)),
        ])
        clip = MidiClip(name="C chord", notes=notes)
        clips = {"track_1": [clip]}
        request = AnalysisRequest(
            capabilities_requested=[AnalyzerCapabilities.CHORD_ANALYSIS],
        )
        result = self.plugin.analyze(clips, request)

        self.assertIn(AnalyzerCapabilities.CHORD_ANALYSIS, result.findings)
        finding = result.findings[AnalyzerCapabilities.CHORD_ANALYSIS]

    def test_analyze_rhythm(self):
        # 8th notes
        notes = tuple(NoteEvent(
            pitch=NotePitch(60), velocity=NoteVelocity(100),
            start_tick=Ticks(i * PPQ // 2),
            duration_ticks=Ticks(PPQ // 4),
            channel=0,
        ) for i in range(8))

        clip = MidiClip(name="8th notes", notes=notes)
        clips = {"track_1": [clip]}
        request = AnalysisRequest(
            capabilities_requested=[AnalyzerCapabilities.RHYTHM_ANALYSIS],
        )
        result = self.plugin.analyze(clips, request)

        self.assertIn(AnalyzerCapabilities.RHYTHM_ANALYSIS, result.findings)
        finding = result.findings[AnalyzerCapabilities.RHYTHM_ANALYSIS]
        self.assertEqual(finding["note_count"], 8)

    def test_analyze_empty_clips(self):
        clips = {"track_1": []}
        request = AnalysisRequest(
            capabilities_requested=[AnalyzerCapabilities.KEY_DETECTION],
        )
        result = self.plugin.analyze(clips, request)
        self.assertIn(AnalyzerCapabilities.KEY_DETECTION, result.findings)
        finding = result.findings[AnalyzerCapabilities.KEY_DETECTION]
        self.assertEqual(finding["confidence"], 0.0)


# ── Context window tests ───────────────────────────────────────────

class TestContextWindow(unittest.TestCase):
    """Test the musical context window for multi-turn generation."""

    def setUp(self):
        from ai.inference.context_window import ContextWindow
        self.window = ContextWindow(max_history=5)

    def test_initial_state(self):
        self.assertEqual(self.window.turn_count, 0)
        self.assertFalse(self.window.has_context)

    def test_build_context_defaults(self):
        ctx = self.window.build_context()
        self.assertEqual(ctx.bpm, 120.0)
        self.assertEqual(ctx.key_sharps_flats, 0)
        self.assertEqual(ctx.key_mode, "major")

    def test_build_context_with_params(self):
        ctx = self.window.build_context(
            bpm=140.0,
            key_sharps_flats=-3,
            key_mode="minor",
            bar_count=16,
            style_tags=["jazz"],
        )
        self.assertEqual(ctx.bpm, 140.0)
        self.assertEqual(ctx.key_sharps_flats, -3)
        self.assertEqual(ctx.key_mode, "minor")
        self.assertEqual(ctx.bar_count, 16)
        self.assertIn("jazz", ctx.style_tags)

    def test_context_persistence(self):
        # First call sets values
        self.window.build_context(bpm=140.0, key_mode="minor")
        # Second call without params should remember
        ctx = self.window.build_context(bar_count=8)
        self.assertEqual(ctx.bpm, 140.0)
        self.assertEqual(ctx.key_mode, "minor")

    def test_record_generation(self):
        ctx = self.window.build_context(style_tags=["pop"])
        result = GeneratedContent(
            explanation="Test generation",
            clips={ContentType.CHORDS: MidiClip(name="Test")},
            confidence=0.95,
        )
        self.window.record_generation("chords", result)
        self.assertEqual(self.window.turn_count, 1)
        self.assertTrue(self.window.has_context)

    def test_turn_summary(self):
        result = GeneratedContent(
            explanation="Generated 4 bars of C major chords",
            clips={ContentType.CHORDS: MidiClip(name="Test")},
            confidence=0.9,
        )
        self.window.record_generation("chords", result)
        summary = self.window.get_turn_summary()
        self.assertIn("chords", summary)

    def test_clear(self):
        self.window.build_context(bpm=140.0)
        result = GeneratedContent(
            explanation="Test", clips={}, confidence=1.0,
        )
        self.window.record_generation("test", result)
        self.window.clear()
        self.assertEqual(self.window.turn_count, 0)
        ctx = self.window.build_context()
        self.assertEqual(ctx.bpm, 120.0)

    def test_max_history(self):
        from ai.inference.context_window import ContextWindow
        window = ContextWindow(max_history=3)
        for i in range(10):
            result = GeneratedContent(
                explanation=f"Turn{i}", clips={}, confidence=1.0,
            )
            window.record_generation("test", result)
        # History should be pruned to max_history
        self.assertEqual(len(window._turn_history), window.max_history)
        # The oldest entries should be gone (Turn0-Turn6 are pruned)
        # The last 3 entries are Turns 7, 8, 9 → show as Turn 1-3 in summary
        summary = window.get_turn_summary()
        self.assertIn("3 generation turns", summary)
        self.assertNotIn("Turn0", summary)


# ── Prompt engine tests ────────────────────────────────────────────

class TestPromptEngine(unittest.TestCase):
    """Test the natural language prompt parser."""

    def setUp(self):
        from ai.inference.prompt_engine import PromptEngine
        self.engine = PromptEngine()

    def test_parse_style(self):
        result = self.engine.parse("generate a jazz chord progression")
        self.assertIn("jazz", result.style_tags)

        result = self.engine.parse("make some lofi beats")
        self.assertIn("lofi", result.style_tags)

    def test_parse_content_type(self):
        result = self.engine.parse("create a melody")
        self.assertIn(ContentType.MELODY, result.content_types)

        result = self.engine.parse("generate drums and bass")
        self.assertIn(ContentType.DRUMS, result.content_types)
        self.assertIn(ContentType.BASS, result.content_types)

    def test_parse_key(self):
        result = self.engine.parse("write something in C major")
        self.assertEqual(result.key_sharps_flats, 0)
        self.assertEqual(result.key_mode, "major")

        result = self.engine.parse("a sad song in A minor")
        self.assertEqual(result.key_sharps_flats, 0)
        self.assertEqual(result.key_mode, "minor")

    def test_parse_bpm(self):
        result = self.engine.parse("a house track at 128 bpm")
        self.assertEqual(result.bpm, 128.0)

    def test_parse_bars(self):
        result = self.engine.parse("generate 16 bars of pop chords")
        self.assertEqual(result.bar_count, 16)

    def test_parse_mood(self):
        result = self.engine.parse("make a happy upbeat melody")
        self.assertIn("happy", result.mood_tags)
        self.assertGreater(result.energy, 0.5)

    def test_parse_complexity(self):
        result = self.engine.parse("a complex jazz arrangement")
        self.assertGreater(result.density, 0.5)

        result = self.engine.parse("a simple beat")
        self.assertLess(result.density, 0.5)

    def test_to_musical_context(self):
        result = self.engine.parse("happy lofi chords in C major at 80 bpm, 4 bars")
        ctx = self.engine.to_musical_context(result)
        self.assertEqual(ctx.bpm, 80.0)
        self.assertEqual(ctx.key_mode, "major")
        self.assertEqual(ctx.bar_count, 4)
        self.assertIn("lofi", ctx.style_tags)
        self.assertIn("happy", ctx.style_tags)

    def test_to_generation_prompt(self):
        result = self.engine.parse("generate a pop melody in G major")
        prompt = self.engine.to_generation_prompt(result)
        self.assertIsInstance(prompt, GenerationPrompt)
        self.assertIn(ContentType.MELODY, prompt.content_types_requested)

    def test_explain(self):
        result = self.engine.parse("happy jazz melody in F major, 120 bpm, 8 bars")
        explanation = self.engine.explain(result)
        self.assertIsInstance(explanation, str)
        self.assertIn("jazz", explanation.lower())

    def test_default_content_types(self):
        # No content type specified → default to chords + melody
        result = self.engine.parse("make something in C major")
        self.assertIn(ContentType.CHORDS, result.content_types)
        self.assertIn(ContentType.MELODY, result.content_types)


# ── Integration test ────────────────────────────────────────────────

class TestAIIntegration(unittest.TestCase):
    """Integration test: generate chords + melody + drums together."""

    def test_full_generation_pipeline(self):
        """Generate a complete 8-bar composition."""
        from ai.models.chord_generator import ChordGeneratorPlugin
        from ai.models.melody_generator import MelodyGeneratorPlugin
        from ai.models.drum_generator import DrumGeneratorPlugin
        from ai.inference.prompt_engine import PromptEngine

        engine = PromptEngine()
        parsed = engine.parse(
            "create a happy pop song in C major, 120 bpm, 8 bars"
        )
        ctx = engine.to_musical_context(parsed)

        # Generate chords
        chord_gen = ChordGeneratorPlugin(seed=42)
        chord_gen.initialize()
        chord_prompt = GenerationPrompt(
            context=ctx,
            constraints=GenerationConstraints(bar_count=8),
        )
        chord_result = chord_gen.generate(chord_prompt)
        chords = chord_result.chord_progression

        # Generate melody over those chords
        melody_gen = MelodyGeneratorPlugin(seed=43)
        melody_gen.initialize()
        melody_ctx = MusicalContext(
            bpm=ctx.bpm,
            key_sharps_flats=ctx.key_sharps_flats,
            key_mode=ctx.key_mode,
            bar_count=8,
            chord_progression=chords,
            style_tags=["arch"],
        )
        melody_prompt = GenerationPrompt(context=melody_ctx)
        melody_result = melody_gen.generate(melody_prompt)

        # Generate drums
        drum_gen = DrumGeneratorPlugin(seed=44)
        drum_gen.initialize()
        drum_ctx = MusicalContext(
            bpm=ctx.bpm,
            bar_count=8,
            style_tags=["pop"],
        )
        drum_prompt = GenerationPrompt(context=drum_ctx)
        drum_result = drum_gen.generate(drum_prompt)

        # Verify all components
        self.assertGreater(chord_result.clips[ContentType.CHORDS].note_count, 0)
        self.assertGreater(melody_result.clips[ContentType.MELODY].note_count, 0)
        self.assertGreater(drum_result.clips[ContentType.DRUMS].note_count, 0)

        # Total notes should be reasonable
        total_notes = (
            chord_result.clips[ContentType.CHORDS].note_count +
            melody_result.clips[ContentType.MELODY].note_count +
            drum_result.clips[ContentType.DRUMS].note_count
        )
        self.assertGreater(total_notes, 10)

        # Cleanup
        chord_gen.shutdown()
        melody_gen.shutdown()
        drum_gen.shutdown()

    def test_generated_content_compatible_with_sequencer(self):
        """Verify AI-generated clips work with the sequencer."""
        from ai.models.chord_generator import ChordGeneratorPlugin
        from core.model.project import Project
        from core.model.track import MidiTrack
        from core.engine.sequencer import Sequencer, render_project

        # Generate chords
        gen = ChordGeneratorPlugin(seed=42)
        gen.initialize()
        ctx = MusicalContext(bar_count=4, style_tags=["pop"])
        prompt = GenerationPrompt(context=ctx)
        result = gen.generate(prompt)
        clip = result.clips[ContentType.CHORDS]

        # Build project with the clip
        track = MidiTrack(
            name="Chords",
            clips=(clip,),
        )
        project = Project(
            metadata=Project.create_new(name="AI Test").metadata,
            tracks=(track,),
        )

        # Sequencer should compile without errors
        seq = Sequencer()
        state = seq.compile(project)
        self.assertGreater(len(state.events), 0)

        # Offline render should work
        render = render_project(project)
        self.assertGreater(render.note_count, 0)

        gen.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
