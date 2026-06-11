"""
Playback clock — converts between ticks and real time during playback.

The clock is the tempo-aware timekeeper. It tracks the current tick position
and provides methods to advance time (in ticks or seconds) while accounting
for the project's tempo map.

This is designed to be called from an audio callback thread:
1. Audio backend requests N samples for the next buffer
2. Clock computes how many ticks those N samples represent
3. Sequencer returns events in that tick range
4. Events are dispatched to the audio graph

Key design:
- All time math is exact (no floating-point drift in tick accumulation)
- Tempo changes take effect at their exact tick positions
- Sample-accurate timing is achievable with sub-tick interpolation
"""

from dataclasses import dataclass, field
from typing import Optional, Iterator

from ..model.time_model import (
    Ticks, Seconds, Samples, PPQ,
    ticks_to_seconds, seconds_to_ticks,
    ticks_to_samples, samples_to_ticks,
    TempoMap, TempoChange,
)


@dataclass
class ClockPosition:
    """A point-in-time snapshot of the playback clock."""

    tick: Ticks = Ticks(0)
    """Current absolute tick position."""

    elapsed_seconds: Seconds = Seconds(0.0)
    """Elapsed wall-clock seconds from tick 0."""

    current_bpm: float = 120.0
    """Effective BPM at the current position."""

    beat: float = 0.0
    """Current beat position (quarter notes from tick 0)."""

    bar: int = 1
    """Current bar number (1-based)."""

    beat_in_bar: float = 1.0
    """Current beat within the current bar (1-based, e.g., 3.5 = beat 3.5)."""

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "elapsed_seconds": self.elapsed_seconds,
            "current_bpm": self.current_bpm,
            "beat": self.beat,
            "bar": self.bar,
            "beat_in_bar": self.beat_in_bar,
        }


@dataclass
class PlaybackClock:
    """
    Tempo-aware playback clock that tracks tick position and real time.

    The clock does NOT run itself — it is advanced externally by the
    audio callback (real-time) or by the render loop (offline). This
    keeps it deterministic and testable.

    Usage:
        clock = PlaybackClock(tempo_map, start_bpm=120.0)
        clock.seek(Ticks(0))

        # In audio callback:
        samples_to_render = 256
        events = clock.advance_by_samples(samples_to_render, sample_rate=44100)
        for event_range in events:
            process(event_range)

        # Or tick-by-tick (for offline rendering):
        while clock.tick < end_tick:
            clock.advance_by_ticks(Ticks(1))
            process(clock.current_events)
    """

    tempo_map: TempoMap = field(default_factory=TempoMap)
    start_bpm: float = 120.0
    _tick: Ticks = Ticks(0)
    _elapsed_seconds: float = 0.0

    # Time-signature parameters for bar/beat display
    beats_per_bar: int = 4
    beat_unit: int = 4  # denominator (4 = quarter note)

    @property
    def tick(self) -> Ticks:
        """Current absolute tick position."""
        return self._tick

    @property
    def elapsed_seconds(self) -> Seconds:
        """Elapsed wall-clock seconds from tick 0."""
        return Seconds(self._elapsed_seconds)

    @property
    def current_bpm(self) -> float:
        """Effective BPM at the current tick position."""
        return self.tempo_map.bpm_at(self._tick)

    @property
    def position(self) -> ClockPosition:
        """Get a full position snapshot."""
        return self._compute_position(self._tick)

    def _compute_position(self, tick: Ticks) -> ClockPosition:
        """Compute the full ClockPosition for a given tick."""
        elapsed = ticks_to_seconds(tick, self.tempo_map, self.start_bpm)
        total_beats = tick / PPQ
        bpb = self.beats_per_bar  # beats per bar

        # Bar number (1-based)
        if bpb > 0:
            bar = int(total_beats / bpb) + 1
            beat_in_bar = (total_beats % bpb) + 1.0
        else:
            bar = 1
            beat_in_bar = 1.0

        return ClockPosition(
            tick=tick,
            elapsed_seconds=elapsed,
            current_bpm=self.tempo_map.bpm_at(tick),
            beat=total_beats,
            bar=bar,
            beat_in_bar=beat_in_bar,
        )

    def seek(self, tick: Ticks) -> ClockPosition:
        """
        Move the playhead to an absolute tick position.

        This is used for transport seek, loop wrap, and stop/reset.
        Returns the new clock position.
        """
        self._tick = tick
        self._elapsed_seconds = ticks_to_seconds(
            tick, self.tempo_map, self.start_bpm
        )
        return self.position

    def seek_seconds(self, seconds: Seconds) -> ClockPosition:
        """
        Move the playhead to a specific wall-clock time.

        Converts seconds to ticks using the tempo map, then seeks.
        """
        tick = seconds_to_ticks(seconds, self.tempo_map, self.start_bpm)
        return self.seek(tick)

    def seek_bar(self, bar: int, beat: float = 1.0) -> ClockPosition:
        """
        Move the playhead to a specific bar and beat.

        Args:
            bar: Bar number (1-based).
            beat: Beat within bar (1-based, e.g., 1.0 = downbeat).
        """
        total_beats = (bar - 1) * self.beats_per_bar + (beat - 1.0)
        tick = Ticks(round(total_beats * PPQ))
        return self.seek(tick)

    def advance_by_ticks(self, delta_ticks: Ticks) -> Ticks:
        """
        Advance the clock by a given number of ticks.

        For offline rendering or tick-precise stepping.

        Args:
            delta_ticks: Number of ticks to advance.

        Returns:
            The new absolute tick position.
        """
        self._tick = Ticks(self._tick + delta_ticks)
        self._elapsed_seconds = ticks_to_seconds(
            self._tick, self.tempo_map, self.start_bpm
        )
        return self._tick

    def advance_by_samples(
        self, sample_count: int, sample_rate: int
    ) -> tuple[Ticks, Ticks]:
        """
        Advance the clock by a given number of audio samples.

        This is the primary real-time advance method. It computes the
        tick range covered by the sample block, accounting for tempo
        changes within the block.

        Args:
            sample_count: Number of audio samples to advance.
            sample_rate: Audio sample rate (e.g., 44100).

        Returns:
            (start_tick, end_tick) — the tick range covered by this
            sample block. The caller should query the sequencer for
            events in [start_tick, end_tick).
        """
        start_tick = self._tick
        start_seconds = self._elapsed_seconds

        # Advance elapsed time
        self._elapsed_seconds += sample_count / sample_rate

        # Convert new elapsed time back to ticks
        new_tick = seconds_to_ticks(
            Seconds(self._elapsed_seconds),
            self.tempo_map,
            self.start_bpm,
        )
        self._tick = new_tick

        return (start_tick, new_tick)

    def ticks_to_seconds(self, ticks: Ticks) -> Seconds:
        """Convert a tick duration to wall-clock seconds at the current tempo."""
        return ticks_to_seconds(ticks, self.tempo_map, self.start_bpm)

    def seconds_to_ticks(self, seconds: Seconds) -> Ticks:
        """Convert wall-clock seconds to ticks at the current tempo."""
        return seconds_to_ticks(seconds, self.tempo_map, self.start_bpm)

    def ticks_to_samples(self, ticks: Ticks, sample_rate: int) -> Samples:
        """Convert ticks to audio samples at the current tempo."""
        return ticks_to_samples(ticks, sample_rate, self.tempo_map, self.start_bpm)

    def is_before(self, tick: Ticks) -> bool:
        """Check if the playhead is before the given tick."""
        return self._tick < tick

    def is_at_or_after(self, tick: Ticks) -> bool:
        """Check if the playhead is at or after the given tick."""
        return self._tick >= tick

    def reset(self) -> None:
        """Reset the clock to tick 0."""
        self._tick = Ticks(0)
        self._elapsed_seconds = 0.0
