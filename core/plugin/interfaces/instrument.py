"""
Instrument plugin interface — sound generators.

Instruments produce audio in response to MIDI note events.
They can be synthesizers, samplers, or bridges to external plugins (VST, AU, etc.).

Each instrument plugin receives note-on/note-off events and renders audio
buffers through its process() method (called from the realtime audio thread).
"""

from abc import abstractmethod
from typing import Optional

from .base import PluginBase, PluginManifest, PluginCategory
from ...model.note import NoteEvent, NotePitch, NoteVelocity
from ...model.time_model import Ticks, Samples


class InstrumentPlugin(PluginBase):
    """
    Abstract instrument plugin — a sound source that responds to MIDI notes.
    
    Lifecycle:
        1. initialize() — allocate synth engine / load samples
        2. process() — called repeatedly from realtime audio thread
        3. note_on() / note_off() — called when MIDI events occur
        4. shutdown() — release resources
    
    The process() method must be realtime-safe — no allocation, no locks,
    no I/O. Use lock-free queues for parameter updates.
    """
    
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="",
            name="Unnamed Instrument",
            version="0.1.0",
            category=PluginCategory.INSTRUMENT,
            description="An instrument plugin",
            capabilities=["instrument.synth"],
        )
    
    @abstractmethod
    def note_on(self, note: NoteEvent, tick: Ticks) -> None:
        """
        Trigger a note-on event.
        
        Called from the sequencer thread (not realtime). The instrument
        should schedule the voice internally.
        
        Args:
            note: The note event with pitch, velocity, channel.
            tick: Current sequencer tick position.
        """
        ...
    
    @abstractmethod
    def note_off(self, note: NoteEvent, tick: Ticks) -> None:
        """
        Trigger a note-off event for a currently playing note.
        
        Args:
            note: The note event to end (matched by pitch + channel + voice).
            tick: Current sequencer tick position.
        """
        ...
    
    @abstractmethod
    def process(self, output_buffer: list[list[float]], num_samples: int,
                sample_rate: int) -> None:
        """
        Render audio into the output buffer.
        
        Called from the realtime audio thread. Must be lock-free and
        allocation-free.
        
        Args:
            output_buffer: Stereo output buffer: [[left_samples], [right_samples]].
                           Pre-allocated to num_samples length.
            num_samples: Number of samples to render.
            sample_rate: Sample rate in Hz.
        """
        ...
    
    def all_notes_off(self) -> None:
        """
        Immediately silence all active voices (panic button).
        
        Default implementation calls note_off for all tracked notes.
        Override for more efficient implementation.
        """
        pass
    
    def set_parameter(self, param_id: str, value: float) -> None:
        """
        Update an instrument parameter (e.g., filter cutoff, envelope attack).
        
        Must be realtime-safe — use atomic writes or lock-free queues.
        
        Args:
            param_id: Parameter identifier.
            value: New value (normalized or in the parameter's native range).
        """
        pass
    
    def get_latency_samples(self) -> int:
        """
        Return the instrument's processing latency in samples.
        
        Used by the audio engine for delay compensation.
        """
        return 0
    
    @property
    def polyphony(self) -> int:
        """Maximum number of simultaneous voices."""
        return 16
    
    @property
    def supports_mpe(self) -> bool:
        """Whether this instrument supports MIDI Polyphonic Expression."""
        return False
