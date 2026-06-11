"""
Effect plugin interface — audio signal processors.

Effects sit in a track's device chain after the instrument (or audio source)
and process the audio stream. Examples: reverb, delay, EQ, compressor, distortion.

Effects receive audio buffers through their process() method and can also
respond to parameter changes and automation.
"""

from abc import abstractmethod
from typing import Optional

from .base import PluginBase, PluginManifest, PluginCategory


class EffectPlugin(PluginBase):
    """
    Abstract effect plugin — processes audio in a device chain.
    
    Lifecycle:
        1. initialize() — allocate DSP resources
        2. prepare() — called when sample rate or block size changes
        3. process() — called repeatedly from realtime audio thread
        4. reset() — clear internal state (e.g., reverb tail)
        5. shutdown() — release resources
    
    All audio processing methods must be realtime-safe.
    """
    
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="",
            name="Unnamed Effect",
            version="0.1.0",
            category=PluginCategory.EFFECT,
            description="An effect plugin",
            capabilities=["effect.process"],
        )
    
    @abstractmethod
    def prepare(self, sample_rate: int, max_block_size: int) -> None:
        """
        Prepare the effect for processing at the given sample rate.
        
        Called when the audio engine is set up or when sample rate changes.
        Allocate internal buffers here.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            max_block_size: Maximum number of samples per process() call.
        """
        ...
    
    @abstractmethod
    def process(self, input_buffer: list[list[float]],
                output_buffer: list[list[float]],
                num_samples: int) -> None:
        """
        Process an audio block through the effect.
        
        Called from the realtime audio thread.
        
        Args:
            input_buffer: Stereo input [[left], [right]].
            output_buffer: Stereo output [[left], [right]] (pre-allocated).
            num_samples: Number of samples in this block.
        """
        ...
    
    def process_mono(self, input_buffer: list[float],
                     output_buffer: list[float],
                     num_samples: int) -> None:
        """
        Mono processing path (optional override).
        
        Default implementation converts to stereo and processes.
        """
        stereo_in = [input_buffer, input_buffer[:]]
        stereo_out = [output_buffer, output_buffer[:]]
        self.process(stereo_in, stereo_out, num_samples)
    
    def reset(self) -> None:
        """
        Reset internal state — clear delay lines, reverb tails, etc.
        
        Called when transport stops or when seeking.
        Default is no-op.
        """
        pass
    
    def set_parameter(self, param_id: str, value: float) -> None:
        """
        Update an effect parameter in a realtime-safe manner.
        
        Args:
            param_id: Parameter identifier.
            value: New value.
        """
        pass
    
    def get_latency_samples(self) -> int:
        """
        Return additional latency introduced by this effect (for delay compensation).
        
        Examples:
            - Look-ahead limiter: typically 512-2048 samples
            - Linear phase EQ: FFT size dependent
            - Most effects: 0
        """
        return 0
    
    @property
    def is_mono_compatible(self) -> bool:
        """Whether this effect supports mono input."""
        return True
    
    @property
    def tail_length_samples(self) -> int:
        """
        Length of the effect tail in samples (for reverb/delay tails).
        
        The audio engine uses this to know how long to keep rendering
        after transport stops.
        """
        return 0
