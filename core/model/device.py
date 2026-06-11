"""
Device chain models — the signal processing pipeline for each track.

Each track has a device chain: Instrument → Effect1 → Effect2 → ... → Output.
Devices can be instruments (sound sources) or effects (signal processors).

The device chain is a directed list — signal flows left to right.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from uuid import uuid4


class DeviceType(Enum):
    """Types of processing devices."""
    INSTRUMENT = auto()  # Sound generator (synth, sampler, etc.)
    EFFECT = auto()      # Signal processor (reverb, delay, EQ, etc.)
    UTILITY = auto()     # Utility (gain, panner, analyzer, etc.)


@dataclass(frozen=True)
class DeviceParameter:
    """
    A single exposed parameter on a device, suitable for automation and UI knobs.

    Attributes:
        id: Unique parameter identifier within the device.
        name: Human-readable parameter name.
        value: Current value.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        default_value: Default/reset value.
        unit: Optional unit string (e.g., "dB", "Hz", "%").
        automation_target: Whether this parameter can be automated.
    """
    id: str
    name: str
    value: float = 0.0
    min_value: float = 0.0
    max_value: float = 1.0
    default_value: float = 0.0
    unit: Optional[str] = None
    automation_target: bool = True
    
    @property
    def normalized(self) -> float:
        """Return the value normalized to 0.0 - 1.0 range."""
        rng = self.max_value - self.min_value
        if rng == 0:
            return 0.5
        return (self.value - self.min_value) / rng
    
    def with_value(self, value: float) -> "DeviceParameter":
        clamped = max(self.min_value, min(self.max_value, value))
        return DeviceParameter(
            id=self.id, name=self.name, value=clamped,
            min_value=self.min_value, max_value=self.max_value,
            default_value=self.default_value, unit=self.unit,
            automation_target=self.automation_target,
        )


@dataclass(frozen=True)
class Device:
    """
    Base device in a track's processing chain.
    
    Attributes:
        id: Unique device instance ID.
        name: Human-readable device name.
        device_type: Instrument, effect, or utility.
        plugin_id: Reference to the plugin that implements this device.
        parameters: Exposed parameters for automation and UI.
        enabled: Whether this device is active in the chain.
        preset_name: Currently loaded preset (if any).
    """
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = "Device"
    device_type: DeviceType = DeviceType.EFFECT
    plugin_id: Optional[str] = None
    parameters: tuple[DeviceParameter, ...] = field(default_factory=tuple)
    enabled: bool = True
    preset_name: Optional[str] = None
    
    def get_parameter(self, param_id: str) -> Optional[DeviceParameter]:
        """Get a parameter by ID."""
        for p in self.parameters:
            if p.id == param_id:
                return p
        return None
    
    def with_parameter(self, param_id: str, value: float) -> "Device":
        """Return a new device with the given parameter updated."""
        new_params = tuple(
            p.with_value(value) if p.id == param_id else p
            for p in self.parameters
        )
        return Device(
            id=self.id, name=self.name, device_type=self.device_type,
            plugin_id=self.plugin_id, parameters=new_params,
            enabled=self.enabled, preset_name=self.preset_name,
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "device_type": self.device_type.name.lower(),
            "plugin_id": self.plugin_id,
            "enabled": self.enabled,
            "preset_name": self.preset_name,
            "parameters": [
                {
                    "id": p.id, "name": p.name, "value": p.value,
                    "min_value": p.min_value, "max_value": p.max_value,
                    "default_value": p.default_value, "unit": p.unit,
                    "automation_target": p.automation_target,
                }
                for p in self.parameters
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Device":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "Device"),
            device_type=DeviceType[data.get("device_type", "EFFECT").upper()],
            plugin_id=data.get("plugin_id"),
            enabled=data.get("enabled", True),
            preset_name=data.get("preset_name"),
            parameters=tuple(
                DeviceParameter(
                    id=p["id"], name=p["name"], value=p["value"],
                    min_value=p.get("min_value", 0.0),
                    max_value=p.get("max_value", 1.0),
                    default_value=p.get("default_value", p["value"]),
                    unit=p.get("unit"),
                    automation_target=p.get("automation_target", True),
                )
                for p in data.get("parameters", [])
            ),
        )


@dataclass(frozen=True)
class InstrumentDevice(Device):
    """An instrument device (synth, sampler, VSTi bridge)."""
    device_type: DeviceType = DeviceType.INSTRUMENT
    
    # Common instrument parameters
    polyphony: int = 16
    transpose: int = 0  # Semitones
    fine_tune: float = 0.0  # Cents
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "polyphony": self.polyphony,
            "transpose": self.transpose,
            "fine_tune": self.fine_tune,
        })
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "InstrumentDevice":
        base = Device.from_dict.__func__(InstrumentDevice, data)
        return cls(
            id=base.id, name=base.name, plugin_id=base.plugin_id,
            parameters=base.parameters, enabled=base.enabled,
            preset_name=base.preset_name,
            polyphony=data.get("polyphony", 16),
            transpose=data.get("transpose", 0),
            fine_tune=data.get("fine_tune", 0.0),
        )


@dataclass(frozen=True)
class EffectDevice(Device):
    """An effect device (reverb, delay, EQ, compressor, etc.)."""
    device_type: DeviceType = DeviceType.EFFECT
    wet_dry: float = 1.0  # 0.0 = fully dry, 1.0 = fully wet
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["wet_dry"] = self.wet_dry
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "EffectDevice":
        base = Device.from_dict.__func__(EffectDevice, data)
        return cls(
            id=base.id, name=base.name, plugin_id=base.plugin_id,
            parameters=base.parameters, enabled=base.enabled,
            preset_name=base.preset_name,
            wet_dry=data.get("wet_dry", 1.0),
        )


@dataclass(frozen=True)
class DeviceChain:
    """
    Ordered list of devices on a track: Instrument → Effects chain.
    
    Invariant: At most one instrument device, always first if present.
    Effects follow the instrument in the order they process the signal.
    """
    devices: tuple[Device, ...] = field(default_factory=tuple)
    
    @property
    def instrument(self) -> Optional[Device]:
        """Return the instrument device, if any."""
        for d in self.devices:
            if d.device_type == DeviceType.INSTRUMENT:
                return d
        return None
    
    @property
    def effects(self) -> tuple[Device, ...]:
        """Return all effect devices in processing order."""
        return tuple(d for d in self.devices if d.device_type == DeviceType.EFFECT)
    
    def with_device(self, device: Device) -> "DeviceChain":
        """Return a new chain with an added device (replaces at same position if ID matches)."""
        existing = tuple(d for d in self.devices if d.id != device.id)
        return DeviceChain(devices=existing + (device,))
    
    def without_device(self, device_id: str) -> "DeviceChain":
        """Return a new chain with the given device removed."""
        return DeviceChain(devices=tuple(d for d in self.devices if d.id != device_id))
    
    def with_parameter(self, device_id: str, param_id: str, value: float) -> "DeviceChain":
        """Return a new chain with a parameter on a specific device updated."""
        return DeviceChain(devices=tuple(
            d.with_parameter(param_id, value) if d.id == device_id else d
            for d in self.devices
        ))
    
    def to_dict(self) -> list[dict]:
        return [d.to_dict() for d in self.devices]
    
    @classmethod
    def from_dict(cls, data: list[dict]) -> "DeviceChain":
        devices = []
        for d in data:
            dtype = d.get("device_type", "effect").upper()
            if dtype == "INSTRUMENT":
                devices.append(InstrumentDevice.from_dict(d))
            else:
                devices.append(EffectDevice.from_dict(d))
        return cls(devices=tuple(devices))
    
    def __bool__(self) -> bool:
        return len(self.devices) > 0
