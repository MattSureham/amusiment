"""
Automation models — time-varying parameter control.

Automation envelopes allow any device parameter to change over time.
Points are stored as (tick, value) pairs with interpolation curves.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from uuid import uuid4

from .time_model import Ticks


class InterpolationMode(Enum):
    """How values are interpolated between automation points."""
    LINEAR = auto()       # Straight line between points
    STEP = auto()         # Hold value until next point (abrupt change)
    SMOOTH = auto()       # Cubic hermite spline (smooth curve)
    EXPONENTIAL = auto()  # Exponential curve (useful for frequency/gain)


@dataclass(frozen=True)
class AutomationPoint:
    """
    A single control point on an automation envelope.
    
    Attributes:
        tick: Absolute tick position.
        value: Parameter value at this point.
        interpolation: How to interpolate FROM this point to the next.
        curve_tension: Curve shaping parameter (-1.0 to 1.0, 0 = linear).
    """
    tick: Ticks
    value: float
    interpolation: InterpolationMode = InterpolationMode.LINEAR
    curve_tension: float = 0.0  # -1.0 (convex) to 1.0 (concave)
    
    def __post_init__(self):
        object.__setattr__(self, "curve_tension",
                           max(-1.0, min(1.0, self.curve_tension)))
    
    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "value": self.value,
            "interpolation": self.interpolation.name.lower(),
            "curve_tension": self.curve_tension,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AutomationPoint":
        return cls(
            tick=Ticks(data["tick"]),
            value=data["value"],
            interpolation=InterpolationMode[data.get("interpolation", "linear").upper()],
            curve_tension=data.get("curve_tension", 0.0),
        )


@dataclass(frozen=True)
class AutomationLane:
    """
    An automation lane for a specific parameter on a specific device.
    
    Lanes are displayed as sub-tracks in the arrangement view.
    Each lane holds an envelope (ordered points).
    """
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = "Automation"
    device_id: str = ""
    parameter_id: str = ""
    color: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "device_id": self.device_id,
            "parameter_id": self.parameter_id,
            "color": self.color,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AutomationLane":
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("name", "Automation"),
            device_id=data.get("device_id", ""),
            parameter_id=data.get("parameter_id", ""),
            color=data.get("color"),
        )


@dataclass(frozen=True)
class AutomationEnvelope:
    """
    An automation envelope: a lane paired with its control points.
    
    The envelope evaluates to a parameter value at any given tick
    by interpolating between the surrounding points.
    
    Attributes:
        lane: Which automated parameter this envelope controls.
        points: Ordered tuple of automation points.
    """
    lane: AutomationLane = field(default_factory=AutomationLane)
    points: tuple[AutomationPoint, ...] = field(default_factory=tuple)
    
    @property
    def sorted_points(self) -> tuple[AutomationPoint, ...]:
        """Return points sorted by tick position."""
        return tuple(sorted(self.points, key=lambda p: p.tick))
    
    def value_at(self, tick: Ticks) -> float:
        """
        Evaluate the envelope at a given tick position.
        
        If tick is before the first point, returns the first point's value.
        If tick is after the last point, returns the last point's value.
        If exactly on a point, returns that point's value.
        Otherwise, interpolates between the two surrounding points.
        """
        sorted_pts = self.sorted_points
        
        if not sorted_pts:
            return 0.0
        
        if tick <= sorted_pts[0].tick:
            return sorted_pts[0].value
        
        if tick >= sorted_pts[-1].tick:
            return sorted_pts[-1].value
        
        # Find surrounding points
        for i in range(len(sorted_pts) - 1):
            p0 = sorted_pts[i]
            p1 = sorted_pts[i + 1]
            
            if p0.tick == p1.tick:
                continue
            
            if p0.tick <= tick <= p1.tick:
                frac = (tick - p0.tick) / (p1.tick - p0.tick)
                
                if p0.interpolation == InterpolationMode.STEP:
                    return p0.value
                
                elif p0.interpolation == InterpolationMode.LINEAR:
                    return p0.value + (p1.value - p0.value) * frac
                
                elif p0.interpolation == InterpolationMode.SMOOTH:
                    # Smoothstep (cubic hermite)
                    t = frac * frac * (3.0 - 2.0 * frac)
                    return p0.value + (p1.value - p0.value) * t
                
                elif p0.interpolation == InterpolationMode.EXPONENTIAL:
                    # Exponential curve (useful for dB/frequency scales)
                    if p0.value <= 0 or p1.value <= 0:
                        return p0.value + (p1.value - p0.value) * frac
                    import math
                    log_v0 = math.log(p0.value)
                    log_v1 = math.log(p1.value)
                    return math.exp(log_v0 + (log_v1 - log_v0) * frac)
        
        return sorted_pts[-1].value
    
    def with_point(self, point: AutomationPoint) -> "AutomationEnvelope":
        """Return a new envelope with the given point added (replaces at same tick)."""
        others = tuple(p for p in self.points if p.tick != point.tick)
        return AutomationEnvelope(lane=self.lane, points=others + (point,))
    
    def without_tick(self, tick: Ticks) -> "AutomationEnvelope":
        """Return a new envelope with the point at the given tick removed."""
        return AutomationEnvelope(
            lane=self.lane,
            points=tuple(p for p in self.points if p.tick != tick),
        )
    
    def to_dict(self) -> dict:
        return {
            "lane": self.lane.to_dict(),
            "points": [p.to_dict() for p in self.points],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AutomationEnvelope":
        return cls(
            lane=AutomationLane.from_dict(data.get("lane", {})),
            points=tuple(AutomationPoint.from_dict(p) for p in data.get("points", [])),
        )
