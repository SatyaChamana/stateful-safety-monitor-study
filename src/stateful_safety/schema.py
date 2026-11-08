"""Research record schema. No labels or target task text belong in a monitor view."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Label = Literal["harmful", "benign"]

@dataclass(frozen=True)
class Sequence:
    sequence_id: str
    split: str
    task_type: str
    label: Label
    steps: tuple[str, ...]
    harmful_index: int | None

    def __post_init__(self):
        if not self.sequence_id:
            raise ValueError("Missing sequence ID")
        if not self.steps or not all(isinstance(s, str) and s.strip() for s in self.steps):
            raise ValueError(f"{self.sequence_id}: steps must be nonempty strings")
        if self.label not in ("harmful", "benign"):
            raise ValueError(f"{self.sequence_id}: unknown label {self.label!r}")
        if self.label == "benign" and self.harmful_index is not None:
            raise ValueError(f"{self.sequence_id}: benign sequence cannot have harmful index")
        if self.harmful_index is not None and not (0 <= self.harmful_index < len(self.steps)):
            raise ValueError(f"{self.sequence_id}: harmful_index outside sequence")

@dataclass(frozen=True)
class MonitorView:
    sequence_id: str
    turn_index: int
    current_request: str
    history: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class Decision:
    score: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    parse_status: str | None = None  # paper mode: ok / refusal_sorry / no_number_mapped_benign
    continuous_score: float | None = None  # Phase 3 readout A: P('1')/(P('0')+P('1')), first token
    readout_mass: float | None = None      # P('0')+P('1') at that position (is the reply a digit?)
