"""Record how many rows each cleaning step removes, so every funnel is auditable."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FilterReport:
    name: str
    initial: int = 0
    steps: list[tuple[str, int]] = field(default_factory=list)

    def start(self, n: int) -> None:
        self.initial += n

    def add(self, step: str, removed: int) -> None:
        for i, (s, n) in enumerate(self.steps):
            if s == step:
                self.steps[i] = (s, n + removed)
                return
        self.steps.append((step, removed))

    @property
    def remaining(self) -> int:
        return self.initial - sum(n for _, n in self.steps)

    def to_frame(self) -> pd.DataFrame:
        rows, left = [("initial", 0, self.initial)], self.initial
        for step, removed in self.steps:
            left -= removed
            rows.append((step, removed, left))
        return pd.DataFrame(rows, columns=["step", "removed", "remaining"])

    @classmethod
    def from_frames(cls, name: str, frames: list[pd.DataFrame]) -> FilterReport:
        """Add up the funnels of separately processed partitions."""
        report = cls(name)
        for frame in frames:
            report.start(int(frame.loc[frame["step"] == "initial", "remaining"].iloc[0]))
            for row in frame[frame["step"] != "initial"].itertuples():
                report.add(row.step, int(row.removed))
        return report

    def __str__(self) -> str:
        return f"{self.name}\n" + self.to_frame().to_string(index=False)
