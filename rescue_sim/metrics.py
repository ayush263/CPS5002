from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import csv
import os

from .utils import ensure_dir, avg


@dataclass
class Metrics:
    survivors_saved: int = 0
    survivors_dead: int = 0
    agents_lost: int = 0
    ethical_score: float = 0.0

    logs: List[Dict[str, Any]] = field(default_factory=list)

    def log_step(self, row: Dict[str, Any]) -> None:
        self.logs.append(row)

    def export_csv(self, path: str) -> str:
        ensure_dir(path)
        out = os.path.join(path, "run_metrics.csv")
        if not self.logs:
            return out
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.logs[0].keys()))
            w.writeheader()
            w.writerows(self.logs)
        return out


@dataclass
class BatchAggregator:
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, outcome: Dict[str, Any], run_dir: str) -> None:
        row = dict(outcome)
        row["run_dir"] = run_dir
        self.rows.append(row)

    def export_csv(self, out_dir: str) -> str:
        ensure_dir(out_dir)
        out = os.path.join(out_dir, "batch_results.csv")
        if not self.rows:
            return out
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        return out
