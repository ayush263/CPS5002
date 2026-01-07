from __future__ import annotations
import csv
import os
from typing import List, Dict

import matplotlib.pyplot as plt


def _read_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if v.replace(".", "", 1).isdigit() else v for k, v in row.items()})
    return rows


def plot_run_metrics(out_dir: str) -> None:
    """
    Generates:
      plot_survivors_saved_vs_time.png
      plot_survivors_alive_vs_time.png
      plot_agents_lost_vs_time.png
      plot_ethical_score_vs_time.png
      plot_total_hazard_intensity_vs_time.png
    """
    csv_path = os.path.join(out_dir, "run_metrics.csv")
    rows = _read_csv(csv_path)
    if not rows:
        return

    steps = [int(r["step"]) for r in rows]
    saved = [int(r["saved"]) for r in rows]
    alive = [int(r["alive"]) for r in rows]
    dead = [int(r["dead"]) for r in rows]
    agents_lost = [int(r["agents_lost"]) for r in rows]
    ethical = [float(r["ethical_score"]) for r in rows]
    hazard_int = [float(r["total_hazard_intensity"]) for r in rows]

    def _plot(x, y, xlabel, ylabel, title, filename):
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    _plot(steps, saved, "Step", "Saved (cumulative)", "Survivors Saved vs Time", "plot_survivors_saved_vs_time.png")
    _plot(steps, alive, "Step", "Alive survivors", "Survivors Alive vs Time", "plot_survivors_alive_vs_time.png")
    _plot(steps, agents_lost, "Step", "Agents lost (cumulative)", "Agent Loss vs Time", "plot_agents_lost_vs_time.png")
    _plot(steps, ethical, "Step", "Ethical score", "Ethical Score vs Time", "plot_ethical_score_vs_time.png")
    _plot(steps, hazard_int, "Step", "Total hazard intensity", "Hazard Intensity vs Time", "plot_total_hazard_intensity_vs_time.png")


def plot_batch_summary(batch_csv: str, out_dir: str) -> None:
    """
    plot_batch_summary.png
    Shows averages across runs (bar chart).
    """
    rows = _read_csv(batch_csv)
    if not rows:
        return

    avg_saved = sum(int(r["survivors_saved"]) for r in rows) / len(rows)
    avg_dead = sum(int(r["survivors_dead"]) for r in rows) / len(rows)
    avg_agents_lost = sum(int(r["agents_lost"]) for r in rows) / len(rows)
    avg_ethics = sum(float(r["ethical_score"]) for r in rows) / len(rows)

    labels = ["Saved", "Dead", "Agents Lost", "Ethical Score"]
    values = [avg_saved, avg_dead, avg_agents_lost, avg_ethics]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Batch Summary (Averages across runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_batch_summary.png"))
    plt.close()
