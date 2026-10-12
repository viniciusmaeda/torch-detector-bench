"""Reproducible experiment metadata and publication artifact writers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    """Atomically publish metadata; reject non-standard JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def json_safe(value):
    """Represent undefined statistics as null, never as zero or invalid JSON."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_figure(figure, path: Path) -> None:
    """Save a raster preview and editable/vector publication formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_table(frame: pd.DataFrame, path: Path, caption: str = "") -> None:
    """Export machine-readable and LaTeX tables without requiring Jinja2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    def escape(value):
        text = str(value)
        return "".join({"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                        "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c) for c in text)
    rows = [" & ".join(escape(c) for c in frame.columns) + r" \\", r"\hline"]
    rows += [" & ".join(escape(v) for v in row) + r" \\" for row in frame.itertuples(index=False, name=None)]
    tex = "\n".join([r"\begin{table}[ht]", r"\centering", r"\caption{" + escape(caption) + "}",
                     r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\hline", *rows,
                     r"\hline", r"\end{tabular}", r"\end{table}"])
    path.with_suffix(".tex").write_text(tex + "\n", encoding="utf-8")
