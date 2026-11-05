"""Resolve historical artifacts after the results-folder reorganization."""

from pathlib import Path

from .filter_assets import ROOT


def historical_path(relative, root=ROOT):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Historical artifact paths must be relative")
    if relative.parts and relative.parts[0] in {"results", "runs"}:
        archived = root / "results/archive" / relative
        if (root / "results/archive" / relative.parts[0]).is_dir():
            return archived
    return root / relative
