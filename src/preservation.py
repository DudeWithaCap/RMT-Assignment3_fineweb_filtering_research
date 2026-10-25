"""Protect research artifacts while allowing source and documentation edits."""

from pathlib import PurePosixPath

from .filter_assets import sha256_file


def is_editable_project_file(relative):
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.parts[0] in {"assignment_sources", "stage6"}:
        return True
    if path.name == ".gitignore":
        return True
    if len(path.parts) == 1 and (path.suffix == ".md" or path.name.startswith("requirements")):
        return True
    return path.suffix == ".py" and path.parts[0] in {
        "src", "stage35", "stage4", "stage5", "tests", "validation"
    }


def verify_research_files(root, records):
    protected = {name: record for name, record in records.items() if not is_editable_project_file(name)}
    # Clones omit local datasets and caches; present files remain protected.
    protected = {name: record for name, record in protected.items()
                 if (root / name).exists() or not is_local_artifact(name)}
    changed = [name for name, record in protected.items()
               if not (root / name).is_file() or sha256_file(root / name) != record["sha256"]]
    if changed:
        raise RuntimeError(f"Research artifacts changed: {changed}")
    return len(protected)


def is_local_artifact(relative):
    path = PurePosixPath(relative.replace("\\", "/"))
    return (path.parts[0] == "data" or "replay" in path.parts or ".mplconfig" in path.parts
            or path.parts[0] == "runs" and "data" in path.parts)
