"""Explicit acquisition of original filter assets; no automatic downloads at run time."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
DATATROVE_REVISION = "0f2c69f8249aa0c53ebcf10afa2394da506a953f"
NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
ASSET_BASE = f"https://raw.githubusercontent.com/huggingface/datatrove/{DATATROVE_REVISION}/src/datatrove/assets/"
ASSETS = {
    "url": {name: ASSET_BASE + name for name in (
        "url_filterblacklists.tar.gz", "banned_words.txt", "banned_subwords.txt", "soft_banned_words.txt")},
    "language": {"lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"},
    "tokenizer": {"punkt_tab.zip": f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_REVISION}/packages/tokenizers/punkt_tab.zip"},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_asset(directory, name):
    """Reject absent, unrecorded or changed assets rather than weakening a filter."""
    directory = Path(directory)
    lock_path = directory / "assets_manifest.json"
    if not lock_path.exists():
        raise FileNotFoundError("Missing filter assets. Run: python -m src.filter_assets")
    record = json.loads(lock_path.read_text(encoding="utf-8")).get(name)
    pinned = json.loads((ROOT / "filter_assets.lock.json").read_text(encoding="utf-8")).get(name)
    path = directory / name
    if (not record or not pinned or record != pinned or not path.is_file()
            or sha256_file(path) != pinned["sha256"]):
        raise ValueError(f"Missing or changed asset: {name}. Run python -m src.filter_assets")
    return path, record


def fetch_assets(directory, group="all"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / "assets_manifest.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    pinned = json.loads((ROOT / "filter_assets.lock.json").read_text(encoding="utf-8"))
    groups = ASSETS if group == "all" else {group: ASSETS[group]}
    for assets in groups.values():
        for name, url in assets.items():
            path = directory / name
            if path.exists():
                _, record = checked_asset(directory, name)
                if record["url"] != url:
                    raise ValueError(f"Asset URL changed for {name}; archive the old asset first")
                print(f"Verified cached {name}")
                continue
            partial = path.with_suffix(path.suffix + ".partial")
            if partial.exists():
                raise FileExistsError(f"Archive incomplete asset before retrying: {partial}")
            digest, size = hashlib.sha256(), 0
            print(f"Downloading {name} from {url}", flush=True)
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with partial.open("wb") as target:
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 64 * 1024 * 1024:
                            raise RuntimeError(f"Asset exceeds 64 MiB limit: {name}")
                        target.write(chunk)
                        digest.update(chunk)
            if name not in pinned or digest.hexdigest() != pinned[name]["sha256"] or size != pinned[name]["bytes"]:
                raise ValueError(f"Downloaded asset differs from pinned checksum: {name}")
            partial.rename(path)
            lock[name] = {"url": url, "sha256": digest.hexdigest(), "bytes": size}
            temporary_lock = lock_path.with_suffix(".json.partial")
            temporary_lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
            temporary_lock.replace(lock_path)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "filter_config.json")
    parser.add_argument("--group", choices=["all", *ASSETS], default="all")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        fetch_assets(ROOT / config["assets_directory"], args.group)
    except Exception as error:
        print(f"Asset acquisition failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
