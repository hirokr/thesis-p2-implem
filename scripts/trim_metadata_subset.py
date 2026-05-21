from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(r"C:\t309\dataSubset")
FILES = ["av1.metadata.json", "dfdc.metadata.json", "faceavceleb.metadata.json"]
LIMIT = 5000


def trim_data(data: object) -> object:
    if isinstance(data, list):
        print(len(data))
        return data[:LIMIT]
    if isinstance(data, dict):
        items = list(data.items())
        print(len(data))
        return dict(items[:LIMIT])
    raise TypeError(f"Unsupported JSON root type: {type(data).__name__}")


def main() -> None:
    for name in FILES:
        path = BASE_DIR / name
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        original_count = len(data)
        trimmed = trim_data(data)
        new_count = len(trimmed)
        if new_count == original_count and original_count <= LIMIT:
            print(f"{name}: {original_count} entries (<= {LIMIT}), unchanged")
            continue
        print(f"{name}: {original_count} -> {new_count}")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(trimmed, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
