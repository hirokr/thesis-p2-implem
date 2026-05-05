import json
import random
from pathlib import Path

BASE = Path(r"c:\t309\dataSubset")

PATHS = {
    "av1": BASE / "av1.metadata.json",
    "dfdc": BASE / "dfdc.metadata.json",
    "faceavceleb": BASE / "faceavceleb.metadata.json",
    "faceforensics": BASE / "faceforensics.metadata.json",
    "lavdf": BASE / "lavdf.metadata.json",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def build_records():
    records = []

    av1 = load_json(PATHS["av1"])
    for entry in av1:
        key = entry.get("file")
        if not key:
            continue
        is_real = entry.get("modify_type") == "real"
        records.append(("av1", key, is_real))

    dfdc = load_json(PATHS["dfdc"])
    for key, entry in dfdc.items():
        label = str(entry.get("label", "")).upper()
        is_real = label == "REAL"
        records.append(("dfdc", key, is_real))

    faceavceleb = load_json(PATHS["faceavceleb"])
    for entry in faceavceleb:
        key = entry.get("file")
        if not key:
            continue
        method = str(entry.get("method", "")).lower()
        vid_type = str(entry.get("type", "")).lower()
        is_real = method == "real" or vid_type == "realvideo-realaudio"
        records.append(("faceavceleb", key, is_real))

    faceforensics = load_json(PATHS["faceforensics"])
    for entry in faceforensics:
        key = entry.get("file")
        if not key:
            continue
        label = str(entry.get("label", "")).upper()
        is_real = label == "REAL"
        records.append(("faceforensics", key, is_real))

    lavdf = load_json(PATHS["lavdf"])
    for entry in lavdf:
        key = entry.get("file")
        if not key:
            continue
        n_fakes = entry.get("n_fakes", 0)
        is_real = n_fakes == 0
        records.append(("lavdf", key, is_real))

    return records


def main():
    rng = random.Random(42)
    records = build_records()

    real = [r for r in records if r[2]]
    fake = [r for r in records if not r[2]]

    min_count = min(len(real), len(fake))
    target_total = min(15000, min_count * 2)
    per_class = target_total // 2

    real_sample = rng.sample(real, per_class)
    fake_sample = rng.sample(fake, per_class)

    selected = {(ds, key) for ds, key, _ in real_sample + fake_sample}

    # Filter and overwrite datasets
    av1 = load_json(PATHS["av1"])
    av1 = [e for e in av1 if ("av1", e.get("file")) in selected]
    dump_json(PATHS["av1"], av1)

    dfdc = load_json(PATHS["dfdc"])
    dfdc = {k: v for k, v in dfdc.items() if ("dfdc", k) in selected}
    dump_json(PATHS["dfdc"], dfdc)

    faceavceleb = load_json(PATHS["faceavceleb"])
    faceavceleb = [e for e in faceavceleb if ("faceavceleb", e.get("file")) in selected]
    dump_json(PATHS["faceavceleb"], faceavceleb)

    faceforensics = load_json(PATHS["faceforensics"])
    faceforensics = [e for e in faceforensics if ("faceforensics", e.get("file")) in selected]
    dump_json(PATHS["faceforensics"], faceforensics)

    lavdf = load_json(PATHS["lavdf"])
    lavdf = [e for e in lavdf if ("lavdf", e.get("file")) in selected]
    dump_json(PATHS["lavdf"], lavdf)

    print("Balanced subset written.")
    print(f"Total: {target_total} (real={per_class}, fake={per_class})")


if __name__ == "__main__":
    main()
