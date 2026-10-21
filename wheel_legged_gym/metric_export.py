"""Export all raw scalar CSVs without Isaac Gym or plotting dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import re


def export_tensorboard(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run_dir = Path(run_dir)
    if not list(run_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError("No TensorBoard event files in " + str(run_dir))
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    out = run_dir / "csv"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for tag in accumulator.Tags()["scalars"]:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag) + ".csv"
        if name in manifest.values():
            name = name[:-4] + "_" + hashlib.sha256(tag.encode()).hexdigest()[:8] + ".csv"
        manifest[tag] = name
        with (out / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["wall_time", "step", "value"])
            writer.writerows((event.wall_time, event.step, event.value)
                             for event in accumulator.Scalars(tag))
    (out / "tags.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_series(run_dir, tag):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "csv/tags.json").read_text())
    if tag not in manifest:
        return [], []
    with (run_dir / "csv" / manifest[tag]).open(newline="") as stream:
        # In a resumed run, retain the last recorded value at each iteration.
        values = {float(row["step"]): float(row["value"]) for row in csv.DictReader(stream)}
    steps = sorted(values)
    return steps, [values[step] for step in steps]

