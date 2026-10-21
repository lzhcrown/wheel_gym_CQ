"""TensorBoard scalars and append-only CSVs with identical raw values."""
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from torch.utils.tensorboard import SummaryWriter


class TrainingSummaryWriter(SummaryWriter):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.csv_dir = Path(self.log_dir) / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._csv_files = {}
        manifest = self.csv_dir / "tags.json"
        self._csv_tags = json.loads(manifest.read_text()) if manifest.exists() else {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, **kwargs):
        walltime = time.time() if walltime is None else walltime
        super().add_scalar(tag, scalar_value, global_step, walltime, **kwargs)
        if tag not in self._csv_files:
            filename = self._csv_tags.get(tag)
            if filename is None:
                filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag) + ".csv"
                if filename in self._csv_tags.values():
                    filename = filename[:-4] + "_" + hashlib.sha256(tag.encode()).hexdigest()[:8] + ".csv"
                self._csv_tags[tag] = filename
                (self.csv_dir / "tags.json").write_text(json.dumps(self._csv_tags, indent=2) + "\n")
            path = self.csv_dir / filename
            new_file = not path.exists() or path.stat().st_size == 0
            stream = path.open("a", newline="", buffering=1)
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(["wall_time", "step", "value"])
            self._csv_files[tag] = (stream, writer)
        value = scalar_value.item() if hasattr(scalar_value, "item") else float(scalar_value)
        self._csv_files[tag][1].writerow([walltime, 0 if global_step is None else global_step, value])

    def flush(self):
        super().flush()
        for stream, _ in getattr(self, "_csv_files", {}).values():
            stream.flush()

    def close(self):
        super().close()
        for stream, _ in getattr(self, "_csv_files", {}).values():
            stream.close()
        self._csv_files = {}
