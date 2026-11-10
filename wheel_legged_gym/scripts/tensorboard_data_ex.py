"""Export raw scalar CSVs, honoring runner.csv_tags in each saved train_cfg.json."""
import argparse
from pathlib import Path
import sys

# Also supports invocation by file path without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wheel_legged_gym.metric_export import export_tensorboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Explicit training run directories to export")
    args = parser.parse_args()
    for run in args.runs:
        manifest = export_tensorboard(run)
        print("Exported {} scalar CSV files to {}".format(len(manifest), Path(run) / "csv"))


if __name__ == "__main__":
    main()
