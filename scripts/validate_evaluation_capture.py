"""Stream-check raw CSV completeness and timing without computing performance metrics."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-8)


def key(row):
    return int(row["env_id"]), int(row["episode_index"])


def validate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata["status"] == "complete", "Collection is not complete")
    protocol = metadata["protocol"]
    with (directory / "episodes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    episodes = {key(row): row for row in rows}
    require(len(episodes) == len(rows) == protocol["total_episodes"] == metadata["recorded_episodes"],
            "Duplicate or missing episodes")
    counts = Counter(row["terrain"] for row in rows)
    require(set(counts) == set(protocol["terrain_types"]), "Terrain set mismatch")
    require(set(counts.values()) == {protocol["envs_per_terrain"] * protocol["episodes_per_env"]},
            "Unequal attempt counts per terrain")
    require(set(episodes) == {(i, j) for i in range(len(counts) * protocol["envs_per_terrain"])
                             for j in range(protocol["episodes_per_env"])}, "Environment/episode ID mismatch")
    for row in rows:
        require(row["status"] in ("success", "failure", "timeout"), "Invalid terminal status")
        require(float(row["mass_kg"]) > 0 and float(row["gravity_m_s2"]) > 0, "Invalid mass/gravity")
        if row["window_started"] == "True":
            require(close(row["elapsed_s"], row["physics_elapsed_s"]), "Physics/control duration mismatch")
        else:
            require(row["window_started"] == "False" and row["elapsed_s"] == row["physics_elapsed_s"] == "" and
                    int(row["control_sample_count"]) == int(row["physics_sample_count"]) == 0,
                    "Unstarted attempt must have no measured duration or samples")
        require(row["status"] != "success" or row["window_started"] == "True", "Success without entry")
    decimation = metadata["env_cfg"]["control"]["decimation"]
    joint_count = len(metadata["joint_columns"])
    report = dict(directory=str(directory.resolve()), status="passed", episodes=len(rows),
                  statuses=dict(Counter(row["status"] for row in rows)), terrains=dict(counts),
                  unstarted_attempts=sum(row["window_started"] == "False" for row in rows),
                  control_samples=0, physics_samples=0, files_bytes={})
    for kind in ("physics", "trajectory"):
        path = directory / (kind + ".csv")
        report["files_bytes"][path.name] = path.stat().st_size
        previous = {}
        seen = Counter()
        if kind == "physics":
            numeric = ["torque_j%02d_nm" % j for j in range(joint_count)] + [
                "joint_vel_%s_j%02d_radps" % (end, j) for end in ("before", "after") for j in range(joint_count)]
            step_name, dt_name, dt = "physics_step", "dt_s", metadata["physics_dt_s"]
        else:
            numeric = [prefix + "world_" + axis + "_m" for prefix in ("", "previous_") for axis in "xyz"] + [
                prefix + "world_v" + axis + "_mps" for prefix in ("", "previous_") for axis in "xyz"] + [
                "body_v" + axis + "_mps" for axis in "xyz"] + [
                "body_w" + axis + "_radps" for axis in "xyz"] + [
                "world_a" + axis + "_mps2" for axis in "xyz"] + [
                "world_q" + axis for axis in "xyzw"] + ["world_w" + axis + "_radps" for axis in "xyz"] + [
                "command_vx_mps", "command_vy_mps", "command_wz_radps",
                "acceleration_xy_mps2", "acceleration_xyz_mps2"]
            step_name, dt_name, dt = "control_step", "control_dt_s", metadata["policy_dt_s"]
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            require(set(numeric).issubset(reader.fieldnames), "Missing raw columns: " + kind)
            for line, row in enumerate(reader, 2):
                identity = key(row)
                context = "{}:{} episode={}".format(path.name, line, identity)
                require(identity in episodes, "Unknown episode at " + context)
                episode = episodes[identity]
                require(row["task"] == episode["task"] == protocol["task"] and
                        row["checkpoint"] == episode["checkpoint"] == metadata["checkpoint"] and
                        row["terrain"] == episode["terrain"] and int(row["eval_seed"]) == protocol["eval_seed"],
                        "Provenance mismatch at " + context)
                require(episode["window_started"] == "True", "Samples before entry at " + context)
                values = {name: float(row[name]) for name in numeric}
                require(all(math.isfinite(x) for x in values.values()), "Nonfinite sample at " + context)
                seen[identity] += 1
                step = seen[identity]
                require(int(row[step_name]) == step and close(row[dt_name], dt) and
                        close(row["time_start_s"], (step - 1) * dt) and close(row["time_end_s"], step * dt),
                        "Missing/duplicate step or bad interval at " + context)
                entry = float(episode["entry_time_s"])
                require(close(row["trial_time_start_s"], entry + (step - 1) * dt) and
                        close(row["trial_time_end_s"], entry + step * dt), "Trial clock mismatch at " + context)
                last = previous.get(identity)
                if kind == "physics":
                    require(int(row["control_step"]) == (step - 1) // decimation + 1 and
                            int(row["substep_index"]) == (step - 1) % decimation + 1,
                            "Substep mapping mismatch at " + context)
                    before = [values["joint_vel_before_j%02d_radps" % j] for j in range(joint_count)]
                    if last is not None:
                        require(all(close(x, y) for x, y in zip(before, last)), "Joint-velocity gap at " + context)
                    previous[identity] = [values["joint_vel_after_j%02d_radps" % j] for j in range(joint_count)]
                else:
                    acceleration = [(values["world_v" + a + "_mps"] - values["previous_world_v" + a + "_mps"]) / dt
                                    for a in "xyz"]
                    require(all(close(values["world_a" + a + "_mps2"], x) for a, x in zip("xyz", acceleration)) and
                            close(values["acceleration_xy_mps2"], math.hypot(*acceleration[:2])) and
                            close(values["acceleration_xyz_mps2"], math.sqrt(sum(x*x for x in acceleration))),
                            "Acceleration cannot be reconstructed at " + context)
                    command = [values[name] for name in ("command_vx_mps", "command_vy_mps", "command_wz_radps")]
                    require(all(close(x, y) for x, y in zip(command, protocol["command_body_vx_vy_wz"])),
                            "Unexpected command at " + context)
                    old_state = [values["previous_world_" + a + "_m"] for a in "xyz"] + [
                        values["previous_world_v" + a + "_mps"] for a in "xyz"]
                    if last is None:
                        last = [float(episode["start_world_" + a + "_m"]) for a in "xyz"] + [
                            float(episode["start_world_v" + a + "_mps"]) for a in "xyz"]
                    require(all(close(x, y) for x, y in zip(old_state, last)), "State gap/reset jump at " + context)
                    previous[identity] = [values["world_" + a + "_m"] for a in "xyz"] + [
                        values["world_v" + a + "_mps"] for a in "xyz"]
        for identity, episode in episodes.items():
            name = "physics" if kind == "physics" else "control"
            require(seen[identity] == int(episode[name + "_sample_count"]), "Sample count mismatch: " + str(identity))
            if episode["window_started"] == "True":
                require(close(seen[identity] * dt, episode["elapsed_s"]), "Window duration mismatch: " + str(identity))
            if kind == "trajectory" and seen[identity]:
                terminal = [float(episode["end_world_" + a + "_m"]) for a in "xyz"] + [
                    float(episode["end_world_v" + a + "_mps"]) for a in "xyz"]
                require(all(close(x, y) for x, y in zip(previous[identity], terminal)), "Terminal sample mismatch")
        report[("physics" if kind == "physics" else "control") + "_samples"] = sum(seen.values())
        print("Validated {} / {}: {} samples".format(directory.name, kind, sum(seen.values())), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.directory)
    except (ValueError, KeyError) as exc:
        result = dict(directory=str(args.directory.resolve()), status="failed", error=str(exc))
    (args.directory / "data_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
