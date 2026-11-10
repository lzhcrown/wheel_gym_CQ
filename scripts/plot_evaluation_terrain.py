#!/usr/bin/env python3
"""Plot the actual evaluation meshes, without creating a physics simulation.

Run from the project root after ``source scripts/activate_train.sh``::

    python scripts/plot_evaluation_terrain.py

The seven meshes are constructed separately to bound memory use. Stairs retain
every converted mesh triangle, including vertical risers. Center profiles are
triangle/plane intersections, not interpolated heightfield approximations.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def terrain_title(item):
    """Describe generated dimensions rather than duplicate protocol constants."""
    name, tile = item["name"], item["tile"]
    if name.startswith("stairs_"):
        concave = tile["central_platform_height_m"] < 0
        return "{}形阶梯｜每级 {:g} cm，{} {:.2f} m".format(
            "凹" if concave else "凸", tile["step_height_m"] * 100,
            "深" if concave else "高", tile["total_height_or_depth_m"])
    if name == "rough":
        low, high = tile["nominal_height_range_m"]
        return "粗糙面｜高度范围 {:g}～{:g} cm".format(low * 100, high * 100)
    if name == "slope":
        return "斜坡｜{:g}°，端部高差 {:.2f} m".format(
            tile["nominal_angle_deg"],
            tile["high_x_platform_height_m"] - tile["low_x_platform_height_m"])
    return "平地"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_font():
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    for filename in candidates:
        if Path(filename).is_file():
            font_manager.fontManager.addfont(filename)
            family = font_manager.FontProperties(fname=filename).get_name()
            plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                                 "font.size": 10, "pdf.fonttype": 42})
            return filename
    raise RuntimeError("Install a Chinese font (Noto Sans CJK) before generating these figures")


def roi_mesh(terrain):
    """Return exact converted triangles in ROI coordinates, with no resampling."""
    vertices = np.asarray(terrain.vertices, dtype=np.float64).copy()
    vertices[:, :2] -= terrain.cfg.border_size
    xmin, xmax, ymin, ymax = terrain.patch_bounds[0]
    inside = ((vertices[:, 0] >= xmin - 1e-5) & (vertices[:, 0] <= xmax + 1e-5)
              & (vertices[:, 1] >= ymin - 1e-5) & (vertices[:, 1] <= ymax + 1e-5))
    triangles = np.asarray(terrain.triangles)[inside[np.asarray(terrain.triangles)].all(axis=1)]
    faces = vertices[triangles]
    faces[:, :, 0] -= xmin
    faces[:, :, 1] -= ymin
    # Snap only floating-point roundoff at exact outer ROI edges.
    for edge in (0., terrain.patch_size_m):
        xy = faces[:, :, :2]
        xy[np.isclose(xy, edge, atol=1e-6, rtol=0.)] = edge
    return faces


def mesh_section(faces, y):
    """Intersect real triangles with the y=center plane, retaining verticals."""
    epsilon = 1e-6
    candidates = faces[(faces[:, :, 1].min(axis=1) <= y + epsilon)
                       & (faces[:, :, 1].max(axis=1) >= y - epsilon)]
    unique = {}
    for triangle in candidates:
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[i], triangle[j]
            da, db = a[1] - y, b[1] - y
            if abs(da) <= epsilon:
                points.append(a[[0, 2]])
            if abs(db) <= epsilon:
                points.append(b[[0, 2]])
            if da * db < 0 and abs(da) > epsilon and abs(db) > epsilon:
                points.append((a + (b - a) * (-da / (db - da)))[[0, 2]])
        if not points:
            continue
        points = np.unique(np.round(points, 6), axis=0)
        if len(points) < 2:
            continue
        # In the degenerate coplanar case retain the longest boundary segment.
        distances = ((points[:, None] - points[None]) ** 2).sum(axis=2)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        ends = sorted((tuple(points[i]), tuple(points[j])))
        unique[tuple(ends)] = np.asarray(ends)
    result = np.array(list(unique.values()))
    if not len(result):
        raise ValueError("Mesh center plane did not intersect the terrain")
    return result


def check_stair_section(segments, tile, roi_size):
    """Require every nominal vertical face in the actual center section."""
    vertical = segments[np.isclose(segments[:, 0, 0], segments[:, 1, 0], atol=1e-5)
                        & (np.abs(segments[:, 0, 1] - segments[:, 1, 1]) > 1e-5)]
    actual = np.unique(np.round(vertical[:, 0, 0], 5))
    inset = np.asarray(tile["terrace_inset_edges_m"])
    expected = np.sort(np.r_[inset, roi_size - inset])
    if len(expected) != 2 * tile["steps_per_side"]:
        raise ValueError("Declared terrace boundaries disagree with the step count")
    if len(actual) != len(expected) or not np.allclose(actual, expected, atol=1e-5):
        raise ValueError("Actual vertical mesh faces disagree with the declared stair geometry: %s" % actual)
    if not np.isclose(np.max(np.abs(segments[:, :, 1])), tile["total_height_or_depth_m"], atol=1e-5):
        raise ValueError("Actual mesh height disagrees with the declared total stair height")
    return {"actual_vertical_faces_x_m": actual.tolist(),
            "actual_height_levels_m": np.unique(np.round(segments[:, :, 1], 6)).tolist()}


def save_figure(fig, output, basename, dpi):
    # Matplotlib 3-D axes omit the projected z label from the tight bounding box.
    labels = list(fig.texts) + [ax.zaxis.label for ax in fig.axes if hasattr(ax, "zaxis")]
    for extension in ("png", "pdf"):
        fig.savefig(output / (basename + "." + extension), dpi=dpi,
                    facecolor="white", bbox_inches="tight", bbox_extra_artists=labels)
    plt.close(fig)


def overview(terrains, output, dpi):
    metadata = terrains[0]["metadata"]
    roi_size = metadata["patch_size_m"]
    stair = next(item["tile"] for item in terrains if item["name"].startswith("stairs_"))
    extreme = max(float(np.max(np.abs(item["faces"][:, :, 2]))) for item in terrains)
    extreme = round(extreme, 6)
    zlimit = extreme + .05
    tick_spacing = max(1, round(roi_size / 3))
    # Keep the exact ROI edge while avoiding a nearly coincident round-number tick.
    xy_ticks = np.r_[np.arange(0, roi_size - .25 * tick_spacing, tick_spacing), roi_size]
    fig = plt.figure(figsize=(19, 10.2))
    fig.suptitle("七类评估地形：实际生成三角网格", fontsize=22, y=.99)
    normalizer = colors.Normalize(-extreme, extreme)
    cmap = plt.get_cmap("coolwarm")
    for index, item in enumerate(terrains):
        ax = fig.add_subplot(2, 4, index + 1, projection="3d")
        faces = item["faces"]
        face_colors = cmap(normalizer(faces[:, :, 2].mean(axis=1)))
        # Lighting improves visibility of real vertical risers; it changes no geometry.
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        length = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(length[:, None], 1e-12)
        light = np.array([-.4, -.5, .77])
        light /= np.linalg.norm(light)
        shade = .70 + .30 * np.abs(normals @ light)
        face_colors[:, :3] *= shade[:, None]
        surface = Poly3DCollection(faces, facecolors=face_colors, edgecolors="none",
                                   linewidths=0, antialiased=False, rasterized=True)
        ax.add_collection3d(surface)
        ax.set(xlim=(0, roi_size), ylim=(0, roi_size), zlim=(-zlimit, zlimit),
               xticks=xy_ticks, yticks=xy_ticks, zticks=(-extreme, 0., extreme))
        ax.set_box_aspect((roi_size, roi_size, 2 * zlimit))
        ax.view_init(elev=29, azim=-57)
        ax.set_xlabel("x / m", labelpad=-1)
        ax.set_ylabel("y / m", labelpad=-1)
        ax.set_zlabel("高程 z / m", labelpad=1)
        ax.tick_params(labelsize=8, pad=0)
        ax.set_title("({}) {}".format(chr(97 + index), terrain_title(item)),
                     fontsize=11, pad=4)
        ax.grid(True, alpha=.2)
    note = fig.add_subplot(2, 4, 8)
    note.axis("off")
    note.text(.02, .96, "统一尺寸与绘制依据", fontsize=15, weight="bold", va="top")
    note.text(.02, .85,
              "每块测量区域：{roi:g} × {roi:g} m\n\n"
              "阶梯：四周同心方形台阶\n"
              "每侧 {steps} 级，踏面 {tread:g} cm\n"
              "中心平台：{platform:g} × {platform:g} m\n\n"
              "所有子图使用相同坐标尺度\n"
              "粗糙面高度未放大\n\n"
              "仅显示测量区域，外部起止平台省略\n"
              "斜坡出口继续连接等高平台\n\n"
              "由实际网格生成器绘制，非仿真截图\n"
              "网格：水平 {horizontal:g} cm / 高度 {vertical:g} cm".format(
                  roi=roi_size, steps=stair["steps_per_side"], tread=stair["tread_width_m"] * 100,
                  platform=stair["central_platform_size_m"],
                  horizontal=metadata["horizontal_scale"] * 100,
                  vertical=metadata["vertical_scale"] * 100),
              fontsize=11, linespacing=1.4, va="top")
    fig.text(.5, .012, "台阶保留实际垂直立面；高程由颜色与 z 轴共同表示（面颜色附加光照阴影）。", ha="center", fontsize=11)
    color_axis = fig.add_axes([.25, .098, .5, .018])
    fig.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=cmap), cax=color_axis,
                 orientation="horizontal", label="未加阴影的高程色标 / m", ticks=np.linspace(-extreme, extreme, 5))
    fig.subplots_adjust(left=.015, right=.955, top=.925, bottom=.20, wspace=.18, hspace=.12)
    save_figure(fig, output, "terrain_overview_3d", dpi)
    return [[0, roi_size], [0, roi_size], [-zlimit, zlimit]]


def horizontal_dimension(ax, start, end, y, label, color="#526475"):
    ax.annotate("", xy=(end, y), xytext=(start, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.))
    ax.text((start + end) / 2, y + .018, label, ha="center", va="bottom", color=color, fontsize=9)


def stair_profiles(terrains, output, dpi):
    stairs = [item for item in terrains if item["name"].startswith("stairs_")]
    common = stairs[0]["tile"]
    roi_size = stairs[0]["metadata"]["patch_size_m"]
    extreme = max(item["tile"]["total_height_or_depth_m"] for item in stairs)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), sharex=True, sharey=True)
    fig.suptitle("凹凸阶梯中心剖面：{} 级 / 侧，{:g} cm 踏面".format(
        common["steps_per_side"], common["tread_width_m"] * 100), fontsize=21, y=.98)
    for index, (ax, item) in enumerate(zip(axes.flat, stairs)):
        tile, segments = item["tile"], item["section"]
        tread = tile["tread_width_m"]
        side_span = tile["steps_per_side"] * tread
        platform_start, platform_end = side_span, roi_size - side_span
        sign = -1 if "concave" in item["name"] else 1
        color = "#2472a5" if sign < 0 else "#ba5439"
        ax.add_collection(LineCollection(segments, colors=color, linewidths=2.7, zorder=3))
        ax.axhline(0, color="#63717b", linewidth=.8, linestyle="--")
        ticks = np.unique(np.round(np.r_[np.arange(0, roi_size, 2 * tread),
                                        platform_start, platform_end, roi_size], 6))
        ax.set(xlim=(-.05, roi_size + .05), ylim=(-1.28 * extreme, 1.38 * extreme),
               xticks=ticks, yticks=np.linspace(-extreme, extreme, 9))
        ax.tick_params(axis="x", labelsize=8, rotation=45, labelbottom=True)
        ax.tick_params(axis="y", labelsize=9, labelleft=True)
        ax.set_xlabel("沿通行方向 x / m")
        ax.set_ylabel("相对外侧平地的高程 z / m")
        ax.grid(True, color="#e0e5e8", linewidth=.6)
        ax.set_title("({}) {}".format(chr(97 + index), terrain_title(item)), fontsize=13, pad=13)
        side_label = "{} 级 × {:g} cm = {:g} m".format(tile["steps_per_side"], tread * 100, side_span)
        horizontal_dimension(ax, 0, platform_start, 1.20 * extreme, side_label)
        horizontal_dimension(ax, platform_start, platform_end, 1.20 * extreme,
                             "中心平台 {:g} m".format(tile["central_platform_size_m"]))
        horizontal_dimension(ax, platform_end, roi_size, 1.20 * extreme, side_label)
        for x in (platform_start, platform_end):
            ax.axvline(x, color="#9ba9b1", linestyle=":", linewidth=.9)
        height, total = tile["step_height_m"], tile["central_platform_height_m"]
        ax.annotate("", xy=(roi_size / 2, total), xytext=(roi_size / 2, 0),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.3))
        ax.text(roi_size / 2 + .10, total / 2, "总{} {:.2f} m".format("深" if sign < 0 else "高", abs(total)),
                color=color, va="center", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
        ax.annotate("每级 {} cm".format(int(round(height * 100))),
                    xy=(3 * tread, sign * 2.5 * height),
                    xytext=(tread, -sign * .30 * extreme),
                    fontsize=10, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, connectionstyle="angle3"))
        ax.text(roi_size / 2, -1.18 * extreme,
                "真实网格与 y = {:g} m 平面的交线".format(roi_size / 2),
                ha="center", color="#526475", fontsize=9)
    fig.text(.5, .025, "剖面来自转换后的三角网格，包含重复 x 坐标的垂直台阶；横纵坐标单位均为米，标注尺寸以轴值为准。", ha="center", fontsize=11)
    fig.subplots_adjust(left=.075, right=.98, bottom=.10, top=.90, hspace=.37, wspace=.19)
    save_figure(fig, output, "stairs_center_profiles", dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/evaluation_terrain")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi < 100:
        parser.error("--dpi must be at least 100")
    font_path = configure_font()
    # Gym must precede Torch imports; only its terrain converter is called.
    import isaacgym
    from isaacgym import terrain_utils
    from wheel_legged_gym import envs
    from wheel_legged_gym.utils import task_registry
    from wheel_legged_gym.evaluation.config import parse_args, configure_evaluation, protocol_dict
    from wheel_legged_gym.evaluation.patch_terrain import PatchEvaluationTerrain
    eval_args = parse_args(["--layout", "patch", "--sim-device", "cpu", "--seed", str(args.seed)])
    env_cfg, _ = task_registry.get_cfgs(eval_args.task)
    cfg = configure_evaluation(env_cfg, eval_args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    terrains, records = [], []
    for name in eval_args.terrain_types:
        terrain_cfg = copy.deepcopy(cfg.terrain)
        terrain_cfg.type_names = [name]
        terrain = PatchEvaluationTerrain(terrain_cfg, 1)
        faces = roi_mesh(terrain)
        section = mesh_section(faces, terrain.patch_size_m / 2)
        tile = terrain.metadata["tiles"][0]
        checks = check_stair_section(section, tile, terrain.patch_size_m) if name.startswith("stairs_") else {}
        terrains.append(dict(name=name, faces=faces, section=section, tile=tile,
                             metadata=terrain.metadata))
        records.append(dict(name=name, generator_metadata=terrain.metadata,
                            mesh_triangles_in_roi=len(faces),
                            mesh_vertex_min_xyz_m=faces.reshape(-1, 3).min(axis=0).tolist(),
                            mesh_vertex_max_xyz_m=faces.reshape(-1, 3).max(axis=0).tolist(),
                            center_profile_segments_xz_m=section.tolist(), checks=checks))
        del terrain
        print("Generated actual mesh and center section:", name, flush=True)
    xyz_limits = overview(terrains, output, args.dpi)
    stair_profiles(terrains, output, args.dpi)
    source_paths = [Path(__file__), ROOT / "wheel_legged_gym/evaluation/presets.py",
                    ROOT / "wheel_legged_gym/evaluation/config.py",
                    ROOT / "wheel_legged_gym/evaluation/patch_terrain.py",
                    ROOT / "wheel_legged_gym/evaluation/terrain.py",
                    Path(terrain_utils.__file__)]
    provenance = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        command="source scripts/activate_train.sh && python scripts/plot_evaluation_terrain.py --seed {} --dpi {}".format(args.seed, args.dpi),
        purpose="Engineering visualization of actual generated evaluation meshes; no simulation, checkpoint, or performance measurements",
        protocol=protocol_dict(eval_args), chinese_font=font_path,
        overview=dict(common_xyz_limits_m=xyz_limits,
                      mesh_resampling=False, rasterized_pdf_faces=True, dpi=args.dpi),
        section_method="Exact triangle intersection with local ROI y={:g} m; preserve vertical faces; round intersection vertices only to 1e-6 m".format(eval_args.size_m / 2),
        sources=[dict(path=str(path), sha256=sha256(path)) for path in source_paths],
        terrains=records,
        artifacts=[dict(path=name, sha256=sha256(output / name)) for name in (
            "terrain_overview_3d.png", "terrain_overview_3d.pdf",
            "stairs_center_profiles.png", "stairs_center_profiles.pdf")],
    )
    (output / "parameters.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved reproducible figures and provenance to", output)


if __name__ == "__main__":
    main()
