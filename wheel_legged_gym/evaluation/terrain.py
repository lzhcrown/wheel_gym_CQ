"""Fixed evaluation terrain with safe, directed edge-to-edge routes.

This builder does not use or change the training terrain curriculum. All
distances are in metres. Robots may share a tile when their collision groups
are separate; the mesh size therefore depends on terrain types, not env count.
"""
import copy
import math

import numpy as np


TERRAIN_TYPES = (
    "flat", "rough", "slope", "stairs_up", "stairs_down", "discrete_obstacles",
)


def _number(cfg, name, default=None, minimum=0., strict=False):
    value = float(getattr(cfg, name, default))
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        comparison = ">" if strict else ">="
        raise ValueError("%s must be finite and %s %s" % (name, comparison, minimum))
    return value


def _grid_cells(distance, scale, name):
    cells = int(round(distance / scale))
    if not math.isclose(cells * scale, distance, rel_tol=0., abs_tol=1e-7):
        raise ValueError("%s must be an integer multiple of horizontal_scale" % name)
    return cells


class EvaluationTerrain:
    """Terrain-compatible arrays plus tile bounds and reproducible spawn plans.

    ``cfg`` is a terrain configuration, not the full environment configuration.
    Required fields are mesh_type, type_names, terrain_length, terrain_width,
    horizontal_scale, vertical_scale and border_size. A private copy is retained.
    Tiles run along +x and are laid out along y with interpolated gaps between
    them. Bounds contain [xmin, xmax, ymin, ymax] in simulation world coordinates.
    """

    def __init__(self, cfg, num_robots):
        self.cfg = copy.deepcopy(cfg)
        self.type = self.cfg.mesh_type
        if self.type not in ("heightfield", "trimesh"):
            raise ValueError("EvaluationTerrain requires heightfield or trimesh")
        self.num_robots = int(num_robots)
        if self.num_robots <= 0 or self.num_robots != num_robots:
            raise ValueError("num_robots must be a positive integer")
        self.type_names = tuple(self.cfg.type_names)
        if not self.type_names or len(set(self.type_names)) != len(self.type_names):
            raise ValueError("type_names must be nonempty and contain no duplicates")
        unknown = set(self.type_names) - set(TERRAIN_TYPES)
        if unknown:
            raise ValueError("Unsupported evaluation terrain: " + ", ".join(sorted(unknown)))

        self.env_length = _number(cfg, "terrain_length", strict=True)
        self.env_width = _number(cfg, "terrain_width", strict=True)
        if not math.isclose(self.env_length, self.env_width, rel_tol=0., abs_tol=1e-7):
            raise ValueError("Evaluation terrain regions must be square")
        self.horizontal_scale = _number(cfg, "horizontal_scale", strict=True)
        self.vertical_scale = _number(cfg, "vertical_scale", strict=True)
        border_size = _number(cfg, "border_size", strict=True)
        gap_size = _number(cfg, "gap_size", 2., strict=True)
        self.platform_length = _number(cfg, "platform_length", 1.5, minimum=1.5)
        self.length_per_env_pixels = _grid_cells(self.env_length, self.horizontal_scale, "terrain_length")
        self.width_per_env_pixels = _grid_cells(self.env_width, self.horizontal_scale, "terrain_width")
        self.border = _grid_cells(border_size, self.horizontal_scale, "border_size")
        gap_cells = _grid_cells(gap_size, self.horizontal_scale, "gap_size")
        _grid_cells(self.platform_length, self.horizontal_scale, "platform_length")
        interior_length = self.env_length - 2 * self.platform_length
        if interior_length < 2 * self.horizontal_scale:
            raise ValueError("Region too small: two >=1.5 m platforms and a nonempty terrain interior are required")
        if self.border < 1 or gap_cells < 1:
            raise ValueError("border_size and gap_size must each span at least one grid cell")

        self.seed = int(getattr(cfg, "seed", 0))
        if self.seed < 0 or self.seed != getattr(cfg, "seed", 0):
            raise ValueError("Terrain seed must be a nonnegative integer")
        self.parameters = {
            "slope_angle_deg": _number(cfg, "slope_angle_deg", 10.),
            "rough_height": _number(cfg, "rough_height", .03),
            "rough_cell_size": _number(cfg, "rough_cell_size", .2, minimum=self.horizontal_scale),
            "step_height": _number(cfg, "step_height", .10),
            "step_width": _number(cfg, "step_width", .4, minimum=self.horizontal_scale),
            "obstacle_height": _number(cfg, "obstacle_height", .10),
            "obstacle_size_min": _number(cfg, "obstacle_size_min", .4, minimum=self.horizontal_scale),
            "obstacle_size_max": _number(cfg, "obstacle_size_max", .8, minimum=self.horizontal_scale),
            "obstacle_density": _number(cfg, "obstacle_density", .15),
        }
        if self.parameters["slope_angle_deg"] >= 90:
            raise ValueError("slope_angle_deg must be less than 90")
        if self.parameters["obstacle_density"] > 1:
            raise ValueError("obstacle_density must be in [0, 1]")
        if self.parameters["obstacle_size_max"] < self.parameters["obstacle_size_min"]:
            raise ValueError("obstacle_size_max must be >= obstacle_size_min")
        if any(name.startswith("stairs_") for name in self.type_names):
            if interior_length < self.parameters["step_width"]:
                raise ValueError("Region interior must fit at least one complete stair tread")
        if "discrete_obstacles" in self.type_names:
            if (interior_length < self.parameters["obstacle_size_max"] + 2 * self.horizontal_scale
                    or self.env_width < self.parameters["obstacle_size_max"]):
                raise ValueError("Region interior is too small for the configured obstacles")

        self.cfg.num_rows, self.cfg.num_cols = 1, len(self.type_names)
        self.cfg.curriculum = False
        self.cfg.num_sub_terrains = len(self.type_names)
        for key, value in self.parameters.items():
            setattr(self.cfg, key, value)
        self.cfg.seed, self.cfg.gap_size = self.seed, gap_size
        self.cfg.platform_length = self.platform_length
        nx, ny = self.length_per_env_pixels, self.width_per_env_pixels
        core = np.zeros((nx + 1, len(self.type_names) * ny + (len(self.type_names) - 1) * gap_cells + 1),
                        dtype=np.float64)
        self.tile_bounds = np.empty((len(self.type_names), 4), dtype=np.float64)
        tile_metadata = []
        previous_right = None
        for index, name in enumerate(self.type_names):
            # A type retains its geometry when other types are omitted/reordered,
            # or the number of evaluation robots changes.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, TERRAIN_TYPES.index(name)]))
            tile, details = self._build_tile(name, rng)
            start = index * (ny + gap_cells)
            if previous_right is not None:
                weights = np.linspace(0., 1., gap_cells + 1)[None, :]
                core[:, start - gap_cells:start + 1] = (
                    previous_right[:, None] * (1 - weights) + tile[:, :1] * weights)
            core[:, start:start + ny + 1] = tile
            previous_right = tile[:, -1]
            y_origin = index * (self.env_width + gap_size)
            self.tile_bounds[index] = (0., self.env_length, y_origin, y_origin + self.env_width)
            tile_metadata.append(dict(type=name, bounds=self.tile_bounds[index].tolist(), **details))

        # Extend the terminal platform through the outer border. Connecting gaps
        # interpolate adjacent edge heights, so tile seams have no artificial cliff.
        height_m = np.pad(core, ((self.border, self.border), (self.border, self.border)), mode="edge")
        units = np.rint(height_m / self.vertical_scale)
        limits = np.iinfo(np.int16)
        if units.min() < limits.min or units.max() > limits.max:
            raise ValueError("Terrain heights exceed int16 heightfield range; reduce slope/steps or increase vertical_scale")
        self.height_field_raw = units.astype(np.int16)
        self.heightsamples = self.height_field_raw
        self.tot_rows, self.tot_cols = self.height_field_raw.shape
        self.env_origins = np.zeros((1, len(self.type_names), 3), dtype=np.float64)
        self.env_origins[0, :, 0] = self.env_length / 2
        self.env_origins[0, :, 1] = (self.tile_bounds[:, 2] + self.tile_bounds[:, 3]) / 2
        self.env_origins[0, :, 2] = self.ground_heights(self.env_origins[0, :, :2])
        self.metadata = {
            "type_names": list(self.type_names), "seed": self.seed,
            "mesh_type": self.type, "slope_threshold": getattr(self.cfg, "slope_treshold", .75),
            "terrain_length": self.env_length, "terrain_width": self.env_width,
            "horizontal_scale": self.horizontal_scale, "vertical_scale": self.vertical_scale,
            "border_size": border_size, "gap_size": gap_size,
            "platform_length": self.platform_length, "direction": "+x",
            "curriculum": False, "parameters": dict(self.parameters), "tiles": tile_metadata,
        }
        if self.type == "trimesh":
            from isaacgym import terrain_utils
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, self.horizontal_scale, self.vertical_scale,
                getattr(self.cfg, "slope_treshold", .75))

    def _build_tile(self, name, rng):
        nx, ny = self.length_per_env_pixels, self.width_per_env_pixels
        x = np.arange(nx + 1) * self.horizontal_scale
        y = np.arange(ny + 1) * self.horizontal_scale
        progress = np.clip(x - self.platform_length, 0., self.env_length - 2 * self.platform_length)
        terrain = np.zeros((nx + 1, ny + 1), dtype=np.float64)
        details = {}
        if name == "slope":
            terrain[:] = (progress * math.tan(math.radians(self.parameters["slope_angle_deg"])))[:, None]
        elif name.startswith("stairs_"):
            levels = np.floor(progress / self.parameters["step_width"] + 1e-9)
            profile = levels * self.parameters["step_height"]
            if name == "stairs_down":
                profile = profile.max() - profile
            terrain[:] = profile[:, None]
            details["step_count"] = int(levels.max())
        elif name == "rough":
            coarse_size = self.parameters["rough_cell_size"]
            cx = np.linspace(0., self.env_length, int(math.ceil(self.env_length / coarse_size)) + 1)
            cy = np.linspace(0., self.env_width, int(math.ceil(self.env_width / coarse_size)) + 1)
            noise = rng.uniform(-self.parameters["rough_height"], self.parameters["rough_height"], (len(cx), len(cy)))
            along_x = np.array([np.interp(x, cx, noise[:, j]) for j in range(len(cy))]).T
            terrain[:] = np.array([np.interp(y, cy, row) for row in along_x])
            terrain[(x <= self.platform_length) | (x >= self.env_length - self.platform_length)] = 0.
        elif name == "discrete_obstacles":
            margin = int(round(self.platform_length / self.horizontal_scale))
            minimum = int(math.ceil(self.parameters["obstacle_size_min"] / self.horizontal_scale))
            maximum = int(math.floor(self.parameters["obstacle_size_max"] / self.horizontal_scale))
            if maximum < minimum:
                raise ValueError("Obstacle size interval contains no grid-aligned sizes")
            average_size = (minimum + maximum) * self.horizontal_scale / 2
            interior_area = (self.env_length - 2 * self.platform_length) * self.env_width
            count = int(math.ceil(interior_area * self.parameters["obstacle_density"] / average_size ** 2))
            for _ in range(count):
                sx, sy = rng.integers(minimum, maximum + 1, size=2)
                start_x = rng.integers(margin + 1, nx - margin - sx + 1)
                start_y = rng.integers(0, ny - sy + 1)
                terrain[start_x:start_x + sx, start_y:start_y + sy] = self.parameters["obstacle_height"]
            details["obstacle_count"] = count
            details["obstacle_coverage_fraction"] = float(np.mean(terrain[margin + 1:nx - margin] > 0))
        quantized = np.rint(terrain / self.vertical_scale) * self.vertical_scale
        details.update(min_height=float(quantized.min()), max_height=float(quantized.max()))
        return terrain, details

    def ground_heights(self, xy):
        """Sample the quantized height grid, exactly flat on both spawn platforms.

        Bilinear values away from the platforms describe the heightfield; they
        are not a raycast against vertically corrected triangle-mesh stair faces.
        """
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("xy must be a finite (N, 2) array")
        grid = (points + self.cfg.border_size) / self.horizontal_scale
        maximum = np.asarray(self.height_field_raw.shape) - 1
        if np.any(grid < -1e-7) or np.any(grid > maximum + 1e-7):
            raise ValueError("Height query lies outside the evaluation terrain mesh")
        grid = np.clip(grid, 0, maximum)
        lower = np.floor(grid).astype(np.int64)
        lower = np.minimum(lower, maximum - 1)
        fraction = grid - lower
        ix, iy = lower[:, 0], lower[:, 1]
        dx, dy = fraction[:, 0], fraction[:, 1]
        samples = self.height_field_raw
        return ((1 - dx) * (1 - dy) * samples[ix, iy] + dx * (1 - dy) * samples[ix + 1, iy]
                + (1 - dx) * dy * samples[ix, iy + 1] + dx * dy * samples[ix + 1, iy + 1]) * self.vertical_scale

    def sample_spawns(self, type_ids, rng, spawn_inset=.75, lateral_margin=.75, goal_inset=.75):
        """Return ground-level starts and goals, with yaw zero and +x direction.

        Success is reaching the opposite x line; matching the goal y coordinate
        is not required. The environment adds its base clearance to start z.
        """
        ids = np.asarray(type_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("type_ids must be a one-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= len(self.type_names)):
            raise ValueError("type_ids contains an unknown terrain index")
        values = np.asarray([spawn_inset, lateral_margin, goal_inset], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Spawn and goal insets and lateral_margin must be finite and positive")
        if spawn_inset >= self.platform_length or goal_inset >= self.platform_length:
            raise ValueError("Spawn and goal must lie inside their flat platforms")
        if spawn_inset + goal_inset >= self.env_length or 2 * lateral_margin >= self.env_width:
            raise ValueError("Region too small for distinct start/goal lines and lateral margins")
        bounds = self.tile_bounds[ids].copy()
        y = rng.uniform(bounds[:, 2] + lateral_margin, bounds[:, 3] - lateral_margin)
        start_xy = np.column_stack((bounds[:, 0] + spawn_inset, y))
        goal_xy = np.column_stack((bounds[:, 1] - goal_inset, y))
        return {
            "ground_positions": np.column_stack((start_xy, self.ground_heights(start_xy))),
            "goal_positions": np.column_stack((goal_xy, self.ground_heights(goal_xy))),
            "directions": np.tile(np.array([[1., 0.]]), (len(ids), 1)),
            "bounds": bounds,
        }
