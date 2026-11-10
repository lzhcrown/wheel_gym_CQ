"""Verify seven 6.2 m ROIs, eight concentric steps and continuous support."""
import copy
import json
import math
from types import SimpleNamespace
import unittest

import isaacgym  # Import before torch, including when running the full suite.
import numpy as np

from wheel_legged_gym.evaluation.patch_terrain import PatchEvaluationTerrain
from wheel_legged_gym.evaluation.presets import (
    PATCH_TERRAIN_TYPES, PATCH_STAIR_TREAD_M, PATCH_STEPS_PER_SIDE, PATCH_STAIR_PLATFORM_M,
)


def config(**overrides):
    values = dict(mesh_type="heightfield", type_names=list(PATCH_TERRAIN_TYPES),
                  terrain_length=6.2, terrain_width=6.2, horizontal_scale=.02,
                  vertical_scale=.005, border_size=2., seed=10)
    values.update(overrides)
    return SimpleNamespace(**values)


def patch_grid(terrain, index):
    bounds = terrain.patch_bounds[index]
    lo = np.rint((bounds[[0, 2]] + terrain.cfg.border_size) / terrain.horizontal_scale).astype(int)
    hi = np.rint((bounds[[1, 3]] + terrain.cfg.border_size) / terrain.horizontal_scale).astype(int)
    return terrain.heightsamples[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]


class PatchTerrainTests(unittest.TestCase):
    def test_seven_exact_rois_and_support_corridors_do_not_modify_input(self):
        cfg = config()
        original = copy.deepcopy(vars(cfg))
        terrain = PatchEvaluationTerrain(cfg, 4096)
        self.assertEqual(vars(cfg), original)
        self.assertEqual(terrain.type_names, ("flat", "stairs_concave_15cm", "stairs_concave_20cm",
                                              "stairs_convex_15cm", "stairs_convex_20cm", "rough", "slope"))
        np.testing.assert_allclose(terrain.patch_bounds[:, 1] - terrain.patch_bounds[:, 0], 6.2)
        np.testing.assert_allclose(terrain.patch_bounds[:, 3] - terrain.patch_bounds[:, 2], 6.2)
        np.testing.assert_allclose(terrain.tile_bounds[:, 1] - terrain.tile_bounds[:, 0], 9.2)
        np.testing.assert_allclose(terrain.tile_bounds[:, 3] - terrain.tile_bounds[:, 2], 8.2)
        np.testing.assert_allclose(terrain.patch_bounds[:, 0] - terrain.tile_bounds[:, 0], 1.5)
        np.testing.assert_allclose(terrain.tile_bounds[:, 1] - terrain.patch_bounds[:, 1], 1.5)
        np.testing.assert_allclose(terrain.patch_bounds[:, 2] - terrain.tile_bounds[:, 2], 1.)
        np.testing.assert_allclose(terrain.tile_bounds[:, 3] - terrain.patch_bounds[:, 3], 1.)
        np.testing.assert_allclose(np.diff(terrain.tile_bounds[:, 2]), 10.2)
        self.assertIs(terrain.tile_bounds, terrain.corridor_bounds)
        self.assertEqual(terrain.env_origins.shape, (1, 7, 3))
        self.assertEqual(terrain.heightsamples.shape, (661, 3671))
        self.assertEqual(terrain.heightsamples.dtype, np.int16)
        self.assertEqual(terrain.metadata["protocol"], "wheel_window_patch")
        self.assertEqual(terrain.metadata["patch_size_m"], 6.2)
        self.assertEqual(terrain.metadata["type_seed_order"], list(PATCH_TERRAIN_TYPES))
        json.dumps(terrain.metadata, allow_nan=False)

    def test_concentric_stairs_have_eight_20cm_treads_and_three_metre_center(self):
        terrain = PatchEvaluationTerrain(config(), 7)
        self.assertEqual(PATCH_STAIR_TREAD_M, .2)
        self.assertEqual(PATCH_STEPS_PER_SIDE, 8)
        self.assertEqual(PATCH_STAIR_PLATFORM_M, 3.)
        self.assertEqual(terrain.metadata["parameters"]["step_width"], .2)
        self.assertEqual(terrain.metadata["parameters"]["steps_per_side"], 8)
        for index, sign, height in ((1, -1, .15), (2, -1, .20), (3, 1, .15), (4, 1, .20)):
            patch = patch_grid(terrain, index).astype(float) * terrain.vertical_scale
            self.assertEqual(patch.shape, (311, 311))
            np.testing.assert_allclose(patch, patch.T)
            np.testing.assert_allclose(patch, patch[::-1, :])
            np.testing.assert_allclose(patch, patch[:, ::-1])
            np.testing.assert_allclose(np.unique(patch), sorted(sign * height * np.arange(9)))
            # Sample tread interiors, avoiding the one-cell heightfield ramps.
            samples = zip((.1, .3, .5, .7, .9, 1.1, 1.3, 1.5, 3.1, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1),
                          (0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0))
            for x, level in samples:
                self.assertAlmostEqual(patch[round(x / .02), 155], sign * height * level)
            # Corners depend on nearest y edge as well as x: this is not an
            # extruded 1-D stair profile.
            self.assertAlmostEqual(patch[155, 5], 0.)
            self.assertAlmostEqual(patch[155, 15], sign * height)
            self.assertAlmostEqual(patch[155, 75], sign * 7 * height)
            self.assertTrue(np.all(patch[[0, -1], :] == 0))
            self.assertTrue(np.all(patch[:, [0, -1]] == 0))
            centerline = patch[:, 155]
            changes = np.diff(centerline)
            self.assertEqual(np.count_nonzero(np.abs(changes) > 1e-9), 16)
            np.testing.assert_allclose(np.abs(changes[np.abs(changes) > 1e-9]), height)
            details = terrain.metadata["tiles"][index]
            self.assertEqual(details["step_height_interpretation"], "per riser")
            self.assertEqual(details["steps_per_side"], 8)
            self.assertEqual(details["tread_width_m"], .2)
            self.assertEqual(details["central_platform_size_m"], 3.)
            self.assertAlmostEqual(details["central_platform_height_m"], sign * 8 * height)
            self.assertAlmostEqual(details["total_height_or_depth_m"], 8 * height)
            self.assertEqual(details["terrace_inset_edges_m"], [.2, .4, .6, .8, 1., 1.2, 1.4, 1.6])
            np.testing.assert_allclose(details["centerline_height_levels_m"],
                                       sign * height * np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

    def test_roughness_limits_coarse_scale_and_flat_reference(self):
        terrain = PatchEvaluationTerrain(config(), 7)
        self.assertFalse(patch_grid(terrain, 0).any())
        patch = patch_grid(terrain, 5) * terrain.vertical_scale
        self.assertGreater(np.std(patch), 0.)
        self.assertLessEqual(np.max(np.abs(patch)), .04)
        self.assertLess(patch.min(), 0.)
        self.assertGreater(patch.max(), 0.)
        self.assertTrue(np.all(patch[[0, -1], :] == 0))
        self.assertTrue(np.all(patch[:, [0, -1]] == 0))
        self.assertEqual(terrain.metadata["parameters"]["rough_cell_size"], .08)
        self.assertEqual(terrain.metadata["tiles"][5]["nominal_height_range_m"], [-.04, .04])

    def test_slope_extends_to_equal_height_exit_and_full_width_support(self):
        terrain = PatchEvaluationTerrain(config(type_names=["slope"]), 1)
        patch = patch_grid(terrain, 0) * terrain.vertical_scale
        expected = np.rint(np.arange(311) * .02 * math.tan(math.radians(12.)) / .005) * .005
        np.testing.assert_allclose(patch[:, 155], expected)
        np.testing.assert_allclose(patch[:, 0], expected)
        np.testing.assert_allclose(patch[:, -1], expected)
        self.assertTrue(np.all(np.diff(expected) >= 0))
        self.assertAlmostEqual(expected[-1], 1.32)
        self.assertLessEqual(np.max(np.diff(expected)), .005 + 1e-9)
        # Both pads, lateral supports and outer longitudinal border continue
        # the profile; crossing the ROI exit never encounters a return cliff.
        x = np.array([-.5, .75, 1.5, 4.6, 7.7, 8.45, 9.2, 9.7])
        profile = np.rint(np.clip(x - 1.5, 0., 6.2) * math.tan(math.radians(12.)) / .005) * .005
        for y in (.1, .5, 1., 4.1, 7.2, 7.7, 8.1):
            np.testing.assert_allclose(terrain.ground_heights(np.column_stack((x, np.full(len(x), y)))), profile)
        details = terrain.metadata["tiles"][0]
        self.assertEqual(details["low_x_platform_height_m"], 0.)
        self.assertEqual(details["high_x_platform_height_m"], 1.32)
        self.assertFalse(details["forced_return_to_zero"])
        self.assertAlmostEqual(details["nominal_height_difference_m"], 6.2 * math.tan(math.radians(12.)))

    def test_centered_spawns_fit_whole_robot_for_both_directions_and_legacy_aliases(self):
        terrain = PatchEvaluationTerrain(config(), 112)
        ids = np.repeat(np.arange(7), 16)
        g = terrain.wheel_geometry
        for sign in (1, -1):
            plans = []
            for side in ("center", "left", "right"):
                plan = terrain.sample_spawns(ids, np.random.default_rng(18), direction=sign, side=side)
                plans.append(plan)
                start, goal, roi, bounds = (plan[key] for key in
                                           ("ground_positions", "goal_positions", "patch_bounds", "bounds"))
                middle_y = (roi[:, 2] + roi[:, 3]) / 2
                self.assertTrue(np.all(np.abs(start[:, 1] - middle_y) <= .04 + 1e-9))
                self.assertTrue(np.all(start[:, 1] + g["wheel_y_min"] > roi[:, 2]))
                self.assertTrue(np.all(start[:, 1] + g["wheel_y_max"] < roi[:, 3]))
                self.assertTrue(np.all(start[:, 0] + g["wheel_x_min"] >= bounds[:, 0]))
                self.assertTrue(np.all(start[:, 0] + g["wheel_x_max"] <= bounds[:, 1]))
                if sign > 0:
                    self.assertTrue(np.all(start[:, 0] + g["wheel_x_max"] < roi[:, 0]))
                    self.assertTrue(np.all(goal[:, 0] + g["wheel_x_min"] > roi[:, 1]))
                else:
                    self.assertTrue(np.all(start[:, 0] + g["wheel_x_min"] > roi[:, 1]))
                    self.assertTrue(np.all(goal[:, 0] + g["wheel_x_max"] < roi[:, 0]))
                np.testing.assert_array_equal(start[ids != 6, 2], 0.)
                np.testing.assert_allclose(start[ids == 6, 2], 0. if sign > 0 else 1.32)
                np.testing.assert_allclose(goal[ids == 6, 2], 1.32 if sign > 0 else 0.)
                np.testing.assert_array_equal(plan["directions"], np.tile([sign, 0.], (len(ids), 1)))
                np.testing.assert_array_equal(plan["quaternions"], np.tile([0., 0., 0., 1.], (len(ids), 1)))
                self.assertEqual(plan["side"], ["center"] * len(ids))
            for key in plans[0]:
                np.testing.assert_array_equal(plans[0][key], plans[1][key])
                np.testing.assert_array_equal(plans[0][key], plans[2][key])

    def test_type_seed_and_spawn_randomness_are_stable_under_selection_and_robot_count(self):
        terrain = PatchEvaluationTerrain(config(), 1)
        other = PatchEvaluationTerrain(config(), 500)
        np.testing.assert_array_equal(terrain.heightsamples, other.heightsamples)
        reordered = PatchEvaluationTerrain(config(type_names=["rough", "stairs_convex_20cm", "flat"]), 3)
        np.testing.assert_array_equal(patch_grid(terrain, 5), patch_grid(reordered, 0))
        np.testing.assert_array_equal(patch_grid(terrain, 4), patch_grid(reordered, 1))
        np.testing.assert_array_equal(patch_grid(terrain, 0), patch_grid(reordered, 2))
        ids = np.arange(7)
        first = terrain.sample_spawns(ids, np.random.default_rng(42), direction=[1, -1, 1, -1, 1, -1, 1])
        second = terrain.sample_spawns(ids, np.random.default_rng(42), direction=[1, -1, 1, -1, 1, -1, 1])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        changed = terrain.sample_spawns(ids, np.random.default_rng(43))
        self.assertFalse(np.array_equal(changed["ground_positions"], first["ground_positions"]))

    def test_mesh_stair_faces_at_nominal_insets_and_no_slope_exit_cliff(self):
        terrain = PatchEvaluationTerrain(config(mesh_type="trimesh",
                                              type_names=["stairs_concave_15cm", "stairs_convex_20cm", "slope"]), 3)
        self.assertEqual(terrain.vertices.dtype, np.float32)
        self.assertEqual(terrain.triangles.dtype, np.uint32)
        self.assertEqual(terrain.vertices.shape, (terrain.tot_rows * terrain.tot_cols, 3))
        self.assertEqual(terrain.triangles.shape, (2 * (terrain.tot_rows - 1) * (terrain.tot_cols - 1), 3))
        np.testing.assert_allclose(terrain.vertices[:, 2], terrain.heightsamples.ravel() * .005)
        self.assertLess(terrain.triangles.max(), len(terrain.vertices))
        grid = terrain.vertices.reshape(terrain.tot_rows, terrain.tot_cols, 3)
        for index in (0, 1):
            x0, x1, y0, y1 = terrain.patch_bounds[index]
            iy = round(((y0 + y1) / 2 + terrain.cfg.border_size) / terrain.horizontal_scale)
            line = grid[:, iy]
            changed = np.flatnonzero(np.abs(np.diff(line[:, 2])) > 1e-6)
            self.assertEqual(len(changed), 16)
            nominal_faces = x0 + np.array([.2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 4.6, 4.8, 5., 5.2, 5.4, 5.6, 5.8, 6.])
            np.testing.assert_allclose(line[changed, 0] - terrain.cfg.border_size, nominal_faces, atol=2e-6)
            np.testing.assert_allclose(line[changed, 0], line[changed + 1, 0], atol=2e-6)
            self.assertAlmostEqual(float(line[changed[8], 0] - line[changed[7], 0]), 3., places=6)
        x0, x1, y0, y1 = terrain.patch_bounds[2]
        ix = round((x1 + terrain.cfg.border_size) / terrain.horizontal_scale)
        iy = round(((y0 + y1) / 2 + terrain.cfg.border_size) / terrain.horizontal_scale)
        np.testing.assert_allclose(grid[ix:, iy, 2], 1.32, atol=1e-6)

    def test_explicit_wheel_envelope_metadata_and_unsafe_routes(self):
        terrain = PatchEvaluationTerrain(config(type_names=["flat"], left_wheel_y=.24, right_wheel_y=-.27,
                                              front_wheel_x=.48, rear_wheel_x=-.50, wheel_radius=.12,
                                              wheel_x_min=-.63, wheel_x_max=.62,
                                              wheel_y_min=-.31, wheel_y_max=.29,
                                              wheel_geometry_source="URDF/test_asset.urdf"), 1)
        self.assertEqual(terrain.metadata["wheel_geometry_fallback_fields"], [])
        self.assertEqual(terrain.metadata["wheel_geometry_source"], "URDF/test_asset.urdf")
        self.assertEqual(terrain.wheel_geometry["wheel_y_min"], -.31)
        plan = terrain.sample_spawns(np.array([0]), np.random.default_rng(1), xy_jitter=0.)
        np.testing.assert_allclose(plan["ground_positions"][0], [.75, 4.1, 0.])
        for changes in (dict(wheel_x_min=-1.), dict(wheel_y_min=-4.)):
            unsafe = PatchEvaluationTerrain(config(type_names=["flat"], **changes), 1)
            with self.assertRaises(ValueError):
                unsafe.sample_spawns(np.array([0]), np.random.default_rng(1), xy_jitter=0.)

    def test_old_patch_types_dimensions_and_incompatible_grids_are_rejected(self):
        for changes in (dict(terrain_length=.4), dict(terrain_length=3.), dict(terrain_width=40.), dict(patch_size_m=.4),
                        dict(horizontal_scale=.03), dict(type_names=[]), dict(type_names=["stairs_up"]),
                        dict(type_names=["discrete_obstacles"]), dict(step_width=.4), dict(step_width=.5), dict(seed=-1),
                        dict(vertical_scale=.03), dict(platform_length=.5), dict(corridor_width=4.)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PatchEvaluationTerrain(config(**changes), 1)
        terrain = PatchEvaluationTerrain(config(type_names=["flat"]), 1)
        for kwargs in (dict(direction=0), dict(side="both"), dict(xy_jitter=3.1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                terrain.sample_spawns(np.array([0]), np.random.default_rng(1), **kwargs)


if __name__ == "__main__":
    unittest.main()
