"""Geometry and spawn checks for evaluation terrain; no simulator is started."""
import copy
import json
from types import SimpleNamespace
import unittest

import isaacgym  # Imported before any other test can import torch.
import numpy as np

from wheel_legged_gym.evaluation.terrain import EvaluationTerrain, TERRAIN_TYPES


def config(**overrides):
    values = dict(mesh_type="heightfield", type_names=list(TERRAIN_TYPES),
                  terrain_length=8., terrain_width=8., horizontal_scale=.1,
                  vertical_scale=.005, border_size=2., seed=31,
                  slope_treshold=.75)
    values.update(overrides)
    return SimpleNamespace(**values)


def tile_grid(terrain, index):
    start = terrain.border + int(round(terrain.tile_bounds[index, 2] / terrain.horizontal_scale))
    return terrain.heightsamples[
        terrain.border:terrain.border + terrain.length_per_env_pixels + 1,
        start:start + terrain.width_per_env_pixels + 1]


class EvaluationTerrainTests(unittest.TestCase):
    def test_layout_units_compatibility_and_shared_robot_geometry(self):
        cfg = config(terrain_length=40., terrain_width=40.)
        original = copy.deepcopy(vars(cfg))
        terrain = EvaluationTerrain(cfg, 4096)
        self.assertEqual(vars(cfg), original)
        self.assertEqual(terrain.env_origins.shape, (1, 6, 3))
        self.assertEqual(terrain.tile_bounds.shape, (6, 4))
        np.testing.assert_allclose(terrain.tile_bounds[:, 1] - terrain.tile_bounds[:, 0], 40.)
        np.testing.assert_allclose(terrain.tile_bounds[:, 3] - terrain.tile_bounds[:, 2], 40.)
        np.testing.assert_allclose(np.diff(terrain.tile_bounds[:, 2]), 42.)
        self.assertEqual(terrain.heightsamples.shape, (441, 2541))
        self.assertIs(terrain.heightsamples, terrain.height_field_raw)
        self.assertEqual(terrain.heightsamples.dtype, np.int16)
        self.assertFalse(terrain.cfg.curriculum)
        self.assertEqual((terrain.cfg.num_rows, terrain.cfg.num_cols), (1, 6))
        np.testing.assert_allclose(terrain.env_origins[0, :, 2],
                                   terrain.ground_heights(terrain.env_origins[0, :, :2]))
        json.dumps(terrain.metadata, allow_nan=False)
        single_robot = EvaluationTerrain(cfg, 1)
        np.testing.assert_array_equal(terrain.heightsamples, single_robot.heightsamples)
        self.assertEqual(terrain.metadata, single_robot.metadata)

    def test_geometry_seed_is_local_and_independent_of_type_order(self):
        np.random.seed(100)
        global_state = np.random.get_state()
        reference = EvaluationTerrain(config(), 8)
        actual_state = np.random.get_state()
        self.assertEqual(actual_state[0], global_state[0])
        np.testing.assert_array_equal(actual_state[1], global_state[1])
        self.assertEqual(actual_state[2:], global_state[2:])
        reordered = EvaluationTerrain(config(type_names=["discrete_obstacles", "rough"]), 24)
        np.testing.assert_array_equal(tile_grid(reference, 5), tile_grid(reordered, 0))
        np.testing.assert_array_equal(tile_grid(reference, 1), tile_grid(reordered, 1))
        different_seed = EvaluationTerrain(config(seed=32), 8)
        for index in (1, 5):
            self.assertFalse(np.array_equal(tile_grid(reference, index), tile_grid(different_seed, index)))

    def test_terrain_direction_and_flat_entry_exit_platforms(self):
        terrain = EvaluationTerrain(config(), 6)
        platform = int(round(terrain.platform_length / terrain.horizontal_scale))
        for index, name in enumerate(TERRAIN_TYPES):
            tile = tile_grid(terrain, index)
            self.assertTrue(np.all(tile[:platform + 1] == tile[0, 0]), name)
            self.assertTrue(np.all(tile[-platform - 1:] == tile[-1, 0]), name)
            if name in ("slope", "stairs_up", "stairs_down"):
                np.testing.assert_array_equal(tile, np.repeat(tile[:, :1], tile.shape[1], axis=1))
                changes = np.diff(tile[:, 0].astype(np.int64))
                if name == "stairs_down":
                    self.assertTrue(np.all(changes <= 0))
                    self.assertGreater(tile[0, 0], tile[-1, 0])
                else:
                    self.assertTrue(np.all(changes >= 0))
                    self.assertGreater(tile[-1, 0], tile[0, 0])
                if name.startswith("stairs_"):
                    np.testing.assert_array_equal(np.unique(np.abs(changes)), [0, 20])
            elif name == "flat":
                self.assertFalse(tile.any())
            elif name == "rough":
                self.assertLessEqual(np.abs(tile).max() * terrain.vertical_scale, .03)
                self.assertGreater(np.std(tile), 0.)
            else:
                np.testing.assert_array_equal(np.unique(tile), [0, 20])

    def test_spawn_height_heading_and_footprint_are_on_safe_platforms(self):
        terrain = EvaluationTerrain(config(), 600)
        type_ids = np.repeat(np.arange(6), 100)
        plans = terrain.sample_spawns(type_ids, np.random.default_rng(46))
        start, goal = plans["ground_positions"], plans["goal_positions"]
        bounds = plans["bounds"]
        self.assertEqual(start.shape, (600, 3))
        np.testing.assert_allclose(start[:, 0], bounds[:, 0] + .75)
        np.testing.assert_allclose(goal[:, 0], bounds[:, 1] - .75)
        np.testing.assert_array_equal(start[:, 1], goal[:, 1])
        self.assertTrue(np.all(start[:, 1] >= bounds[:, 2] + .75))
        self.assertTrue(np.all(start[:, 1] <= bounds[:, 3] - .75))
        self.assertGreater(np.std(start[:100, 1]), 1.)
        np.testing.assert_array_equal(plans["directions"], np.tile([1., 0.], (600, 1)))
        np.testing.assert_allclose(start[:, 2], terrain.ground_heights(start[:, :2]))
        np.testing.assert_allclose(goal[:, 2], terrain.ground_heights(goal[:, :2]))
        self.assertTrue(np.all(goal[type_ids == 3, 2] > start[type_ids == 3, 2]))
        self.assertTrue(np.all(goal[type_ids == 4, 2] < start[type_ids == 4, 2]))
        # The full 1 m by 1 m support footprint remains on the same flat height
        # at both endpoints, including the elevated entrance of downstairs.
        for positions in (start, goal):
            for offset in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)):
                np.testing.assert_allclose(terrain.ground_heights(positions[:, :2] + offset), positions[:, 2])
        # The existing 17 x 11 Critic height scan spans x +/-0.8, y +/-0.5.
        # At the entry, its rear row reaches x=-0.05 into the extended border;
        # the front row may correctly observe the start of the slope/steps.
        scan_x, scan_y = np.meshgrid(np.linspace(-.8, .8, 17), np.linspace(-.5, .5, 11), indexing="ij")
        scan = np.column_stack((scan_x.ravel(), scan_y.ravel()))
        queries = start[:, None, :2] + scan[None]
        measured = terrain.ground_heights(queries.reshape(-1, 2)).reshape(600, 187)
        self.assertTrue(np.isfinite(measured).all())
        self.assertTrue(np.all(queries[:, :, 1] >= bounds[:, None, 2]))
        self.assertTrue(np.all(queries[:, :, 1] <= bounds[:, None, 3]))
        repeat = terrain.sample_spawns(type_ids, np.random.default_rng(46))
        for key in plans:
            np.testing.assert_array_equal(plans[key], repeat[key])

    def test_seams_and_outer_border_extend_ground_without_cliffs(self):
        terrain = EvaluationTerrain(config(type_names=["flat", "stairs_down"]), 2)
        x = 2.
        y = np.arange(8., 10. + .05, .1)
        actual = terrain.ground_heights(np.column_stack((np.full(len(y), x), y)))
        expected = np.linspace(actual[0], actual[-1], len(y))
        np.testing.assert_allclose(actual, expected, atol=terrain.vertical_scale / 2 + 1e-9)
        for tile in range(2):
            middle_y = (terrain.tile_bounds[tile, 2] + terrain.tile_bounds[tile, 3]) / 2
            entry = terrain.ground_heights(np.array([[0., middle_y]]))[0]
            exit_height = terrain.ground_heights(np.array([[8., middle_y]]))[0]
            np.testing.assert_allclose(terrain.ground_heights(np.array([[-2., middle_y], [-1., middle_y]])), entry)
            np.testing.assert_allclose(terrain.ground_heights(np.array([[9., middle_y], [10., middle_y]])), exit_height)

    def test_triangle_mesh_arrays_match_heightfield_and_offset(self):
        terrain = EvaluationTerrain(config(mesh_type="trimesh", type_names=["flat", "stairs_up"]), 4)
        self.assertEqual(terrain.vertices.shape, (terrain.tot_rows * terrain.tot_cols, 3))
        self.assertEqual(terrain.triangles.shape, (2 * (terrain.tot_rows - 1) * (terrain.tot_cols - 1), 3))
        self.assertEqual(terrain.vertices.dtype, np.float32)
        self.assertEqual(terrain.triangles.dtype, np.uint32)
        self.assertGreaterEqual(terrain.triangles.min(), 0)
        self.assertLess(terrain.triangles.max(), len(terrain.vertices))
        np.testing.assert_allclose(terrain.vertices[:, 2], terrain.heightsamples.ravel() * terrain.vertical_scale)
        first_world = terrain.vertices[0, :2] - terrain.cfg.border_size
        np.testing.assert_allclose(first_world, [-2., -2.])
        self.assertTrue(np.isfinite(terrain.vertices).all())

    def test_invalid_region_and_parameter_inputs_fail_explicitly(self):
        cases = (
            dict(terrain_length=.4, terrain_width=.4),
            dict(terrain_length=8., terrain_width=7.),
            dict(terrain_length=8.01, terrain_width=8.01),
            dict(type_names=[]), dict(type_names=["flat", "flat"]),
            dict(type_names=["unknown"]), dict(platform_length=.5),
            dict(rough_height=-.1), dict(step_width=0.), dict(horizontal_scale=0.),
            dict(slope_angle_deg=90.), dict(obstacle_density=2.), dict(seed=-1),
            dict(obstacle_size_min=1., obstacle_size_max=.5),
            dict(terrain_length=3.8, terrain_width=3.8, type_names=["discrete_obstacles"]),
            dict(terrain_length=40., terrain_width=40., slope_angle_deg=89., type_names=["slope"]),
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                EvaluationTerrain(config(**changes), 2)

    def test_invalid_spawn_and_height_queries_fail_explicitly(self):
        terrain = EvaluationTerrain(config(type_names=["flat"]), 1)
        invalid = (dict(type_ids=[-1]), dict(type_ids=[1]), dict(type_ids=[.1]),
                   dict(spawn_inset=1.5), dict(goal_inset=0.), dict(lateral_margin=4.))
        for changes in invalid:
            arguments = dict(type_ids=np.array([0]), rng=np.random.default_rng(1))
            arguments.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                terrain.sample_spawns(**arguments)
        with self.assertRaises(ValueError):
            terrain.ground_heights(np.array([[-2.1, 1.]]))
        empty = terrain.sample_spawns(np.array([], dtype=np.int64), np.random.default_rng(1))
        self.assertEqual(empty["ground_positions"].shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
