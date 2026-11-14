"""Verify actual training stair spacing through Isaac Gym's grid conversion."""
import unittest

import isaacgym  # Must precede torch imported by the registered environments.
from isaacgym import terrain_utils
import numpy as np

from wheel_legged_gym import envs
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.utils.helpers import class_to_dict
from wheel_legged_gym.utils.terrain import Terrain


def terrain_builder(cfg):
    builder = Terrain.__new__(Terrain)
    builder.cfg = cfg
    builder.width_per_env_pixels = round(cfg.terrain_width / cfg.horizontal_scale)
    builder.length_per_env_pixels = round(cfg.terrain_length / cfg.horizontal_scale)
    builder.proportions = np.cumsum(cfg.terrain_proportions)
    return builder


class TrainingStairTests(unittest.TestCase):
    def test_all_variants_generate_three_cell_treads_at_every_curriculum_level(self):
        for variant in ('full', 'no_latent_input', 'no_gate', 'no_state_input', 'no_mode_loss'):
            cfg, _ = task_registry.get_cfgs('mine_ablation_' + variant)
            self.assertEqual(class_to_dict(cfg)['terrain']['stair_tread_width'], .3)
            self.assertEqual(cfg.terrain.horizontal_scale, .1)
            builder = terrain_builder(cfg.terrain)
            for level in range(cfg.terrain.num_rows):
                for choice, sign in ((.6, -1), (.8, 1)):
                    with self.subTest(variant=variant, level=level, sign=sign):
                        tile = builder.make_terrain(choice, level / cfg.terrain.num_rows)
                        profile = tile.height_field_raw[:, tile.length // 2]
                        edges = np.flatnonzero(np.diff(profile)) + 1
                        np.testing.assert_array_equal(edges, np.r_[np.arange(3, 31, 3), np.arange(70, 98, 3)])
                        height_units = int((.03 + .18 * level / cfg.terrain.num_rows) / .005)
                        self.assertEqual(int(profile[50]), sign * 10 * height_units)
                        # Check the converted mesh as well as the heightfield.
                        vertices, _ = terrain_utils.convert_heightfield_to_trimesh(
                            tile.height_field_raw, tile.horizontal_scale, tile.vertical_scale,
                            cfg.terrain.slope_treshold)
                        line = vertices.reshape(tile.width, tile.length, 3)[:, tile.length // 2]
                        changed = np.flatnonzero(np.abs(np.diff(line[:, 2])) > 1e-6)
                        np.testing.assert_allclose(np.diff(line[changed[:10], 0]), .3, atol=1e-6)
                        np.testing.assert_allclose(np.diff(line[changed[10:], 0]), .3, atol=1e-6)

    def test_unrepresentable_tread_is_rejected(self):
        cfg, _ = task_registry.get_cfgs('mine_ablation_full')
        cfg.terrain.stair_tread_width = .35
        with self.assertRaisesRegex(ValueError, 'positive multiple'):
            terrain_builder(cfg.terrain).make_terrain(.8, .9)


if __name__ == '__main__':
    unittest.main()
