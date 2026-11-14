"""Stable identifiers and dimensions for evaluation, separate from training."""

PATCH_TERRAIN_TYPES = (
    "flat",
    "stairs_concave_15cm",
    "stairs_concave_20cm",
    "stairs_convex_15cm",
    "stairs_convex_20cm",
    "rough",
    "slope",
)
PATCH_STAIR_TREAD_M = 0.3
PATCH_STEPS_PER_SIDE = 8
PATCH_STAIR_PLATFORM_M = 3.0
# Preserve all eight risers and the central platform when changing tread width.
PATCH_SIZE_M = 2 * PATCH_STEPS_PER_SIDE * PATCH_STAIR_TREAD_M + PATCH_STAIR_PLATFORM_M
PATCH_PLATFORM_LENGTH_M = 1.5
PATCH_CORRIDOR_WIDTH_M = PATCH_SIZE_M + 2.0
