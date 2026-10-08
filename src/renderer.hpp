#pragma once

// Higher values retain finer terrain farther away: 0 maps to 0.005, 100 to 0.003.
inline constexpr float LOD_DETAIL_PERCENT = 40.f;
static_assert(LOD_DETAIL_PERCENT >= 0.0f && LOD_DETAIL_PERCENT <= 100.0f);

inline constexpr float TARGET_SCREEN_PERCENTAGE = 0.005f - (LOD_DETAIL_PERCENT / 100.0f) * (0.005f - 0.003f);
inline constexpr bool TAA_ENABLED = true;
inline constexpr float TAA_JITTER_PIXELS = 0.25f;
inline constexpr float TAA_DEPTH_REJECTION_THRESHOLD = 0.001f;
inline constexpr double MIN_SUPPORTED_TAN_HALF_FOV = 0.70;
// Per LOD step, blend this fraction of the remaining AO toward fully lit; total lift is capped at 50%.
inline constexpr float LOD_AO_LIGHTENING_PER_LEVEL = 0.15f;
static_assert(LOD_AO_LIGHTENING_PER_LEVEL >= 0.0f && LOD_AO_LIGHTENING_PER_LEVEL <= 0.5f);

class Renderer {
public:
  void run();
};
