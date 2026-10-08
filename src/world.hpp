#pragma once

#include <cstdint>
#include <utility>

inline constexpr int CHUNK_SIZE = 32;

namespace world {
using ChunkCoord = std::pair<int32_t, int32_t>;
struct ChunkLod {
  int32_t x;
  int32_t z;
  int32_t lod;
  int32_t scale;

  bool operator==(const ChunkLod &) const = default;
};
inline constexpr int32_t kRenderDistance = 6;
inline constexpr int32_t kVerticalRenderDistance = kRenderDistance;
// Conservative world-space height envelope for the height function in terrain_noise.hlsl.
inline constexpr int32_t kMinTerrainHeight = -400;
inline constexpr int32_t kMaxTerrainHeight = 600;
} // namespace world
