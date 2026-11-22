#pragma once

#include <array>
#include <cstdint>
#include <string_view>
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
// Chunk columns within this many chunks of the player (a square, like Minecraft's render distance) are generated and
// meshed on the CPU (cpu_world.hpp) and kept in RAM; the GPU clipmap skips them.
inline constexpr int32_t kCpuChunkRadius = 8;
// Border heights across a CPU <-> GPU seam are floored from (height - this), on both sides, so a few ulp of
// disagreement between the two noise implementations can only lengthen a seam wall (hidden in terrain), never open
// a gap. Must match CPU_SEAM_BIAS in chunk_compute.slang.
inline constexpr float kCpuSeamBias = 0.02f;
// ClipmapResult::neighborLods: set in an edge's byte when CPU-owned columns lie across it.
inline constexpr uint32_t kNeighborCpuFlag = 0x8u;
// Ocean surface: every cell below this height that the terrain leaves empty is ocean (the land floor in
// terrain_noise.hlsl keeps inland ground above it). Must match SEA_LEVEL in chunk_compute.slang.
inline constexpr int32_t kSeaLevel = 0;
// Conservative world-space height envelope for the height function in terrain_noise.hlsl.
inline constexpr int32_t kMinTerrainHeight = -400;
inline constexpr int32_t kMaxTerrainHeight = 1100;
// Most vertical chunks one GPU clipmap column can need: LOD 0 over the whole envelope. The GPU picks the chunks
// each column actually meshes from its generated heights (allocateWork in chunk_compute.slang). Must match
// MAX_COLUMN_CHUNKS there.
inline constexpr int32_t kMaxColumnChunks =
    (kMaxTerrainHeight - 1) / CHUNK_SIZE - (kMinTerrainHeight - CHUNK_SIZE + 1) / CHUNK_SIZE + 1;
static_assert(kMaxColumnChunks == 48, "update MAX_COLUMN_CHUNKS in chunk_compute.slang");
// Material layering, mirrored from chunk_compute.slang (TRANSITION_HEIGHTS, TRANSITION_DITHER, DITHER_WAVELENGTH,
// FILLER_DEPTH, SEED_DITHER*). Must match.
inline constexpr std::array<int32_t, 3> kTransitionHeights{2, 320, 600};
inline constexpr std::array<float, 3> kTransitionDither{0.0f, 40.0f, 40.0f};
inline constexpr float kDitherWavelength = 160.0f;
inline constexpr int32_t kFillerDepth = 3;
inline constexpr uint32_t kSeedDither = 0x4cf5ad43u;
inline constexpr uint32_t kSeedDitherWhite = 0x2545f491u;
// Surface block per height layer, bottom to top. Layer boundaries are TRANSITION_HEIGHTS in chunk_compute.slang
// (one fewer entry than this list).
inline constexpr std::array<std::string_view, 4> kTerrainLayerBlocks{"sand", "grass", "stone", "snow"};
// Block for the FILLER_DEPTH blocks under a column's top block, per layer (grass on top, dirt underneath).
inline constexpr std::array<std::string_view, kTerrainLayerBlocks.size()> kTerrainLayerFillerBlocks{"sand", "dirt",
                                                                                                    "stone", "snow"};
// Block deeper than the filler, per layer (grass, 3 dirt, then stone; sand / snow: 3 more, then stone).
inline constexpr std::array<std::string_view, kTerrainLayerBlocks.size()> kTerrainLayerDeepBlocks{"stone", "stone",
                                                                                                  "stone", "stone"};
} // namespace world
