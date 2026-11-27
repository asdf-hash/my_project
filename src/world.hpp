#pragma once

#include <array>
#include <cstdint>
#include <string_view>

inline constexpr int CHUNK_SIZE = 32;

namespace world {
struct ChunkLod {
  int32_t x;     // chunk column x, in chunks
  int32_t z;     // chunk column z, in chunks
  int32_t lod;   // level of detail; a tile spans `scale` = 1 << lod chunks per axis
  int32_t scale;

  bool operator==(const ChunkLod &) const = default;
};
inline constexpr int32_t kRenderDistance = 6;
// Square radius (in chunks) around the player whose columns are generated and meshed on the CPU (cpu_world.hpp);
// the GPU clipmap skips them.
inline constexpr int32_t kCpuChunkRadius = 8;
// Camera more than this many chunks above the CPU square's top: the CPU chunks are hidden and the clipmap picks every
// tile's LOD from distance alone instead of forcing LOD 0 around the square.
inline constexpr int32_t kCpuHideHeightChunks = 4;
// Seam border heights are floored from (height - this) on both sides, so a few ulp of CPU/GPU noise disagreement can
// only lengthen a seam wall, never open a gap. Must match CPU_SEAM_BIAS in chunk_compute.slang.
inline constexpr float kCpuSeamBias = 0.02f;
// ClipmapResult::neighborLods: set in an edge's byte when CPU-owned columns lie across it.
inline constexpr uint32_t kNeighborCpuFlag = 0x8u;
// Empty cells below this height are ocean. Must match SEA_LEVEL in chunk_compute.slang.
inline constexpr int32_t kSeaLevel = 0;
// Conservative world-space envelope of the height function in terrain_noise.hlsl.
inline constexpr int32_t kMinTerrainHeight = -400;
inline constexpr int32_t kMaxTerrainHeight = 1100;
// Most vertical chunks one GPU clipmap column can need (LOD 0 over the whole envelope). Must match MAX_COLUMN_CHUNKS
// in chunk_compute.slang.
inline constexpr int32_t kMaxColumnChunks =
    (kMaxTerrainHeight - 1) / CHUNK_SIZE - (kMinTerrainHeight - CHUNK_SIZE + 1) / CHUNK_SIZE + 1;
static_assert(kMaxColumnChunks == 48, "update MAX_COLUMN_CHUNKS in chunk_compute.slang");
// Material layering. Must match TRANSITION_HEIGHTS, TRANSITION_DITHER, DITHER_WAVELENGTH, FILLER_DEPTH and
// SEED_DITHER* in chunk_compute.slang.
inline constexpr std::array<int32_t, 3> kTransitionHeights{2, 320, 600};
inline constexpr std::array<float, 3> kTransitionDither{0.0f, 40.0f, 40.0f};
inline constexpr float kDitherWavelength = 160.0f;
inline constexpr int32_t kFillerDepth = 3;
inline constexpr uint32_t kSeedDither = 0x4cf5ad43u;
inline constexpr uint32_t kSeedDitherWhite = 0x2545f491u;
// Surface block per height layer, bottom to top (one more entry than TRANSITION_HEIGHTS).
inline constexpr std::array<std::string_view, 4> kTerrainLayerBlocks{"sand", "grass", "stone", "snow"};
// The kFillerDepth blocks under a column's top block, per layer.
inline constexpr std::array<std::string_view, kTerrainLayerBlocks.size()> kTerrainLayerFillerBlocks{"sand", "dirt",
                                                                                                    "stone", "snow"};
// Everything deeper than the filler, per layer.
inline constexpr std::array<std::string_view, kTerrainLayerBlocks.size()> kTerrainLayerDeepBlocks{"stone", "stone",
                                                                                                  "stone", "stone"};
// Tree blocks (placement: shaders/vegetation_shared.hlsl).
inline constexpr std::string_view kTreeLogBlock = "log";
inline constexpr std::string_view kTreeLeavesBlock = "leaves";
} // namespace world
