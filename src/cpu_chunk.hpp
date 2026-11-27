#pragma once

// RAM data of the CPU-owned columns (cpu_world.hpp): a padded heightmap plus voxels for the chunks crossing the
// surface. Chunks outside that range store nothing and follow the heightmap.

#include "block_registry.hpp"
#include "vegetation.hpp"
#include "world.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace cpuworld {

inline constexpr int32_t kPad = CHUNK_SIZE + 2;                 // 34: one border column on each side
inline constexpr uint16_t kAir = 0xffffu;                       // block id of air in VoxelChunk::ids (bit 9 set)
// Stands for "some opaque terrain block" where a cell has no voxel storage (below a column's stored chunks).
inline constexpr uint16_t kBuriedTerrain = 0;
inline constexpr int32_t kVoxelsPerChunk = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/** Floor division; `divisor` must be positive. */
inline int32_t floorDiv(int32_t value, int32_t divisor) {
  return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

// Block ids for the height layers (world::kTerrainLayer*Blocks resolved through the block registry).
struct TerrainBlockIds {
  std::array<uint16_t, 4> top{};
  std::array<uint16_t, 4> filler{};
  std::array<uint16_t, 4> deep{};
  // Non-opaque; fills empty cells below world::kSeaLevel. kAir disables the ocean.
  uint16_t ocean = 0xffffu;
  // Log is opaque, leaves are not. kAir in either disables trees.
  uint16_t log = 0xffffu;
  uint16_t leaves = 0xffffu;
};

struct VoxelChunk {
  // Column-major: index (z * 32 + x) * 32 + y, so a column's cells are contiguous.
  std::array<uint16_t, kVoxelsPerChunk> ids;
  // Bit y set = opaque (a block whose id has bit 9 clear; air has it set), index z * 32 + x.
  std::array<uint32_t, CHUNK_SIZE * CHUNK_SIZE> solid;
  // Bit y set = ocean, index z * 32 + x. Ocean is non-opaque, so these cells are never in `solid`.
  std::array<uint32_t, CHUNK_SIZE * CHUNK_SIZE> ocean;
  // Cells holding a non-opaque block other than air and ocean; the per-cell non-opaque mesher skips chunks at 0.
  uint16_t nonOpaqueCount = 0;

  /** Index into `ids` of chunk-local cell (x, y, z). */
  static constexpr int32_t index(int32_t x, int32_t y, int32_t z) { return (z * CHUNK_SIZE + x) * CHUNK_SIZE + y; }
};

struct Column {
  int32_t cx = 0; // chunk column x, in chunks
  int32_t cz = 0; // chunk column z, in chunks
  // Raw surface heights over the padded 34x34 area, index padIndex(lx, lz) for column-local lx, lz in -1..32.
  std::array<float, kPad * kPad> rawHeights{};
  // Interior column tops in blocks (first air block), as generateHeights computes them at LOD 0.
  std::array<int32_t, CHUNK_SIZE * CHUNK_SIZE> heights{};
  std::array<float, CHUNK_SIZE * CHUNK_SIZE> dither{}; // layer-boundary dither per interior column, index lz * 32 + lx
  // Tree candidates by slot (vegetation_shared.hlsl treeCandidate); trunkHeight 0 where there is none. Every tree
  // lies inside this column; canopies may overlap.
  std::array<vegetation::TreeInstance, vegetation::TREE_SLOTS> trees{};
  // Block id of ocean cells (TerrainBlockIds::ocean); unstored cells between the column top and sea level are ocean.
  uint16_t oceanId = kAir;
  // Voxel chunks for chunkY in [firstChunkY, firstChunkY + chunks.size()); null slots are unstored and follow the
  // heightmap like chunks outside the range (e.g. fully ocean chunks between seabed and surface).
  int32_t firstChunkY = 0;
  std::vector<std::unique_ptr<VoxelChunk>> chunks;
  // Blocks were broken, placed or flooded here, so the voxels no longer follow the heightmap.
  bool edited = false;
  // Per interior column (index lz * 32 + lx): lowest world y with no opaque cell at or above it, so cells from here
  // up see open sky (skylight seeds). Kept current by rescanSkyTop after every opacity change.
  std::array<int32_t, CHUNK_SIZE * CHUNK_SIZE> skyTop{};
  // World y range [shadowLow, shadowHigh) covering every non-opaque cell below its column's skyTop (cells skylight
  // reaches only sideways). Empty for plain heightmap terrain, so computeSkyLight can skip most chunks; after edits
  // it only grows (a conservative superset).
  int32_t shadowLow = std::numeric_limits<int32_t>::max();
  int32_t shadowHigh = std::numeric_limits<int32_t>::min();

  /** Index into `rawHeights` of column-local (lx, lz), each in -1..32. */
  static constexpr int32_t padIndex(int32_t lx, int32_t lz) { return (lz + 1) * kPad + (lx + 1); }

  /** Height of a border cell (lx or lz of -1 or 32), biased down by kCpuSeamBias so a seam wall can only lengthen. */
  int32_t borderHeight(int32_t lx, int32_t lz) const {
    return std::max(static_cast<int32_t>(std::floor(rawHeights[padIndex(lx, lz)] - world::kCpuSeamBias)),
                    world::kMinTerrainHeight);
  }
  /** Interior column top (first air block). */
  int32_t height(int32_t lx, int32_t lz) const { return heights[lz * CHUNK_SIZE + lx]; }

  /** Stored voxels of chunk `chunkY`, or null if unstored. */
  const VoxelChunk *chunk(int32_t chunkY) const {
    const int32_t i = chunkY - firstChunkY;
    return i >= 0 && i < static_cast<int32_t>(chunks.size()) ? chunks[i].get() : nullptr;
  }

  /** Ocean bits of column (lx, lz) within chunk `chunkY` (bit y = chunk-local cell y). */
  uint32_t ocean32(int32_t lx, int32_t lz, int32_t chunkY) const {
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return voxels->ocean[lz * CHUNK_SIZE + lx];
    }
    return oceanId == kAir ? 0u : oceanMask(height(lx, lz), chunkY);
  }

  /** Opaque bits of column (lx, lz) within chunk `chunkY` (bit y = chunk-local cell y). */
  uint32_t solid32(int32_t lx, int32_t lz, int32_t chunkY) const {
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return voxels->solid[lz * CHUNK_SIZE + lx];
    }
    return heightMask(height(lx, lz), chunkY);
  }

  /** Whether column-local cell (lx, y, lz) is opaque; y is in world blocks. */
  bool isSolid(int32_t lx, int32_t y, int32_t lz) const {
    const int32_t chunkY = floorDiv(y, CHUNK_SIZE);
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return (voxels->solid[lz * CHUNK_SIZE + lx] >> (y - chunkY * CHUNK_SIZE)) & 1u;
    }
    return y < height(lx, lz);
  }

  /**
   * Block at column-local cell (lx, y, lz). Unstored cells follow the heightmap: kBuriedTerrain below the top, then
   * ocean below sea level, then kAir.
   */
  uint16_t blockId(int32_t lx, int32_t y, int32_t lz) const {
    const int32_t chunkY = floorDiv(y, CHUNK_SIZE);
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return voxels->ids[VoxelChunk::index(lx, y - chunkY * CHUNK_SIZE, lz)];
    }
    return y < height(lx, lz) ? kBuriedTerrain : emptyCell(y);
  }
  /** What a non-terrain cell at height y holds: ocean below sea level, else air. */
  uint16_t emptyCell(int32_t y) const { return y < world::kSeaLevel ? oceanId : kAir; }

  /**
   * Recompute skyTop of column (lx, lz) by scanning the stored range top down (below it the heightmap holds), and
   * grow [shadowLow, shadowHigh) by the column's non-opaque cells under it.
   */
  void rescanSkyTop(int32_t lx, int32_t lz) {
    const int32_t lastChunkY = firstChunkY + static_cast<int32_t>(chunks.size()) - 1;
    int32_t top = std::min(height(lx, lz), firstChunkY * CHUNK_SIZE);
    for (int32_t chunkY = lastChunkY; chunkY >= firstChunkY; --chunkY) {
      if (const uint32_t bits = solid32(lx, lz, chunkY); bits != 0u) {
        top = chunkY * CHUNK_SIZE + 32 - std::countl_zero(bits);
        break;
      }
    }
    skyTop[lz * CHUNK_SIZE + lx] = top;
    // Below the stored range everything is opaque (heights never drop below firstChunkY's base).
    for (int32_t chunkY = firstChunkY; chunkY * CHUNK_SIZE < top; ++chunkY) {
      const int32_t base = chunkY * CHUNK_SIZE;
      uint32_t open = ~solid32(lx, lz, chunkY);
      if (top - base < CHUNK_SIZE) {
        open &= (1u << (top - base)) - 1u;
      }
      if (open != 0u) {
        shadowLow = std::min(shadowLow, base + std::countr_zero(open));
        shadowHigh = std::max(shadowHigh, base + CHUNK_SIZE - std::countl_zero(open));
      }
    }
  }

  /** Bits of chunk `chunkY` below a column top at `height` (all 32 set when the top is above the chunk). */
  static uint32_t heightMask(int32_t height, int32_t chunkY) {
    const int32_t n = std::clamp(height - chunkY * CHUNK_SIZE, 0, CHUNK_SIZE);
    return n >= 32 ? 0xffffffffu : (1u << n) - 1u;
  }
  /** Bits of chunk `chunkY` between a column top at `height` and sea level (the implicit ocean). */
  static uint32_t oceanMask(int32_t height, int32_t chunkY) {
    return heightMask(world::kSeaLevel, chunkY) & ~heightMask(height, chunkY);
  }
};

// One QuadCluster (chunk_compute.slang / geometry_shared.hlsl): 16 per chunk, one per 8x8 column tile.
struct QuadCluster {
  uint32_t base; // first quad, relative to ChunkMesh::quads until the renderer rebases it
  uint32_t counts01; // quad counts per face direction, two per word (layout: geometry_shared.hlsl)
  uint32_t counts23;
  uint32_t counts45;
  std::array<float, 4> boundsMin;
  std::array<float, 4> boundsMax;
};
static_assert(sizeof(QuadCluster) == 48);
inline constexpr uint32_t kClustersPerChunk = 16;

struct ChunkMesh {
  int32_t cy = 0; // chunk y, in chunks
  // Packed quads (QuadBuffer layout) in cluster order; every quad shares chunkOffset (ChunkOffsets layout).
  std::vector<uint64_t> quads;
  uint64_t chunkOffset = 0;
  std::array<QuadCluster, kClustersPerChunk> clusters{};
  // Translucent (ocean) faces for the blended pass: same tile layout, quads stored after the opaque ones.
  std::array<QuadCluster, kClustersPerChunk> translucentClusters{};
};

// The column being meshed and its 8 neighbors; null where a neighbor isn't loaded (the center's border heights are
// used instead).
struct Neighborhood {
  const Column *columns[3][3]{}; // [dz + 1][dx + 1]
  /** The column being meshed. */
  const Column &center() const { return *columns[1][1]; }
};

} // namespace cpuworld
