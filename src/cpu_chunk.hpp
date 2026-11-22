#pragma once

// Data held in RAM for the CPU-owned chunk columns (cpu_world.hpp): per column a padded heightmap and voxel arrays for
// the chunks that cross the surface. Chunks below them are "buried" and chunks above "empty": both follow the
// heightmap and store nothing.

#include "block_registry.hpp"
#include "vegetation.hpp"
#include "world.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace cpuworld {

inline constexpr int32_t kPad = CHUNK_SIZE + 2;                 // 34: one border column on each side
inline constexpr uint16_t kAir = 0xffffu;                       // block id of air in VoxelChunk::ids (bit 9 set)
// Stands for "some opaque terrain block" where a cell has no voxel storage (below a column's stored chunks).
inline constexpr uint16_t kBuriedTerrain = 0;
inline constexpr int32_t kVoxelsPerChunk = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

inline int32_t floorDiv(int32_t value, int32_t divisor) { // divisor > 0
  return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

// Block ids for the height layers (world::kTerrainLayer*Blocks resolved through the block registry).
struct TerrainBlockIds {
  std::array<uint16_t, 4> top{};
  std::array<uint16_t, 4> filler{};
  std::array<uint16_t, 4> deep{};
  // Ocean block (non-opaque): fills empty cells below world::kSeaLevel. kAir disables the ocean.
  uint16_t ocean = 0xffffu;
  // Tree blocks (world::kTreeLogBlock opaque, world::kTreeLeavesBlock non-opaque). kAir in either disables trees.
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
  // Cells holding a non-opaque block other than air and ocean: the per-cell non-opaque mesher skips chunks without
  // any (ocean has its own bitmask pass).
  uint16_t nonOpaqueCount = 0;

  static constexpr int32_t index(int32_t x, int32_t y, int32_t z) { return (z * CHUNK_SIZE + x) * CHUNK_SIZE + y; }
};

struct Column {
  int32_t cx = 0;
  int32_t cz = 0;
  // Raw surface heights over the padded 34x34 area, index (lz + 1) * 34 + (lx + 1) for lx, lz in -1..32.
  std::array<float, kPad * kPad> rawHeights{};
  // Interior column tops in blocks (first air block), as generateHeights computes them at LOD 0.
  std::array<int32_t, CHUNK_SIZE * CHUNK_SIZE> heights{};
  std::array<float, CHUNK_SIZE * CHUNK_SIZE> dither{};
  // One tree candidate per 8x8 cell (index cellZ * 4 + cellX), ground set, trunkHeight 0 where there is none
  // (vegetation_shared.hlsl). Every tree lies inside its cell, so inside this column.
  static constexpr int32_t kTreeCellsPerSide = CHUNK_SIZE >> vegetation::TREE_CELL_SHIFT;
  std::array<vegetation::TreeInstance, kTreeCellsPerSide * kTreeCellsPerSide> trees{};
  // Block id of ocean cells (TerrainBlockIds::ocean); unstored cells between the column top and sea level are ocean.
  uint16_t oceanId = kAir;
  // Voxel chunks cover chunkY in [firstChunkY, firstChunkY + chunks.size()); null slots are unstored (follow the
  // heightmap, like chunks outside the range: fully ocean chunks between the seabed and the surface).
  int32_t firstChunkY = 0;
  std::vector<std::unique_ptr<VoxelChunk>> chunks;
  // Blocks were broken or placed here, so the voxels no longer follow the heightmap (CpuWorld::breakBlock /
  // placeBlock).
  bool edited = false;

  static constexpr int32_t padIndex(int32_t lx, int32_t lz) { return (lz + 1) * kPad + (lx + 1); }

  // Height used for columns outside this one (border cells): biased down so a seam wall can only get longer.
  int32_t borderHeight(int32_t lx, int32_t lz) const {
    return std::max(static_cast<int32_t>(std::floor(rawHeights[padIndex(lx, lz)] - world::kCpuSeamBias)),
                    world::kMinTerrainHeight);
  }
  int32_t height(int32_t lx, int32_t lz) const { return heights[lz * CHUNK_SIZE + lx]; }

  const VoxelChunk *chunk(int32_t chunkY) const {
    const int32_t i = chunkY - firstChunkY;
    return i >= 0 && i < static_cast<int32_t>(chunks.size()) ? chunks[i].get() : nullptr;
  }

  // Ocean bits of column (lx, lz) within chunk chunkY (bit y = chunk-local cell y).
  uint32_t ocean32(int32_t lx, int32_t lz, int32_t chunkY) const {
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return voxels->ocean[lz * CHUNK_SIZE + lx];
    }
    return oceanId == kAir ? 0u : oceanMask(height(lx, lz), chunkY);
  }

  // Solid bits of column (lx, lz) within chunk chunkY (bit y = chunk-local cell y).
  uint32_t solid32(int32_t lx, int32_t lz, int32_t chunkY) const {
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return voxels->solid[lz * CHUNK_SIZE + lx];
    }
    return heightMask(height(lx, lz), chunkY);
  }

  bool isSolid(int32_t lx, int32_t y, int32_t lz) const {
    const int32_t chunkY = floorDiv(y, CHUNK_SIZE);
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return (voxels->solid[lz * CHUNK_SIZE + lx] >> (y - chunkY * CHUNK_SIZE)) & 1u;
    }
    return y < height(lx, lz);
  }

  // Block at a cell; unstored cells follow the heightmap (kBuriedTerrain below the top, then ocean below sea level,
  // then kAir).
  uint16_t blockId(int32_t lx, int32_t y, int32_t lz) const {
    const int32_t chunkY = floorDiv(y, CHUNK_SIZE);
    if (const VoxelChunk *voxels = chunk(chunkY)) {
      return voxels->ids[VoxelChunk::index(lx, y - chunkY * CHUNK_SIZE, lz)];
    }
    return y < height(lx, lz) ? kBuriedTerrain : emptyCell(y);
  }
  // What an empty (not terrain) cell at height y holds: ocean below sea level, else air.
  uint16_t emptyCell(int32_t y) const { return y < world::kSeaLevel ? oceanId : kAir; }

  // Bits of chunk chunkY below a column top at `height`.
  static uint32_t heightMask(int32_t height, int32_t chunkY) {
    const int32_t n = std::clamp(height - chunkY * CHUNK_SIZE, 0, CHUNK_SIZE);
    return n >= 32 ? 0xffffffffu : (1u << n) - 1u;
  }
  // Bits of chunk chunkY between a column top at `height` and sea level (the implicit ocean).
  static uint32_t oceanMask(int32_t height, int32_t chunkY) {
    return heightMask(world::kSeaLevel, chunkY) & ~heightMask(height, chunkY);
  }
};

// One QuadCluster (chunk_compute.slang / geometry_shared.hlsl): 16 per chunk, one per 8x8 column tile.
struct QuadCluster {
  uint32_t base; // first quad, relative to ChunkMesh::quads until the renderer rebases it
  uint32_t counts01;
  uint32_t counts23;
  uint32_t counts45;
  std::array<float, 4> boundsMin;
  std::array<float, 4> boundsMax;
};
static_assert(sizeof(QuadCluster) == 48);
inline constexpr uint32_t kClustersPerChunk = 16;

struct ChunkMesh {
  int32_t cx = 0;
  int32_t cy = 0;
  int32_t cz = 0;
  // Packed quads (QuadBuffer layout) in cluster order; every quad shares chunkOffset (ChunkOffsets layout).
  std::vector<uint64_t> quads;
  uint64_t chunkOffset = 0;
  std::array<QuadCluster, kClustersPerChunk> clusters{};
  // Translucent faces (ocean), drawn in a separate blended pass: same tile layout, their quads follow the opaque
  // clusters' quads in `quads`.
  std::array<QuadCluster, kClustersPerChunk> translucentClusters{};
};

// The column being meshed and its 8 neighbors; null where a neighbor isn't loaded (its border heights are used).
struct Neighborhood {
  const Column *columns[3][3]{}; // [dz + 1][dx + 1]
  const Column &center() const { return *columns[1][1]; }
};

} // namespace cpuworld
