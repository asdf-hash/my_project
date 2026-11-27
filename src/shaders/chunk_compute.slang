#include "terrain_noise.hlsl"

static const uint CHUNK_SIZE = 32u;
static const uint HEIGHT_STRIDE = 34u;
static const uint HEIGHTS_PER_CHUNK = HEIGHT_STRIDE * HEIGHT_STRIDE;
static const uint MAX_QUAD_COUNT = 1u << 24u;
// Must match VERTEX_QUADS_PER_INSTANCE in slang.slang and renderer.cpp.
static const uint VERTEX_QUADS_PER_INSTANCE = 16384u;
// world::kMinTerrainHeight / kMaxTerrainHeight.
static const int MIN_TERRAIN_HEIGHT = -400;
static const int MAX_TERRAIN_HEIGHT = 1100;
// world::kSeaLevel. The GPU clipmap draws only the ocean surface: one +Y quad per column into WaterClusters.
static const int SEA_LEVEL = 0;
// Most vertical chunks one column can need (LOD 0 over the whole envelope): world::kMaxColumnChunks.
static const uint MAX_COLUMN_CHUNKS = 48u;
static const uint INVALID_WORK_SLOT = 0xffffffffu;
// generateHeights groups per column: 16 interior tiles + 3 border lines.
static const uint HEIGHT_GROUPS = 19u;
static const uint GROUP_LIMIT = 65535u;
// WorkPool layout (ints), matching renderer.cpp: free-stack top, work slot high water, ChunkWork entries this remesh,
// overflow flag, then the indirect arguments finalizeWork writes (each 3 ints, 16-byte aligned) with the
// relocation counters (collectRelocations) in the gaps.
static const uint WORK_POOL_FREE_TOP = 0u;
static const uint WORK_POOL_HIGH_WATER = 1u;
static const uint WORK_POOL_NEW_WORK = 2u;
static const uint WORK_POOL_OVERFLOW = 3u;
static const uint WORK_POOL_MESH_ARGS = 4u;    // meshColumns: (16 tiles, items folded into y / z)
static const uint WORK_POOL_HOLES = 7u;        // free slots below the live count
static const uint WORK_POOL_COMPACT_ARGS = 8u; // compactQuads: (16 clusters, work slots folded into y / z)
static const uint WORK_POOL_MOVERS = 11u;      // live slots at or above the live count
static const uint WORK_POOL_TASK_ARGS = 12u;   // drawMeshTasksIndirectEXT over the GPU clipmap's clusters

// Layer tops: sand to 2 (beaches), grass to 320, stone to 600, snow above.
static const int TRANSITION_HEIGHTS[3] = { 2, 320, 600 };
static const int heightCount = TRANSITION_HEIGHTS.getCount();
// Per-column boundary shift in blocks (transitionDither) for ragged layer bands; the beach line is not dithered.
static const float TRANSITION_DITHER[3] = { 0.0f, 40.0f, 40.0f };
static const float DITHER_WAVELENGTH = 160.0f;
// Blocks of filler (dirt under grass) between a column's top block and its deep block (stone).
static const int FILLER_DEPTH = 3;
static const uint SEED_DITHER = 0x4cf5ad43u;
static const uint SEED_DITHER_WHITE = 0x2545f491u;
// Edges next to CPU-meshed columns (world::kNeighborCpuFlag) floor border heights from height - CPU_SEAM_BIAS
// (world::kCpuSeamBias), so ulp differences between the noise ports can only lengthen a hidden seam wall.
static const uint NEIGHBOR_CPU_FLAG = 0x8u;
static const float CPU_SEAM_BIAS = 0.02f;

struct DispatchParams {
  uint horizontalChunkCount; // entries in ColumnList (new columns this remesh)
  uint retireCount;          // entries in RetireList (column slots that left)
  uint workPoolCapacity;     // work slots in FreeWorkSlots / Clusters (initWorkPool)
  uint oceanBlock;           // block ID of the ocean surface quads
  // Block ID per height layer, from world::kTerrainLayerBlocks (one more entry than TRANSITION_HEIGHTS), and the
  // block used below a column's top block (world::kTerrainLayerFillerBlocks).
  uint4 layerBlocks;
  uint4 layerFillerBlocks;
  uint4 layerDeepBlocks;  // world::kTerrainLayerDeepBlocks, deeper than FILLER_DEPTH below the top
  // Dispatches with more items than the Z group limit (65535) run in batches; the item is groupID.z + zBase.
  uint zBase;
  uint logBlock;    // world::kTreeLogBlock
  uint leavesBlock; // world::kTreeLeavesBlock
};

// Heights for one 8x8 meshColumns group plus a 1-column border (AO and neighbor faces reach +-1).
static const uint TILE_SIZE = 10u;
static const int NO_MATERIAL_HEIGHT = int(0x80000000u);
// Bit 9 of a block ID marks a non-opaque block (blocks::kNonOpaqueBit), drawn in the alpha cutout pass.
static const uint NON_OPAQUE_BLOCK_BIT = 0x200u;

// Prefix-sum total per wave (up to 8 waves of 8+ lanes) and packed direction pair.
groupshared uint scanValues[8u * 3u];
// Top face of each column in the group (index lane): height and merge key (see topFaceKey), for greedy merging.
groupshared int topFaceHeight[64];
groupshared uint topFaceKey[64];
// Length of the x-merged top face run starting at each lane (0: none starts there), for merging runs along z.
groupshared uint topRunLength[64];
groupshared uint waterRunLength[64]; // ocean surface: x-run length at a run's first lane, else 0
groupshared uint waterFlag[64];      // ocean surface on this lane's column
groupshared uint waterQuadCount;
groupshared uint waterOutputBase;
// Any lane emitted tree faces (sets the cluster's cutout flag when a tree block is non-opaque).
groupshared uint groupHasTreeFaces;
// Lowest face bottom / highest face top of the group's columns (chunk-local cells), for tight cluster bounds.
groupshared int tileFaceLow;
groupshared int tileFaceHigh;
groupshared uint groupOutputBase;
groupshared int heightTile[TILE_SIZE * TILE_SIZE];
// Trees over each column of the group (index lane), in this LOD's cells (see computeTreeColumn): log cells
// [treeLogLo, treeLogHi), leaf cells treeLeafBase + i for each bit i of treeLeafMask.
groupshared int treeLogLo[64];
// LOD 0-3: the tree slots of the regions the tile covers (stageTreeSlots; treeSlotHeight 0 = no tree). Ground: LOD 0
// the trunk column's height, LOD 1-2 its coarse column's height in cells.
groupshared int treeSlotTrunkX[64];
groupshared int treeSlotTrunkZ[64];
groupshared int treeSlotGround[64];
groupshared int treeSlotHeight[64];
groupshared uint treeSlotShape[64];
groupshared int treeLogHi[64];
groupshared int treeLeafBase[64];
groupshared uint treeLeafMask[64];

[[vk::push_constant]]
ConstantBuffer<DispatchParams> dispatchParams;

[[vk::binding(0, 0)]]
RWStructuredBuffer<int> HeightBuffer;
[[vk::binding(1, 0)]]
StructuredBuffer<int4> ChunkCoords;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint64_t> QuadBuffer;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint64_t> ChunkOffsets;
// [0]: quads appended since the last full rebuild (bump allocator), [1]: of those, still live.
[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> QuadCount;
// New work this remesh, appended by allocateWork: column slot, chunkY, work slot (its clusters are
// Clusters[slot * 16 ..]), unused. WorkPool[WORK_POOL_NEW_WORK] entries.
[[vk::binding(5, 0)]]
RWStructuredBuffer<int4> ChunkWork;
// One record per meshColumns group (8x8 columns of one chunk), indexed workSlot * 16 + tile. The group's quads
// are stored as six contiguous runs by direction: +Y, -X, +X, -Z, +Z, -Y. Read by taskMain in task.slang.
struct QuadCluster {
  uint base;
  uint counts01; // +Y | -X << 16
  uint counts23; // +X | -Z << 16
  uint counts45; // +Z | -Y << 16
  float4 boundsMin;
  float4 boundsMax;
};
[[vk::binding(6, 0)]]
RWStructuredBuffer<QuadCluster> Clusters;
// VkDrawIndexedIndirectCommand for vertMain, written by finalizeVertexDraw after each remesh.
[[vk::binding(7, 0)]]
RWStructuredBuffer<uint> VertexDrawArguments;
// Per column (parallel to ChunkCoords): LOD range of the tiles across each edge. Edge e (-X, +X, -Z, +Z) at bits
// e*8: min LOD | max LOD << 4. Written by buildVisibleClipmap (ClipmapResult::neighborLods).
[[vk::binding(8, 0)]]
StructuredBuffer<uint> ChunkNeighborLods;
// Column slots that need heights this remesh (generateHeights), and column slots that left the clipmap (retireWork).
[[vk::binding(9, 0)]]
StructuredBuffer<uint> ColumnList;
[[vk::binding(10, 0)]]
StructuredBuffer<uint> RetireList;
// Scratch for compactQuads: [i] is the compacted QuadBuffer, [MAX_QUAD_COUNT + i] the compacted ChunkOffsets.
[[vk::binding(11, 0)]]
RWStructuredBuffer<uint64_t> CompactScratch;
// Work slot allocation, all on the GPU (see WORK_POOL_*).
[[vk::binding(12, 0)]]
RWStructuredBuffer<int> WorkPool;
// Free work slots as a stack: [0, WorkPool[WORK_POOL_FREE_TOP]).
[[vk::binding(13, 0)]]
RWStructuredBuffer<uint> FreeWorkSlots;
// Per column slot: the work slot of each of its vertical chunks in order (INVALID_WORK_SLOT past the last), written by
// allocateWork and read back by retireWork when the column leaves.
[[vk::binding(14, 0)]]
RWStructuredBuffer<uint> ColumnWorkSlots;
// Per column slot and generateHeights group: lowest / highest stored height of the group's cells.
[[vk::binding(15, 0)]]
RWStructuredBuffer<int2> ColumnHeightRange;
// Per work slot: who holds it (column slot * MAX_COLUMN_CHUNKS + vertical index), INVALID_WORK_SLOT while free.
[[vk::binding(16, 0)]]
RWStructuredBuffer<uint> SlotOwner;
// collectRelocations output: holes at [0, capacity), movers at [capacity, 2 * capacity).
[[vk::binding(17, 0)]]
RWStructuredBuffer<uint> RelocationScratch;
// Per work slot, like Clusters: the translucent ocean surface quads of each tile (+Y run only), drawn in the blended
// pass. Their quads live in QuadBuffer like any other cluster's.
[[vk::binding(18, 0)]]
RWStructuredBuffer<QuadCluster> WaterClusters;

/**
 * HeightBuffer index of a padded cell.
 *
 * @param x Padded column-local x (0 = the -X border).
 * @param z Padded column-local z (0 = the -Z border).
 */
uint heightIndex(uint columnIndex, uint x, uint z) { return columnIndex * HEIGHTS_PER_CHUNK + z * HEIGHT_STRIDE + x; }

/**
 * Stored height at a chunk-local cell (-1..32 reaches the border), in this LOD's cells.
 */
int sampleHeight(uint columnIndex, int x, int z) {
  return HeightBuffer[heightIndex(columnIndex, uint(x + 1), uint(z + 1))];
}

/**
 * Height from the group's staged heightTile.
 *
 * @param tileOrigin Chunk-local (x, z) of heightTile[0].
 * @param x Chunk-local x.
 * @param z Chunk-local z.
 */
int tileHeight(int2 tileOrigin, int x, int z) {
  return heightTile[uint(z - tileOrigin.y) * TILE_SIZE + uint(x - tileOrigin.x)];
}

/**
 * Whether a cell is terrain (heightmap columns are solid all the way down).
 *
 * @param y Chunk-local cell y.
 */
bool isSolidAt(int2 tileOrigin, int x, int y, int z, int chunkY) {
  const int worldY = chunkY * int(CHUNK_SIZE) + y;
  return worldY < tileHeight(tileOrigin, x, z);
}

/**
 * Classic voxel vertex AO from the two edge neighbours and the corner.
 *
 * @return occlusion 0..3 (two sides alone give 3).
 */
uint vertexAo(bool sideA, bool sideB, bool corner) {
  return sideA && sideB ? 3u : uint(sideA) + uint(sideB) + uint(corner);
}

/**
 * Per-column layer-boundary dither: a smooth wiggle plus white noise.
 *
 * @param lodScale Voxel size, in blocks.
 * @return dither in [-1, 1].
 */
float transitionDither(int worldX, int worldZ, int lodScale) {
  const float wiggle = perlinFbm(float2(float(worldX), float(worldZ)), DITHER_WAVELENGTH, 2u, SEED_DITHER, lodScale).x;
  const float speckle = float(hashCell(int2(worldX, worldZ), SEED_DITHER_WHITE) >> 8u) * (2.0f / 16777216.0f) - 1.0f;
  return clamp(0.65f * wiggle + 0.35f * speckle, -1.0f, 1.0f);
}

/** Dithered world height of layer boundary i. */
int transitionHeight(uint i, float dither) { return TRANSITION_HEIGHTS[i] + int(round(dither * TRANSITION_DITHER[i])); }

/**
 * Depth of a cell's top block below the column's top block. Using the cell's top block (not its bottom) keeps
 * coarse LODs' grass / dirt / stone bands the same as full detail.
 *
 * @param cellsBelowTop Cells between this cell and the column's top cell.
 * @param scale Voxel size, in blocks.
 * @return depth in blocks.
 */
int cellDepthBelowTop(int cellsBelowTop, int scale) {
  return cellsBelowTop == 0 ? 0 : (cellsBelowTop - 1) * scale + 1;
}

/**
 * Terrain block for a world height: the layer's top, filler or deep block.
 *
 * @param depthBelowTop Blocks between this block and the column's top block (0 = the top block itself).
 * @return block ID.
 */
uint selectBlockType(int worldY, float dither, int depthBelowTop) {
  uint layerId = 0;
  [unroll]
  for (uint i = 0u; i < uint(heightCount); i++) {
    layerId += int(worldY >= transitionHeight(i, dither));
  }

  const uint layer = min(heightCount, layerId);
  if (depthBelowTop == 0) {
    return dispatchParams.layerBlocks[layer];
  }
  return depthBelowTop <= FILLER_DEPTH ? dispatchParams.layerFillerBlocks[layer] : dispatchParams.layerDeepBlocks[layer];
}

/**
 * Whether a block lies within one voxel of a dithered layer boundary (its material needs the exact surface height).
 *
 * @param scale Voxel size, in blocks.
 */
bool isNearLayerTransition(int blockWorldY, int scale, float dither) {
  bool nearTransition = false;
  [unroll]
  for (uint i = 0u; i < uint(heightCount); ++i) {
    const int transition = transitionHeight(i, dither);
    nearTransition |= blockWorldY >= transition - scale && blockWorldY < transition + scale;
  }
  return nearTransition;
}

/** calculateAoWith over terrain only. */
uint calculateAo(int2 tileOrigin, int x, int y, int z, int chunkY, uint normalIndex) {
  return calculateAoWith(tileOrigin, x, y, z, chunkY, normalIndex, false);
}

/** Whether a cell occludes for AO: terrain, plus opaque tree blocks when includeTrees (as on the CPU). */
bool aoOpaqueAt(int2 tileOrigin, int x, int y, int z, int chunkY, bool includeTrees) {
  return isSolidAt(tileOrigin, x, y, z, chunkY) ||
         (includeTrees && treeBlockOpaque(treeBlockAtColumn(tileOrigin, x, chunkY * int(CHUNK_SIZE) + y, z)));
}

/**
 * Packed AO of a face's four corners (layout at aoConstantAlongU).
 *
 * @param y Chunk-local cell y.
 * @param normalIndex Face direction 0..5 (+X, -X, +Y, -Y, +Z, -Z).
 * @return 8 bits, 2 per corner.
 */
uint calculateAoWith(int2 tileOrigin, int x, int y, int z, int chunkY, uint normalIndex, bool includeTrees) {
  const int3 uAxes[6] = { int3(0, 0, 1), int3(0, 0, 1), int3(1, 0, 0), int3(1, 0, 0), int3(1, 0, 0), int3(1, 0, 0) };
  const int3 vAxes[6] = { int3(0, 1, 0), int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1), int3(0, 1, 0), int3(0, 1, 0) };
  const int3 normalAxes[6] = { int3(1, 0, 0),  int3(-1, 0, 0), int3(0, 1, 0),
                               int3(0, -1, 0), int3(0, 0, 1),  int3(0, 0, -1) };
  const int3 outside = int3(x, y, z) + normalAxes[normalIndex];
  const int3 u = uAxes[normalIndex];
  const int3 v = vAxes[normalIndex];

  const bool sideU0 = aoOpaqueAt(tileOrigin, outside.x - u.x, outside.y - u.y, outside.z - u.z, chunkY, includeTrees);
  const bool sideU1 = aoOpaqueAt(tileOrigin, outside.x + u.x, outside.y + u.y, outside.z + u.z, chunkY, includeTrees);
  const bool sideV0 = aoOpaqueAt(tileOrigin, outside.x - v.x, outside.y - v.y, outside.z - v.z, chunkY, includeTrees);
  const bool sideV1 = aoOpaqueAt(tileOrigin, outside.x + v.x, outside.y + v.y, outside.z + v.z, chunkY, includeTrees);
  const bool corner0 =
      aoOpaqueAt(tileOrigin, outside.x - u.x - v.x, outside.y - u.y - v.y, outside.z - u.z - v.z, chunkY, includeTrees);
  const bool corner1 =
      aoOpaqueAt(tileOrigin, outside.x + u.x - v.x, outside.y + u.y - v.y, outside.z + u.z - v.z, chunkY, includeTrees);
  const bool corner2 =
      aoOpaqueAt(tileOrigin, outside.x + u.x + v.x, outside.y + u.y + v.y, outside.z + u.z + v.z, chunkY, includeTrees);
  const bool corner3 =
      aoOpaqueAt(tileOrigin, outside.x - u.x + v.x, outside.y - u.y + v.y, outside.z - u.z + v.z, chunkY, includeTrees);
  return vertexAo(sideU0, sideV0, corner0) | (vertexAo(sideU1, sideV0, corner1) << 2u) |
         (vertexAo(sideU1, sideV1, corner2) << 4u) | (vertexAo(sideU0, sideV1, corner3) << 6u);
}

/**
 * Surface height used to pick materials near layer transitions; per column, so evaluated at most once per thread.
 *
 * @param cachedHeight Per-thread cache, NO_MATERIAL_HEIGHT until filled.
 * @param chunkCoord (chunk x, chunk z, LOD, scale).
 * @return height in blocks (at scale 1, exactly the stored height).
 */
int columnMaterialHeight(inout int cachedHeight, int4 chunkCoord, uint x, uint z, int columnHeight) {
  if (cachedHeight == NO_MATERIAL_HEIGHT) {
    const int scale = chunkCoord.w;
    if (scale == 1) {
      cachedHeight = columnHeight;
    } else {
      const int worldCellX = chunkCoord.x * int(CHUNK_SIZE) + int(x) * scale;
      const int worldCellZ = chunkCoord.y * int(CHUNK_SIZE) + int(z) * scale;
      cachedHeight = int(floor(sampleTerrainHeight(worldCellX + scale / 2, worldCellZ + scale / 2, scale)));
    }
  }
  return cachedHeight;
}

/**
 * Block ID of the face at a cell of this column.
 *
 * @param worldY World cell y, in this LOD's cells.
 * @param columnHeight The column's stored height, in cells.
 * @return block ID.
 */
uint faceBlockType(uint x, uint z, int worldY, int4 chunkCoord, int columnHeight, inout int cachedMaterialHeight,
                   float dither) {
  const int scale = chunkCoord.w;
  const int blockWorldY = worldY * scale + scale - 1;
  const bool nearLayerTransition = isNearLayerTransition(blockWorldY, scale, dither);
  const int materialWorldY =
      nearLayerTransition ? columnMaterialHeight(cachedMaterialHeight, chunkCoord, x, z, columnHeight) : blockWorldY;
  return selectBlockType(materialWorldY, dither, cellDepthBelowTop(columnHeight - 1 - worldY, scale));
}

// AO corners: bits 0-1 (u0,v0), 2-3 (u1,v0), 4-5 (u1,v1), 6-7 (u0,v1). A merged quad interpolates AO over its whole
// extent, so faces only merge along an axis the AO doesn't vary on (and with identical AO).
/** Whether packed AO is the same at u0 and u1. */
bool aoConstantAlongU(uint ao) { return (ao & 3u) == ((ao >> 2u) & 3u) && ((ao >> 6u) & 3u) == ((ao >> 4u) & 3u); }
/** Whether packed AO is the same at v0 and v1. */
bool aoConstantAlongV(uint ao) { return (ao & 3u) == ((ao >> 6u) & 3u) && ((ao >> 2u) & 3u) == ((ao >> 4u) & 3u); }

/**
 * Pack one quad into QuadBuffer / ChunkOffsets (unpacked by decodePackedQuad in quad_decode.hlsl). Dropped past
 * MAX_QUAD_COUNT.
 *
 * @param x Chunk-local cell x (likewise y, z).
 * @param ao Packed corner AO (8 bits).
 * @param length Merged cells along the greedy direction, 1..32 (top faces: +x; side faces: +y).
 * @param width Merged cells across it, 1..8 (top faces: toward -z from the written cell; side faces: greedy axis).
 * @param chunkCoord (chunk x, chunk z, LOD, scale).
 */
void writeQuad(uint outputIndex, uint x, uint y, uint z, uint normalIndex, uint ao, uint blockType, uint length,
               uint width, int chunkY, int4 chunkCoord) {
  if (outputIndex >= MAX_QUAD_COUNT) {
    return;
  }
  uint packed = (x & 0x1fu) | ((y & 0x1fu) << 5u) | ((z & 0x1fu) << 10u) | ((normalIndex & 0x7u) << 15u);
  packed |= ao << 18u;
  packed |= ((length - 1u) & 0x1fu) << 26u;
  // Texture rotation is chosen per cell in fragMain, so merged faces don't share one. Bits 26-28: width - 1.
  const uint lod = uint(chunkCoord.z) & 0x7u;
  // Skylight 15 at every corner: heightmap terrain has no overhangs, so every face sees open sky. Once GPU chunks
  // hold voxels (3D noise, edited LODs), compute real levels with cpu_light.cpp's rules.
  const uint metadata = blockType | (0xffffu << 10u) | (((width - 1u) & 0x7u) << 26u) | (lod << 29u);
  QuadBuffer[outputIndex] = (uint64_t(metadata) << 32u) | uint64_t(packed);

  const int worldX = chunkCoord.x * int(CHUNK_SIZE);
  const int worldZ = chunkCoord.y * int(CHUNK_SIZE);
  const uint64_t offsetX = uint64_t(uint(worldX) & 0x00ffffffu);
  const uint64_t offsetZ = uint64_t(uint(worldZ) & 0x00ffffffu) << 24u;
  const uint64_t offsetY = uint64_t(uint(chunkY) & 0x1fffu) << 48u;
  const uint64_t packedLod = uint64_t(chunkCoord.z & 0x7) << 61u;
  ChunkOffsets[outputIndex] = offsetX | offsetZ | offsetY | packedLod;
}

// ---- Side faces ----
// Merged vertically into runs of equal block and AO, then across neighbouring columns along the greedy axis (z for x
// faces, x for z faces) when the neighbour has an identical run. Only each lane's first SIDE_RUN_SLOTS runs per
// direction are shared, so runs past that never merge across columns.
static const uint SIDE_RUN_SLOTS = 4u;
static const uint SIDE_RUN_VALID = 1u << 28u;
// [direction][lane][slot]: SIDE_RUN_VALID | first row (chunk-local) | (length - 1) << 5 | block << 10 | AO << 20.
groupshared uint sideRunDescs[4u * 64u * SIDE_RUN_SLOTS];

static const uint SIDE_MODE_RECORD = 0u; // store this lane's run descriptors
static const uint SIDE_MODE_COUNT = 1u;  // count the quads this lane emits
static const uint SIDE_MODE_WRITE = 2u;  // write them

/** sideRunDescs index of a lane's run slot. */
uint sideRunSlot(uint direction, uint lane, uint slot) { return (direction * 64u + lane) * SIDE_RUN_SLOTS + slot; }

/** Whether a lane recorded exactly this run descriptor in any slot. */
bool laneHasSideRun(uint direction, uint lane, uint desc) {
  bool found = false;
  [unroll]
  for (uint slot = 0u; slot < SIDE_RUN_SLOTS; ++slot) {
    found = found || sideRunDescs[sideRunSlot(direction, lane, slot)] == desc;
  }
  return found;
}

// Where this lane sits along the direction's greedy axis, and the lane step to the next column along it.
struct SideMergeAxis {
  uint position;
  uint laneStep;
};

/**
 * Handle one vertical run of this column. RECORD stores it; WRITE also writes its quad, widened over identical runs
 * of the following columns.
 *
 * @param mode SIDE_MODE_*.
 * @param direction Side direction 0..3 (-X, +X, -Z, +Z).
 * @param runIndex Run index within the column (only the first SIDE_RUN_SLOTS merge).
 * @param desc Run descriptor (sideRunDesc).
 * @return quads emitted (COUNT / WRITE): 0 when the previous column along the axis covers it, else 1; 0 for RECORD.
 */
uint sideRun(uint mode, uint direction, uint lane, SideMergeAxis axis, uint runIndex, uint desc, uint outputIndex, uint x,
             uint z, uint normalIndex, int chunkY, int4 chunkCoord) {
  if (mode == SIDE_MODE_RECORD) {
    if (runIndex < SIDE_RUN_SLOTS) {
      sideRunDescs[sideRunSlot(direction, lane, runIndex)] = desc;
    }
    return 0u;
  }
  const uint ao = (desc >> 20u) & 0xffu;
  const bool mergeable = runIndex < SIDE_RUN_SLOTS && aoConstantAlongU(ao);
  if (mergeable && axis.position > 0u && laneHasSideRun(direction, lane - axis.laneStep, desc)) {
    return 0u;
  }
  if (mode == SIDE_MODE_WRITE) {
    uint width = 1u;
    if (mergeable) {
      while (axis.position + width < 8u && laneHasSideRun(direction, lane + width * axis.laneStep, desc)) {
        ++width;
      }
    }
    writeQuad(outputIndex, x, desc & 0x1fu, z, normalIndex, ao, (desc >> 10u) & 0x3ffu, ((desc >> 5u) & 0x1fu) + 1u,
              width, chunkY, chunkCoord);
  }
  return 1u;
}

/**
 * Pack a side run: SIDE_RUN_VALID | first row | (length - 1) << 5 | block << 10 | AO << 20.
 *
 * @param runStart First world cell y of the run.
 * @param runEnd Exclusive last world cell y.
 * @param chunkBottom World cell y of the chunk's first row.
 */
uint sideRunDesc(int runStart, int runEnd, int chunkBottom, uint blockType, uint ao) {
  return SIDE_RUN_VALID | uint(runStart - chunkBottom) | (uint(runEnd - runStart - 1) << 5u) | (blockType << 10u) |
         (ao << 20u);
}

/**
 * Walk one column's side faces in one direction, evaluating blocks and AO. Every mode sees identical inputs, so the
 * runs match across modes.
 *
 * @param faceStart First world cell y with a face (the neighbour's height, clamped to the chunk).
 * @param faceEnd Exclusive last world cell y.
 * @return RECORD: run count; COUNT / WRITE: quads emitted (written from outputIndex on).
 */
uint sideFaceRuns(uint mode, uint direction, uint lane, SideMergeAxis axis, uint outputIndex, int2 tileOrigin, uint x,
                  uint z, uint normalIndex, int faceStart, int faceEnd, int chunkY, int4 chunkCoord, int columnHeight,
                  inout int cachedMaterialHeight, float dither) {
  const int chunkBottom = chunkY * int(CHUNK_SIZE);
  uint runs = 0u;
  uint quads = 0u;
  int runStart = faceStart;
  uint runBlock = 0u;
  uint runAo = 0u;
  bool runMergeable = false;
  for (int worldY = faceStart; worldY < faceEnd; ++worldY) {
    const uint blockType = faceBlockType(x, z, worldY, chunkCoord, columnHeight, cachedMaterialHeight, dither);
    const uint ao = calculateAo(tileOrigin, int(x), worldY - chunkBottom, int(z), chunkY, normalIndex);
    const bool mergeable = aoConstantAlongV(ao);
    if (worldY > faceStart && runMergeable && mergeable && blockType == runBlock && ao == runAo) {
      continue; // extends the current run
    }
    if (worldY > faceStart) {
      quads += sideRun(mode, direction, lane, axis, runs, sideRunDesc(runStart, worldY, chunkBottom, runBlock, runAo),
                       outputIndex + quads, x, z, normalIndex, chunkY, chunkCoord);
      ++runs;
    }
    runStart = worldY;
    runBlock = blockType;
    runAo = ao;
    runMergeable = mergeable;
  }
  if (faceEnd > faceStart) {
    quads += sideRun(mode, direction, lane, axis, runs, sideRunDesc(runStart, faceEnd, chunkBottom, runBlock, runAo),
                     outputIndex + quads, x, z, normalIndex, chunkY, chunkCoord);
    ++runs;
  }
  return mode == SIDE_MODE_RECORD ? runs : quads;
}

/**
 * sideFaceRuns in COUNT / WRITE mode from the RECORDed descriptors, skipping block and AO evaluation. Only valid when
 * every run fit in the slots.
 *
 * @return quads emitted.
 */
uint recordedSideRuns(uint mode, uint direction, uint lane, SideMergeAxis axis, uint runCount, uint outputIndex, uint x,
                      uint z, uint normalIndex, int chunkY, int4 chunkCoord) {
  uint quads = 0u;
  for (uint run = 0u; run < runCount; ++run) {
    quads += sideRun(mode, direction, lane, axis, run, sideRunDescs[sideRunSlot(direction, lane, run)],
                     outputIndex + quads, x, z, normalIndex, chunkY, chunkCoord);
  }
  return quads;
}

/** Integer division rounding toward -infinity; divisor must be > 0. */
int floorDivide(int value, int divisor) {
  return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

/**
 * Top of the column containing a world block in a tile of another scale, computed exactly as generateHeights does
 * for that tile's own columns (sample point, noise detail, rounding).
 *
 * @param scale The tile's voxel size, in blocks.
 * @param bias Subtracted from the height before rounding (CPU_SEAM_BIAS next to CPU chunks).
 * @return column top, in blocks.
 */
int columnTopBlocks(int worldX, int worldZ, int scale, float bias) {
  const int originX = floorDivide(worldX, scale) * scale;
  const int originZ = floorDivide(worldZ, scale) * scale;
  const int cells = int(floor((sampleTerrainHeight(originX, originZ, scale) - bias) / float(scale)));
  return max(cells, floorDivide(MIN_TERRAIN_HEIGHT, scale)) * scale;
}

/**
 * Height to store for a padded cell of a column.
 *
 * @param localX Padded column-local x (0 and HEIGHT_STRIDE - 1 are borders).
 * @param localZ Padded column-local z.
 * @return height in the tile's cells, clamped to the envelope floor.
 */
int columnCellHeight(uint columnIndex, uint localX, uint localZ) {
  const int4 chunkCoord = ChunkCoords[columnIndex];
  const int scale = chunkCoord.w;
  const int worldX = chunkCoord.x * int(CHUNK_SIZE) + (int(localX) - 1) * scale;
  const int worldZ = chunkCoord.y * int(CHUNK_SIZE) + (int(localZ) - 1) * scale;
  const int minimumHeight = floorDivide(MIN_TERRAIN_HEIGHT, scale);

  // Edge border cells (not corners, which only feed AO) take the lowest neighbour-LOD top along the edge, so seam
  // walls reach the neighbour's real surface; extra wall height hides inside the neighbour's terrain.
  const bool borderX = localX == 0u || localX == HEIGHT_STRIDE - 1u;
  const bool borderZ = localZ == 0u || localZ == HEIGHT_STRIDE - 1u;
  int cells;
  if (borderX != borderZ) {
    const uint edge = borderX ? (localX == 0u ? 0u : 1u) : (localZ == 0u ? 2u : 3u);
    const uint edgeBits = (ChunkNeighborLods[columnIndex] >> (edge * 8u)) & 0xffu;
    const uint lodRange = edgeBits & 0x77u;
    const float bias = (edgeBits & NEIGHBOR_CPU_FLAG) != 0u ? CPU_SEAM_BIAS : 0.0f;
    const int tileOrigin = (borderX ? chunkCoord.x : chunkCoord.y) * int(CHUNK_SIZE);
    // First block just outside the edge, and this border cell's start along the edge.
    const int across = (edge & 1u) == 0u ? tileOrigin - 1 : tileOrigin + int(CHUNK_SIZE) * scale;
    const int alongStart = borderX ? worldZ : worldX;
    int lowestTop = 0x7fffffff;
    for (uint lod = lodRange & 0xfu; lod <= (lodRange >> 4u); ++lod) {
      const int neighborScale = 1 << lod;
      // A finer neighbor has scale / neighborScale columns along this cell; a coarser one covers it with one.
      const int samples = neighborScale >= scale ? 1 : scale / neighborScale;
      for (int k = 0; k < samples; ++k) {
        const int along = alongStart + k * neighborScale;
        lowestTop = min(lowestTop, borderX ? columnTopBlocks(across, along, neighborScale, bias)
                                           : columnTopBlocks(along, across, neighborScale, bias));
      }
    }
    cells = floorDivide(lowestTop, scale); // at or below the neighbor's top, in this tile's cells
  } else {
    cells = int(floor(sampleTerrainHeight(worldX, worldZ, scale) / float(scale)));
  }
  return max(cells, minimumHeight);
}

// ---- Trees (placement shared with the CPU: vegetation_shared.hlsl) ----
// Each group column gets a TreeColumn (treeLog* / treeLeaf*) from the tile's regions' staged slots. Trees past the
// tile count as none for the neighbouring tile's faces (drawn twice at worst). LOD 0 matches the CPU's stampTrees
// exactly (logs win); LOD 1+ see "Coarse LODs" in vegetation_shared.hlsl.
// Tree cells over one column, in this LOD's cells: logs [logLo, logHi), leaves leafBase + bit i of leafMask.
struct TreeColumn {
  int logLo;
  int logHi;
  int leafBase;
  uint leafMask;
};

static const int TREE_LEAF_WINDOW = 16; // leaf cells a column can hold above treeLeafBase (overlapping trees)

/**
 * Union another tree's cells into a column. Leaves that would not fit the TREE_LEAF_WINDOW are dropped.
 *
 * @param logLo First log cell (logLo >= logHi: no logs).
 * @param logHi Exclusive last log cell.
 * @param leafBase Cell of leafMask's bit 0.
 */
void mergeTreeColumn(inout TreeColumn column, int logLo, int logHi, int leafBase, uint leafMask) {
  if (logLo < logHi) {
    // Two trunks in one column share its ground, so the union is the taller one.
    column.logLo = column.logLo < column.logHi ? min(column.logLo, logLo) : logLo;
    column.logHi = max(column.logHi, logHi);
  }
  if (leafMask == 0u) {
    return;
  }
  if (column.leafMask == 0u) {
    column.leafBase = leafBase;
    column.leafMask = leafMask;
    return;
  }
  const int base = min(column.leafBase, leafBase);
  if (column.leafBase - base >= TREE_LEAF_WINDOW || leafBase - base >= TREE_LEAF_WINDOW) {
    return;
  }
  const uint merged = (column.leafMask << uint(column.leafBase - base)) | (leafMask << uint(leafBase - base));
  if (firstbithigh(merged) < TREE_LEAF_WINDOW) {
    column.leafBase = base;
    column.leafMask = merged;
  }
}

/** Tree regions per tile side: 1 up to scale 4 (tile 8 * scale <= 32 blocks), 2 at scale 8. */
int treeTileRegionsPerSide(int scale) { return max(1, (8 * scale) >> TREE_REGION_SHIFT); }

/**
 * Fill the treeSlot* arrays for the tile, one lane per slot of its regions (16 or 64); call before the barrier
 * after height staging. Ground, band and slope come from a full-octave sample at the trunk, so every LOD and the
 * CPU agree (LOD 0 takes the ground from HeightBuffer, the same value).
 *
 * @param tileX Tile index along x within the chunk, 0..3 (likewise tileZ).
 * @param chunkCoord (chunk x, chunk z, LOD, scale).
 */
void stageTreeSlots(uint lane, uint tileX, uint tileZ, uint columnIndex, int4 chunkCoord) {
  const int scale = chunkCoord.w;
  const int regionsPerSide = treeTileRegionsPerSide(scale);
  if (scale > (1 << TREE_CELL_SHIFT) || lane >= uint(regionsPerSide * regionsPerSide * TREE_SLOTS)) {
    return;
  }
  const int originX = chunkCoord.x * int(CHUNK_SIZE);
  const int originZ = chunkCoord.y * int(CHUNK_SIZE);
  const int regionIndex = int(lane) / TREE_SLOTS;
  const int2 region = int2(((originX + int(tileX) * 8 * scale) >> TREE_REGION_SHIFT) + regionIndex % regionsPerSide,
                           ((originZ + int(tileZ) * 8 * scale) >> TREE_REGION_SHIFT) + regionIndex / regionsPerSide);
  TreeInstance tree = treeCandidate(region, lane % uint(TREE_SLOTS));
  if (tree.trunkHeight != 0) {
    float2 grad;
    const int exactGround = int(floor(sampleTerrainHeightGrad(tree.trunkX, tree.trunkZ, 1, grad)));
    tree.ground = scale == 1 ? sampleHeight(columnIndex, tree.trunkX - originX, tree.trunkZ - originZ) : exactGround;
    if (!treeGroundOk(tree.ground, treeSteepness(grad.x, grad.y), tree.shape)) {
      tree.trunkHeight = 0;
    }
  }
  treeSlotTrunkX[lane] = tree.trunkX;
  treeSlotTrunkZ[lane] = tree.trunkZ;
  treeSlotGround[lane] = tree.ground;
  treeSlotHeight[lane] = tree.trunkHeight;
  treeSlotShape[lane] = tree.shape;
}

/**
 * Tree cells over one column of the group at this LOD: exact blocks at LOD 0, covering canopies at LOD 1-2, one
 * leaf cell per shown tree at LOD 3+ (LOD 4+ re-derive candidates instead of using the staged slots).
 *
 * @param height The column's stored height, in cells.
 * @return the column's tree cells (empty past TREE_MAX_LOD).
 */
TreeColumn computeTreeColumn(int2 tileOrigin, uint x, uint z, int4 chunkCoord, int height) {
  TreeColumn column = { 0, 0, 0, 0u };
  const int scale = chunkCoord.w;
  if (scale > (1 << TREE_MAX_LOD)) {
    return column;
  }
  const int columnX = chunkCoord.x * int(CHUNK_SIZE) + int(x) * scale; // the column's low corner, world blocks
  const int columnZ = chunkCoord.y * int(CHUNK_SIZE) + int(z) * scale;
  if (scale > (1 << TREE_CELL_SHIFT)) {
    // LOD 4+: the column is filled when it holds an accepted trunk whose LOD roll passes. Candidates come from the
    // regions it covers (position, then roll, then density); slope and band from one full-octave sample at its
    // centre, taken lazily.
    const int regionLowX = columnX >> TREE_REGION_SHIFT;
    const int regionLowZ = columnZ >> TREE_REGION_SHIFT;
    const int regionHighX = (columnX + scale - 1) >> TREE_REGION_SHIFT;
    const int regionHighZ = (columnZ + scale - 1) >> TREE_REGION_SHIFT;
    int steepness = -1; // not sampled yet
    for (int regionZ = regionLowZ; regionZ <= regionHighZ; ++regionZ) {
      for (int regionX = regionLowX; regionX <= regionHighX; ++regionX) {
        const int2 region = int2(regionX, regionZ);
        for (uint slot = 0u; slot < uint(TREE_SLOTS); ++slot) {
          const TreeInstance tree = treeCandidateUnfiltered(region, slot);
          if (tree.trunkX < columnX || tree.trunkX >= columnX + scale || tree.trunkZ < columnZ ||
              tree.trunkZ >= columnZ + scale || !treeShowsAtScale(tree.trunkX, tree.trunkZ, scale)) {
            continue;
          }
          if (int(treeCandidateHash(region, slot) >> 24u) >= treeChanceAt(tree.trunkX, tree.trunkZ)) {
            continue;
          }
          if (steepness < 0) {
            float2 grad;
            const int ground =
                int(floor(sampleTerrainHeightGrad(columnX + scale / 2, columnZ + scale / 2, 1, grad)));
            if (ground < TREE_MIN_GROUND || ground > TREE_MAX_GROUND) {
              return column;
            }
            steepness = treeSteepness(grad.x, grad.y);
          }
          if (treeSlopeOk(steepness, treeSlopeRoll(tree.shape))) {
            column.leafBase = height;
            column.leafMask = 1u;
            return column;
          }
        }
      }
    }
    return column;
  }
  const int regionsPerSide = treeTileRegionsPerSide(scale);
  const uint slotCount = uint(regionsPerSide * regionsPerSide * TREE_SLOTS);
  if (scale == 1) {
    for (uint slot = 0u; slot < slotCount; ++slot) {
      const int trunkHeight = treeSlotHeight[slot];
      const int dx = columnX - treeSlotTrunkX[slot];
      const int dz = columnZ - treeSlotTrunkZ[slot];
      if (trunkHeight == 0 || abs(dx) > TREE_RADIUS || abs(dz) > TREE_RADIUS) {
        continue;
      }
      TreeInstance tree;
      tree.trunkX = treeSlotTrunkX[slot];
      tree.trunkZ = treeSlotTrunkZ[slot];
      tree.ground = treeSlotGround[slot];
      tree.trunkHeight = trunkHeight;
      tree.shape = treeSlotShape[slot];
      const uint style = treeStyle(tree.shape);
      const int leafBase = tree.ground + treeCanopyLow(style, trunkHeight);
      const uint leafCount = uint(treeCanopyHigh(style, trunkHeight) - treeCanopyLow(style, trunkHeight));
      uint leafMask = 0u;
      for (uint i = 0u; i < leafCount; ++i) {
        if (treeBlockAt(tree, columnX, leafBase + int(i), columnZ) == TREE_LEAVES) {
          leafMask |= 1u << i;
        }
      }
      const bool trunk = dx == 0 && dz == 0;
      mergeTreeColumn(column, trunk ? tree.ground : 0, trunk ? tree.ground + trunkHeight : 0, leafBase, leafMask);
    }
    return column;
  }
  if (scale == (1 << TREE_CELL_SHIFT)) {
    // LOD 3: one leaf cell on the column's top when it holds an accepted trunk whose LOD roll passes.
    for (uint slot = 0u; slot < slotCount; ++slot) {
      const int trunkX = treeSlotTrunkX[slot];
      const int trunkZ = treeSlotTrunkZ[slot];
      if (treeSlotHeight[slot] != 0 && trunkX >= columnX && trunkX < columnX + scale && trunkZ >= columnZ &&
          trunkZ < columnZ + scale && treeShowsAtScale(trunkX, trunkZ, scale)) {
        column.leafBase = height;
        column.leafMask = 1u;
        return column;
      }
    }
    return column;
  }
  // LOD 1-2: every tree whose canopy covers at least half of the column, or whose trunk stands in it, at its real
  // height; logs in the trunk's column from the terrain top up to its canopy.
  for (uint slot = 0u; slot < slotCount; ++slot) {
    const int trunkHeight = treeSlotHeight[slot];
    if (trunkHeight == 0) {
      continue;
    }
    const int trunkX = treeSlotTrunkX[slot];
    const int trunkZ = treeSlotTrunkZ[slot];
    const bool trunkHere = trunkX >= columnX && trunkX < columnX + scale && trunkZ >= columnZ && trunkZ < columnZ + scale;
    if (!trunkHere && !treeCanopyShows(treeCanopyOverlap(trunkX, trunkZ, columnX, columnZ, scale), scale)) {
      continue;
    }
    const int2 leaves = treeLeafCells(treeSlotGround[slot], trunkHeight, treeSlotShape[slot], scale, height);
    const bool logs = trunkHere && leaves.x > height;
    mergeTreeColumn(column, logs ? height : 0, logs ? leaves.x : 0, leaves.x, (1u << uint(leaves.y)) - 1u);
  }
  return column;
}

/**
 * Tree block of the group's column `lane` at a world cell; cells below the terrain top are terrain (it wins).
 *
 * @return TREE_NONE, TREE_LOG or TREE_LEAVES.
 */
uint treeCellBlock(uint lane, int y, int columnHeight) {
  if (y < columnHeight) {
    return TREE_NONE;
  }
  if (y >= treeLogLo[lane] && y < treeLogHi[lane]) {
    return TREE_LOG;
  }
  const int i = y - treeLeafBase[lane];
  return i >= 0 && i < TREE_LEAF_WINDOW && ((treeLeafMask[lane] >> uint(i)) & 1u) != 0u ? TREE_LEAVES : TREE_NONE;
}

/**
 * Whether a TREE_* block maps to an opaque block ID (non-opaque bit 9 clear, see blocks.txt).
 */
bool treeBlockOpaque(uint block) {
  const uint blockType = block == TREE_LOG ? dispatchParams.logBlock : dispatchParams.leavesBlock;
  return block != TREE_NONE && (blockType & 0x200u) == 0u;
}

/**
 * Tree block at a chunk-local column and world cell y; columns outside the group's tile hold none.
 *
 * @return TREE_NONE, TREE_LOG or TREE_LEAVES.
 */
uint treeBlockAtColumn(int2 tileOrigin, int x, int y, int z) {
  const int tileLocalX = x - tileOrigin.x - 1;
  const int tileLocalZ = z - tileOrigin.y - 1;
  if (tileLocalX < 0 || tileLocalX >= 8 || tileLocalZ < 0 || tileLocalZ >= 8) {
    return TREE_NONE;
  }
  return treeCellBlock(uint(tileLocalZ * 8 + tileLocalX), y, tileHeight(tileOrigin, x, z));
}

/**
 * World cells that may hold tree blocks over the group's column `lane`.
 *
 * @return [x, y) cell range, empty if none.
 */
int2 treeCellRange(uint lane) {
  const bool hasLog = treeLogLo[lane] < treeLogHi[lane];
  const bool hasLeaves = treeLeafMask[lane] != 0u;
  const int leafHi = treeLeafBase[lane] + int(firstbithigh(treeLeafMask[lane])) + 1;
  if (hasLog && hasLeaves) {
    return int2(min(treeLogLo[lane], treeLeafBase[lane]), max(treeLogHi[lane], leafHi));
  }
  return hasLog ? int2(treeLogLo[lane], treeLogHi[lane]) : (hasLeaves ? int2(treeLeafBase[lane], leafHi) : int2(0, 0));
}

/**
 * Count (and optionally write) this lane's tree faces within the chunk as 1x1 quads in the cluster's direction runs
 * (+Y, -X, +X, -Z, +Z, -Y). A face shows unless the neighbour is terrain, an opaque tree block or the same block
 * (like the CPU's binary mesher and emitNonOpaque).
 *
 * @param write Also write each quad at runBase[run] + counts[run].
 * @param height The column's stored height, in cells.
 * @param counts Incremented per emitted face, per run.
 */
void treeFaces(bool write, int2 tileOrigin, uint lane, uint x, uint z, int height, int chunkY, int4 chunkCoord,
               uint runBase[6], inout uint counts[6]) {
  const int chunkBottom = chunkY * int(CHUNK_SIZE);
  const int2 range = treeCellRange(lane);
  const int yBegin = max(max(range.x, height), chunkBottom);
  const int yEnd = min(range.y, chunkBottom + int(CHUNK_SIZE));
  const uint runNormals[6] = { 2u, 1u, 0u, 5u, 4u, 3u };
  const int3 normalAxes[6] = { int3(1, 0, 0),  int3(-1, 0, 0), int3(0, 1, 0),
                               int3(0, -1, 0), int3(0, 0, 1),  int3(0, 0, -1) };
  for (int y = yBegin; y < yEnd; ++y) {
    const uint block = treeCellBlock(lane, y, height);
    if (block == TREE_NONE) {
      continue;
    }
    const uint blockType = block == TREE_LOG ? dispatchParams.logBlock : dispatchParams.leavesBlock;
    [unroll]
    for (uint run = 0u; run < 6u; ++run) {
      const int3 n = normalAxes[runNormals[run]];
      const int neighborX = int(x) + n.x;
      const int neighborY = y + n.y;
      const int neighborZ = int(z) + n.z;
      if (neighborY < tileHeight(tileOrigin, neighborX, neighborZ)) {
        continue; // terrain
      }
      const uint neighbor = treeBlockAtColumn(tileOrigin, neighborX, neighborY, neighborZ);
      if (treeBlockOpaque(neighbor) || neighbor == block) {
        continue;
      }
      if (write) {
        const uint ao = calculateAoWith(tileOrigin, int(x), y - chunkBottom, int(z), chunkY, runNormals[run], true);
        writeQuad(runBase[run] + counts[run], x, uint(y - chunkBottom), z, runNormals[run], ao, blockType, 1u, 1u,
                  chunkY, chunkCoord);
      }
      ++counts[run];
    }
  }
}

// Lowest / highest stored height of this generateHeights group's cells, for allocateWork.
groupshared int heightGroupLow;
groupshared int heightGroupHigh;

/**
 * Compute and store the padded heights of each new column, and each group's height range for allocateWork.
 * Dispatched as (HEIGHT_GROUPS, 1, columns), batched through zBase.
 */
[shader("compute")]
[numthreads(8, 8, 1)]
void generateHeights(uint3 groupThreadID: SV_GroupThreadID, uint3 groupID: SV_GroupID) {
  const uint item = groupID.z + dispatchParams.zBase;
  if (item >= dispatchParams.horizontalChunkCount) {
    return;
  }
  const uint columnIndex = ColumnList[item];
  // Groups 0-15: 8x8 interior tiles (2D keeps noise branches coherent; 1D measured slower). Groups 16-18: the 132
  // border cells as lines (rows z = 0, 33, then columns x = 0, 33), so costly edge cells share warps and 1216 threads
  // cover 1156 cells instead of 1600.
  const uint group = groupID.x;
  const uint lane = groupThreadID.y * 8u + groupThreadID.x;
  uint localX = 0u;
  uint localZ = 0u;
  bool active = true; // the last border group has more lanes than cells; the rest still reach the barriers
  if (group < 16u) {
    localX = 1u + (group & 3u) * 8u + groupThreadID.x;
    localZ = 1u + (group >> 2u) * 8u + groupThreadID.y;
  } else {
    const uint border = (group - 16u) * 64u + lane;
    if (border >= 4u * HEIGHT_STRIDE - 4u) {
      active = false;
    } else if (border < 2u * HEIGHT_STRIDE) {
      localX = border % HEIGHT_STRIDE;
      localZ = border < HEIGHT_STRIDE ? 0u : HEIGHT_STRIDE - 1u;
    } else {
      const uint along = border - 2u * HEIGHT_STRIDE; // 0 .. 2 * 32 - 1
      localX = along < CHUNK_SIZE ? 0u : HEIGHT_STRIDE - 1u;
      localZ = 1u + along % CHUNK_SIZE;
    }
  }

  if (lane == 0u) {
    heightGroupLow = 0x7fffffff;
    heightGroupHigh = int(0x80000000u);
  }
  GroupMemoryBarrierWithGroupSync();
  if (active) {
    const int stored = columnCellHeight(columnIndex, localX, localZ);
    HeightBuffer[heightIndex(columnIndex, localX, localZ)] = stored;
    InterlockedMin(heightGroupLow, stored);
    InterlockedMax(heightGroupHigh, stored);
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane == 0u) {
    ColumnHeightRange[columnIndex * HEIGHT_GROUPS + group] = int2(heightGroupLow, heightGroupHigh);
  }
}

/**
 * Pick each new column's vertical chunks from its height range (plus one chunk of slack each end, and the ocean
 * surface chunk) and pop work slots for them; appends the work to ChunkWork for meshColumns. One thread per new
 * column, after generateHeights.
 */
[numthreads(64, 1, 1)]
[shader("compute")]
void allocateWork(uint3 dispatchThreadID: SV_DispatchThreadID) {
  if (dispatchThreadID.x >= dispatchParams.horizontalChunkCount) {
    return;
  }
  const uint columnIndex = ColumnList[dispatchThreadID.x];
  int low = 0x7fffffff;
  int high = int(0x80000000u);
  for (uint g = 0u; g < HEIGHT_GROUPS; ++g) {
    const int2 range = ColumnHeightRange[columnIndex * HEIGHT_GROUPS + g];
    low = min(low, range.x);
    high = max(high, range.y);
  }
  const int scale = ChunkCoords[columnIndex].w;
  // The same envelope buildVisibleClipmap used to queue every chunk of a column.
  const int envelopeLow = floorDivide(MIN_TERRAIN_HEIGHT, scale);
  const int envelopeHigh = -floorDivide(-MAX_TERRAIN_HEIGHT, scale);
  const int firstChunkY = max(floorDivide(low, int(CHUNK_SIZE)) - 1, floorDivide(envelopeLow, int(CHUNK_SIZE)));
  // Columns below sea level also need the chunk holding the ocean surface cell.
  const int seaCell = floorDivide(SEA_LEVEL, scale);
  const int surfaceChunkY = low < seaCell ? floorDivide(seaCell - 1, int(CHUNK_SIZE)) : int(0x80000000u);
  const int lastChunkY = min(max(floorDivide(high - 1, int(CHUNK_SIZE)) + 1, surfaceChunkY),
                             floorDivide(envelopeHigh - 1, int(CHUNK_SIZE)));
  int count = clamp(lastChunkY - firstChunkY + 1, 0, int(MAX_COLUMN_CHUNKS));

  // Pops and pushes (retireWork) never share a pass, so the stack holds exactly [0, top). Running dry (now or
  // earlier: the stack may be stale) drops the column and flags overflow; the CPU then rebuilds everything.
  int top = 0;
  if (WorkPool[WORK_POOL_OVERFLOW] != 0) {
    count = 0;
  } else if (count != 0) {
    InterlockedAdd(WorkPool[WORK_POOL_FREE_TOP], -count, top);
    if (top < count) {
      InterlockedAdd(WorkPool[WORK_POOL_OVERFLOW], 1);
      count = 0;
    }
  }
  int workBase = 0;
  InterlockedAdd(WorkPool[WORK_POOL_NEW_WORK], count, workBase);
  uint slotEnd = 0u;
  for (uint i = 0u; i < MAX_COLUMN_CHUNKS; ++i) {
    uint slot = INVALID_WORK_SLOT;
    if (int(i) < count) {
      slot = FreeWorkSlots[top - 1 - int(i)];
      ChunkWork[workBase + int(i)] = int4(int(columnIndex), firstChunkY + int(i), int(slot), 0);
      SlotOwner[slot] = columnIndex * MAX_COLUMN_CHUNKS + i;
      slotEnd = max(slotEnd, slot + 1u);
    }
    ColumnWorkSlots[columnIndex * MAX_COLUMN_CHUNKS + i] = slot;
  }
  if (slotEnd != 0u) {
    InterlockedMax(WorkPool[WORK_POOL_HIGH_WATER], int(slotEnd));
  }
}

/**
 * Write the indirect arguments for this remesh's meshColumns, compactQuads and the task-shader draw (the latter two
 * walk work slots below the high water). Item counts fold into y (up to 65535) and z.
 */
[numthreads(1, 1, 1)]
[shader("compute")]
void finalizeWork() {
  const uint newWork = uint(WorkPool[WORK_POOL_NEW_WORK]);
  const uint highWater = uint(WorkPool[WORK_POOL_HIGH_WATER]);
  WorkPool[WORK_POOL_MESH_ARGS + 0] = 16;
  WorkPool[WORK_POOL_MESH_ARGS + 1] = int(min(newWork, GROUP_LIMIT));
  WorkPool[WORK_POOL_MESH_ARGS + 2] = int((newWork + GROUP_LIMIT - 1u) / GROUP_LIMIT);
  WorkPool[WORK_POOL_COMPACT_ARGS + 0] = 16;
  WorkPool[WORK_POOL_COMPACT_ARGS + 1] = int(min(highWater, GROUP_LIMIT));
  WorkPool[WORK_POOL_COMPACT_ARGS + 2] = int((highWater + GROUP_LIMIT - 1u) / GROUP_LIMIT);
  const uint taskGroups = (highWater * 16u + 31u) / 32u; // CLUSTERS_PER_TASK in geometry_shared.hlsl
  WorkPool[WORK_POOL_TASK_ARGS + 0] = int(min(taskGroups, GROUP_LIMIT));
  WorkPool[WORK_POOL_TASK_ARGS + 1] = int((taskGroups + GROUP_LIMIT - 1u) / GROUP_LIMIT);
  WorkPool[WORK_POOL_TASK_ARGS + 2] = 1;
  WorkPool[WORK_POOL_HOLES] = 0; // for the next collectRelocations
  WorkPool[WORK_POOL_MOVERS] = 0;
}

/** Full rebuild: every work slot free, lowest on top so the high water stays low. Dispatched over the capacity. */
[numthreads(256, 1, 1)]
[shader("compute")]
void initWorkPool(uint3 dispatchThreadID: SV_DispatchThreadID) {
  const uint i = dispatchThreadID.x;
  const uint capacity = dispatchParams.workPoolCapacity;
  if (i < capacity) {
    FreeWorkSlots[i] = capacity - 1u - i;
    SlotOwner[i] = INVALID_WORK_SLOT;
  }
  if (i == 0u) {
    WorkPool[WORK_POOL_FREE_TOP] = int(capacity);
    WorkPool[WORK_POOL_HIGH_WATER] = 0;
    WorkPool[WORK_POOL_NEW_WORK] = 0;
    WorkPool[WORK_POOL_OVERFLOW] = 0;
    WorkPool[WORK_POOL_HOLES] = 0;
    WorkPool[WORK_POOL_MOVERS] = 0;
  }
}

// Slot defragmentation, run with quad compaction (after retireWork): moves live work slots below the live count so
// the high water drops back to it. Three passes over the capacity; all skip after an overflow (stale stack).

/** Pass 1: list free slots below the live count (holes) and live slots at or above it (movers); counts are equal. */
[numthreads(256, 1, 1)]
[shader("compute")]
void collectRelocations(uint3 dispatchThreadID: SV_DispatchThreadID) {
  const uint i = dispatchThreadID.x;
  const uint capacity = dispatchParams.workPoolCapacity;
  if (WorkPool[WORK_POOL_OVERFLOW] != 0 || i >= uint(WorkPool[WORK_POOL_HIGH_WATER])) {
    return;
  }
  const uint live = capacity - uint(WorkPool[WORK_POOL_FREE_TOP]);
  const bool free = SlotOwner[i] == INVALID_WORK_SLOT;
  int index = 0;
  if (i < live && free) {
    InterlockedAdd(WorkPool[WORK_POOL_HOLES], 1, index);
    RelocationScratch[index] = i;
  } else if (i >= live && !free) {
    InterlockedAdd(WorkPool[WORK_POOL_MOVERS], 1, index);
    RelocationScratch[capacity + uint(index)] = i;
  }
}

/** Pass 2: mover k takes hole k: its clusters (quads stay put), its owner's ColumnWorkSlots entry, its SlotOwner. */
[numthreads(256, 1, 1)]
[shader("compute")]
void relocateSlots(uint3 dispatchThreadID: SV_DispatchThreadID) {
  const uint k = dispatchThreadID.x;
  const uint capacity = dispatchParams.workPoolCapacity;
  if (WorkPool[WORK_POOL_OVERFLOW] != 0 || k >= uint(min(WorkPool[WORK_POOL_HOLES], WorkPool[WORK_POOL_MOVERS]))) {
    return;
  }
  const uint from = RelocationScratch[capacity + k];
  const uint to = RelocationScratch[k];
  const uint owner = SlotOwner[from];
  QuadCluster empty = (QuadCluster)0;
  for (uint c = 0u; c < 16u; ++c) {
    Clusters[to * 16u + c] = Clusters[from * 16u + c];
    Clusters[from * 16u + c] = empty;
    WaterClusters[to * 16u + c] = WaterClusters[from * 16u + c];
    WaterClusters[from * 16u + c] = empty;
  }
  ColumnWorkSlots[owner] = to;
  SlotOwner[to] = owner;
  SlotOwner[from] = INVALID_WORK_SLOT;
}

/** Pass 3: free slots are now exactly [live, capacity): restack them lowest on top and lower the high water. */
[numthreads(256, 1, 1)]
[shader("compute")]
void rebuildFreeStack(uint3 dispatchThreadID: SV_DispatchThreadID) {
  const uint i = dispatchThreadID.x;
  const uint capacity = dispatchParams.workPoolCapacity;
  if (WorkPool[WORK_POOL_OVERFLOW] != 0) {
    return;
  }
  const uint freeCount = uint(WorkPool[WORK_POOL_FREE_TOP]); // unchanged by relocation
  if (i < freeCount) {
    FreeWorkSlots[i] = capacity - 1u - i;
  }
  if (i == 0u) {
    WorkPool[WORK_POOL_HIGH_WATER] = int(capacity - freeCount);
  }
}

/**
 * Mesh one 8x8-column tile of one chunk (one thread per column): greedy-merged top and side faces, tree faces and
 * the ocean surface, written as one QuadCluster (six direction runs) plus one WaterClusters entry. Dispatched
 * indirectly (finalizeWork) as (16 tiles, items folded into y / z).
 */
[numthreads(8, 8, 1)]
[shader("compute")]
void meshColumns(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID) {
  const uint item = groupID.y + groupID.z * GROUP_LIMIT;
  if (item >= uint(WorkPool[WORK_POOL_NEW_WORK])) {
    return;
  }
  const uint tileX = groupID.x & 3u;
  const uint tileZ = groupID.x >> 2u;

  const int4 work = ChunkWork[item];
  const uint columnIndex = uint(work.x);
  const int chunkY = work.y;
  const uint workSlot = uint(work.z);
  const int chunkBottom = chunkY * int(CHUNK_SIZE);
  const int chunkTop = chunkBottom + int(CHUNK_SIZE);
  const uint x = tileX * 8u + groupThreadID.x;
  const uint z = tileZ * 8u + groupThreadID.y;
  const uint lane = groupThreadID.y * 8u + groupThreadID.x;
  // Any wave size from 8 to 64 lanes (NVIDIA 32, AMD 32/64, Intel 8/16); waves are assigned in thread order.
  const uint waveLanes = WaveGetLaneCount();
  const uint waveIndex = lane / waveLanes;
  const uint waveCount = 64u / waveLanes;

  // Stage heights (+1 border); AO and neighbour faces read only the tile. The early return is group-uniform, so
  // every thread reaches the barriers.
  const int2 tileOrigin = int2(int(tileX * 8u) - 1, int(tileZ * 8u) - 1);
  for (uint i = lane; i < TILE_SIZE * TILE_SIZE; i += 64u) {
    heightTile[i] = sampleHeight(columnIndex, tileOrigin.x + int(i % TILE_SIZE), tileOrigin.y + int(i / TILE_SIZE));
  }
  stageTreeSlots(lane, tileX, tileZ, columnIndex, ChunkCoords[columnIndex]);
  GroupMemoryBarrierWithGroupSync();

  const int height = tileHeight(tileOrigin, int(x), int(z));
  const int negXHeight = tileHeight(tileOrigin, int(x) - 1, int(z));
  const int posXHeight = tileHeight(tileOrigin, int(x) + 1, int(z));
  const int negZHeight = tileHeight(tileOrigin, int(x), int(z) - 1);
  const int posZHeight = tileHeight(tileOrigin, int(x), int(z) + 1);

  const int4 chunkCoord = ChunkCoords[columnIndex];
  const int scale = chunkCoord.w;
  const float dither = transitionDither(chunkCoord.x * int(CHUNK_SIZE) + int(x) * scale + scale / 2,
                                        chunkCoord.y * int(CHUNK_SIZE) + int(z) * scale + scale / 2, scale);
  int cachedMaterialHeight = NO_MATERIAL_HEIGHT;

  // This column's trees, for the whole group (read across lanes after the barrier below).
  const TreeColumn treeColumn = computeTreeColumn(tileOrigin, x, z, chunkCoord, height);
  treeLogLo[lane] = treeColumn.logLo;
  treeLogHi[lane] = treeColumn.logHi;
  treeLeafBase[lane] = treeColumn.leafBase;
  treeLeafMask[lane] = treeColumn.leafMask;

  // Top faces merge along x within the group's 8-column rows: a column whose left neighbour has an identical top
  // face (height, block, AO constant along x) is covered by that neighbour's quad.
  const bool hasTop = height > chunkBottom && height <= chunkTop;
  uint topBlock = 0u;
  uint topAo = 0u;
  if (hasTop) {
    topBlock = faceBlockType(x, z, height - 1, chunkCoord, height, cachedMaterialHeight, dither);
    topAo = calculateAo(tileOrigin, int(x), height - 1 - chunkBottom, int(z), chunkY, 2u);
  }
  // bit 19: has a top face, bit 18: may merge, bits 10-17: AO, bits 0-9: block.
  topFaceKey[lane] = hasTop ? (1u << 19u) | (aoConstantAlongU(topAo) ? 1u << 18u : 0u) | (topAo << 10u) | topBlock : 0u;
  topFaceHeight[lane] = height;
  if (lane == 0u) {
    tileFaceLow = int(CHUNK_SIZE);
    tileFaceHigh = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  // This column's faces span from the lowest neighbour height (side faces) or its top cell up to its height.
  const int columnLow = min(min(min(negXHeight, posXHeight), min(negZHeight, posZHeight)), height - 1);
  if (height > chunkBottom && columnLow < chunkTop) {
    InterlockedMin(tileFaceLow, clamp(columnLow, chunkBottom, chunkTop) - chunkBottom);
    InterlockedMax(tileFaceHigh, clamp(height, chunkBottom, chunkTop) - chunkBottom);
  }
  {
    const int2 treeRange = treeCellRange(lane);
    const int treeLow = max(max(treeRange.x, height), chunkBottom);
    const int treeHigh = min(treeRange.y, chunkTop);
    if (treeLow < treeHigh) {
      InterlockedMin(tileFaceLow, treeLow - chunkBottom);
      InterlockedMax(tileFaceHigh, treeHigh - chunkBottom);
    }
  }
  const uint mergeableTop = (1u << 19u) | (1u << 18u);
  const uint myTopKey = topFaceKey[lane];
  const bool mergesLeft = groupThreadID.x > 0u && (myTopKey & mergeableTop) == mergeableTop &&
                          topFaceKey[lane - 1u] == myTopKey && topFaceHeight[lane - 1u] == height;
  uint topLength = 0u;
  if (hasTop && !mergesLeft) {
    topLength = 1u;
    if ((myTopKey & mergeableTop) == mergeableTop) {
      while (groupThreadID.x + topLength < 8u && topFaceKey[lane + topLength] == myTopKey &&
             topFaceHeight[lane + topLength] == height) {
        ++topLength;
      }
    }
  }
  topRunLength[lane] = topLength;

  // Side faces, pass 1: record this column's runs for the neighbours (unused slots stay 0, which matches nothing).
  const uint faceNormalIndices[5] = { 2u, 1u, 0u, 5u, 4u };
  const int sideNeighbor[4] = { negXHeight, posXHeight, negZHeight, posZHeight };
  // x faces merge along z (lane step 8), z faces along x (lane step 1).
  const SideMergeAxis xFaceAxis = { groupThreadID.y, 8u };
  const SideMergeAxis zFaceAxis = { groupThreadID.x, 1u };
  uint sideRunCounts[4];
  [unroll]
  for (uint f = 0u; f < 4u; ++f) {
    [unroll]
    for (uint slot = 0u; slot < SIDE_RUN_SLOTS; ++slot) {
      sideRunDescs[sideRunSlot(f, lane, slot)] = 0u;
    }
    sideRunCounts[f] = sideFaceRuns(SIDE_MODE_RECORD, f, lane, f < 2u ? xFaceAxis : zFaceAxis, 0u, tileOrigin, x, z,
                                    faceNormalIndices[f + 1u], max(sideNeighbor[f], chunkBottom), min(height, chunkTop),
                                    chunkY, chunkCoord, height, cachedMaterialHeight, dither);
  }
  GroupMemoryBarrierWithGroupSync();

  // Top faces: stack identical x runs along z. A run is covered when the row before it (z - 1) starts the same run
  // (same start, length, height, block and AO); otherwise it extends over the identical runs after it. Merging along
  // z needs AO constant along z (the key holds the AO, so neighbours with equal keys agree).
  const bool topMergesZ = aoConstantAlongV(topAo);
  uint topWidth = 0u;
  if (topLength != 0u) {
    const bool covered = topMergesZ && groupThreadID.y > 0u && topRunLength[lane - 8u] == topLength &&
                         topFaceKey[lane - 8u] == myTopKey && topFaceHeight[lane - 8u] == height;
    if (!covered) {
      topWidth = 1u;
      while (topMergesZ && groupThreadID.y + topWidth < 8u && topRunLength[lane + 8u * topWidth] == topLength &&
             topFaceKey[lane + 8u * topWidth] == myTopKey && topFaceHeight[lane + 8u * topWidth] == height) {
        ++topWidth;
      }
    }
  }
  const uint topCount = topWidth != 0u ? 1u : 0u;

  // Side faces, pass 2: quads per direction after merging across columns. Columns whose runs all fit in the slots
  // reuse the recorded descriptors instead of re-evaluating blocks and AO.
  uint sideCounts[4];
  [unroll]
  for (uint f = 0u; f < 4u; ++f) {
    const SideMergeAxis axis = f < 2u ? xFaceAxis : zFaceAxis;
    sideCounts[f] =
        sideRunCounts[f] <= SIDE_RUN_SLOTS
            ? recordedSideRuns(SIDE_MODE_COUNT, f, lane, axis, sideRunCounts[f], 0u, x, z, faceNormalIndices[f + 1u],
                               chunkY, chunkCoord)
            : sideFaceRuns(SIDE_MODE_COUNT, f, lane, axis, 0u, tileOrigin, x, z, faceNormalIndices[f + 1u],
                           max(sideNeighbor[f], chunkBottom), min(height, chunkTop), chunkY, chunkCoord, height,
                           cachedMaterialHeight, dither);
  }
  const uint negXCount = sideCounts[0];
  const uint posXCount = sideCounts[1];
  const uint negZCount = sideCounts[2];
  const uint posZCount = sideCounts[3];
  // -Y faces: none for heightmap columns; the run is plumbed through the cluster, task and mesh shaders for future
  // overhangs / caves (write them at faceRunOffsets[5]).
  const uint negYCount = 0u;

  // Tree faces go after this lane's terrain faces in each run.
  const uint noRunBase[6] = { 0u, 0u, 0u, 0u, 0u, 0u };
  uint treeCounts[6] = { 0u, 0u, 0u, 0u, 0u, 0u };
  treeFaces(false, tileOrigin, lane, x, z, height, chunkY, chunkCoord, noRunBase, treeCounts);

  // One run per direction so the per-frame cull drops whole runs. Two directions per uint (16 bits each): a run holds
  // at most 64 columns * 32 quads, so fields never carry.
  const uint packedCounts[3] = { (topCount + treeCounts[0]) | ((negXCount + treeCounts[1]) << 16u),
                                 (posXCount + treeCounts[2]) | ((negZCount + treeCounts[3]) << 16u),
                                 (posZCount + treeCounts[4]) | ((negYCount + treeCounts[5]) << 16u) };
  if (lane == 0u) {
    groupHasTreeFaces = 0u; // the other lanes set it only after the barrier below
  }
  uint packedLanePrefix[3];
  [unroll]
  for (uint k = 0u; k < 3u; ++k) {
    packedLanePrefix[k] = WavePrefixSum(packedCounts[k]);
    if (WaveGetLaneIndex() == waveLanes - 1u) {
      scanValues[waveIndex * 3u + k] = packedLanePrefix[k] + packedCounts[k];
    }
  }
  GroupMemoryBarrierWithGroupSync();

  uint packedTotals[3] = { 0u, 0u, 0u };
  for (uint w = 0u; w < waveCount; ++w) {
    [unroll]
    for (uint k = 0u; k < 3u; ++k) {
      const uint waveTotal = scanValues[w * 3u + k];
      packedTotals[k] += waveTotal;
      packedLanePrefix[k] += w < waveIndex ? waveTotal : 0u;
    }
  }
  uint faceRunOffsets[6];
  uint groupQuadCount = 0u;
  [unroll]
  for (uint d = 0u; d < 6u; ++d) {
    const uint shift = (d & 1u) * 16u;
    faceRunOffsets[d] = groupQuadCount + ((packedLanePrefix[d >> 1u] >> shift) & 0xffffu);
    groupQuadCount += (packedTotals[d >> 1u] >> shift) & 0xffffu;
  }

  const uint laneTreeFaces =
      treeCounts[0] + treeCounts[1] + treeCounts[2] + treeCounts[3] + treeCounts[4] + treeCounts[5];
  if (WaveActiveAnyTrue(laneTreeFaces != 0u) && WaveIsFirstLane()) {
    InterlockedOr(groupHasTreeFaces, 1u);
  }
  if (lane == 63u) {
    InterlockedAdd(QuadCount[0], groupQuadCount, groupOutputBase);
    InterlockedAdd(QuadCount[1], groupQuadCount);
  }
  GroupMemoryBarrierWithGroupSync();
  const uint chunkBase = groupOutputBase;

  if (lane == 0u) {
    const float tileScale = float(chunkCoord.w);
    // Y covers only the cells the group's faces occupy (whole chunk if it has none; then it has no quads either).
    const int faceLow = tileFaceLow <= tileFaceHigh ? tileFaceLow : 0;
    const int faceHigh = tileFaceLow <= tileFaceHigh ? tileFaceHigh : int(CHUNK_SIZE);
    const float3 boundsMin =
        float3(float(chunkCoord.x * int(CHUNK_SIZE)) + float(tileX * 8u) * tileScale,
               float(chunkBottom + faceLow) * tileScale,
               float(chunkCoord.y * int(CHUNK_SIZE)) + float(tileZ * 8u) * tileScale);
    QuadCluster cluster;
    cluster.base = chunkBase;
    cluster.counts01 = packedTotals[0];
    cluster.counts23 = packedTotals[1];
    cluster.counts45 = packedTotals[2];
    cluster.boundsMin = float4(boundsMin, 0.0f);
    // w: CLUSTER_HAS_CUTOUT (geometry_shared.hlsl) when it may hold non-opaque quads; conservative.
    const uint terrainBlocks = dispatchParams.layerBlocks.x | dispatchParams.layerBlocks.y |
                               dispatchParams.layerBlocks.z | dispatchParams.layerBlocks.w |
                               dispatchParams.layerFillerBlocks.x | dispatchParams.layerFillerBlocks.y |
                               dispatchParams.layerFillerBlocks.z | dispatchParams.layerFillerBlocks.w |
                               dispatchParams.layerDeepBlocks.x | dispatchParams.layerDeepBlocks.y |
                               dispatchParams.layerDeepBlocks.z | dispatchParams.layerDeepBlocks.w;
    const bool treeCutout = groupHasTreeFaces != 0u &&
                            ((dispatchParams.logBlock | dispatchParams.leavesBlock) & NON_OPAQUE_BLOCK_BIT) != 0u;
    const bool terrainCutout = (terrainBlocks & NON_OPAQUE_BLOCK_BIT) != 0u && groupQuadCount != 0u;
    cluster.boundsMax = float4(boundsMin + float3(8.0f, float(faceHigh - faceLow), 8.0f) * tileScale,
                               treeCutout || terrainCutout ? 1.0f : 0.0f);
    Clusters[workSlot * 16u + groupID.x] = cluster;
  }

  if (topCount != 0u) {
    // A top face extends toward -z from the cell it is written at (faceV), so write it at the run's last row.
    writeQuad(chunkBase + faceRunOffsets[0], x, uint(height - 1 - chunkBottom), z + topWidth - 1u, 2u, topAo, topBlock,
              topLength, topWidth, chunkY, chunkCoord);
  }
  [unroll]
  for (uint f = 0u; f < 4u; ++f) {
    const SideMergeAxis axis = f < 2u ? xFaceAxis : zFaceAxis;
    if (sideRunCounts[f] <= SIDE_RUN_SLOTS) {
      recordedSideRuns(SIDE_MODE_WRITE, f, lane, axis, sideRunCounts[f], chunkBase + faceRunOffsets[f + 1u], x, z,
                       faceNormalIndices[f + 1u], chunkY, chunkCoord);
    } else {
      sideFaceRuns(SIDE_MODE_WRITE, f, lane, axis, chunkBase + faceRunOffsets[f + 1u], tileOrigin, x, z,
                   faceNormalIndices[f + 1u], max(sideNeighbor[f], chunkBottom), min(height, chunkTop), chunkY,
                   chunkCoord, height, cachedMaterialHeight, dither);
    }
  }

  {
    const uint terrainCounts[6] = { topCount, negXCount, posXCount, negZCount, posZCount, negYCount };
    uint treeRunBase[6];
    [unroll]
    for (uint d = 0u; d < 6u; ++d) {
      treeRunBase[d] = chunkBase + faceRunOffsets[d] + terrainCounts[d];
    }
    uint written[6] = { 0u, 0u, 0u, 0u, 0u, 0u };
    treeFaces(true, tileOrigin, lane, x, z, height, chunkY, chunkCoord, treeRunBase, written);
  }

  // Ocean surface: columns lower than sea level get a +Y quad on the cell below it, merged into x runs and then
  // stacked along z (no AO, one block), written to this tile's WaterClusters entry.
  const int seaCell = floorDivide(SEA_LEVEL, scale);
  const int surfaceCell = seaCell - 1;
  const bool hasWater = height < seaCell && surfaceCell >= chunkBottom && surfaceCell < chunkTop;
  waterFlag[lane] = hasWater ? 1u : 0u;
  if (lane == 0u) {
    waterQuadCount = 0u;
  }
  GroupMemoryBarrierWithGroupSync();
  uint waterLength = 0u;
  if (hasWater && (groupThreadID.x == 0u || waterFlag[lane - 1u] == 0u)) {
    waterLength = 1u;
    while (groupThreadID.x + waterLength < 8u && waterFlag[lane + waterLength] != 0u) {
      ++waterLength;
    }
  }
  waterRunLength[lane] = waterLength;
  GroupMemoryBarrierWithGroupSync();
  uint waterWidth = 0u;
  uint waterSlot = 0u;
  if (waterLength != 0u && !(groupThreadID.y > 0u && waterRunLength[lane - 8u] == waterLength)) {
    waterWidth = 1u;
    while (groupThreadID.y + waterWidth < 8u && waterRunLength[lane + 8u * waterWidth] == waterLength) {
      ++waterWidth;
    }
    InterlockedAdd(waterQuadCount, 1u, waterSlot);
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane == 0u) {
    uint base = 0u;
    if (waterQuadCount != 0u) {
      InterlockedAdd(QuadCount[0], waterQuadCount, base);
      InterlockedAdd(QuadCount[1], waterQuadCount);
    }
    waterOutputBase = base;
    const float tileScale = float(chunkCoord.w);
    const float3 boundsMin = float3(float(chunkCoord.x * int(CHUNK_SIZE)) + float(tileX * 8u) * tileScale,
                                    float(surfaceCell) * tileScale,
                                    float(chunkCoord.y * int(CHUNK_SIZE)) + float(tileZ * 8u) * tileScale);
    QuadCluster cluster;
    cluster.base = base;
    cluster.counts01 = waterQuadCount; // +Y run only
    cluster.counts23 = 0u;
    cluster.counts45 = 0u;
    cluster.boundsMin = float4(boundsMin, 0.0f);
    cluster.boundsMax = float4(boundsMin + float3(8.0f, 1.0f, 8.0f) * tileScale, 0.0f);
    WaterClusters[workSlot * 16u + groupID.x] = cluster;
  }
  GroupMemoryBarrierWithGroupSync();
  if (waterWidth != 0u) {
    // Like top faces: written at the run's last row (faceV points toward -z).
    writeQuad(waterOutputBase + waterSlot, x, uint(surfaceCell - chunkBottom), z + waterWidth - 1u, 2u, 0u,
              dispatchParams.oceanBlock, waterLength, waterWidth, chunkY, chunkCoord);
  }
}

/**
 * Retire the work slots of columns that left the clipmap: their quads stay in QuadBuffer (until compaction) but get
 * the invalid normal 7 so vertMain discards them, their clusters drop to zero quads, and the slots return to the
 * free stack. Dispatched as (16 clusters, MAX_COLUMN_CHUNKS, retireCount) before this remesh's allocateWork.
 */
[numthreads(64, 1, 1)]
[shader("compute")]
void retireWork(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID) {
  const uint item = groupID.z + dispatchParams.zBase;
  if (item >= dispatchParams.retireCount) {
    return;
  }
  const uint workSlot = ColumnWorkSlots[RetireList[item] * MAX_COLUMN_CHUNKS + groupID.y];
  if (workSlot == INVALID_WORK_SLOT) {
    return;
  }
  const uint clusterIndex = workSlot * 16u + groupID.x;
  const QuadCluster cluster = Clusters[clusterIndex];
  const uint total = (cluster.counts01 & 0xffffu) + (cluster.counts01 >> 16u) + (cluster.counts23 & 0xffffu) +
                     (cluster.counts23 >> 16u) + (cluster.counts45 & 0xffffu) + (cluster.counts45 >> 16u);
  const uint written = cluster.base >= MAX_QUAD_COUNT ? 0u : min(total, MAX_QUAD_COUNT - cluster.base);
  for (uint i = groupThreadID.x; i < written; i += 64u) {
    QuadBuffer[cluster.base + i] = QuadBuffer[cluster.base + i] | (7ull << 15u);
  }
  // The tile's ocean surface quads (+Y run only).
  const QuadCluster water = WaterClusters[clusterIndex];
  const uint waterTotal = water.counts01 & 0xffffu;
  const uint waterWritten = water.base >= MAX_QUAD_COUNT ? 0u : min(waterTotal, MAX_QUAD_COUNT - water.base);
  for (uint i = groupThreadID.x; i < waterWritten; i += 64u) {
    QuadBuffer[water.base + i] = QuadBuffer[water.base + i] | (7ull << 15u);
  }
  GroupMemoryBarrierWithGroupSync(); // every lane has read the cluster before it is cleared
  if (groupThreadID.x == 0u) {
    InterlockedAdd(QuadCount[1], 0u - total - waterTotal);
    Clusters[clusterIndex].counts01 = 0u;
    Clusters[clusterIndex].counts23 = 0u;
    Clusters[clusterIndex].counts45 = 0u;
    WaterClusters[clusterIndex].counts01 = 0u;
    // After an overflow the stack is stale until the rebuild; leave it alone.
    if (groupID.x == 0u && WorkPool[WORK_POOL_OVERFLOW] == 0) {
      SlotOwner[workSlot] = INVALID_WORK_SLOT;
      int top = 0;
      InterlockedAdd(WorkPool[WORK_POOL_FREE_TOP], 1, top);
      FreeWorkSlots[top] = workSlot;
    }
  }
}

groupshared uint compactBase;
groupshared uint compactWaterBase;
/**
 * Garbage-collect quads without remeshing: pack every live cluster's quads into CompactScratch and point the
 * clusters at their new runs. QuadCount[0] must be zeroed first and becomes the live quad count. Cluster order
 * doesn't matter; each stays contiguous. Dispatched indirectly as (16 clusters, work slots folded into y / z).
 */
[numthreads(64, 1, 1)]
[shader("compute")]
void compactQuads(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID) {
  const uint item = groupID.y + groupID.z * GROUP_LIMIT;
  if (item >= uint(WorkPool[WORK_POOL_HIGH_WATER])) {
    return;
  }
  const uint clusterIndex = item * 16u + groupID.x;
  const QuadCluster cluster = Clusters[clusterIndex];
  const uint total = (cluster.counts01 & 0xffffu) + (cluster.counts01 >> 16u) + (cluster.counts23 & 0xffffu) +
                     (cluster.counts23 >> 16u) + (cluster.counts45 & 0xffffu) + (cluster.counts45 >> 16u);
  const QuadCluster water = WaterClusters[clusterIndex];
  const uint waterTotal = water.counts01 & 0xffffu; // +Y run only
  if (groupThreadID.x == 0u) {
    InterlockedAdd(QuadCount[0], total, compactBase);
    InterlockedAdd(QuadCount[0], waterTotal, compactWaterBase);
  }
  GroupMemoryBarrierWithGroupSync(); // also: every lane has read the clusters before lane 0 rewrites their bases
  const uint base = compactBase;
  const uint waterBase = compactWaterBase;
  // The CPU falls back to a full rebuild once QuadBuffer overflowed, so every live run is complete and in range.
  for (uint i = groupThreadID.x; i < total; i += 64u) {
    CompactScratch[base + i] = QuadBuffer[cluster.base + i];
    CompactScratch[MAX_QUAD_COUNT + base + i] = ChunkOffsets[cluster.base + i];
  }
  for (uint i = groupThreadID.x; i < waterTotal; i += 64u) {
    CompactScratch[waterBase + i] = QuadBuffer[water.base + i];
    CompactScratch[MAX_QUAD_COUNT + waterBase + i] = ChunkOffsets[water.base + i];
  }
  if (groupThreadID.x == 0u) {
    Clusters[clusterIndex].base = base;
    WaterClusters[clusterIndex].base = waterBase;
  }
}

static const uint COPY_COMPACTED_GROUPS = 1024u;
/** Copy compactQuads' output back over QuadBuffer / ChunkOffsets (grid-stride over the packed size). */
[numthreads(256, 1, 1)]
[shader("compute")]
void copyCompactedQuads(uint3 dispatchThreadID: SV_DispatchThreadID) {
  const uint count = min(QuadCount[0], MAX_QUAD_COUNT);
  for (uint i = dispatchThreadID.x; i < count; i += COPY_COMPACTED_GROUPS * 256u) {
    QuadBuffer[i] = CompactScratch[i];
    ChunkOffsets[i] = CompactScratch[MAX_QUAD_COUNT + i];
  }
}

/** Write vertMain's indexed indirect draw for the current quad count, and reset the new-work counter. */
[numthreads(1, 1, 1)]
[shader("compute")]
void finalizeVertexDraw() {
  const uint quadCount = min(QuadCount[0], MAX_QUAD_COUNT);
  VertexDrawArguments[0] = VERTEX_QUADS_PER_INSTANCE * 6u; // indexCount
  VertexDrawArguments[1] = (quadCount + VERTEX_QUADS_PER_INSTANCE - 1u) / VERTEX_QUADS_PER_INSTANCE; // instanceCount
  VertexDrawArguments[2] = 0u; // firstIndex
  VertexDrawArguments[3] = 0u; // vertexOffset
  VertexDrawArguments[4] = 0u; // firstInstance
  WorkPool[WORK_POOL_NEW_WORK] = 0; // this remesh's meshColumns is done; the next allocateWork appends from 0
}
