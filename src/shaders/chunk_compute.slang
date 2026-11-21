#include "terrain_noise.hlsl"

static const uint CHUNK_SIZE = 32u;
static const uint HEIGHT_STRIDE = 34u;
static const uint HEIGHTS_PER_CHUNK = HEIGHT_STRIDE * HEIGHT_STRIDE;
static const uint MAX_QUAD_COUNT = 1u << 23u;
// Must match VERTEX_QUADS_PER_INSTANCE in slang.slang and renderer.cpp.
static const uint VERTEX_QUADS_PER_INSTANCE = 16384u;
// world::kMinTerrainHeight / kMaxTerrainHeight.
static const int MIN_TERRAIN_HEIGHT = -400;
static const int MAX_TERRAIN_HEIGHT = 1100;
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

// sand up to 2 blocks above sea level (beaches), grass to 320, stone to 600, snow above (Big Globe-scale
// mountains reach ~1000).
static const int TRANSITION_HEIGHTS[3] = { 2, 320, 600 };
static const int heightCount = TRANSITION_HEIGHTS.getCount();
// Each column shifts every boundary by up to this many blocks (transitionDither), so layers meet in ragged,
// speckled bands instead of flat contour lines. The beach boundary is not dithered: a clean line at sea level.
static const float TRANSITION_DITHER[3] = { 0.0f, 40.0f, 40.0f };
static const float DITHER_WAVELENGTH = 160.0f;
// Blocks of filler (dirt under grass) between a column's top block and its deep block (stone).
static const int FILLER_DEPTH = 3;
static const uint SEED_DITHER = 0x4cf5ad43u;
static const uint SEED_DITHER_WHITE = 0x2545f491u;
// Edges whose neighbor columns are meshed on the CPU (world::kNeighborCpuFlag in ChunkNeighborLods) floor their
// border heights from (height - CPU_SEAM_BIAS), as the CPU does across the same seam (world::kCpuSeamBias): the two
// noise implementations may differ by a few ulp, and this way that can only lengthen a hidden seam wall.
static const uint NEIGHBOR_CPU_FLAG = 0x8u;
static const float CPU_SEAM_BIAS = 0.02f;

struct DispatchParams {
  uint horizontalChunkCount; // entries in ColumnList (new columns this remesh)
  uint retireCount;          // entries in RetireList (column slots that left)
  uint workPoolCapacity;     // work slots in FreeWorkSlots / Clusters (initWorkPool)
  uint unused;
  // Block ID per height layer, from world::kTerrainLayerBlocks (one more entry than TRANSITION_HEIGHTS), and the
  // block used below a column's top block (world::kTerrainLayerFillerBlocks).
  uint4 layerBlocks;
  uint4 layerFillerBlocks;
  uint4 layerDeepBlocks;  // world::kTerrainLayerDeepBlocks, deeper than FILLER_DEPTH below the top
  // Dispatches with more items than the Z group limit (65535) run in batches; the item is groupID.z + zBase.
  uint zBase;
};

// Heights for one 8x8 meshColumns group plus a 1-column border (AO and neighbor faces reach +-1).
static const uint TILE_SIZE = 10u;
static const int NO_MATERIAL_HEIGHT = int(0x80000000u);

// Prefix-sum total per wave (up to 8 waves of 8+ lanes) and direction pair.
groupshared uint scanValues[8u * 3u];
// Top face of each column in the group (index lane): height and merge key (see topFaceKey), for greedy merging.
groupshared int topFaceHeight[64];
groupshared uint topFaceKey[64];
// Length of the x-merged top face run starting at each lane (0: none starts there), for merging runs along z.
groupshared uint topRunLength[64];
// Lowest face bottom / highest face top of the group's columns (chunk-local cells), for tight cluster bounds.
groupshared int tileFaceLow;
groupshared int tileFaceHigh;
groupshared uint groupOutputBase;
groupshared int heightTile[TILE_SIZE * TILE_SIZE];

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

uint heightIndex(uint columnIndex, uint x, uint z) { return columnIndex * HEIGHTS_PER_CHUNK + z * HEIGHT_STRIDE + x; }

int sampleHeight(uint columnIndex, int x, int z) {
  return HeightBuffer[heightIndex(columnIndex, uint(x + 1), uint(z + 1))];
}

// x/z are chunk-local; tileOrigin is the chunk-local coordinate of heightTile[0].
int tileHeight(int2 tileOrigin, int x, int z) {
  return heightTile[uint(z - tileOrigin.y) * TILE_SIZE + uint(x - tileOrigin.x)];
}

bool isSolidAt(int2 tileOrigin, int x, int y, int z, int chunkY) {
  const int worldY = chunkY * int(CHUNK_SIZE) + y;
  return worldY < tileHeight(tileOrigin, x, z);
}

uint vertexAo(bool sideA, bool sideB, bool corner) {
  return sideA && sideB ? 3u : uint(sideA) + uint(sideB) + uint(corner);
}

// Per-column dither in [-1, 1]: a smooth wiggle plus per-column white noise.
float transitionDither(int worldX, int worldZ, int lodScale) {
  const float wiggle = perlinFbm(float2(float(worldX), float(worldZ)), DITHER_WAVELENGTH, 2u, SEED_DITHER, lodScale).x;
  const float speckle = float(hashCell(int2(worldX, worldZ), SEED_DITHER_WHITE) >> 8u) * (2.0f / 16777216.0f) - 1.0f;
  return clamp(0.65f * wiggle + 0.35f * speckle, -1.0f, 1.0f);
}

int transitionHeight(uint i, float dither) { return TRANSITION_HEIGHTS[i] + int(round(dither * TRANSITION_DITHER[i])); }

// Depth in blocks of a cell's top block below the column's top block. A coarse-LOD cell spans `scale` blocks; using
// its top block (not its bottom) gives the same grass / dirt / stone bands as full detail.
int cellDepthBelowTop(int cellsBelowTop, int scale) {
  return cellsBelowTop == 0 ? 0 : (cellsBelowTop - 1) * scale + 1;
}

// depthBelowTop: blocks between this block and the column's top block (0 = the top block itself).
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

bool isNearLayerTransition(int blockWorldY, int scale, float dither) {
  bool nearTransition = false;
  [unroll]
  for (uint i = 0u; i < uint(heightCount); ++i) {
    const int transition = transitionHeight(i, dither);
    nearTransition |= blockWorldY >= transition - scale && blockWorldY < transition + scale;
  }
  return nearTransition;
}

uint calculateAo(int2 tileOrigin, int x, int y, int z, int chunkY, uint normalIndex) {
  const int3 uAxes[6] = { int3(0, 0, 1), int3(0, 0, 1), int3(1, 0, 0), int3(1, 0, 0), int3(1, 0, 0), int3(1, 0, 0) };
  const int3 vAxes[6] = { int3(0, 1, 0), int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1), int3(0, 1, 0), int3(0, 1, 0) };
  const int3 normalAxes[6] = { int3(1, 0, 0),  int3(-1, 0, 0), int3(0, 1, 0),
                               int3(0, -1, 0), int3(0, 0, 1),  int3(0, 0, -1) };
  const int3 outside = int3(x, y, z) + normalAxes[normalIndex];
  const int3 u = uAxes[normalIndex];
  const int3 v = vAxes[normalIndex];

  const bool sideU0 = isSolidAt(tileOrigin, outside.x - u.x, outside.y - u.y, outside.z - u.z, chunkY);
  const bool sideU1 = isSolidAt(tileOrigin, outside.x + u.x, outside.y + u.y, outside.z + u.z, chunkY);
  const bool sideV0 = isSolidAt(tileOrigin, outside.x - v.x, outside.y - v.y, outside.z - v.z, chunkY);
  const bool sideV1 = isSolidAt(tileOrigin, outside.x + v.x, outside.y + v.y, outside.z + v.z, chunkY);
  const bool corner0 =
      isSolidAt(tileOrigin, outside.x - u.x - v.x, outside.y - u.y - v.y, outside.z - u.z - v.z, chunkY);
  const bool corner1 =
      isSolidAt(tileOrigin, outside.x + u.x - v.x, outside.y + u.y - v.y, outside.z + u.z - v.z, chunkY);
  const bool corner2 =
      isSolidAt(tileOrigin, outside.x + u.x + v.x, outside.y + u.y + v.y, outside.z + u.z + v.z, chunkY);
  const bool corner3 =
      isSolidAt(tileOrigin, outside.x - u.x + v.x, outside.y - u.y + v.y, outside.z - u.z + v.z, chunkY);
  return vertexAo(sideU0, sideV0, corner0) | (vertexAo(sideU1, sideV0, corner1) << 2u) |
         (vertexAo(sideU1, sideV1, corner2) << 4u) | (vertexAo(sideU0, sideV1, corner3) << 6u);
}

// Surface height used to pick materials near layer transitions. It depends only on the column, so it is
// evaluated at most once per thread. At scale 1 the sample point matches the height buffer exactly.
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
bool aoConstantAlongU(uint ao) { return (ao & 3u) == ((ao >> 2u) & 3u) && ((ao >> 6u) & 3u) == ((ao >> 4u) & 3u); }
bool aoConstantAlongV(uint ao) { return (ao & 3u) == ((ao >> 6u) & 3u) && ((ao >> 2u) & 3u) == ((ao >> 4u) & 3u); }

// length: merged cells along the greedy direction, 1..32 (top faces: +x, side faces: +y). width: merged cells across
// it, 1..8 (top faces: toward -z from the written cell, see faceV; side faces: along the greedy axis). See
// decodePackedQuad.
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
  const uint metadata = blockType | (15u << 10u) | (0xfffu << 14u) | (((width - 1u) & 0x7u) << 26u) | (lod << 29u);
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
// Each column's side faces in one direction are merged vertically into runs of equal block type and AO. Runs are
// then merged across neighbouring columns of the group along the face's greedy axis (z for x faces, x for z faces)
// when a neighbour has an identical run (same rows, block and AO, AO constant along that axis). Every lane records
// its first SIDE_RUN_SLOTS runs per direction in shared memory so its neighbours can look them up; runs past that
// (very tall, varied walls) are never merged across columns.
static const uint SIDE_RUN_SLOTS = 4u;
static const uint SIDE_RUN_VALID = 1u << 28u;
// [direction][lane][slot]: SIDE_RUN_VALID | first row (chunk-local) | (length - 1) << 5 | block << 10 | AO << 20.
groupshared uint sideRunDescs[4u * 64u * SIDE_RUN_SLOTS];

static const uint SIDE_MODE_RECORD = 0u; // store this lane's run descriptors
static const uint SIDE_MODE_COUNT = 1u;  // count the quads this lane emits
static const uint SIDE_MODE_WRITE = 2u;  // write them

uint sideRunSlot(uint direction, uint lane, uint slot) { return (direction * 64u + lane) * SIDE_RUN_SLOTS + slot; }

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

// One vertical run (desc) of this column. RECORD stores it; COUNT / WRITE return the quads it emits: 0 when the
// previous column along the greedy axis has the same run (that column's quad covers it), else 1.
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

uint sideRunDesc(int runStart, int runEnd, int chunkBottom, uint blockType, uint ao) {
  return SIDE_RUN_VALID | uint(runStart - chunkBottom) | (uint(runEnd - runStart - 1) << 5u) | (blockType << 10u) |
         (ao << 20u);
}

// Walks one column's side faces in one direction. RECORD returns the number of runs; COUNT / WRITE the quads emitted
// (written from outputIndex on). Every mode sees identical inputs, so the runs match.
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

// Same as sideFaceRuns in COUNT / WRITE mode, from the descriptors RECORD stored: no block or AO evaluation. Only
// valid when every run fit in the slots.
uint recordedSideRuns(uint mode, uint direction, uint lane, SideMergeAxis axis, uint runCount, uint outputIndex, uint x,
                      uint z, uint normalIndex, int chunkY, int4 chunkCoord) {
  uint quads = 0u;
  for (uint run = 0u; run < runCount; ++run) {
    quads += sideRun(mode, direction, lane, axis, run, sideRunDescs[sideRunSlot(direction, lane, run)],
                     outputIndex + quads, x, z, normalIndex, chunkY, chunkCoord);
  }
  return quads;
}

int floorDivide(int value, int divisor) { // divisor > 0
  return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

// Top of the column containing (worldX, worldZ) in a tile of the given scale, in blocks: the same sample point
// (cell origin), noise detail and rounding generateHeights uses for that tile's own columns.
int columnTopBlocks(int worldX, int worldZ, int scale, float bias) {
  const int originX = floorDivide(worldX, scale) * scale;
  const int originZ = floorDivide(worldZ, scale) * scale;
  const int cells = int(floor((sampleTerrainHeight(originX, originZ, scale) - bias) / float(scale)));
  return max(cells, floorDivide(MIN_TERRAIN_HEIGHT, scale)) * scale;
}

// Stored height (in the tile's cells, clamped to the envelope floor) of padded cell (localX, localZ) of a column.
int columnCellHeight(uint columnIndex, uint localX, uint localZ) {
  const int4 chunkCoord = ChunkCoords[columnIndex];
  const int scale = chunkCoord.w;
  const int worldX = chunkCoord.x * int(CHUNK_SIZE) + (int(localX) - 1) * scale;
  const int worldZ = chunkCoord.y * int(CHUNK_SIZE) + (int(localZ) - 1) * scale;
  const int minimumHeight = floorDivide(MIN_TERRAIN_HEIGHT, scale);

  // Edge border cells (not corners, which only feed AO) stand for the neighbor tile's columns. Across an LOD seam
  // that tile has another resolution, so sample it there: walls built against this height then reach down to the
  // neighbor's real surface, and the seam has no gap. With several LODs along the edge the lowest one wins; any
  // extra wall height ends up hidden inside the neighbor's terrain.
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

// Lowest / highest stored height of this generateHeights group's cells, for allocateWork.
groupshared int heightGroupLow;
groupshared int heightGroupHigh;

[shader("compute")]
[numthreads(8, 8, 1)]
void generateHeights(uint3 groupThreadID: SV_GroupThreadID, uint3 groupID: SV_GroupID) {
  const uint item = groupID.z + dispatchParams.zBase;
  if (item >= dispatchParams.horizontalChunkCount) {
    return;
  }
  const uint columnIndex = ColumnList[item];
  // Dispatched as (HEIGHT_GROUPS, 1, columns). Groups 0-15 cover the 32x32 interior as 8x8 tiles (2D tiles keep the
  // noise branches coherent; a plain 1D mapping measured slower). Groups 16-18 walk the 132 border cells as lines:
  // rows z = 0 and z = 33, then columns x = 0 and x = 33. 1216 threads for 1156 cells instead of 1600 for a 5x5 grid
  // of 8x8 tiles, and the costly edge cells (they sample neighbour LODs) share warps.
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

// Picks each new column's vertical chunks from its height range and gives them work slots from the free stack.
// One thread per new column, after generateHeights. A column's faces span from its lowest neighbour height (side
// faces start there) to its top cell (height - 1), all inside [low, high - 1] of the padded heights; one more chunk
// of slack is kept on each end. Appends the work to ChunkWork for meshColumns (indirect, see finalizeWork).
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
  const int lastChunkY = min(floorDivide(high - 1, int(CHUNK_SIZE)) + 1, floorDivide(envelopeHigh - 1, int(CHUNK_SIZE)));
  int count = clamp(lastChunkY - firstChunkY + 1, 0, int(MAX_COLUMN_CHUNKS));

  // Pop count slots. Pops and pushes (retireWork) never share a pass, so the stack holds exactly [0, top).
  // Running dry (or having run dry before: the stack may hold stale entries then) drops the column's chunks and
  // flags it; the CPU reads the flag back and rebuilds everything.
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

// Indirect arguments for this remesh's meshColumns, and for compactQuads and the task-shader draw (both walk the work
// slots below the high water). Dispatches fold the item count into y (up to 65535) and z.
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

// Full rebuild: every work slot free, lowest on top so the high water stays low. Dispatched over the capacity.
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

// Slot defragmentation, run with quad compaction (after retireWork): moves every live work slot below the live count
// so the high water, which compactQuads and the task-shader draw walk, drops back to it. Three passes, each
// dispatched over the capacity (256 threads per group); all skip after an overflow (the stack is stale then).
//
// 1. Free slots below the live count (holes) and live slots at or above it (movers). There are as many of each.
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

// 2. Mover k takes hole k: its clusters (their quads stay put), its owner's ColumnWorkSlots entry, its SlotOwner.
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
  }
  ColumnWorkSlots[owner] = to;
  SlotOwner[to] = owner;
  SlotOwner[from] = INVALID_WORK_SLOT;
}

// 3. The free slots are now exactly [live, capacity): restack them lowest on top and lower the high water.
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

[numthreads(8, 8, 1)]
[shader("compute")]
void meshColumns(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID) {
  // Dispatched indirectly (finalizeWork) as (16 tiles, items folded into y / z).
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

  // Stage this group's heights (+1 border) in shared memory; AO and neighbor faces read only from the tile.
  // The early return above is uniform per group, so these barriers are reached by every thread.
  const int2 tileOrigin = int2(int(tileX * 8u) - 1, int(tileZ * 8u) - 1);
  for (uint i = lane; i < TILE_SIZE * TILE_SIZE; i += 64u) {
    heightTile[i] = sampleHeight(columnIndex, tileOrigin.x + int(i % TILE_SIZE), tileOrigin.y + int(i / TILE_SIZE));
  }
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
  // -Y (bottom) faces: a cell whose cell below is air. Heightmap columns are solid all the way down (isSolidAt), so
  // they have none; the run is plumbed through the cluster, task and mesh shaders for terrain that does
  // (overhangs, caves). Count them here and write them at faceRunOffsets[5] below.
  const uint negYCount = 0u;

  // The group's quads are laid out as one run per direction so the per-frame cull can drop a whole run at once.
  // Two directions share a uint (16 bits each): a run holds at most 64 columns * 32 quads, so fields never carry.
  const uint packedCounts[3] = { topCount | (negXCount << 16u), posXCount | (negZCount << 16u),
                                 posZCount | (negYCount << 16u) };
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
    cluster.boundsMax = float4(boundsMin + float3(8.0f, float(faceHigh - faceLow), 8.0f) * tileScale, 0.0f);
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
}

// Retires the work slots of columns that left the clipmap: their quads stay in QuadBuffer (reclaimed by compaction or
// the next full rebuild) but get an invalid normal (7) so vertMain discards them, the clusters drop to zero quads so
// taskMain skips them, and the slots go back on the free stack. Dispatched as (16 clusters, MAX_COLUMN_CHUNKS,
// retireCount) before this remesh's allocateWork, which may reuse the slots.
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
  GroupMemoryBarrierWithGroupSync(); // every lane has read the cluster before it is cleared
  if (groupThreadID.x == 0u) {
    InterlockedAdd(QuadCount[1], 0u - total);
    Clusters[clusterIndex].counts01 = 0u;
    Clusters[clusterIndex].counts23 = 0u;
    Clusters[clusterIndex].counts45 = 0u;
    // After an overflow the stack is stale until the rebuild; leave it alone.
    if (groupID.x == 0u && WorkPool[WORK_POOL_OVERFLOW] == 0) {
      SlotOwner[workSlot] = INVALID_WORK_SLOT;
      int top = 0;
      InterlockedAdd(WorkPool[WORK_POOL_FREE_TOP], 1, top);
      FreeWorkSlots[top] = workSlot;
    }
  }
}

// Garbage collection without remeshing: packs the quads of every live cluster into CompactScratch and points the
// clusters at their new runs. QuadCount[0] must be zeroed first; it becomes the packed size (the live quad count).
// Dispatched indirectly (finalizeWork) as (16 clusters, work slots below the high water folded into y / z). Cluster order in the output doesn't matter: each cluster keeps its
// quads contiguous (six runs by direction), which is all taskMain and retireWork rely on.
groupshared uint compactBase;
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
  if (groupThreadID.x == 0u) {
    InterlockedAdd(QuadCount[0], total, compactBase);
  }
  GroupMemoryBarrierWithGroupSync(); // also: every lane has read the cluster before lane 0 rewrites its base
  const uint base = compactBase;
  // The CPU falls back to a full rebuild once QuadBuffer overflowed, so every live run is complete and in range.
  for (uint i = groupThreadID.x; i < total; i += 64u) {
    CompactScratch[base + i] = QuadBuffer[cluster.base + i];
    CompactScratch[MAX_QUAD_COUNT + base + i] = ChunkOffsets[cluster.base + i];
  }
  if (groupThreadID.x == 0u) {
    Clusters[clusterIndex].base = base;
  }
}

// Copies compactQuads' output back over QuadBuffer / ChunkOffsets. Grid-stride over the packed size.
static const uint COPY_COMPACTED_GROUPS = 1024u;
[numthreads(256, 1, 1)]
[shader("compute")]
void copyCompactedQuads(uint3 dispatchThreadID: SV_DispatchThreadID) {
  const uint count = min(QuadCount[0], MAX_QUAD_COUNT);
  for (uint i = dispatchThreadID.x; i < count; i += COPY_COMPACTED_GROUPS * 256u) {
    QuadBuffer[i] = CompactScratch[i];
    ChunkOffsets[i] = CompactScratch[MAX_QUAD_COUNT + i];
  }
}

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
