#include "terrain_noise.hlsl"

static const uint CHUNK_SIZE = 32u;
static const uint HEIGHT_STRIDE = 34u;
static const uint HEIGHTS_PER_CHUNK = HEIGHT_STRIDE * HEIGHT_STRIDE;
static const uint MAX_QUAD_COUNT = 1u << 22u;
static const int MIN_TERRAIN_HEIGHT = -400;
static const int DITHER_RADIUS = 3;

const static int TRANSISTION_HEIGHTS[3] = { 0, 160, 250 };
const static uint LAYER_TYPES[4] = { 3u, 0u, 2u, 4u };
const static int heightCount = TRANSISTION_HEIGHTS.getCount();

struct DispatchParams {
  uint chunkCount;
  uint horizontalChunkCount;
};

groupshared uint scanValues[64];
groupshared uint groupOutputBase;

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
[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> QuadCount;
[[vk::binding(5, 0)]]
RWStructuredBuffer<uint3> DrawArguments;
[[vk::binding(6, 0)]]
StructuredBuffer<int2> ChunkWork;

uint heightIndex(uint columnIndex, uint x, uint z) { return columnIndex * HEIGHTS_PER_CHUNK + z * HEIGHT_STRIDE + x; }

int sampleHeight(uint columnIndex, int x, int z) {
  return HeightBuffer[heightIndex(columnIndex, uint(x + 1), uint(z + 1))];
}

uint hashFace(uint x, uint y, uint z, uint normalIndex, uint blockType) {
  uint hash = x * 0x8da6b343u ^ y * 0xd8163841u ^ z * 0xcb1ab31fu ^ normalIndex ^ (blockType * 0x9e3779b9u);
  hash ^= hash >> 16u;
  hash *= 0x7feb352du;
  hash ^= hash >> 15u;
  hash *= 0x846ca68bu;
  hash ^= hash >> 16u;
  return hash;
}

uint faceQuadCount(int height, int neighborHeight, int chunkY) {
  const int chunkBottom = chunkY * int(CHUNK_SIZE);
  const int chunkTop = chunkBottom + int(CHUNK_SIZE);
  const int faceBottom = max(neighborHeight, chunkBottom);
  const int faceTop = min(height, chunkTop);
  return uint(max(faceTop - faceBottom, 0));
}

bool isSolidAt(uint columnIndex, int x, int y, int z, int chunkY) {
  const int worldY = chunkY * int(CHUNK_SIZE) + y;
  return worldY < sampleHeight(columnIndex, x, z);
}

uint vertexAo(bool sideA, bool sideB, bool corner) {
  return sideA && sideB ? 3u : uint(sideA) + uint(sideB) + uint(corner);
}

uint selectBlockType(int worldY, int worldX, int worldZ) {
  uint layerId = 0;
  [unroll]
  for (uint i = 0u; i < uint(heightCount); i++) {
    layerId += int(worldY >= TRANSISTION_HEIGHTS[i]); // layerId ends up being [0,1,2,3...]
  }

  return LAYER_TYPES[min(heightCount, layerId)];
}

bool isNearLayerTransition(int blockWorldY, int scale) {
  bool nearTransition = false;
  [unroll]
  for (uint i = 0u; i < uint(heightCount); ++i) {
    const int transition = TRANSISTION_HEIGHTS[i];
    nearTransition |= blockWorldY >= transition - scale && blockWorldY < transition + scale;
  }
  return nearTransition;
}

uint calculateAo(uint columnIndex, int x, int y, int z, int chunkY, uint normalIndex) {
  const int3 uAxes[6] = { int3(0, 0, 1), int3(0, 0, 1), int3(1, 0, 0), int3(1, 0, 0), int3(1, 0, 0), int3(1, 0, 0) };
  const int3 vAxes[6] = { int3(0, 1, 0), int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1), int3(0, 1, 0), int3(0, 1, 0) };
  const int3 normalAxes[6] = { int3(1, 0, 0),  int3(-1, 0, 0), int3(0, 1, 0),
                               int3(0, -1, 0), int3(0, 0, 1),  int3(0, 0, -1) };
  const int3 outside = int3(x, y, z) + normalAxes[normalIndex];
  const int3 u = uAxes[normalIndex];
  const int3 v = vAxes[normalIndex];

  const bool sideU0 = isSolidAt(columnIndex, outside.x - u.x, outside.y - u.y, outside.z - u.z, chunkY);
  const bool sideU1 = isSolidAt(columnIndex, outside.x + u.x, outside.y + u.y, outside.z + u.z, chunkY);
  const bool sideV0 = isSolidAt(columnIndex, outside.x - v.x, outside.y - v.y, outside.z - v.z, chunkY);
  const bool sideV1 = isSolidAt(columnIndex, outside.x + v.x, outside.y + v.y, outside.z + v.z, chunkY);
  const bool corner0 =
      isSolidAt(columnIndex, outside.x - u.x - v.x, outside.y - u.y - v.y, outside.z - u.z - v.z, chunkY);
  const bool corner1 =
      isSolidAt(columnIndex, outside.x + u.x - v.x, outside.y + u.y - v.y, outside.z + u.z - v.z, chunkY);
  const bool corner2 =
      isSolidAt(columnIndex, outside.x + u.x + v.x, outside.y + u.y + v.y, outside.z + u.z + v.z, chunkY);
  const bool corner3 =
      isSolidAt(columnIndex, outside.x - u.x + v.x, outside.y - u.y + v.y, outside.z - u.z + v.z, chunkY);
  return vertexAo(sideU0, sideV0, corner0) | (vertexAo(sideU1, sideV0, corner1) << 2u) |
         (vertexAo(sideU1, sideV1, corner2) << 4u) | (vertexAo(sideU0, sideV1, corner3) << 6u);
}

void writeQuad(uint outputIndex, uint columnIndex, uint x, uint y, uint z, int worldY, uint normalIndex,
               int columnHeight, int chunkY, int4 chunkCoord) {
  const int scale = chunkCoord.w;
  const int worldCellX = chunkCoord.x * int(CHUNK_SIZE) + int(x) * scale;
  const int worldCellZ = chunkCoord.y * int(CHUNK_SIZE) + int(z) * scale;
  const int blockWorldY = worldY * scale + scale - 1;
  const bool nearLayerTransition = isNearLayerTransition(blockWorldY, scale);
  const int materialWorldY = nearLayerTransition
                                 ? int(floor(sampleTerrainHeight(worldCellX + scale / 2, worldCellZ + scale / 2, 0u)))
                                 : blockWorldY;
  const uint actualBlockType = selectBlockType(materialWorldY, worldCellX + scale / 2, worldCellZ + scale / 2);
  uint packed = (x & 0x1fu) | ((y & 0x1fu) << 5u) | ((z & 0x1fu) << 10u) | ((normalIndex & 0x7u) << 15u);
  packed |= calculateAo(columnIndex, int(x), int(y), int(z), chunkY, normalIndex) << 18u;
  const uint rotationGroupIndex = (normalIndex >> 1u) == 1u ? 1u : 0u;
  const uint rotationMask = actualBlockType == 0u ? 2u : 0u;
  const uint rotationEnabled = (rotationMask >> rotationGroupIndex) & 1u;
  const uint rotation = hashFace(x, y, z, normalIndex, actualBlockType) & 3u & (0u - rotationEnabled);
  const uint lod = uint(chunkCoord.z) & 0x7u;
  const uint metadata = actualBlockType | (15u << 10u) | (0xfffu << 14u) | (rotation << 26u) | (lod << 29u);
  QuadBuffer[outputIndex] = (uint64_t(metadata) << 32u) | uint64_t(packed);

  const int worldX = chunkCoord.x * int(CHUNK_SIZE);
  const int worldZ = chunkCoord.y * int(CHUNK_SIZE);
  const uint64_t offsetX = uint64_t(uint(worldX) & 0x00ffffffu);
  const uint64_t offsetZ = uint64_t(uint(worldZ) & 0x00ffffffu) << 24u;
  const uint64_t offsetY = uint64_t(uint(chunkY) & 0x1fffu) << 48u;
  const uint64_t packedLod = uint64_t(chunkCoord.z & 0x7) << 61u;
  ChunkOffsets[outputIndex] = offsetX | offsetZ | offsetY | packedLod;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void generateHeights(uint3 dispatchThreadID: SV_DispatchThreadID, uint3 groupID: SV_GroupID) {
  const uint columnIndex = groupID.z;
  const uint localX = dispatchThreadID.x;
  const uint localZ = dispatchThreadID.y;
  if (columnIndex >= dispatchParams.horizontalChunkCount || localX >= HEIGHT_STRIDE || localZ >= HEIGHT_STRIDE) {
    return;
  }

  const int4 chunkCoord = ChunkCoords[columnIndex];
  const int scale = chunkCoord.w;
  const int worldX = chunkCoord.x * int(CHUNK_SIZE) + (int(localX) - 1) * scale;
  const int worldZ = chunkCoord.y * int(CHUNK_SIZE) + (int(localZ) - 1) * scale;
  const float height = sampleTerrainHeight(worldX, worldZ, uint(chunkCoord.z));
  const int minimumHeight = int(floor(float(MIN_TERRAIN_HEIGHT) / float(chunkCoord.w)));
  const float scaledHeight = height / float(chunkCoord.w);
  HeightBuffer[heightIndex(columnIndex, localX, localZ)] = max(int(floor(scaledHeight)), minimumHeight);
}

[numthreads(8, 8, 1)]
[shader("compute")]
void meshColumns(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID) {
  const uint chunkIndex = groupID.z;
  if (chunkIndex >= dispatchParams.chunkCount || dispatchParams.horizontalChunkCount == 0u) {
    return;
  }

  const int2 work = ChunkWork[chunkIndex];
  const uint columnIndex = uint(work.x);
  const int chunkY = work.y;
  const int chunkBottom = chunkY * int(CHUNK_SIZE);
  const int chunkTop = chunkBottom + int(CHUNK_SIZE);
  const uint x = groupID.x * 8u + groupThreadID.x;
  const uint z = groupID.y * 8u + groupThreadID.y;
  const uint lane = groupThreadID.y * 8u + groupThreadID.x;
  const uint waveIndex = lane >> 5u;
  const uint laneInWave = lane & 31u;
  const int height = sampleHeight(columnIndex, int(x), int(z));
  const int negXHeight = sampleHeight(columnIndex, int(x) - 1, int(z));
  const int posXHeight = sampleHeight(columnIndex, int(x) + 1, int(z));
  const int negZHeight = sampleHeight(columnIndex, int(x), int(z) - 1);
  const int posZHeight = sampleHeight(columnIndex, int(x), int(z) + 1);

  const uint topCount = height > chunkBottom && height <= chunkTop ? 1u : 0u;
  const uint negXCount = faceQuadCount(height, negXHeight, chunkY);
  const uint posXCount = faceQuadCount(height, posXHeight, chunkY);
  const uint negZCount = faceQuadCount(height, negZHeight, chunkY);
  const uint posZCount = faceQuadCount(height, posZHeight, chunkY);
  const uint columnFaceCount = topCount + negXCount + posXCount + negZCount + posZCount;

  const uint lanePrefix = WavePrefixSum(columnFaceCount);
  if (laneInWave == 31u) {
    scanValues[waveIndex] = lanePrefix + columnFaceCount;
  }
  GroupMemoryBarrierWithGroupSync();

  const uint previousWaveTotal = waveIndex == 0u ? 0u : scanValues[0];
  const uint laneStart = previousWaveTotal + lanePrefix;
  uint faceOffset = laneStart;

  if (lane == 63u) {
    InterlockedAdd(QuadCount[0], scanValues[0] + (waveIndex == 1u ? scanValues[1] : 0u), groupOutputBase);
  }
  GroupMemoryBarrierWithGroupSync();
  const int4 chunkCoord = ChunkCoords[columnIndex];
  const uint chunkBase = groupOutputBase;

  const uint faceNormalIndices[5] = { 2u, 1u, 0u, 5u, 4u };
  const int faceStartValues[5] = { topCount != 0u ? height - 1 : 0, max(negXHeight, chunkBottom),
                                   max(posXHeight, chunkBottom), max(negZHeight, chunkBottom),
                                   max(posZHeight, chunkBottom) };
  const int faceEndValues[5] = { topCount != 0u ? height : 0, min(height, chunkTop), min(height, chunkTop),
                                 min(height, chunkTop), min(height, chunkTop) };

  for (uint faceIndex = 0u; faceIndex < 5u; ++faceIndex) {
    const int faceStart = faceStartValues[faceIndex];
    const int faceEnd = faceEndValues[faceIndex];
    const uint faceNormal = faceNormalIndices[faceIndex];
    for (int worldY = faceStart; worldY < faceEnd; ++worldY) {
      const uint outputIndex = chunkBase + faceOffset++;
      if (outputIndex < MAX_QUAD_COUNT)
        writeQuad(outputIndex, columnIndex, x, uint(worldY - chunkBottom), z, worldY, faceNormal, height, chunkY,
                  chunkCoord);
    }
  }
}

[numthreads(1, 1, 1)]
[shader("compute")]
void finalizeMeshDispatch() {
  const uint clampedQuadCount = min(QuadCount[0], MAX_QUAD_COUNT);
  const uint meshletCount = (clampedQuadCount + 31u) / 32u;
  const uint maxGroupsX = 65535u;
  DrawArguments[0] = uint3(min(meshletCount, maxGroupsX), (meshletCount + maxGroupsX - 1u) / maxGroupsX, 1u);
}
