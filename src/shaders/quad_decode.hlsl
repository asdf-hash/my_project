// Unpacking of one packed quad (QuadBuffer + ChunkOffsets entries written by meshColumns in chunk_compute.slang).
// Shared by the mesh shader and the vertex shader so both paths agree exactly.
#pragma once

struct DecodedQuad {
  float3 faceOrigin;
  float3 greedyAxis;
  float3 faceV;
  float2 quadSize;
  float voxelScale;
  uint normalIndex;
  uint lighting;
  uint material;
  // Cell the quad was written at, in cells of this LOD (blocks at LOD 0); with the face's uv cell it names the block
  // a fragment belongs to, independent of how faces were merged (fragMain's texture rotation).
  int3 baseCell;
  bool frontFacing;
};

DecodedQuad decodePackedQuad(uint2 packed, uint64_t chunkOffsetPacked, float3 cameraPosition) {
  // Chunk origin fields are signed 24/24/15-bit values packed by World::packChunkOffset.
  const int32_t chunkOffsetX = int32_t(int32_t(chunkOffsetPacked & 0xFFFFFFul) << 8) >> 8;
  const int32_t chunkOffsetZ = int32_t(int32_t((chunkOffsetPacked >> 24u) & 0xFFFFFFul) << 8) >> 8;
  const int32_t chunkY = int32_t(int32_t((chunkOffsetPacked >> 48u) & 0x1FFFul) << 19) >> 19;
  const uint lod = uint(chunkOffsetPacked >> 61u);
  const float voxelScale = float(1u << lod);
  const float chunkOffsetY = float(chunkY * 32) * voxelScale;
  const uint3 subchunkPos = uint3(packed.x & 0x1fu, (packed.x >> 5u) & 0x1fu, (packed.x >> 10u) & 0x1fu);
  const uint normalIndex = (packed.x >> 15u) & 0x7u;
  const uint aoBits = (packed.x >> 18u) & 0xffu;
  const uint metadata = packed.y;
  const uint blockType = metadata & 0x3ffu;
  // Metadata bits 26-28: merged width - 1, across the greedy length (see writeQuad in chunk_compute.slang).
  const uint greedyWidth = ((metadata >> 26u) & 0x7u) + 1u;
  const uint coloredLight = (metadata >> 14u) & 0xfffu;
  const uint encodedLength = (packed.x >> 26u) & 0x1fu;
  const uint greedyLength = encodedLength + 1u;
  const uint faceAxis = normalIndex >> 1u;
  const uint faceSign = normalIndex & 1u;
  const bool xFacing = faceAxis == 0u;

  const float3 blockPos =
      float3(subchunkPos) * voxelScale + float3(float(chunkOffsetX), chunkOffsetY, float(chunkOffsetZ));
  const float positiveFace = float(1u - faceSign);
  const float3 faceOrigin =
      blockPos + float3(faceAxis == 0u ? positiveFace * voxelScale : 0.0f,
                        faceAxis == 1u ? positiveFace * voxelScale : 0.0f,
                        faceAxis == 2u ? positiveFace * voxelScale : (normalIndex == 2u ? voxelScale : 0.0f));
  const float3 faceV = float3(0.0f, faceAxis != 1u ? 1.0f : 0.0f, faceAxis == 1u ? float(int(faceSign) * 2 - 1) : 0.0f);

  // The face plane sits at faceOrigin[faceAxis]; normals with faceSign 0 point toward +axis.
  const float cameraToPlane = cameraPosition[faceAxis] - faceOrigin[faceAxis];

  DecodedQuad quad;
  quad.faceOrigin = faceOrigin;
  quad.greedyAxis = xFacing ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
  quad.faceV = faceV;
  // Top faces: length along the greedy axis, width along faceV. Side faces: width along the greedy (horizontal)
  // axis, length up the face.
  quad.quadSize = faceAxis == 1u ? float2(float(greedyLength), float(greedyWidth))
                                 : float2(float(greedyWidth), float(greedyLength));
  quad.voxelScale = voxelScale;
  quad.normalIndex = normalIndex;
  quad.lighting = aoBits | (((metadata >> 10u) & 0xfu) << 8u) | (coloredLight << 12u);
  // Chunk origins are multiples of 32 cells, so the shifts are exact (and floor for negative coordinates).
  quad.baseCell = int3(subchunkPos) + int3(chunkOffsetX >> int(lod), chunkY * 32, chunkOffsetZ >> int(lod));
  // Bits 15-19 hold length - 1: a full-chunk run (32) would spill into the LOD bits.
  quad.material = blockType | (normalIndex << 10u) | (encodedLength << 15u) | (lod << 20u) | (greedyWidth << 23u);
  quad.frontFacing = (faceSign == 0u ? cameraToPlane : -cameraToPlane) > 0.0f;
  return quad;
}

// Corner order: 0 = (0,0), 1 = (1,0), 2 = (1,1), 3 = (0,1) in (greedy, faceV) quad space.
float2 quadCornerCoord(DecodedQuad quad, uint corner) {
  return float2((corner == 1u || corner == 2u) ? 1.0f : 0.0f, corner >= 2u ? 1.0f : 0.0f) * quad.quadSize;
}

float3 quadCornerWorld(DecodedQuad quad, uint corner) {
  const float2 scaled = quadCornerCoord(quad, corner) * quad.voxelScale;
  return quad.faceOrigin + quad.greedyAxis * scaled.x + quad.faceV * scaled.y;
}

// Faces 0 and 5 have the opposite winding for the same corner order.
bool flippedWinding(uint normalIndex) {
  return normalIndex == 0u || normalIndex == 5u;
}
