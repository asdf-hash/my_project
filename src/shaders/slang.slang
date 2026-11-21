
// Per (block, face) data from blocks.txt, indexed blockId * 6 + normalIndex. Matches GpuBlockFace in renderer.cpp.
struct BlockFace {
  uint texture;
  uint lodTexture;
  uint flags;
  uint padding;
  float4 lodTint;
};
static const uint BLOCK_FACE_RANDOM_ROTATION = 1u << 0u;
static const uint BLOCK_FACE_HAS_LOD_TEXTURE = 1u << 1u;

#include "geometry_shared.hlsl"
#include "quad_decode.hlsl"

[[vk::binding(0, 0)]]
StructuredBuffer<uint2> QuadBuffer;
[[vk::binding(3, 0)]]
StructuredBuffer<uint64_t> ChunkOffsets;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> QuadCount;

[[vk::binding(5, 0)]]
StructuredBuffer<BlockFace> BlockFaces;

struct MeshVertex {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD3;
  float2 uv : TEXCOORD0;
  nointerpolation uint lighting : TEXCOORD1;
  nointerpolation uint material : TEXCOORD2;
  // Clip w = view depth; affine in world space, so perspective-correct interpolation is exact. fragMain turns it
  // back into the camera distance for fog (replaces a float3 world position).
  float viewDepth : TEXCOORD4;
  nointerpolation int3 baseCell : TEXCOORD5; // DecodedQuad::baseCell
};

[[vk::binding(2, 0)]]
SamplerState sampler;
[[vk::binding(11, 0)]]
Texture2D textures[];

static const float aoValues[4] = { 0.1f, 0.25f, 0.45f, 1.0f };

float2 rotateUv(float2 uv, uint rotation) {
  if (rotation == 1u) {
    return float2(uv.y, 1.0 - uv.x);
  }
  if (rotation == 2u) {
    return 1.0 - uv;
  }
  if (rotation == 3u) {
    return float2(1.0 - uv.y, uv.x);
  }
  return uv;
}

float2 rotateUvGradient(float2 gradient, uint rotation) {
  if (rotation == 1u) {
    return float2(gradient.y, -gradient.x);
  }
  if (rotation == 2u) {
    return -gradient;
  }
  if (rotation == 3u) {
    return float2(-gradient.y, gradient.x);
  }
  return gradient;
}

DecodedQuad decodeQuad(uint quadIndex) {
  return decodePackedQuad(QuadBuffer[quadIndex], ChunkOffsets[quadIndex], ubo.cameraPosition.xyz);
}

MeshVertex makeVertex(DecodedQuad quad, uint corner, float3 worldPos, float4 clipPos) {
  MeshVertex v;
  v.pos = clipPos;
  v.previousClip = mul(ubo.previousMvp, float4(worldPos, 1.0));
  v.uv = quadCornerCoord(quad, corner);
  v.lighting = quad.lighting;
  v.material = quad.material;
  v.viewDepth = clipPos.w;
  v.baseCell = quad.baseCell;
  return v;
}

// Greedy merging leaves T-junctions: a merged quad's corner sits partway along its neighbour's longer edge, and the
// rasterizer's sub-pixel snapping puts it a hair off that edge, so single samples of sky show through. Growing the
// quad in world space fails at grazing angles (the growth foreshortens to nothing), so the projected quad is grown
// in screen space instead: every edge moves outward by QUAD_DILATION_PIXELS. Neighbours then overlap by a sliver at
// any angle or distance; coplanar overlap is invisible.
static const float QUAD_DILATION_PIXELS = 1.0f / 32.0f;
static const float QUAD_DILATION_MAX_MITER = 4.0f; // caps the corner push at very sharp projected corners

void quadClipCorners(DecodedQuad quad, out float3 world[4], out float4 clip[4]) {
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    world[i] = quadCornerWorld(quad, i);
    clip[i] = mul(ubo.mvp, float4(world[i], 1.0));
  }
  if (min(min(clip[0].w, clip[1].w), min(clip[2].w, clip[3].w)) <= 0.0f) {
    return; // crosses the camera plane: large on screen, and the screen-space outline is not defined
  }
  // Corners in pixels (the y flip doesn't matter: orientation comes from the signed area below).
  float2 p[4];
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    p[i] = clip[i].xy / clip[i].w * (0.5f * ubo.renderSize);
  }
  float area = 0.0f;
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    const float2 a = p[i];
    const float2 b = p[(i + 1u) & 3u];
    area += a.x * b.y - a.y * b.x;
  }
  const float orientation = area >= 0.0f ? 1.0f : -1.0f;
  // Outward unit normal of edge i (corner i -> i + 1); zero for an edge with no screen length.
  float2 normals[4];
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    const float2 edge = p[(i + 1u) & 3u] - p[i];
    const float edgeLength = length(edge);
    normals[i] = edgeLength > 1e-6f ? float2(edge.y, -edge.x) * (orientation / edgeLength) : float2(0.0f, 0.0f);
  }
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    // Miter of the two edges meeting at corner i: moves both edges out by the dilation.
    const float2 n0 = normals[(i + 3u) & 3u];
    const float2 n1 = normals[i];
    const float2 sum = n0 + n1;
    const float scale = 1.0f / max(1.0f + dot(n0, n1), 1.0f / QUAD_DILATION_MAX_MITER);
    const float2 offsetPixels = sum * (QUAD_DILATION_PIXELS * scale);
    clip[i].xy += offsetPixels / (0.5f * ubo.renderSize) * clip[i].w;
  }
}

// ---- Mesh path -------------------------------------------------------------------------------------------------
// taskMain (task.slang) culls tiles and launches meshlets; see geometry_shared.hlsl for the payload.

// Mesh shader: one meshlet = up to 32 consecutive quads of one surviving run (so one face direction). Per-quad
// backface rejection happens here, before any vertex or primitive is emitted; frustum culling is the task shader's.
groupshared uint visibleQuadCount;

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(QUADS_PER_MESHLET, 1, 1)]
void meshMain(out vertices MeshVertex verts[128], out indices uint3 triangles[64], in payload TaskPayload payload,
              uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID) {
  // Last run whose firstMeshlet <= this meshlet.
  const uint meshlet = groupID.x;
  const uint runCount = min(payload.runCount, CLUSTERS_PER_TASK * RUNS_PER_CLUSTER);
  if (runCount == 0u) {
    SetMeshOutputCounts(0u, 0u);
    return;
  }
  uint low = 0u;
  uint high = runCount - 1u;
  while (low < high) {
    const uint mid = low + (high - low + 1u) / 2u;
    if (payload.firstMeshlet[mid] <= meshlet) {
      low = mid;
    } else {
      high = mid - 1u;
    }
  }
  const uint meshletInRun = meshlet - payload.firstMeshlet[low];
  const uint firstQuad = payload.firstQuad[low] + meshletInRun * QUADS_PER_MESHLET;
  const uint quadCount = min(QUADS_PER_MESHLET, payload.quadCount[low] - meshletInRun * QUADS_PER_MESHLET);

  const uint lane = groupThreadID.x;
  const bool active = lane < quadCount;
  // Inactive lanes decode the meshlet's first quad so every lane takes part in the compaction below.
  const DecodedQuad quad = decodeQuad(firstQuad + (active ? lane : 0u));
  const bool visible = active && quad.frontFacing;

  // Compact the surviving quads to the front of the meshlet. When one subgroup spans the whole group (NVIDIA
  // wave32, AMD wave64) this is two wave intrinsics; narrower subgroups (e.g. Intel SIMD8/16) use shared memory.
  uint survivorCount;
  uint slot;
  if (WaveGetLaneCount() >= QUADS_PER_MESHLET) {
    survivorCount = WaveActiveCountBits(visible);
    slot = WavePrefixCountBits(visible);
  } else {
    if (lane == 0u) {
      visibleQuadCount = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    slot = 0u;
    if (visible) {
      InterlockedAdd(visibleQuadCount, 1u, slot);
    }
    GroupMemoryBarrierWithGroupSync();
    survivorCount = visibleQuadCount;
  }
  SetMeshOutputCounts(survivorCount * 4u, survivorCount * 2u);
  if (!visible) {
    return;
  }

  const uint vertexBase = slot * 4u;
  float3 world[4];
  float4 clip[4];
  quadClipCorners(quad, world, clip);
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    verts[vertexBase + i] = makeVertex(quad, i, world[i], clip[i]);
  }

  const uint primitiveBase = slot * 2u;
  if (flippedWinding(quad.normalIndex)) {
    triangles[primitiveBase] = uint3(vertexBase + 0u, vertexBase + 2u, vertexBase + 1u);
    triangles[primitiveBase + 1u] = uint3(vertexBase + 0u, vertexBase + 3u, vertexBase + 2u);
  } else {
    triangles[primitiveBase] = uint3(vertexBase + 0u, vertexBase + 1u, vertexBase + 2u);
    triangles[primitiveBase + 1u] = uint3(vertexBase + 0u, vertexBase + 2u, vertexBase + 3u);
  }
}

// ---- Vertex path -----------------------------------------------------------------------------------------------
// Drawn with drawIndexedIndirect over a static uint16 index buffer holding VERTEX_QUADS_PER_INSTANCE quads
// ({0,1,2, 0,2,3} + 4q), instanced over every quad in QuadBuffer (args from finalizeVertexDraw). Indexed so the
// post-transform cache shares the two diagonal corners: 4 invocations per quad, not 6. No GPU culling on this
// path: the CPU clipmap culler (ClipmapView::frustumCull) only meshes chunks inside the view frustum.
// Must match VERTEX_QUADS_PER_INSTANCE in renderer.cpp and chunk_compute.slang.
static const uint VERTEX_QUADS_PER_INSTANCE = 16384u;

[shader("vertex")]
MeshVertex vertMain(uint vertexID: SV_VertexID, uint instanceID: SV_InstanceID) {
  const uint drawQuad = instanceID * VERTEX_QUADS_PER_INSTANCE + (vertexID >> 2u);
  const uint quadIndex = drawParams.quadBase + drawQuad;
  MeshVertex dead = (MeshVertex)0;
  dead.pos = float4(asfloat(0x7fc00000u).xxxx); // NaN position: the primitive is discarded before setup.
  if (quadIndex >= min(QuadCount[0], MAX_QUAD_COUNT) ||
      (drawParams.quadCount != 0u && drawQuad >= drawParams.quadCount)) {
    return dead; // Tail of the last instance.
  }

  const DecodedQuad quad = decodeQuad(quadIndex);
  if (quad.normalIndex >= 6u) {
    return dead; // retired by retireWork; reclaimed at the next full rebuild
  }
  // Back faces still cost their invocations, but skip the transforms and attribute exports.
  if (!quad.frontFacing) {
    return dead;
  }
  // The index buffer has a fixed winding, so flipped faces mirror corners 1 <-> 3 instead:
  // {0,1,2},{0,2,3} becomes {0,3,2},{0,2,1}, which is exactly the mesh path's flipped triangles.
  uint corner = vertexID & 3u;
  if (flippedWinding(quad.normalIndex) && (corner & 1u) != 0u) {
    corner ^= 2u;
  }
  // Every corner invocation projects all four to dilate the outline (see quadClipCorners).
  float3 world[4];
  float4 clip[4];
  quadClipCorners(quad, world, clip);
  return makeVertex(quad, corner, world[corner], clip[corner]);
}

struct FragmentInput {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD3;
  float2 uv : TEXCOORD0;
  nointerpolation uint lighting : TEXCOORD1;
  nointerpolation uint material : TEXCOORD2;
  // Clip w = view depth; affine in world space, so perspective-correct interpolation is exact. fragMain turns it
  // back into the camera distance for fog (replaces a float3 world position).
  float viewDepth : TEXCOORD4;
  nointerpolation int3 baseCell : TEXCOORD5;
};

struct SceneFragmentOutput {
  float4 color : SV_Target0;
  float4 motion : SV_Target1;
};

[shader("fragment")]
SceneFragmentOutput fragMain(FragmentInput input) {
  const uint aoBits = input.lighting & 0xffu;
  const float skylight = float((input.lighting >> 8u) & 0xfu) / 15.0f;
  const uint coloredLightBits = (input.lighting >> 12u) & 0xfffu;
  const uint blockType = input.material & 0x3ffu;
  const uint normalIndex = (input.material >> 10u) & 0x7u;
  const uint faceAxis = normalIndex >> 1u;
  const uint faceSign = normalIndex & 1u;
  const uint greedyLength = ((input.material >> 15u) & 0x1fu) + 1u;
  const uint chunkLod = (input.material >> 20u) & 0x7u;
  const uint greedyWidth = (input.material >> 23u) & 0xfu;
  const BlockFace face = BlockFaces[blockType * 6u + normalIndex];
  const bool useLodTexture = chunkLod > 0u && (face.flags & BLOCK_FACE_HAS_LOD_TEXTURE) != 0u;
  const uint textureIndex = useLodTexture ? face.lodTexture : face.texture;
  // Merged quads cover several cells (uv spans 0..length on the merged axis). Each cell's rotation hashes the block
  // it belongs to and the face direction, so remeshing (a broken block re-merging its neighbors) never changes it.
  const float2 cell = floor(input.uv);
  uint rotation = 0u;
  if ((face.flags & BLOCK_FACE_RANDOM_ROTATION) != 0u) {
    // uv.x runs along the greedy axis (z for x faces, else x); uv.y along faceV (up for sides, -z for +y, +z for -y).
    const int3 greedyAxis = faceAxis == 0u ? int3(0, 0, 1) : int3(1, 0, 0);
    const int3 faceV = faceAxis != 1u ? int3(0, 1, 0) : int3(0, 0, faceSign == 0u ? -1 : 1);
    const int3 block = input.baseCell + greedyAxis * int(cell.x) + faceV * int(cell.y);
    uint h = uint(block.x) * 0x8da6b343u ^ uint(block.y) * 0xd8163841u ^ uint(block.z) * 0xcb1ab31fu ^
             (normalIndex | (chunkLod << 3u)) * 0x165667b1u;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    h ^= h >> 15u;
    rotation = h & 3u;
  }

  float2 textureUv = input.uv - cell;
  float2 textureUvDx = ddx(input.uv);
  float2 textureUvDy = ddy(input.uv);
  if (rotation != 0u) {
    textureUv = rotateUv(textureUv, rotation);
    textureUvDx = rotateUvGradient(textureUvDx, rotation);
    textureUvDy = rotateUvGradient(textureUvDy, rotation);
  }
  textureUv.y = 1.0f - textureUv.y;
  textureUvDx.y = -textureUvDx.y;
  textureUvDy.y = -textureUvDy.y;
  const float textureMipSharpeningBias = 0.5f;
  const float gradientScale = exp2(-textureMipSharpeningBias);
  float4 color = textures[NonUniformResourceIndex(textureIndex)].SampleGrad(
      sampler, textureUv, textureUvDx * gradientScale, textureUvDy * gradientScale);

  if (useLodTexture) {
    color.rgb *= face.lodTint.rgb;
  }
  // Debug view (X): keep the texture's detail but shift it toward the chunk source's color.
  if (drawParams.debugTint != DEBUG_TINT_OFF) {
    const float3 tint = drawParams.debugTint == DEBUG_TINT_CPU
                            ? float3(0.1f, 1.0f, 0.1f)
                            : float3(1.0f, float(chunkLod) / float(MAX_LOD_LEVELS - 1u), 0.05f);
    const float detail = dot(color.rgb, float3(0.299f, 0.587f, 0.114f));
    color.rgb = tint * (0.35f + 0.9f * detail);
  }

  // AO corners span the whole merged quad (see quadSize in decodePackedQuad).
  const float2 quadSize = faceAxis == 1u ? float2(float(greedyLength), float(greedyWidth))
                                         : float2(float(greedyWidth), float(greedyLength));
  const float u = saturate(input.uv.x / quadSize.x);
  const float v = saturate(input.uv.y / quadSize.y);
  const float ao00 = aoValues[3u - (aoBits & 0x3u)];
  const float ao10 = aoValues[3u - ((aoBits >> 2u) & 0x3u)];
  const float ao11 = aoValues[3u - ((aoBits >> 4u) & 0x3u)];
  const float ao01 = aoValues[3u - ((aoBits >> 6u) & 0x3u)];
  const float ambientOcclusion = lerp(lerp(ao00, ao10, u), lerp(ao01, ao11, u), v);
  const float lodAoLightening = min(float(chunkLod) * ubo.lodAoLighteningPerLevel, 0.5f);
  const float aoFactor = lerp(ambientOcclusion, 1.0f, lodAoLightening) * skylight;
  const float3 coloredLight = float3(float(coloredLightBits & 0xfu), float((coloredLightBits >> 4u) & 0xfu),
                                     float((coloredLightBits >> 8u) & 0xfu)) /
                              15.0f;
  const float shading = faceAxis == 0u ? 0.6f : (faceAxis == 1u ? (faceSign == 0u ? 1.0f : 0.4f) : 0.8f);
  const float2 currentUv = input.pos.xy / ubo.renderSize;
  const float2 previousNdc = input.previousClip.xy / input.previousClip.w;
  const float2 previousUv = float2(previousNdc.x * 0.5f + 0.5f, 0.5f - previousNdc.y * 0.5f);
  const float fogStart = 1200.0f;
  const float fogEnd = 12000.0f;
  const float2 ndc = currentUv * 2.0f - 1.0f;
  const float cameraDistance = input.viewDepth * length(float3(ndc * ubo.viewRayScale.xy, 1.0f));
  const float fogAmount = saturate((cameraDistance - fogStart) / (fogEnd - fogStart));
  const float3 skyColor = float3(0.55f, 0.72f, 0.9f);
  SceneFragmentOutput output;
  const float3 litColor = color.rgb * coloredLight * aoFactor * shading;
  output.color = float4(lerp(litColor, skyColor, fogAmount), color.a);
  output.motion = float4(previousUv - currentUv, input.previousClip.z / input.previousClip.w, input.pos.z);
  return output;
}
