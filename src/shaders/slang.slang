// Terrain geometry and shading: mesh path (meshMain, culled by taskMain in task.slang), vertex path (vertMain), and
// the opaque / cutout / water fragment shaders.

// Per (block, face) data from blocks.txt, indexed blockId * 6 + normalIndex. Matches GpuBlockFace in renderer.cpp.
struct BlockFace {
  uint texture;
  uint lodTexture;
  uint flags;
  uint animation; // frames of texture (bits 0-7) and lodTexture (8-15), ms per frame (16-31)
  float4 lodTint;
  float4 tint;    // every LOD
  float4 farTint; // far away, see farTintStart
};
// far_tint fades in by camera distance over [ubo.farTintStart, 4 x it), around LOD 3 where a tree becomes one leaf
// block (vegetation_shared.hlsl). By distance rather than LOD, so no seam.
static const uint BLOCK_FACE_RANDOM_ROTATION = 1u << 0u;
static const uint BLOCK_FACE_HAS_LOD_TEXTURE = 1u << 1u;
static const uint BLOCK_FACE_FLUID = 1u << 2u;
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
  // Clip w; affine in world space, so interpolation is exact. fragMain turns it into the camera distance for fog.
  float viewDepth : TEXCOORD4;
  nointerpolation int3 baseCell : TEXCOORD5; // DecodedQuad::baseCell
  float worldY : TEXCOORD6;                  // elevation view (H)
};

[[vk::binding(2, 0)]]
SamplerState sampler;
[[vk::binding(11, 0)]]
Texture2D textures[];

// Display-space brightness by occlusion level, fully lit last (Minecraft's: 0.2 per occluding sample, 4 samples).
static const float aoValues[4] = { 0.4f, 0.6f, 0.8f, 1.0f };

/**
 * Rotates a texture coordinate in the unit square.
 *
 * @param rotation Quarter turns, 0-3.
 */
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

/** Rotates a uv derivative to match rotateUv(uv, rotation). */
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

/** Decodes QuadBuffer[quadIndex] relative to the culling camera. */
DecodedQuad decodeQuad(uint quadIndex) {
  return decodePackedQuad(QuadBuffer[quadIndex], ChunkOffsets[quadIndex], ubo.cullCameraPosition.xyz);
}

/**
 * Builds the output vertex for one quad corner.
 *
 * @param corner Corner index, 0-3.
 * @param clipPos This frame's clip position (already dilated by quadClipCorners).
 */
MeshVertex makeVertex(DecodedQuad quad, uint corner, float3 worldPos, float4 clipPos) {
  MeshVertex v;
  v.pos = clipPos;
  v.previousClip = mul(ubo.previousMvp, float4(worldPos, 1.0));
  v.uv = quadCornerCoord(quad, corner);
  v.lighting = quad.lighting;
  v.material = quad.material;
  v.viewDepth = clipPos.w;
  v.baseCell = quad.baseCell;
  v.worldY = worldPos.y;
  return v;
}

// Greedy merging leaves T-junctions whose sub-pixel snapping lets single sky samples through. Growing the quad in
// world space fails at grazing angles, so each projected edge moves outward by QUAD_DILATION_PIXELS in screen space;
// neighbours then overlap by a sliver (invisible when coplanar) at any angle or distance.
static const float QUAD_DILATION_PIXELS = 1.0f / 32.0f;
static const float QUAD_DILATION_MAX_MITER = 4.0f; // caps the corner push at very sharp projected corners

/**
 * World and clip positions of a quad's four corners, with the clip positions dilated in screen space (see
 * QUAD_DILATION_PIXELS). Quads crossing the camera plane are left undilated.
 */
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

groupshared uint visibleQuadCount;

/**
 * Emits one meshlet: up to QUADS_PER_MESHLET consecutive quads of one surviving run (one face direction), with
 * per-quad backface rejection before anything is emitted. Frustum and occlusion culling are taskMain's.
 *
 * @param groupID x = meshlet index within the task group's payload.
 * @param groupThreadID x = quad within the meshlet.
 */
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
  const bool visible = active && (quad.frontFacing || drawParams.pass == GEOMETRY_PASS_TRANSLUCENT) &&
                       quadInPass(quad.material);

  // Compact survivors to the front. Wave intrinsics when one subgroup spans the group (NVIDIA wave32, AMD wave64);
  // shared memory for narrower subgroups (e.g. Intel SIMD8/16).
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
// drawIndexedIndirect over a static uint16 index buffer of VERTEX_QUADS_PER_INSTANCE quads ({0,1,2, 0,2,3} + 4q),
// instanced over QuadBuffer (args from finalizeVertexDraw). Indexed so the post-transform cache shares the diagonal:
// 4 invocations per quad, not 6. No GPU culling; ClipmapView::frustumCull only meshes chunks in the frustum.
// Must match VERTEX_QUADS_PER_INSTANCE in renderer.cpp and chunk_compute.slang.
static const uint VERTEX_QUADS_PER_INSTANCE = 16384u;

/**
 * One corner of one quad; culled quads return a NaN position so the primitive is discarded.
 *
 * @param vertexID quad within the instance * 4 + corner.
 * @param instanceID Block of VERTEX_QUADS_PER_INSTANCE quads.
 */
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
  if (!quad.frontFacing && drawParams.pass != GEOMETRY_PASS_TRANSLUCENT) {
    return dead;
  }
  // The GPU clipmap's vertex draw covers all its quads in both passes; each keeps its own (liquids are translucent).
  const bool liquid = (BlockFaces[(quad.material & 0x3ffu) * 6u + quad.normalIndex].flags & BLOCK_FACE_FLUID) != 0u;
  if (liquid != (drawParams.pass == GEOMETRY_PASS_TRANSLUCENT) || !quadInPass(quad.material)) {
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

// Same layout as MeshVertex.
struct FragmentInput {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD3;
  float2 uv : TEXCOORD0;
  nointerpolation uint lighting : TEXCOORD1;
  nointerpolation uint material : TEXCOORD2;
  float viewDepth : TEXCOORD4; // see MeshVertex
  nointerpolation int3 baseCell : TEXCOORD5;
  float worldY : TEXCOORD6;
};

struct SceneFragmentOutput {
  float4 color : SV_Target0;
  float4 motion : SV_Target1;
};

// Water fog color and the distance through water at which it is opaque. Matches the ocean tint in blocks.txt and the
// underwater clear color in renderer.cpp.
static const float3 WATER_FOG_COLOR = float3(0.06f, 0.18f, 0.34f);
static const float WATER_FOG_DISTANCE = 48.0f;

// This frame's opaque depth (farthest MSAA sample per pixel, reverse-Z), resolved by the opaque pass. Only fragWater
// reads it; the opaque pass writes it.
[[vk::binding(8, 0)]]
Texture2D<float> SceneDepth;

/** sRGB transfer function, linear to encoded (per channel). */
float3 linearToSrgb(float3 c) {
  return select(c <= 0.0031308f, c * 12.92f, 1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f);
}
/** sRGB transfer function, encoded to linear (per channel). */
float3 srgbToLinear(float3 c) {
  return select(c <= 0.04045f, c / 12.92f, pow(max((c + 0.055f) / 1.055f, 0.0f), 2.4f));
}

// Vanilla 26.x lightmap, block light 0, sky level 0..15 at noon (sampled from its F3 + 4 debug view; brightness 50%).
static const float SKY_LIGHTMAP[16] = { 24.0f / 255.0f,  34.0f / 255.0f,  44.0f / 255.0f,  56.0f / 255.0f,
                                        67.0f / 255.0f,  80.0f / 255.0f,  94.0f / 255.0f,  108.0f / 255.0f,
                                        123.0f / 255.0f, 139.0f / 255.0f, 155.0f / 255.0f, 172.0f / 255.0f,
                                        190.0f / 255.0f, 210.0f / 255.0f, 231.0f / 255.0f, 1.0f };

/**
 * Vanilla's on-screen brightness for a skylight level, blended linearly between levels like its filtered lightmap.
 *
 * @param level Skylight level, 0..15 (fractional between corners).
 */
float vanillaLightmap(float level) {
  const float clamped = clamp(level, 0.0f, 15.0f);
  const uint low = min(uint(clamped), 14u);
  // Tuned against vanilla screenshots: the brightest levels read ~5% hot, so fade to 95% from level 12 to 15.
  const float topTrim = lerp(1.0f, 0.95f, saturate((clamped - 12.0f) / 3.0f));
  return lerp(SKY_LIGHTMAP[low], SKY_LIGHTMAP[low + 1u], clamped - float(low)) * topTrim;
}

/** View depth (clip w) from a reverse-Z depth (1 = near, 0 = far): w = n f / (n (1 - d) + f d). */
float viewDepthFromDepth(float depth) {
  return ubo.nearPlane * ubo.farPlane / (ubo.nearPlane * (1.0f - depth) + ubo.farPlane * depth);
}

/**
 * Shared body of fragMain, fragCutout and fragWater. The flags are constants at each entry point, so fragMain compiles
 * without water fog or discard.
 *
 * @param waterPass Add water fog behind liquid faces.
 * @param cutout Alpha-test (discard below 0.5).
 * @return lit, fogged color and motion (uv delta to last frame, last frame's depth, this depth).
 */
[ForceInline]
SceneFragmentOutput shadeFragment(FragmentInput input, bool waterPass, bool cutout) {
  // lighting: ao (2 bits per corner) 0-7, skylight (4 bits per corner, same order) 8-23.
  // material: block type 0-9, normal 10-12, greedy length - 1 15-19, LOD 20-22, greedy width 23-26.
  const uint aoBits = input.lighting & 0xffu;
  const uint blockType = input.material & 0x3ffu;
  const uint normalIndex = (input.material >> 10u) & 0x7u;
  const uint faceAxis = normalIndex >> 1u;
  const uint faceSign = normalIndex & 1u;
  const uint greedyLength = ((input.material >> 15u) & 0x1fu) + 1u;
  const uint chunkLod = (input.material >> 20u) & 0x7u;
  const uint greedyWidth = (input.material >> 23u) & 0xfu;
  const BlockFace face = BlockFaces[blockType * 6u + normalIndex];
  const bool useLodTexture = chunkLod > 0u && (face.flags & BLOCK_FACE_HAS_LOD_TEXTURE) != 0u;
  // Frame strips: frame k of a texture lives at its slot + k.
  const uint frameCount = max((face.animation >> (useLodTexture ? 8u : 0u)) & 0xffu, 1u);
  const float frameSeconds = float(face.animation >> 16u) * 0.001f;
  const uint frame = uint(ubo.cameraPosition.w / frameSeconds) % frameCount;
  const uint textureIndex = (useLodTexture ? face.lodTexture : face.texture) + frame;
  // Merged quads cover several cells (uv spans 0..length). Each cell's rotation hashes its block and face direction,
  // so remeshing never changes it.
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

  // Alpha cutout for non-opaque blocks (glass, leaves). Only the cutout pass draws those: a discard anywhere in the
  // opaque pass's shader would turn off early depth testing for all terrain.
  if (cutout && color.a < 0.5f) {
    discard;
  }
  color.rgb *= face.tint.rgb;
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

  // Elevation view (H): height color ramp, alternating 4-block bands, a dark line every 16 blocks. Height is sampled
  // half a block inside the face, so a top face counts as its block's level.
  if (ubo.contourMode != 0u && (face.flags & BLOCK_FACE_FLUID) == 0u) {
    const float faceNormalY = faceAxis == 1u ? (faceSign == 0u ? 1.0f : -1.0f) : 0.0f;
    const float y = input.worldY - 0.5f * faceNormalY;
    const float t = saturate((y + 64.0f) / 704.0f); // -64 .. 640
    const float3 ramp[6] = { float3(0.1f, 0.2f, 0.7f), float3(0.15f, 0.65f, 0.2f), float3(0.85f, 0.85f, 0.15f),
                             float3(0.9f, 0.45f, 0.1f), float3(0.65f, 0.15f, 0.1f), float3(0.97f, 0.97f, 0.97f) };
    const float scaled = t * 5.0f;
    const uint index = min(uint(scaled), 4u);
    float3 elevation = lerp(ramp[index], ramp[index + 1u], scaled - float(index));
    elevation *= (uint(floor(y / 4.0f)) & 1u) != 0u ? 0.82f : 1.0f;
    // Lines: within ~1.5 pixels (or 0.12 blocks on top faces' edges) of a multiple of 16.
    const float lineCoord = y / 16.0f;
    const float lineWidth = max(fwidth(lineCoord) * 1.5f, 0.12f / 16.0f);
    const float distanceToLine = abs(frac(lineCoord + 0.5f) - 0.5f);
    if (distanceToLine < lineWidth) {
      elevation *= 0.25f;
    }
    const float detail = dot(color.rgb, float3(0.299f, 0.587f, 0.114f));
    color.rgb = elevation * (0.75f + 0.5f * detail);
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
  const float aoFactor = lerp(ambientOcclusion, 1.0f, lodAoLightening);
  const uint cornerLight = (input.lighting >> 8u) & 0xffffu;
  const float skyLevel = lerp(lerp(float(cornerLight & 0xfu), float((cornerLight >> 4u) & 0xfu), u),
                              lerp(float((cornerLight >> 12u) & 0xfu), float((cornerLight >> 8u) & 0xfu), u), v);
  const float skyBrightness = vanillaLightmap(skyLevel);
  // Face shade: top 1.0, north / south 0.8, east / west 0.6, bottom 0.5.
  const float shading = faceAxis == 0u ? 0.6f : (faceAxis == 1u ? (faceSign == 0u ? 1.0f : 0.5f) : 0.8f);
  const float2 currentUv = input.pos.xy / ubo.renderSize;
  const float2 previousNdc = input.previousClip.xy / input.previousClip.w;
  const float2 previousUv = float2(previousNdc.x * 0.5f + 0.5f, 0.5f - previousNdc.y * 0.5f);
  // Underwater: short linear water fog (must match the clear colors in renderer.cpp). In air: a weak linear haze
  // reaching AIR_FOG_HAZE at ubo.viewRayScale.w, plus a smoothstep border fog over the last AIR_FOG_BORDER_WIDTH that
  // hides the clipmap's edge.
  const bool underwater = ubo.viewRayScale.z > 0.5f;
  const float2 ndc = currentUv * 2.0f - 1.0f;
  const float cameraDistance = input.viewDepth * length(float3(ndc * ubo.viewRayScale.xy, 1.0f));
  const float AIR_FOG_HAZE = 0.3f;
  const float AIR_FOG_BORDER_WIDTH = 0.15f; // of the border fog's end distance
  const float borderFogEnd = ubo.viewRayScale.w;
  const float haze = AIR_FOG_HAZE * saturate(cameraDistance / borderFogEnd);
  const float border = smoothstep(borderFogEnd * (1.0f - AIR_FOG_BORDER_WIDTH), borderFogEnd, cameraDistance);
  const float fogAmount = underwater ? saturate(cameraDistance / WATER_FOG_DISTANCE)
                                     : 1.0f - (1.0f - haze) * (1.0f - border);
  const float3 skyColor = underwater ? WATER_FOG_COLOR : float3(0.55f, 0.72f, 0.9f);
  SceneFragmentOutput output;
  // Light, AO and face shade multiply the on-screen (sRGB-encoded) color, as in vanilla; color here is linear (sRGB
  // texture, sRGB target), so encode, multiply, decode.
  float3 litColor = srgbToLinear(linearToSrgb(color.rgb) * (skyBrightness * aoFactor * shading));
  litColor *= lerp(float3(1.0f, 1.0f, 1.0f), face.farTint.rgb,
                   saturate((cameraDistance - ubo.farTintStart) / (3.0f * ubo.farTintStart)));
  // From above water: fog the view ray between this liquid face and the terrain behind it. Only drawn water surfaces
  // add it, so dry pits below sea level stay clear.
  if (waterPass && !underwater && (face.flags & BLOCK_FACE_FLUID) != 0u) {
    const float sceneW = viewDepthFromDepth(SceneDepth.Load(int3(int2(input.pos.xy), 0)));
    // View-depth difference to distance along the view ray: distances scale with w along a ray.
    const float submerged = max(sceneW - input.viewDepth, 0.0f) * cameraDistance / max(input.viewDepth, 1e-3f);
    const float waterFog = saturate(submerged / WATER_FOG_DISTANCE);
    // Composite: the texture (alpha) over fogged terrain (fog fraction of the fog color), as straight alpha.
    const float3 premultiplied = color.a * litColor + (1.0f - color.a) * waterFog * WATER_FOG_COLOR;
    color.a = 1.0f - (1.0f - color.a) * (1.0f - waterFog);
    litColor = premultiplied / max(color.a, 1e-4f);
  }
  output.color = float4(lerp(litColor, skyColor, fogAmount), color.a);
  output.motion = float4(previousUv - currentUv, input.previousClip.z / input.previousClip.w, input.pos.z);
  return output;
}

/** Opaque pass: no discard, so early depth testing rejects hidden fragments before shading. */
[shader("fragment")]
[earlydepthstencil]
SceneFragmentOutput fragMain(FragmentInput input) { return shadeFragment(input, false, false); }

/** Cutout pass (glass, leaves): alpha-tested. */
[shader("fragment")]
SceneFragmentOutput fragCutout(FragmentInput input) { return shadeFragment(input, false, true); }

/** Translucent (water) pass: adds water fog behind liquid faces. */
[shader("fragment")]
SceneFragmentOutput fragWater(FragmentInput input) { return shadeFragment(input, true, false); }
