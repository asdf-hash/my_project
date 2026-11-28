// Terrain geometry and shading: mesh path (meshMain, culled by taskMain in task.slang), smooth terrain
// (smoothVertMain / smoothFragMain), and the opaque / cutout / water fragment shaders.

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
// far_tint fades in by camera distance over [ubo.farTintStart, 4 x it), around LOD 3 where canopies become leaf-topped
// ground (vegetation_shared.hlsl). By distance rather than LOD, so no seam.
static const uint BLOCK_FACE_RANDOM_ROTATION = 1u << 0u;
static const uint BLOCK_FACE_HAS_LOD_TEXTURE = 1u << 1u;
static const uint BLOCK_FACE_FLUID = 1u << 2u;
#include "geometry_shared.hlsl"
#include "quad_decode.hlsl"

[[vk::binding(0, 0)]]
StructuredBuffer<uint2> QuadBuffer;
[[vk::binding(3, 0)]]
StructuredBuffer<uint64_t> ChunkOffsets;
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
 * Debug views over a texture color: the source tint (X) and the elevation view (H, a height color ramp with
 * alternating 4-block bands and a dark line every 16 blocks).
 *
 * @param heightY World height the elevation view reads.
 */
void applyDebugViews(inout float3 rgb, BlockFace face, uint chunkLod, float heightY) {
  // Debug view (X): keep the texture's detail but shift it toward the chunk source's color.
  if (drawParams.debugTint != DEBUG_TINT_OFF) {
    const float3 tint = drawParams.debugTint == DEBUG_TINT_CPU
                            ? float3(0.1f, 1.0f, 0.1f)
                            : float3(1.0f, float(chunkLod) / float(MAX_LOD_LEVELS - 1u), 0.05f);
    const float detail = dot(rgb, float3(0.299f, 0.587f, 0.114f));
    rgb = tint * (0.35f + 0.9f * detail);
  }
  if (ubo.contourMode != 0u && (face.flags & BLOCK_FACE_FLUID) == 0u) {
    const float y = heightY;
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
    const float detail = dot(rgb, float3(0.299f, 0.587f, 0.114f));
    rgb = elevation * (0.75f + 0.5f * detail);
  }
}

/**
 * Light a shaded texel and finish it: far tint, air / water fog, the water pass's fog behind liquid faces, and motion.
 *
 * @param color Texture color (linear) with tint and debug views applied; alpha is kept.
 * @param lightFactor Skylight brightness x AO x face shade, applied in display (sRGB) space as vanilla does.
 * @param pos SV_Position of the fragment.
 * @param previousClip Last frame's clip position (TAA motion).
 * @param viewDepth Clip w.
 * @param waterPass Add water fog behind liquid faces.
 */
[ForceInline]
SceneFragmentOutput composeFragment(float4 color, BlockFace face, float lightFactor, float4 pos, float4 previousClip,
                                    float viewDepth, bool waterPass) {
  const float2 currentUv = pos.xy / ubo.renderSize;
  const float2 previousNdc = previousClip.xy / previousClip.w;
  const float2 previousUv = float2(previousNdc.x * 0.5f + 0.5f, 0.5f - previousNdc.y * 0.5f);
  // Underwater: short linear water fog (must match the clear colors in renderer.cpp). In air: a weak linear haze
  // reaching AIR_FOG_HAZE at ubo.viewRayScale.w, plus a smoothstep border fog over the last AIR_FOG_BORDER_WIDTH that
  // hides the clipmap's edge.
  const bool underwater = ubo.viewRayScale.z > 0.5f;
  const float2 ndc = currentUv * 2.0f - 1.0f;
  const float cameraDistance = viewDepth * length(float3(ndc * ubo.viewRayScale.xy, 1.0f));
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
  float3 litColor = srgbToLinear(linearToSrgb(color.rgb) * lightFactor);
  litColor *= lerp(float3(1.0f, 1.0f, 1.0f), face.farTint.rgb,
                   saturate((cameraDistance - ubo.farTintStart) / (3.0f * ubo.farTintStart)));
  // From above water: fog the view ray between this liquid face and the terrain behind it. Only drawn water surfaces
  // add it, so dry pits below sea level stay clear.
  if (waterPass && !underwater && (face.flags & BLOCK_FACE_FLUID) != 0u) {
    const float sceneW = viewDepthFromDepth(SceneDepth.Load(int3(int2(pos.xy), 0)));
    // View-depth difference to distance along the view ray: distances scale with w along a ray.
    const float submerged = max(sceneW - viewDepth, 0.0f) * cameraDistance / max(viewDepth, 1e-3f);
    const float waterFog = saturate(submerged / WATER_FOG_DISTANCE);
    // Composite: the texture (alpha) over fogged terrain (fog fraction of the fog color), as straight alpha.
    const float3 premultiplied = color.a * litColor + (1.0f - color.a) * waterFog * WATER_FOG_COLOR;
    color.a = 1.0f - (1.0f - color.a) * (1.0f - waterFog);
    litColor = premultiplied / max(color.a, 1e-4f);
  }
  output.color = float4(lerp(litColor, skyColor, fogAmount), color.a);
  output.motion = float4(previousUv - currentUv, previousClip.z / previousClip.w, pos.z);
  return output;
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
  {
    const float faceNormalY = faceAxis == 1u ? (faceSign == 0u ? 1.0f : -1.0f) : 0.0f;
    // Height sampled half a block inside the face, so a top face counts as its block's level.
    applyDebugViews(color.rgb, face, chunkLod, input.worldY - 0.5f * faceNormalY);
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
  return composeFragment(color, face, skyBrightness * aoFactor * shading, input.pos, input.previousClip,
                         input.viewDepth, waterPass);
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

// ---- Smooth terrain ---------------------------------------------------------------------------------------------
// The CPU-meshed chunks' opaque faces as Surface Nets quads (cpuworld::SmoothVertex, emitSmooth in cpu_mesher.cpp),
// drawn from a vertex buffer in smooth mode (V). Textures are projected triplanar since faces are no longer
// axis-aligned.

struct SmoothVertexInput {
  // x, y, z: chunk-local position in 1/256 blocks, biased by +256; w: block id | face direction << 10.
  [[vk::location(0)]] uint4 positionMaterial : POSITION;
  // Normal as signed bytes x, y, z (bits 0-23) | skylight 0..15 (26-29).
  [[vk::location(1)]] uint normalLight : NORMAL;
};

struct SmoothVertex {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD3;
  float3 world : TEXCOORD0;
  float3 normal : TEXCOORD1;
  float skyLevel : TEXCOORD5;  // 0..15
  float viewDepth : TEXCOORD4; // clip w
  // Near terrain: block id | face direction << 10. LOD terrain: unused (the block comes from height and slope).
  nointerpolation uint material : TEXCOORD6;
  float dither : TEXCOORD7;            // LOD terrain: layer dither, -1..1
  nointerpolation uint lod : TEXCOORD8; // LOD terrain: tile LOD (0 for near terrain)
};

static const int SMOOTH_FIXED_ONE = 256; // cpuworld::kSmoothFixedOne

/** Signed byte in bits [shift, shift + 8) of `word`, as -1..1. */
float unpackSnorm8(uint word, uint shift) {
  return max(float(int(word << (24u - shift)) >> 24) / 127.0f, -1.0f);
}

[shader("vertex")]
SmoothVertex smoothVertMain(SmoothVertexInput input) {
  // Integer world position in fixed point first, so chunks sharing a corner produce the identical float.
  const int3 fixed = drawParams.smoothOrigin * SMOOTH_FIXED_ONE + int3(input.positionMaterial.xyz) - SMOOTH_FIXED_ONE;
  const float3 world = float3(fixed >> 8) + float3(fixed & 255) / float(SMOOTH_FIXED_ONE);
  SmoothVertex output;
  output.pos = mul(ubo.mvp, float4(world, 1.0f));
  output.previousClip = mul(ubo.previousMvp, float4(world, 1.0f));
  output.world = world;
  output.normal = float3(unpackSnorm8(input.normalLight, 0u), unpackSnorm8(input.normalLight, 8u),
                         unpackSnorm8(input.normalLight, 16u));
  output.skyLevel = float((input.normalLight >> 26u) & 0xfu);
  output.viewDepth = output.pos.w;
  output.material = input.positionMaterial.w;
  output.dither = 0.0f;
  output.lod = 0u;
  return output;
}

/**
 * One triplanar projection of a block face texture, with the mesh path's per-block random rotation.
 *
 * @param uv Projected world coordinates (blocks); the integer part picks the texture cell.
 * @param uvDx, uvDy Screen derivatives of uv.
 * @param depthCell Integer coordinate along the projection axis, so stacked blocks rotate independently.
 * @param normalIndex Face direction whose texture is used, 0..5.
 */
float4 sampleProjection(BlockFace face, float2 uv, float2 uvDx, float2 uvDy, int depthCell, uint normalIndex,
                        uint chunkLod) {
  const bool useLodTexture = chunkLod > 0u && (face.flags & BLOCK_FACE_HAS_LOD_TEXTURE) != 0u;
  const uint frameCount = max((face.animation >> (useLodTexture ? 8u : 0u)) & 0xffu, 1u);
  const float frameSeconds = float(face.animation >> 16u) * 0.001f;
  const uint frame = uint(ubo.cameraPosition.w / frameSeconds) % frameCount;
  const uint textureIndex = (useLodTexture ? face.lodTexture : face.texture) + frame;
  const float2 cell = floor(uv);
  uint rotation = 0u;
  if ((face.flags & BLOCK_FACE_RANDOM_ROTATION) != 0u) {
    uint h = uint(int(cell.x)) * 0x8da6b343u ^ uint(int(cell.y)) * 0xd8163841u ^ uint(depthCell) * 0xcb1ab31fu ^
             normalIndex * 0x165667b1u;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    h ^= h >> 15u;
    rotation = h & 3u;
  }
  float2 textureUv = uv - cell;
  if (rotation != 0u) {
    textureUv = rotateUv(textureUv, rotation);
    uvDx = rotateUvGradient(uvDx, rotation);
    uvDy = rotateUvGradient(uvDy, rotation);
  }
  textureUv.y = 1.0f - textureUv.y;
  uvDx.y = -uvDx.y;
  uvDy.y = -uvDy.y;
  const float gradientScale = exp2(-0.5f); // mip sharpening, as in shadeFragment
  float4 texel = textures[NonUniformResourceIndex(textureIndex)].SampleGrad(sampler, textureUv, uvDx * gradientScale,
                                                                            uvDy * gradientScale);
  texel.rgb *= face.tint.rgb;
  if (useLodTexture) {
    texel.rgb *= face.lodTint.rgb;
  }
  return texel;
}

/**
 * Shade a smooth surface: triplanar textures of block `blockType` (its top texture on gentle slopes, so they read as
 * ground), interpolated skylight, face shade blended by the normal.
 *
 * @param cutout Alpha-test (discard below 0.5): the foliage surface.
 */
[ForceInline]
SceneFragmentOutput shadeSmooth(SmoothVertex input, uint blockType, bool cutout) {
  const float3 n = normalize(input.normal);
  // Sharp blend: mostly one projection, mixing only near 45 degrees.
  float3 weights = pow(abs(n), 4.0f);
  weights /= weights.x + weights.y + weights.z;
  const bool groundLike = n.y > 0.5f;
  const uint xFace = groundLike ? 2u : (n.x >= 0.0f ? 0u : 1u);
  const uint yFace = n.y >= 0.0f ? 2u : 3u;
  const uint zFace = groundLike ? 2u : (n.z >= 0.0f ? 4u : 5u);
  // Projections (u, v) matching the mesh path's faces: v up on sides, -z on top, +z on the bottom.
  const float2 uvX = float2(input.world.z, input.world.y);
  const float2 uvY = float2(input.world.x, n.y >= 0.0f ? -input.world.z : input.world.z);
  const float2 uvZ = float2(input.world.x, input.world.y);
  // Derivatives in uniform control flow, before the weighted branches below.
  const float2 uvXDx = ddx(uvX), uvXDy = ddy(uvX);
  const float2 uvYDx = ddx(uvY), uvYDy = ddy(uvY);
  const float2 uvZDx = ddx(uvZ), uvZDy = ddy(uvZ);
  // Cell along each projection axis, taken half a block inside the surface.
  const int3 inside = int3(floor(input.world - 0.5f * n));
  float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);
  BlockFace dominant = BlockFaces[blockType * 6u + yFace];
  float dominantWeight = -1.0f;
  [unroll]
  for (uint axis = 0u; axis < 3u; ++axis) {
    const float weight = weights[axis];
    const uint faceIndex = axis == 0u ? xFace : (axis == 1u ? yFace : zFace);
    const BlockFace face = BlockFaces[blockType * 6u + faceIndex];
    if (weight > dominantWeight) {
      dominant = face;
      dominantWeight = weight;
    }
    if (weight > 0.001f) {
      const float4 texel =
          axis == 0u ? sampleProjection(face, uvX, uvXDx, uvXDy, inside.x, faceIndex, input.lod)
                     : (axis == 1u ? sampleProjection(face, uvY, uvYDx, uvYDy, inside.y, faceIndex, input.lod)
                                   : sampleProjection(face, uvZ, uvZDx, uvZDy, inside.z, faceIndex, input.lod));
      color += texel * weight;
    }
  }
  if (cutout && color.a < 0.5f) {
    discard;
  }
  color.a = 1.0f;
  applyDebugViews(color.rgb, dominant, input.lod, input.world.y - 0.5f * n.y);

  const float skyBrightness = vanillaLightmap(input.skyLevel);
  // The block faces' shade (top 1.0, north / south 0.8, east / west 0.6, bottom 0.5), blended by the normal.
  const float3 n2 = n * n;
  const float shading = n2.x * 0.6f + n2.z * 0.8f + n2.y * (n.y >= 0.0f ? 1.0f : 0.5f);
  return composeFragment(color, dominant, skyBrightness * shading, input.pos, input.previousClip, input.viewDepth,
                         false);
}

/** Opaque smooth near terrain. */
[shader("fragment")]
[earlydepthstencil]
SceneFragmentOutput smoothFragMain(SmoothVertex input) { return shadeSmooth(input, input.material & 0x3ffu, false); }

/** Smooth foliage (leaves, glass): alpha-tested. */
[shader("fragment")]
SceneFragmentOutput smoothFragCutout(SmoothVertex input) { return shadeSmooth(input, input.material & 0x3ffu, true); }

// ---- Smooth LOD terrain -----------------------------------------------------------------------------------------
// The GPU clipmap's columns as heightfields: per tile a grid over its 33 x 33 column corners (SmoothHeights, from
// generateHeights) plus a skirt ring hanging below the edges, which hides cracks against coarser or finer
// neighbors. Heights follow the CPU's smooth convention: a column's float surface h sits at h - 0.5, half a block
// into the column (the CPU places it at the cell center).

[[vk::binding(4, 0)]]
StructuredBuffer<float2> SmoothHeights;

static const uint LOD_GRID_SIDE = 35u;          // renderer.cpp LOD_GRID_SIDE: 33 corners + the skirt ring
static const uint SMOOTH_HEIGHT_STRIDE = 34u;   // padded column grid (chunk_compute.slang HEIGHT_STRIDE)
// Layer tops and their dither (chunk_compute.slang TRANSITION_HEIGHTS / TRANSITION_DITHER).
static const int SMOOTH_TRANSITION_HEIGHTS[3] = { 2, 320, 600 };
static const float SMOOTH_TRANSITION_DITHER[3] = { 0.0f, 40.0f, 40.0f };

/** SmoothHeights entry of padded grid point (l, m) (0..33 each: corner l - 1) of column slot `slot`. */
float2 smoothHeightAt(uint slot, uint l, uint m) {
  return SmoothHeights[slot * SMOOTH_HEIGHT_STRIDE * SMOOTH_HEIGHT_STRIDE + m * SMOOTH_HEIGHT_STRIDE + l];
}

/**
 * One grid vertex of one tile.
 *
 * @param tile Instance: column slot, tile origin x / z (blocks), scale (blocks per column).
 */
[shader("vertex")]
SmoothVertex smoothLodVertMain(uint vertexID: SV_VertexID, [[vk::location(0)]] int4 tile: TILE) {
  const uint slot = uint(tile.x);
  const int scale = tile.w;
  const uint i = vertexID % LOD_GRID_SIDE;
  const uint j = vertexID / LOD_GRID_SIDE;
  const bool skirt = i == 0u || j == 0u || i == LOD_GRID_SIDE - 1u || j == LOD_GRID_SIDE - 1u;
  // Corner 0..32 along each axis; its padded grid point is corner + 1.
  const uint cx = clamp(i, 1u, LOD_GRID_SIDE - 2u) - 1u;
  const uint cz = clamp(j, 1u, LOD_GRID_SIDE - 2u) - 1u;
  const float2 sample = smoothHeightAt(slot, cx + 1u, cz + 1u);
  // Normal from central differences (one-sided at the far edge, which has no padded point beyond).
  const uint left = cx, right = min(cx + 2u, SMOOTH_HEIGHT_STRIDE - 1u);
  const uint back = cz, front = min(cz + 2u, SMOOTH_HEIGHT_STRIDE - 1u);
  const float dx = (smoothHeightAt(slot, right, cz + 1u).x - smoothHeightAt(slot, left, cz + 1u).x) /
                   (float(right - left) * float(scale));
  const float dz = (smoothHeightAt(slot, cx + 1u, front).x - smoothHeightAt(slot, cx + 1u, back).x) /
                   (float(front - back) * float(scale));
  // Skirts drop far enough to cover the worst crack against a neighbor LOD on steep ground.
  const float skirtDepth = skirt ? 4.0f + 4.0f * float(scale) : 0.0f;
  const float3 world = float3(float(tile.y + int(cx) * scale) + 0.5f, sample.x - 0.5f - skirtDepth,
                              float(tile.z + int(cz) * scale) + 0.5f);
  SmoothVertex output;
  output.pos = mul(ubo.mvp, float4(world, 1.0f));
  output.previousClip = mul(ubo.previousMvp, float4(world, 1.0f));
  output.world = world;
  output.normal = normalize(float3(-dx, 1.0f, -dz));
  output.skyLevel = 15.0f;
  output.viewDepth = output.pos.w;
  output.material = 0u;
  output.dither = sample.y;
  output.lod = firstbitlow(uint(scale));
  return output;
}

/** LOD terrain: the layer's top block, its filler on steep ground and its deep block on cliffs (by the normal). */
[shader("fragment")]
[earlydepthstencil]
SceneFragmentOutput smoothLodFragMain(SmoothVertex input) {
  uint layer = 0u;
  [unroll]
  for (uint t = 0u; t < 3u; ++t) {
    layer += uint(input.world.y + 0.5f >=
                  float(SMOOTH_TRANSITION_HEIGHTS[t]) + round(input.dither * SMOOTH_TRANSITION_DITHER[t]));
  }
  const float up = normalize(input.normal).y;
  const uint blockType = up > 0.8f ? drawParams.layerBlocks[layer]
                                   : (up > 0.6f ? drawParams.layerFillerBlocks[layer] : drawParams.layerDeepBlocks[layer]);
  return shadeSmooth(input, blockType, false);
}
