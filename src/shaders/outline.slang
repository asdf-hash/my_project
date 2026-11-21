// Outline of the targeted block, drawn last in the scene pass (MSAA, jittered like the terrain, so TAA smooths it)
// and depth-tested (no writes), so edges behind the block or buried in the terrain stay hidden. Each of the 12 box
// edges is a screen-space strip (lines wider than 1 px aren't guaranteed in Vulkan). It writes motion vectors like
// fragMain, so TAA reprojects it instead of smearing it while the camera moves.

#include "scene_ubo.hlsl"

// Matches OutlineParams in renderer.cpp.
struct OutlineParams {
  float4 boxMin; // xyz: world-space box corner; w: line width in pixels
  float4 boxMax; // xyz: opposite corner
};

[[vk::push_constant]]
ConstantBuffer<OutlineParams> params;

// Corner bits: 1 = x max, 2 = y max, 4 = z max.
static const uint2 EDGES[12] = {
  uint2(0u, 1u), uint2(2u, 3u), uint2(4u, 5u), uint2(6u, 7u), // along x
  uint2(0u, 2u), uint2(1u, 3u), uint2(4u, 6u), uint2(5u, 7u), // along y
  uint2(0u, 4u), uint2(1u, 5u), uint2(2u, 6u), uint2(3u, 7u), // along z
};

// Clip-space w is the view depth; keep strips in front of the 0.1 near plane.
static const float NEAR_W = 0.12f;
// Pulls the outline toward the camera so it wins the depth test against the faces it lies on.
static const float DEPTH_PULL = 0.995f;

struct OutlineVertex {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD0;
};

float3 corner(uint index) {
  const float3 p = float3((index & 1u) != 0u ? params.boxMax.x : params.boxMin.x,
                          (index & 2u) != 0u ? params.boxMax.y : params.boxMin.y,
                          (index & 4u) != 0u ? params.boxMax.z : params.boxMin.z);
  const float3 camera = ubo.cameraPosition.xyz;
  return camera + (p - camera) * DEPTH_PULL;
}

[shader("vertex")]
OutlineVertex outlineVertexMain(uint vertexID: SV_VertexID) {
  const uint2 edge = EDGES[vertexID / 6u];
  const uint local = vertexID % 6u;
  const uint quadCorner = local < 3u ? local : local - 2u; // 0 1 2, 1 2 3
  const bool atEnd = (quadCorner & 1u) != 0u;
  const float side = (quadCorner & 2u) != 0u ? 1.0f : -1.0f;

  const float3 worldA = corner(edge.x);
  const float3 worldB = corner(edge.y);
  float4 a = mul(ubo.mvp, float4(worldA, 1.0f));
  float4 b = mul(ubo.mvp, float4(worldB, 1.0f));
  float4 previousA = mul(ubo.previousMvp, float4(worldA, 1.0f));
  float4 previousB = mul(ubo.previousMvp, float4(worldB, 1.0f));
  OutlineVertex output;
  if (a.w < NEAR_W && b.w < NEAR_W) {
    output.pos = float4(0.0f, 0.0f, -1.0f, 1.0f); // behind the camera: outside the depth range, culled
    output.previousClip = output.pos;
    return output;
  }
  // Clip the segment to the near plane (the previous-frame ends move along with it).
  if (a.w < NEAR_W) {
    const float t = (NEAR_W - a.w) / (b.w - a.w);
    a = lerp(a, b, t);
    previousA = lerp(previousA, previousB, t);
  } else if (b.w < NEAR_W) {
    const float t = (NEAR_W - b.w) / (a.w - b.w);
    b = lerp(b, a, t);
    previousB = lerp(previousB, previousA, t);
  }

  const float2 halfScreen = 0.5f * ubo.renderSize;
  const float2 pa = a.xy / a.w * halfScreen;
  const float2 pb = b.xy / b.w * halfScreen;
  const float2 delta = pb - pa;
  const float2 dir = dot(delta, delta) > 1e-6f ? normalize(delta) : float2(1.0f, 0.0f);
  const float2 normal = float2(-dir.y, dir.x);
  const float halfWidth = 0.5f * params.boxMin.w;
  // Extend past both ends by half the width so neighboring edges meet in a square corner.
  const float2 offsetNdc = (normal * side * halfWidth + dir * (atEnd ? halfWidth : -halfWidth)) / halfScreen;
  const float4 p = atEnd ? b : a;
  const float4 previous = atEnd ? previousB : previousA;
  output.pos = float4(p.xy + offsetNdc * p.w, p.z, p.w);
  // Same screen-space offset last frame: the strip's motion is its edge's motion.
  output.previousClip = float4(previous.xy + offsetNdc * previous.w, previous.z, previous.w);
  return output;
}

struct OutlineFragmentOutput {
  float4 color : SV_Target0;
  float4 motion : SV_Target1;
};

[shader("fragment")]
OutlineFragmentOutput outlineFragmentMain(OutlineVertex input) {
  // Motion as in fragMain (slang.slang).
  const float2 currentUv = input.pos.xy / ubo.renderSize;
  const float2 previousNdc = input.previousClip.xy / input.previousClip.w;
  const float2 previousUv = float2(previousNdc.x * 0.5f + 0.5f, 0.5f - previousNdc.y * 0.5f);
  OutlineFragmentOutput output;
  output.color = float4(0.0f, 0.0f, 0.0f, 0.8f);
  output.motion = float4(previousUv - currentUv, input.previousClip.z / input.previousClip.w, input.pos.z);
  return output;
}
