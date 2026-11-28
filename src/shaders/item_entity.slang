// Dropped item entities in the opaque scene pass: one textured cube per instance (36 vertices) in its rigid body's
// orientation, lit by the skylight at the body and the terrain's face shading (by world normal), fogged like
// terrain, with motion vectors for TAA.

#include "scene_ubo.hlsl"

// Bindless block textures (renderer.cpp createTextureImages).
[[vk::binding(11, 0)]]
Texture2D textures[];

// Must match ItemInstance in renderer.cpp (instance-rate vertex attributes).
struct ItemInstance {
  [[vk::location(0)]] float4 center : CENTER;             // xyz: cube center (world), w: edge length
  [[vk::location(1)]] float4 previousCenter : PREVIOUS;   // xyz: last frame's center
  [[vk::location(2)]] float4 rotation : ROTATION;         // quaternion (x, y, z, w)
  [[vk::location(3)]] float4 previousRotation : PREVROT;  // last frame's
  [[vk::location(4)]] uint4 faces : FACES;                // top, side, bottom texture slots; tint RGBA8 (R low)
  [[vk::location(5)]] float4 extra : EXTRA;               // x: skylight 0..15
};

struct ItemVertex {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD0;
  float2 uv : TEXCOORD1;
  nointerpolation uint texture : TEXCOORD2;
  nointerpolation float light : TEXCOORD3; // skylight brightness x face shade
  nointerpolation float3 tint : TEXCOORD4;
  float viewDepth : TEXCOORD5;
};

// Must match slang.slang (vanilla lightmap and fog).
static const float SKY_LIGHTMAP[16] = { 24.0f / 255.0f,  34.0f / 255.0f,  44.0f / 255.0f,  56.0f / 255.0f,
                                        67.0f / 255.0f,  80.0f / 255.0f,  94.0f / 255.0f,  108.0f / 255.0f,
                                        123.0f / 255.0f, 139.0f / 255.0f, 155.0f / 255.0f, 172.0f / 255.0f,
                                        190.0f / 255.0f, 210.0f / 255.0f, 231.0f / 255.0f, 1.0f };
static const float3 WATER_FOG_COLOR = float3(0.06f, 0.18f, 0.34f);
static const float WATER_FOG_DISTANCE = 48.0f;

float vanillaLightmap(float level) {
  const float clamped = clamp(level, 0.0f, 15.0f);
  const uint low = min(uint(clamped), 14u);
  const float topTrim = lerp(1.0f, 0.95f, saturate((clamped - 12.0f) / 3.0f));
  return lerp(SKY_LIGHTMAP[low], SKY_LIGHTMAP[low + 1u], clamped - float(low)) * topTrim;
}
float3 linearToSrgb(float3 c) {
  return select(c <= 0.0031308f, c * 12.92f, 1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f);
}
float3 srgbToLinear(float3 c) {
  return select(c <= 0.04045f, c / 12.92f, pow(max((c + 0.055f) / 1.055f, 0.0f), 2.4f));
}

// Faces: +x, -x, +y, -y, +z, -z; per face the 4 corners (unit cube, -0.5..0.5) counter-clockwise from outside, with
// the texture's bottom-left first.
static const float3 FACE_NORMALS[6] = {float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0),
                                       float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1)};
static const float3 FACE_U[6] = {float3(0, 0, -1), float3(0, 0, 1), float3(1, 0, 0),
                                 float3(1, 0, 0), float3(1, 0, 0), float3(-1, 0, 0)};
static const float3 FACE_V[6] = {float3(0, 1, 0), float3(0, 1, 0), float3(0, 0, -1),
                                 float3(0, 0, 1), float3(0, 1, 0), float3(0, 1, 0)};

/** Rotate p by unit quaternion q (x, y, z, w). */
float3 rotate(float4 q, float3 p) {
  const float3 t = 2.0f * cross(q.xyz, p);
  return p + q.w * t + cross(q.xyz, t);
}

[shader("vertex")]
ItemVertex itemVertexMain(ItemInstance instance, uint vertexID: SV_VertexID) {
  const uint face = vertexID / 6u;
  const uint local = vertexID % 6u;
  const uint corner = local < 3u ? local : local - 2u; // 0 1 2, 1 2 3
  const float2 unit = float2(float(corner & 1u), float(corner >> 1u));
  const float3 normal = FACE_NORMALS[face];
  const float3 cubePoint = 0.5f * normal + (unit.x - 0.5f) * FACE_U[face] + (unit.y - 0.5f) * FACE_V[face];
  const float size = instance.center.w;
  const float3 world = instance.center.xyz + rotate(instance.rotation, cubePoint) * size;
  const float3 previousWorld = instance.previousCenter.xyz + rotate(instance.previousRotation, cubePoint) * size;

  ItemVertex output;
  output.pos = mul(ubo.mvp, float4(world, 1.0f));
  output.previousClip = mul(ubo.previousMvp, float4(previousWorld, 1.0f));
  output.uv = float2(unit.x, 1.0f - unit.y);
  output.texture = face == 2u ? instance.faces.x : (face == 3u ? instance.faces.z : instance.faces.y);
  // Terrain face shade on the world normal (top 1.0, bottom 0.5, x 0.6, z 0.8), blended while tumbling.
  const float3 n = rotate(instance.rotation, normal);
  const float shade = n.x * n.x * 0.6f + n.z * n.z * 0.8f + n.y * n.y * (n.y > 0.0f ? 1.0f : 0.5f);
  output.light = vanillaLightmap(instance.extra.x) * shade;
  const uint tint = instance.faces.w;
  output.tint = float3(float(tint & 0xffu), float((tint >> 8u) & 0xffu), float((tint >> 16u) & 0xffu)) / 255.0f;
  output.viewDepth = output.pos.w;
  return output;
}

struct ItemFragmentOutput {
  float4 color : SV_Target0;
  float4 motion : SV_Target1;
};

[shader("fragment")]
ItemFragmentOutput itemFragmentMain(ItemVertex input) {
  const int2 texel = clamp(int2(floor(input.uv * 16.0f)), int2(0, 0), int2(15, 15));
  float4 color = textures[NonUniformResourceIndex(input.texture)].Load(int3(texel, 0));
  if (color.a < 0.5f) {
    discard; // cutout (leaves)
  }
  color.rgb *= input.tint;
  float3 lit = srgbToLinear(linearToSrgb(color.rgb) * input.light);

  // Fog like slang.slang composeFragment (no water-surface term: items are drawn in the opaque pass).
  const float2 currentUv = input.pos.xy / ubo.renderSize;
  const bool underwater = ubo.viewRayScale.z > 0.5f;
  const float2 ndc = currentUv * 2.0f - 1.0f;
  const float cameraDistance = input.viewDepth * length(float3(ndc * ubo.viewRayScale.xy, 1.0f));
  const float borderFogEnd = ubo.viewRayScale.w;
  const float haze = 0.3f * saturate(cameraDistance / borderFogEnd);
  const float border = smoothstep(borderFogEnd * 0.85f, borderFogEnd, cameraDistance);
  const float fogAmount = underwater ? saturate(cameraDistance / WATER_FOG_DISTANCE)
                                     : 1.0f - (1.0f - haze) * (1.0f - border);
  const float3 skyColor = underwater ? WATER_FOG_COLOR : float3(0.55f, 0.72f, 0.9f);

  const float2 previousNdc = input.previousClip.xy / input.previousClip.w;
  const float2 previousUv = float2(previousNdc.x * 0.5f + 0.5f, 0.5f - previousNdc.y * 0.5f);
  ItemFragmentOutput output;
  output.color = float4(lerp(lit, skyColor, fogAmount), 1.0f);
  output.motion = float4(previousUv - currentUv, input.previousClip.z / input.previousClip.w, input.pos.z);
  return output;
}
