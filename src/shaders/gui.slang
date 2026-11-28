// 2D GUI: one instanced draw of quads built on the CPU (gui.hpp Batch), 6 vertices per quad. Kinds: sprite (a texture
// region, tinted, optionally tiled), solid fill, and isometric block icon. Colors multiply in sRGB space like
// vanilla; the swapchain is sRGB, so the result is converted back to linear.

// Bindless textures (block faces, GUI sprites, font); renderer.cpp createTextureImages.
[[vk::binding(11, 0)]]
Texture2D textures[];

// First bytes of OverlayParams in renderer.cpp (the GUI shares its pipeline layout).
struct GuiParams {
  float2 screenSize;
};

[[vk::push_constant]]
ConstantBuffer<GuiParams> params;

// Must match gui::Quad / gui::QuadKind in gui.hpp.
static const uint KIND_SPRITE = 0u;
static const uint KIND_FILL = 1u;
static const uint KIND_BLOCK_ICON = 2u;

struct QuadInput {
  [[vk::location(0)]] float4 rect : RECT;     // screen pixels: x, y, w, h
  [[vk::location(1)]] float4 source : SOURCE; // texels: x, y, w, h
  [[vk::location(2)]] float4 extent : EXTENT; // xy: texels spanned
  [[vk::location(3)]] uint4 data : DATA;      // texture, color RGBA8, kind, extra
};

struct GuiVertex {
  float4 position : SV_Position;
  float2 local : TEXCOORD0; // 0..1 across the quad
  nointerpolation float4 source : TEXCOORD1;
  nointerpolation float2 extent : TEXCOORD2;
  nointerpolation uint4 data : TEXCOORD3;
};

/** sRGB transfer function, linear to encoded. */
float3 linearToSrgb(float3 c) {
  return select(c <= 0.0031308f, c * 12.92f, 1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f);
}
/** sRGB transfer function, encoded to linear. */
float3 srgbToLinear(float3 c) {
  return select(c <= 0.04045f, c / 12.92f, pow(max((c + 0.055f) / 1.055f, 0.0f), 2.4f));
}

/** RGBA8 (R in the low byte) to 0..1. */
float4 unpackColor(uint bits) {
  return float4(float(bits & 0xffu), float((bits >> 8u) & 0xffu), float((bits >> 16u) & 0xffu), float(bits >> 24u)) /
         255.0f;
}

[shader("vertex")]
GuiVertex guiVertexMain(QuadInput quad, uint vertexID: SV_VertexID) {
  const uint corner = vertexID < 3u ? vertexID : vertexID - 2u; // 0 1 2, 1 2 3
  const float2 unit = float2(float(corner & 1u), float(corner >> 1u));
  const float2 pixel = quad.rect.xy + unit * quad.rect.zw;
  GuiVertex output;
  output.position = float4(pixel / params.screenSize * 2.0f - 1.0f, 0.0f, 1.0f);
  output.local = unit;
  output.source = quad.source;
  output.extent = quad.extent.xy;
  output.data = quad.data;
  return output;
}

/**
 * A block icon like vanilla's GUI item: an orthographic cube seen 30 degrees from above at a 45 degree yaw, 10 GUI
 * pixels per block, its top face full bright and its left / right faces at 0.64 / 0.41 (display space).
 *
 * @param local Position relative to the icon center, in GUI pixels (y down).
 * @return the icon color (linear, straight alpha), transparent where the cube is missed or the texture is cut out.
 */
float4 blockIcon(float2 local, uint topTexture, uint sideTexture, float3 tint) {
  // Camera direction (toward the viewer), screen right and up, in block space.
  const float3 toCamera = float3(0.6123724f, 0.5f, 0.6123724f);
  const float3 right = float3(0.7071068f, 0.0f, -0.7071068f);
  const float3 up = float3(-0.3535534f, 0.8660254f, -0.3535534f);
  const float3 origin = (local.x * right - local.y * up) / 10.0f + 4.0f * toCamera;
  const float3 dir = -toCamera;
  const float3 tLow = (-0.5f - origin) / dir;
  const float3 tHigh = (0.5f - origin) / dir;
  const float3 tNear3 = min(tLow, tHigh);
  const float3 tFar3 = max(tLow, tHigh);
  const float tNear = max(tNear3.x, max(tNear3.y, tNear3.z));
  const float tFar = min(tFar3.x, min(tFar3.y, tFar3.z));
  if (tNear > tFar) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const float3 hit = origin + dir * tNear;
  float2 uv;
  uint texture;
  float shade;
  if (tNear == tNear3.y) { // top (+y)
    uv = float2(hit.x + 0.5f, hit.z + 0.5f);
    texture = topTexture;
    shade = 1.0f;
  } else if (tNear == tNear3.z) { // +z: the left face on screen
    uv = float2(hit.x + 0.5f, 0.5f - hit.y);
    texture = sideTexture;
    shade = 0.64f;
  } else { // +x: the right face
    uv = float2(0.5f - hit.z, 0.5f - hit.y);
    texture = sideTexture;
    shade = 0.41f;
  }
  const int2 texel = clamp(int2(floor(uv * 16.0f)), int2(0, 0), int2(15, 15));
  const float4 color = textures[NonUniformResourceIndex(texture)].Load(int3(texel, 0));
  if (color.a < 0.5f) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  return float4(srgbToLinear(linearToSrgb(color.rgb) * tint * shade), 1.0f);
}

[shader("fragment")]
float4 guiFragmentMain(GuiVertex input) : SV_Target0 {
  const float4 tint = unpackColor(input.data.y);
  const uint kind = input.data.z;
  if (kind == KIND_FILL) {
    return float4(srgbToLinear(tint.rgb), tint.a);
  }
  if (kind == KIND_BLOCK_ICON) {
    return blockIcon((input.local - 0.5f) * 16.0f, input.data.x, input.data.w, tint.rgb);
  }
  // Sprite: texel within the source region; extent > source size repeats it (tiling).
  const float2 spanned = min(floor(input.local * input.extent), input.extent - 1.0f);
  const int2 texel = int2(input.source.xy + fmod(spanned, input.source.zw));
  const float4 color = textures[NonUniformResourceIndex(input.data.x)].Load(int3(texel, 0));
  return float4(srgbToLinear(linearToSrgb(color.rgb) * tint.rgb), color.a * tint.a);
}
