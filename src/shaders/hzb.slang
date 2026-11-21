// Hierarchical depth (Hi-Z) for the task shader's occlusion test. Every texel holds the FARTHEST depth of the
// screen area it covers (depth 0 = near, 1 = far, compare LESS). Mip 0 is half the next power of two of the
// screen, so every level after it is an exact 2x2 reduction. Built after the scene pass of the mesh path; the next
// frame's taskMain reads it together with ubo.previousMvp, the matrices it was rendered with.
//
// Two dispatches build the whole pyramid, with one barrier between them:
//   hzbBuildLow:  one 16x16 group per 16x16 block of mip 0, one thread per texel: depth -> mips 0..4.
// The depth is the scene pass's single-sample MAX resolve (renderer.cpp, resolvedDepthImages).
//   hzbBuildHigh: a single group: mip 4 -> mips 5..last.
// Both reduce in shared memory. Depth is >= 0, so 0 is the identity for max: texels outside a level (non-square or
// tiny pyramids) read as 0.

static const uint MAX_HZB_MIPS = 16u;  // MAX_HZB_MIPS in renderer.cpp
static const uint LOW_MIPS = 5u;       // mips written by hzbBuildLow (HZB_LOW_MIPS in renderer.cpp)
static const uint HIGH_SIDE = 64u;     // largest mip 5 hzbBuildHigh holds (mip 0 up to 2048, HZB_MAX_MIP0_SIDE)

struct HzbParams {
  uint2 mip0Size;
  uint2 depthSize;
  uint mipCount;
};

[[vk::push_constant]]
ConstantBuffer<HzbParams> params;

// Single-sample MAX resolve of the scene's MSAA depth: already the farthest sample of each pixel.
[[vk::binding(0, 0)]]
Texture2D<float> Depth;
[[vk::binding(1, 0)]]
RWTexture2D<float> Mips[MAX_HZB_MIPS];

uint2 mipSize(uint mip) { return max(params.mip0Size >> mip, uint2(1u, 1u)); }

float farthestSample(uint2 pixel) { return Depth.Load(int3(int2(pixel), 0)); }

// Farthest depth over the depth pixels covered by mip 0 texel `texel`, rounding outward so nothing is missed.
float farthestDepth(uint2 texel) {
  const uint2 first = texel * params.depthSize / params.mip0Size;
  const uint2 last =
      min(((texel + 1u) * params.depthSize + params.mip0Size - 1u) / params.mip0Size, params.depthSize) - 1u;
  float farthest = 0.0f;
  if (all(last - first < 3u)) {
    // Usual case: the screen is at most twice mip 0, so a footprint spans at most 3x3 pixels. Fixed bounds let
    // every load issue at once; clamped repeats hit the cache.
    [unroll]
    for (uint y = 0u; y < 3u; ++y) {
      [unroll]
      for (uint x = 0u; x < 3u; ++x) {
        farthest = max(farthest, farthestSample(min(first + uint2(x, y), last)));
      }
    }
  } else {
    // Mip 0 clamped to HZB_MAX_MIP0_SIDE on a very large screen.
    for (uint y = first.y; y <= last.y; ++y) {
      for (uint x = first.x; x <= last.x; ++x) {
        farthest = max(farthest, farthestSample(uint2(x, y)));
      }
    }
  }
  return farthest;
}

groupshared float lowA[16 * 16];
groupshared float lowB[8 * 8];

[shader("compute")]
[numthreads(16, 16, 1)]
void hzbBuildLow(uint3 groupID: SV_GroupID, uint3 threadID: SV_GroupThreadID) {
  const uint2 t = threadID.xy;
  const uint2 texel0 = groupID.xy * 16u + t;
  float m0 = 0.0f;
  if (all(texel0 < params.mip0Size)) {
    m0 = farthestDepth(texel0);
    Mips[0][texel0] = m0;
  }
  lowA[t.y * 16u + t.x] = m0;
  GroupMemoryBarrierWithGroupSync();

  // Mips 1..4 ping-pong between lowA (16x16 source, 4x4) and lowB (8x8, 2x2); row stride = the level's side.
  // Mip indices stay compile-time constants (unrolled), so the image array needs no dynamic indexing here.
  uint side = 8u;
  [unroll]
  for (uint mip = 1u; mip < LOW_MIPS; ++mip) {
    if (all(t < side)) {
      const uint srcSide = side * 2u;
      const uint i = t.y * 2u * srcSide + t.x * 2u;
      float v;
      if ((mip & 1u) != 0u) {
        v = max(max(lowA[i], lowA[i + 1u]), max(lowA[i + srcSide], lowA[i + srcSide + 1u]));
        lowB[t.y * side + t.x] = v;
      } else {
        v = max(max(lowB[i], lowB[i + 1u]), max(lowB[i + srcSide], lowB[i + srcSide + 1u]));
        lowA[t.y * side + t.x] = v;
      }
      const uint2 texel = groupID.xy * side + t;
      if (mip < params.mipCount && all(texel < mipSize(mip))) {
        Mips[mip][texel] = v;
      }
    }
    GroupMemoryBarrierWithGroupSync();
    side >>= 1u;
  }
}

// Level `level` (>= LOW_MIPS) lives in highA when odd, highB when even, row stride = its width.
groupshared float highA[HIGH_SIDE * HIGH_SIDE];
groupshared float highB[(HIGH_SIDE / 2u) * (HIGH_SIDE / 2u)];

float highRead(uint level, uint2 texel) {
  const uint2 size = mipSize(level);
  if (any(texel >= size)) {
    return 0.0f;
  }
  const uint i = texel.y * size.x + texel.x;
  return (level & 1u) != 0u ? highA[i] : highB[i];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void hzbBuildHigh(uint3 threadID: SV_GroupThreadID) {
  const uint lane = threadID.x;
  // First high mip from the last low one, read back from the image hzbBuildLow wrote.
  const uint2 size = mipSize(LOW_MIPS);
  const uint2 srcSize = mipSize(LOW_MIPS - 1u);
  for (uint i = lane; i < size.x * size.y; i += 256u) {
    const uint2 texel = uint2(i % size.x, i / size.x);
    float v = 0.0f;
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
      const uint2 s = texel * 2u + uint2(k & 1u, k >> 1u);
      if (all(s < srcSize)) {
        v = max(v, Mips[LOW_MIPS - 1u][s]);
      }
    }
    highA[i] = v;
    Mips[LOW_MIPS][texel] = v;
  }
  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint mip = LOW_MIPS + 1u; mip < MAX_HZB_MIPS; ++mip) {
    if (mip >= params.mipCount) {
      break;
    }
    const uint2 levelSize = mipSize(mip);
    for (uint i = lane; i < levelSize.x * levelSize.y; i += 256u) {
      const uint2 texel = uint2(i % levelSize.x, i / levelSize.x);
      const uint2 src = texel * 2u;
      const float v = max(max(highRead(mip - 1u, src), highRead(mip - 1u, src + uint2(1u, 0u))),
                          max(highRead(mip - 1u, src + uint2(0u, 1u)), highRead(mip - 1u, src + uint2(1u, 1u))));
      if ((mip & 1u) != 0u) {
        highA[i] = v;
      } else {
        highB[i] = v;
      }
      Mips[mip][texel] = v;
    }
    GroupMemoryBarrierWithGroupSync();
  }
}

