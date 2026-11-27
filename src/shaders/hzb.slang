// Hierarchical depth (Hi-Z) for the task shader's occlusion test. Each texel holds the FARTHEST depth it covers
// (reverse-Z: 1 = near, 0 = far, so farthest = min). Mip 0 is half the next power of two of the screen, so every later
// level is an exact 2x2 reduction. Built after the mesh path's scene pass; next frame's taskMain reads it with
// ubo.previousMvp.
//
// Two dispatches, one barrier between them:
//   hzbBuildLow:  one 16x16 group per 16x16 block of mip 0: depth -> mips 0..4.
//   hzbBuildHigh: a single group: mip 4 -> mips 5..last.
// Texels outside a level (non-square or tiny pyramids) read as 1, the identity of farther().

static const uint MAX_HZB_MIPS = 16u;  // MAX_HZB_MIPS in renderer.cpp
static const uint LOW_MIPS = 5u;       // mips written by hzbBuildLow (HZB_LOW_MIPS in renderer.cpp)
static const uint HIGH_SIDE = 64u;     // largest mip 5 hzbBuildHigh holds (mip 0 up to 2048, HZB_MAX_MIP0_SIDE)

struct HzbParams {
  uint2 mip0Size;
  uint2 depthSize;
  uint mipCount;
  // Reverse-Z depth of HZB_IGNORE_NEAR_DISTANCE (renderer.cpp): nearer samples store the far plane (0), so geometry
  // right in front of the camera, which disoccludes fast, never culls anything.
  float ignoreNearerDepth;
};

[[vk::push_constant]]
ConstantBuffer<HzbParams> params;

static const float NEAREST = 1.0f; // identity of farther()

/** Farther of two reverse-Z depths. */
float farther(float a, float b) { return min(a, b); }

// Scene pass's single-sample MIN resolve of the MSAA depth (renderer.cpp, resolvedDepthImage): already the farthest
// sample of each pixel.
[[vk::binding(0, 0)]]
Texture2D<float> Depth;
[[vk::binding(1, 0)]]
RWTexture2D<float> Mips[MAX_HZB_MIPS];

/** Size of mip level `mip`, at least 1x1. */
uint2 mipSize(uint mip) { return max(params.mip0Size >> mip, uint2(1u, 1u)); }

/** Depth of one scene pixel, with samples nearer than params.ignoreNearerDepth pushed to the far plane. */
float farthestSample(uint2 pixel) {
  const float depth = Depth.Load(int3(int2(pixel), 0));
  return depth > params.ignoreNearerDepth ? 0.0f : depth;
}

/**
 * Farthest depth over the depth pixels covered by a mip 0 texel, rounding outward so nothing is missed.
 *
 * @param texel Mip 0 texel coordinate.
 * @return farthest (smallest) reverse-Z depth in the texel's footprint.
 */
float farthestDepth(uint2 texel) {
  const uint2 first = texel * params.depthSize / params.mip0Size;
  const uint2 last =
      min(((texel + 1u) * params.depthSize + params.mip0Size - 1u) / params.mip0Size, params.depthSize) - 1u;
  float farthest = NEAREST;
  if (all(last - first < 3u)) {
    // Usual case: screen <= 2x mip 0, so a footprint spans <= 3x3 pixels. Fixed bounds let every load issue at once;
    // clamped repeats hit the cache.
    [unroll]
    for (uint y = 0u; y < 3u; ++y) {
      [unroll]
      for (uint x = 0u; x < 3u; ++x) {
        farthest = farther(farthest, farthestSample(min(first + uint2(x, y), last)));
      }
    }
  } else {
    // Mip 0 clamped to HZB_MAX_MIP0_SIDE on a very large screen.
    for (uint y = first.y; y <= last.y; ++y) {
      for (uint x = first.x; x <= last.x; ++x) {
        farthest = farther(farthest, farthestSample(uint2(x, y)));
      }
    }
  }
  return farthest;
}

groupshared float lowA[16 * 16];
groupshared float lowB[8 * 8];

/**
 * Builds mips 0..LOW_MIPS-1 from the scene depth, one thread per mip 0 texel.
 *
 * @param groupID 16x16 block of mip 0 this group reduces.
 * @param threadID Texel within the block.
 */
[shader("compute")]
[numthreads(16, 16, 1)]
void hzbBuildLow(uint3 groupID: SV_GroupID, uint3 threadID: SV_GroupThreadID) {
  const uint2 t = threadID.xy;
  const uint2 texel0 = groupID.xy * 16u + t;
  float m0 = NEAREST;
  if (all(texel0 < params.mip0Size)) {
    m0 = farthestDepth(texel0);
    Mips[0][texel0] = m0;
  }
  lowA[t.y * 16u + t.x] = m0;
  GroupMemoryBarrierWithGroupSync();

  // Mips 1..4 ping-pong between lowA (16x16 source, 4x4) and lowB (8x8, 2x2); row stride = the level's side.
  // Unrolled so mip indices are compile-time constants: no dynamic indexing of the image array.
  uint side = 8u;
  [unroll]
  for (uint mip = 1u; mip < LOW_MIPS; ++mip) {
    if (all(t < side)) {
      const uint srcSide = side * 2u;
      const uint i = t.y * 2u * srcSide + t.x * 2u;
      float v;
      if ((mip & 1u) != 0u) {
        v = farther(farther(lowA[i], lowA[i + 1u]), farther(lowA[i + srcSide], lowA[i + srcSide + 1u]));
        lowB[t.y * side + t.x] = v;
      } else {
        v = farther(farther(lowB[i], lowB[i + 1u]), farther(lowB[i + srcSide], lowB[i + srcSide + 1u]));
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

/** Texel of high level `level` from shared memory, or NEAREST outside the level. */
float highRead(uint level, uint2 texel) {
  const uint2 size = mipSize(level);
  if (any(texel >= size)) {
    return NEAREST;
  }
  const uint i = texel.y * size.x + texel.x;
  return (level & 1u) != 0u ? highA[i] : highB[i];
}

/**
 * Builds mips LOW_MIPS..mipCount-1 in a single 256-thread group, from the last mip hzbBuildLow wrote.
 *
 * @param threadID Lane within the group.
 */
[shader("compute")]
[numthreads(256, 1, 1)]
void hzbBuildHigh(uint3 threadID: SV_GroupThreadID) {
  const uint lane = threadID.x;
  // First high mip from the last low one, read back from the image.
  const uint2 size = mipSize(LOW_MIPS);
  const uint2 srcSize = mipSize(LOW_MIPS - 1u);
  for (uint i = lane; i < size.x * size.y; i += 256u) {
    const uint2 texel = uint2(i % size.x, i / size.x);
    float v = NEAREST;
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
      const uint2 s = texel * 2u + uint2(k & 1u, k >> 1u);
      if (all(s < srcSize)) {
        v = farther(v, Mips[LOW_MIPS - 1u][s]);
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
      const float v = farther(farther(highRead(mip - 1u, src), highRead(mip - 1u, src + uint2(1u, 0u))),
                              farther(highRead(mip - 1u, src + uint2(0u, 1u)), highRead(mip - 1u, src + uint2(1u, 1u))));
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

