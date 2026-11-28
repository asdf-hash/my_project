struct TaaInput {
  float4 pos : SV_Position; // the fragment stage derives its uv from this
};

struct TaaOutput {
  float4 color : SV_Target0;
  float4 history : SV_Target1;
};

[[vk::binding(0, 0)]]
Texture2D<float4> SceneColor;
[[vk::binding(1, 0)]]
Texture2D<float4> HistoryColor;
[[vk::binding(2, 0)]]
Texture2D<float4> MotionVectors;
[[vk::binding(3, 0)]]
SamplerState taaSampler;

// Matches TaaParams in renderer.cpp.
struct TaaParams {
  float2 inverseRenderSize;
  uint historyValid; // bit 0: history usable; bit 1: TAA enabled (TAA_ENABLED)
  float depthRejectionThreshold;
};

[[vk::push_constant]]
ConstantBuffer<TaaParams> taaParams;

/** Linear RGB to YCoCg (x = luma Y, y = Co, z = Cg). */
float3 toYCoCg(float3 color) {
  return float3(dot(color, float3(0.25f, 0.5f, 0.25f)), color.r - color.b,
                color.g - (color.r + color.b) * 0.5f);
}

/** Inverse of toYCoCg. */
float3 fromYCoCg(float3 color) {
  const float temp = color.x - color.z * 0.5f;
  return float3(temp + color.y * 0.5f, color.x + color.z * 0.5f, temp - color.y * 0.5f);
}

/** Fullscreen triangle from three vertex IDs, no vertex buffer. */
[shader("vertex")]
void taaVertexMain(uint vertexId : SV_VertexID, out TaaInput output) {
  const float2 uv = vertexId == 0u ? float2(0.0f, 0.0f)
                     : vertexId == 1u ? float2(2.0f, 0.0f)
                                      : float2(0.0f, 2.0f);
  output.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
}

/**
 * Resolves TAA: sharpens the current frame's luma, then blends in reprojected history clamped to the 3x3 YCoCg
 * neighborhood. History whose stored depth differs from motion.z by more than depthRejectionThreshold is rejected.
 *
 * @return color: the displayed frame; history: the same color with this pixel's depth (motion.w) in alpha.
 */
[shader("fragment")]
TaaOutput taaFragmentMain(TaaInput input) {
  const float2 uv = input.pos.xy * taaParams.inverseRenderSize;
  const float4 current = SceneColor.SampleLevel(taaSampler, uv, 0.0f);
  const float4 motion = MotionVectors.Load(int3(uint2(input.pos.xy), 0));
  const float2 historyUv = uv + motion.xy;

  float3 outputColor = current.rgb;
  if ((taaParams.historyValid & 2u) != 0u) {
    const float3 currentYCoCg = toYCoCg(current.rgb);
    float3 neighborhoodMin = float3(1.0e20f, 1.0e20f, 1.0e20f);
    float3 neighborhoodMax = float3(-1.0e20f, -1.0e20f, -1.0e20f);
    float3 neighborhoodMean = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; ++y) {
      [unroll]
      for (int x = -1; x <= 1; ++x) {
        const float3 sampleColor =
            x == 0 && y == 0
                ? currentYCoCg
                : toYCoCg(
                      SceneColor.SampleLevel(taaSampler, uv + float2(x, y) * taaParams.inverseRenderSize, 0.0f).rgb);
        neighborhoodMin = min(neighborhoodMin, sampleColor);
        neighborhoodMax = max(neighborhoodMax, sampleColor);
        neighborhoodMean += sampleColor;
      }
    }
    neighborhoodMean /= 9.0f;

    const float sharpenedLuma =
        currentYCoCg.x + 0.15f * (currentYCoCg.x - neighborhoodMean.x);
    outputColor = fromYCoCg(float3(sharpenedLuma, currentYCoCg.yz));
    const bool validHistory = (taaParams.historyValid & 1u) != 0u && all(historyUv >= 0.0f) && all(historyUv < 1.0f);
    if (validHistory) {
      const uint2 historyPixel = uint2(historyUv / taaParams.inverseRenderSize);
      const float historyDepth = HistoryColor.Load(int3(historyPixel, 0)).a;
      if (abs(historyDepth - motion.z) <= taaParams.depthRejectionThreshold) {
        const float3 history = toYCoCg(HistoryColor.SampleLevel(taaSampler, historyUv, 0.0f).rgb);
        const float3 clampedHistory = clamp(history, neighborhoodMin, neighborhoodMax);
        outputColor = lerp(outputColor, fromYCoCg(clampedHistory), 0.75f);
      }
    }
  }

  TaaOutput output;
  output.color = float4(saturate(outputColor), 1.0f);
  output.history = float4(saturate(outputColor), motion.w);
  return output;
}
