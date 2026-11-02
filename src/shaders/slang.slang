struct UniformBuffer {
  float4x4 mvp;
  float4x4 previousMvp;
  float2 renderSize;
  uint quadCount;
  uint historyValid;
  float lodAoLighteningPerLevel;
  float4 cameraPosition;
  uint4 blockFaceTextures[10];
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint2> QuadBuffer;
[[vk::binding(3, 0)]]
StructuredBuffer<uint64_t> ChunkOffsets;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> QuadCount;
[[vk::binding(1, 0)]]
ConstantBuffer<UniformBuffer> ubo;

struct MeshVertex {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD3;
  float2 uv : TEXCOORD0;
  nointerpolation uint lighting : TEXCOORD1;
  nointerpolation uint material : TEXCOORD2;
  float3 worldPosition : TEXCOORD4;
};

[[vk::binding(2, 0)]]
SamplerState sampler;
[[vk::binding(11, 0)]]
Texture2D textures[];

static const float aoValues[4] = { 0.1f, 0.25f, 0.5f, 1.0f };

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

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void meshMain(out vertices MeshVertex verts[128], out indices uint3 triangles[64], uint3 groupID: SV_GroupID,
              uint3 groupThreadID: SV_GroupThreadID) {
  const uint quadsPerMeshlet = 32u;
  const uint linearMeshletID = groupID.x + groupID.y * 65535u + groupID.z * 65535u * 65535u;
  const uint firstQuad = linearMeshletID * quadsPerMeshlet;
  const uint activeQuadCount = firstQuad < QuadCount[0] ? min(QuadCount[0] - firstQuad, quadsPerMeshlet) : 0u;
  SetMeshOutputCounts(activeQuadCount * 4u, activeQuadCount * 2u);

  const uint quadInMeshlet = groupThreadID.x;
  if (quadInMeshlet >= activeQuadCount) {
    return;
  }

  const uint2 packed = QuadBuffer[firstQuad + quadInMeshlet];
  const uint64_t chunkOffsetPacked = ChunkOffsets[firstQuad + quadInMeshlet];
  // Chunk origin fields are signed 24/24/15-bit values packed by World::packChunkOffset.
  const int32_t chunkOffsetX = int32_t(int32_t(chunkOffsetPacked & 0xFFFFFFul) << 8) >> 8;
  const int32_t chunkOffsetZ = int32_t(int32_t((chunkOffsetPacked >> 24u) & 0xFFFFFFul) << 8) >> 8;
  const int32_t chunkY = int32_t(int32_t((chunkOffsetPacked >> 48u) & 0x1FFFul) << 19) >> 19;
  const uint lod = uint(chunkOffsetPacked >> 61u);
  const float voxelScale = float(1u << lod);
  const float chunkOffsetY = float(chunkY * 32) * voxelScale;
  uint3 subchunkPos = uint3(packed.x & 0x1fu, (packed.x >> 5u) & 0x1fu, (packed.x >> 10u) & 0x1fu);
  uint normalIndex = (packed.x >> 15u) & 0x7u;
  const uint aoBits = (packed.x >> 18u) & 0xffu;
  const uint metadata = packed.y;
  const uint blockType = metadata & 0x3ffu;
  const uint rotation = (metadata >> 26u) & 0x3u;
  const float skylight = float((metadata >> 10u) & 0xfu) / 15.0f;
  const uint coloredLight = (metadata >> 14u) & 0xfffu;
  const uint encodedLength = (packed.x >> 26u) & 0x1fu;
  const uint greedyLength = encodedLength + 1u;
  const uint faceAxis = normalIndex >> 1u;
  const uint faceSign = normalIndex & 1u;
  const bool xFacing = faceAxis == 0u;

  float3 blockPos = float3(subchunkPos) * voxelScale + float3(float(chunkOffsetX), chunkOffsetY, float(chunkOffsetZ));
  const float positiveFace = float(1u - faceSign);
  const float3 faceOrigin =
      blockPos + float3(faceAxis == 0u ? positiveFace * voxelScale : 0.0f,
                        faceAxis == 1u ? positiveFace * voxelScale : 0.0f,
                        faceAxis == 2u ? positiveFace * voxelScale : (normalIndex == 2u ? voxelScale : 0.0f));
  const float3 faceV = float3(0.0f, faceAxis != 1u ? 1.0f : 0.0f, faceAxis == 1u ? float(int(faceSign) * 2 - 1) : 0.0f);

  const float3 greedyAxis = xFacing ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
  const float2 quadSize = float2(float(greedyLength), 1.0);
  const uint vertexBase = quadInMeshlet * 4u;
  const uint lighting = aoBits | (((metadata >> 10u) & 0xfu) << 8u) | (coloredLight << 12u);
  const uint material = blockType | (normalIndex << 10u) | (rotation << 13u) | (greedyLength << 15u) | (lod << 20u);

  [unroll]
  for (uint i = 0; i < 4; ++i) {
    const float2 quadOffset = float2((i == 1u || i == 2u) ? 1.0f : 0.0f, i >= 2u ? 1.0f : 0.0f);
    const float2 quadCoord = quadOffset * quadSize;
    const float2 scaledQuadCoord = quadCoord * voxelScale;
    float3 worldPos = faceOrigin + greedyAxis * scaledQuadCoord.x + faceV * scaledQuadCoord.y;
    verts[vertexBase + i].pos = mul(ubo.mvp, float4(worldPos, 1.0));
    verts[vertexBase + i].previousClip = mul(ubo.previousMvp, float4(worldPos, 1.0));
    verts[vertexBase + i].uv = quadCoord;
    verts[vertexBase + i].lighting = lighting;
    verts[vertexBase + i].material = material;
    verts[vertexBase + i].worldPosition = worldPos;
  }

  const uint primitiveBase = quadInMeshlet * 2u;
  if (normalIndex == 0u || normalIndex == 5u) {
    triangles[primitiveBase] = uint3(vertexBase + 0u, vertexBase + 2u, vertexBase + 1u);
    triangles[primitiveBase + 1u] = uint3(vertexBase + 0u, vertexBase + 3u, vertexBase + 2u);
  } else {
    triangles[primitiveBase] = uint3(vertexBase + 0u, vertexBase + 1u, vertexBase + 2u);
    triangles[primitiveBase + 1u] = uint3(vertexBase + 0u, vertexBase + 2u, vertexBase + 3u);
  }
}

struct FragmentInput {
  float4 pos : SV_Position;
  float4 previousClip : TEXCOORD3;
  float2 uv : TEXCOORD0;
  nointerpolation uint lighting : TEXCOORD1;
  nointerpolation uint material : TEXCOORD2;
  float3 worldPosition : TEXCOORD4;
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
  const uint rotation = (input.material >> 13u) & 0x3u;
  const uint greedyLength = (input.material >> 15u) & 0x1fu;
  const uint chunkLod = (input.material >> 20u) & 0x7u;
  const uint useDirtForLodGrassSide = uint(blockType == 0u && chunkLod > 0u && faceAxis != 1u);
  const uint textureBlockType = blockType + useDirtForLodGrassSide;
  const uint textureIndex = ubo.blockFaceTextures[textureBlockType * 2u + normalIndex / 4u][normalIndex % 4u];

  float2 textureUv = input.uv;
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

  const float tintWeight = float(useDirtForLodGrassSide) * 1.5f;
  const float3 dirtTint = float3(0.772f, 0.887f, 0.760f);

  color.rgb *= lerp(float3(1.0f, 1.0f, 1.0f), dirtTint, tintWeight);

  const float u = saturate(input.uv.x / float(greedyLength));
  const float v = saturate(input.uv.y);
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
  const float fogAmount =
      saturate((distance(input.worldPosition, ubo.cameraPosition.xyz) - fogStart) / (fogEnd - fogStart));
  const float3 skyColor = float3(0.55f, 0.72f, 0.9f);
  SceneFragmentOutput output;
  const float3 litColor = color.rgb * coloredLight * aoFactor * shading;
  output.color = float4(lerp(litColor, skyColor, fogAmount), color.a);
  output.motion = float4(previousUv - currentUv, input.previousClip.z / input.previousClip.w, input.pos.z);
  return output;
}
