// The scene uniform buffer (binding 1 of the scene descriptor set): slang.slang, task.slang and outline.slang.
// Matches UniformBufferObject in renderer.cpp.
#pragma once

struct UniformBuffer {
  float4x4 mvp;
  float4x4 previousMvp;
  float2 renderSize;
  uint clusterCount; // unused: each draw pushes its own cluster count (task.slang drawParams)
  uint historyValid;
  float lodAoLighteningPerLevel;
  float nearPlane; // projection planes (reverse-Z: depth 1 at near, 0 at far), for view depth from the depth buffer
  float farPlane;
  float4 cameraPosition; // w: time in seconds (wraps every 3600 s), for texture animation
  float4 viewRayScale; // xy: view-space ray per NDC unit (1 / proj[0][0], 1 / proj[1][1]); z: 1 when underwater
  float4 hzbParams;    // xy: Hi-Z mip 0 size, z: mip count, w: 1 when the Hi-Z holds last frame's depth
};

[[vk::binding(1, 0)]]
ConstantBuffer<UniformBuffer> ubo;
