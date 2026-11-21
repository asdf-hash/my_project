// Declarations shared by slang.slang (mesh/vertex/fragment) and task.slang (task shader).
#pragma once

#include "scene_ubo.hlsl"

// Per draw: which cluster set the task shader walks (GPU-meshed clipmap or CPU-meshed near chunks, each with its own
// descriptor set), and the debug view's tint. Matches GeometryPushConstants in renderer.cpp.
static const uint DEBUG_TINT_OFF = 0u;
static const uint DEBUG_TINT_LOD = 1u; // GPU-meshed: red (LOD 0) to yellow (LOD 7)
static const uint DEBUG_TINT_CPU = 2u; // CPU-meshed: green
struct GeometryPushConstants {
  uint clusterCount;
  uint debugTint;
  uint quadBase;  // vertex path: first quad of this draw
  uint quadCount; // vertex path: quads in this draw (0: up to QuadCount[0])
};
[[vk::push_constant]]
ConstantBuffer<GeometryPushConstants> drawParams;

// One record per meshColumns group (8x8 columns of one chunk); its quads are six contiguous runs by direction:
// +Y, -X, +X, -Z, +Z, -Y. Matches QuadCluster in chunk_compute.slang.
static const uint RUNS_PER_CLUSTER = 6u;
struct QuadCluster {
  uint base;
  uint counts01; // +Y | -X << 16
  uint counts23; // +X | -Z << 16
  uint counts45; // +Z | -Y << 16
  float4 boundsMin;
  float4 boundsMax;
};
[[vk::binding(6, 0)]]
StructuredBuffer<QuadCluster> Clusters;

static const uint MAX_QUAD_COUNT = 1u << 23u; // matches chunk_compute.slang
static const uint MAX_LOD_LEVELS = 8u;          // MAX_LOD in lod.hpp

// Task -> mesh hand-off. One task group covers CLUSTERS_PER_TASK clusters; each meshlet is up to
// QUADS_PER_MESHLET quads of one surviving direction run.
static const uint CLUSTERS_PER_TASK = 32u;
static const uint QUADS_PER_MESHLET = 32u;

struct TaskPayload {
  uint runCount;
  // Surviving runs in lane order; firstMeshlet is strictly increasing, so meshMain can binary search it.
  uint firstQuad[CLUSTERS_PER_TASK * RUNS_PER_CLUSTER];
  uint quadCount[CLUSTERS_PER_TASK * RUNS_PER_CLUSTER];
  uint firstMeshlet[CLUSTERS_PER_TASK * RUNS_PER_CLUSTER];
};
