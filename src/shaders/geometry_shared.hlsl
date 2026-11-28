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
  uint pass;        // GEOMETRY_PASS_*
  uint clusterBase; // mesh path: first cluster of this draw (CPU frustum culling draws runs of visible chunk slots)
  int3 smoothOrigin; // smooth path: world block origin of the drawn chunk
  uint padding;
  // Smooth LOD terrain (smoothLodFragMain): block ID per height layer (as chunk_compute.slang's DispatchParams).
  uint4 layerBlocks;
  uint4 layerFillerBlocks;
  uint4 layerDeepBlocks;
};
static const uint GEOMETRY_PASS_OPAQUE = 0u;      // opaque blocks only: no discard, so early depth testing holds
static const uint GEOMETRY_PASS_TRANSLUCENT = 1u; // blended water: no back-face culling
static const uint GEOMETRY_PASS_CUTOUT = 2u;      // non-opaque blocks (leaves, glass): alpha-tested with discard

// Bit 9 of a block ID marks a non-opaque block (blocks::kNonOpaqueBit): its texture alpha is cut out.
static const uint NON_OPAQUE_BIT = 0x200u;
/**
 * Whether a quad belongs to the current draw's pass. Opaque and cutout passes walk the same clusters and split
 * their quads by the block's non-opaque bit; the translucent pass keeps everything.
 *
 * @param material Packed DecodedQuad::material (block ID in the low bits).
 * @return true if the quad is drawn in drawParams.pass.
 */
bool quadInPass(uint material) {
  return drawParams.pass == GEOMETRY_PASS_TRANSLUCENT ||
         ((material & NON_OPAQUE_BIT) != 0u) == (drawParams.pass == GEOMETRY_PASS_CUTOUT);
}
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
  float4 boundsMax; // w: CLUSTER_HAS_CUTOUT when it may hold non-opaque quads (else 0)
};
static const float CLUSTER_HAS_CUTOUT = 1.0f;
[[vk::binding(6, 0)]]
StructuredBuffer<QuadCluster> Clusters;

static const uint MAX_QUAD_COUNT = 1u << 24u; // matches chunk_compute.slang and renderer.cpp
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
