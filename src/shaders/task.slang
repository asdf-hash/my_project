// Task shader for the mesh path. Own SPIR-V module: with taskMain and meshMain in one module the NVIDIA driver handed
// meshMain an empty payload.
//
// One workgroup per CLUSTERS_PER_TASK QuadClusters. CPU-meshed chunks are drawn with drawMeshTasksEXT over runs of
// CPU-frustum-visible chunk slots (cpuVisibleSlotRuns in renderer.cpp, offset by clusterBase); the GPU clipmap uses
// drawMeshTasksIndirectEXT. Each lane culls its tile and back-facing direction runs, then the group launches one
// meshlet per QUADS_PER_MESHLET quads of every surviving run.
#include "geometry_shared.hlsl"

// Last frame's Hi-Z (hzb.slang): farthest depth per texel, rendered with ubo.previousMvp.
[[vk::binding(7, 0)]]
Texture2D<float> Hzb;

groupshared TaskPayload taskPayload;
groupshared uint laneRunCount[CLUSTERS_PER_TASK];
groupshared uint laneMeshletCount[CLUSTERS_PER_TASK];

/**
 * Conservative frustum test against ubo.cullMvp (reverse-Z: near at z = w, far at 0).
 *
 * @return true only when all 8 box corners are outside the same clip plane.
 */
bool boxOutsideFrustum(float3 boundsMin, float3 boundsMax) {
  bool allLeft = true, allRight = true, allBelow = true, allAbove = true, allNear = true, allFar = true;
  [unroll]
  for (uint i = 0u; i < 8u; ++i) {
    const float3 corner = float3((i & 1u) != 0u ? boundsMax.x : boundsMin.x, (i & 2u) != 0u ? boundsMax.y : boundsMin.y,
                                 (i & 4u) != 0u ? boundsMax.z : boundsMin.z);
    const float4 c = mul(ubo.cullMvp, float4(corner, 1.0));
    allLeft = allLeft && c.x < -c.w;
    allRight = allRight && c.x > c.w;
    allBelow = allBelow && c.y < -c.w;
    allAbove = allAbove && c.y > c.w;
    allNear = allNear && c.z > c.w;
    allFar = allFar && c.z < 0.0f;
  }
  return allLeft || allRight || allBelow || allAbove || allNear || allFar;
}

/**
 * Occlusion test against last frame's Hi-Z, projecting with last frame's matrices. Boxes crossing the near plane or
 * the screen edge are never culled; newly uncovered tiles can show up one frame late.
 *
 * @return true when the box's nearest point lies behind the farthest depth over its whole screen rectangle.
 */
bool boxOccluded(float3 boundsMin, float3 boundsMax) {
  if (ubo.hzbParams.w == 0.0f) {
    return false;
  }
  float2 uvMin = float2(1.0f, 1.0f);
  float2 uvMax = float2(0.0f, 0.0f);
  float nearestDepth = 0.0f; // reverse-Z: nearer is larger
  [unroll]
  for (uint i = 0u; i < 8u; ++i) {
    const float3 corner = float3((i & 1u) != 0u ? boundsMax.x : boundsMin.x, (i & 2u) != 0u ? boundsMax.y : boundsMin.y,
                                 (i & 4u) != 0u ? boundsMax.z : boundsMin.z);
    const float4 c = mul(ubo.cullPreviousMvp, float4(corner, 1.0));
    if (c.w <= 0.1f) {
      return false;
    }
    // The viewport flips y: NDC +1 is the top row.
    const float2 uv = float2(c.x / c.w * 0.5f + 0.5f, 0.5f - c.y / c.w * 0.5f);
    uvMin = min(uvMin, uv);
    uvMax = max(uvMax, uv);
    nearestDepth = max(nearestDepth, c.z / c.w);
  }
  // Only boxes fully on screen last frame are tested: clamping the off-screen part (no depth there) to the screen
  // made tiles pop in while the camera turned.
  if (any(uvMin < 0.0f) || any(uvMax > 1.0f)) {
    return false;
  }
  // One level finer than where the rectangle fits in one texel: <= 3x3 texels there, a much tighter fit than the
  // coarser level's 2x2 texels covering up to 4x the area.
  const float2 extent = (uvMax - uvMin) * ubo.hzbParams.xy;
  const uint level =
      min(uint(max(ceil(log2(max(max(extent.x, extent.y), 1.0f))) - 1.0f, 0.0f)), uint(ubo.hzbParams.z) - 1u);
  const int2 levelSize = max(int2(ubo.hzbParams.xy) >> int(level), int2(1, 1));
  const int2 first = int2(uvMin * float2(levelSize));
  const int2 last = min(int2(uvMax * float2(levelSize)), levelSize - 1);
  float farthest = 1.0f;
  for (int y = first.y; y <= last.y; ++y) {
    for (int x = first.x; x <= last.x; ++x) {
      farthest = min(farthest, Hzb.Load(int3(x, y, int(level))));
    }
  }
  return nearestDepth < farthest;
}

/**
 * Culls one QuadCluster per lane and dispatches a meshlet per QUADS_PER_MESHLET quads of every surviving run.
 *
 * @param groupID Spread over x and y to stay under the 65535 groups-per-dimension limit.
 * @param groupThreadID Lane = cluster within the group.
 */
[shader("amplification")]
[numthreads(CLUSTERS_PER_TASK, 1, 1)]
void taskMain(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID) {
  const uint lane = groupThreadID.x;
  const uint drawCluster = (groupID.x + groupID.y * 65535u) * CLUSTERS_PER_TASK + lane;
  const uint clusterIndex = drawParams.clusterBase + drawCluster;

  // Per run: first quad and quad count, 0 when the run is culled. Indexed only with compile-time constants.
  uint runFirst[RUNS_PER_CLUSTER] = { 0u, 0u, 0u, 0u, 0u, 0u };
  uint runQuads[RUNS_PER_CLUSTER] = { 0u, 0u, 0u, 0u, 0u, 0u };
  if (drawCluster < drawParams.clusterCount) {
    const QuadCluster cluster = Clusters[clusterIndex];
    const float3 boundsMin = cluster.boundsMin.xyz;
    const float3 boundsMax = cluster.boundsMax.xyz;
    // Skip the culls' 16 corner projections for empty clusters (air tiles, freed slots below the high water) and, in
    // the cutout pass, clusters not flagged as holding cutout quads (most hold none).
    const bool empty = (cluster.counts01 | cluster.counts23 | cluster.counts45) == 0u ||
                       (drawParams.pass == GEOMETRY_PASS_CUTOUT && cluster.boundsMax.w != CLUSTER_HAS_CUTOUT);
    if (!empty && !boxOutsideFrustum(boundsMin, boundsMax) && !boxOccluded(boundsMin, boundsMax)) {
      const float3 camera = ubo.cullCameraPosition.xyz;
      const uint counts[RUNS_PER_CLUSTER] = { cluster.counts01 & 0xffffu, cluster.counts01 >> 16u,
                                              cluster.counts23 & 0xffffu, cluster.counts23 >> 16u,
                                              cluster.counts45 & 0xffffu, cluster.counts45 >> 16u };
      // Same order as the runs. Conservative: a run survives if any face plane in the tile could face the camera.
      // Translucent pass draws both sides (water is seen from below too).
      const bool bothSides = drawParams.pass == GEOMETRY_PASS_TRANSLUCENT;
      const bool facing[RUNS_PER_CLUSTER] = { bothSides || camera.y > boundsMin.y, bothSides || camera.x < boundsMax.x,
                                              bothSides || camera.x > boundsMin.x, bothSides || camera.z < boundsMax.z,
                                              bothSides || camera.z > boundsMin.z, bothSides || camera.y < boundsMax.y };
      uint cursor = cluster.base;
      [unroll]
      for (uint d = 0u; d < RUNS_PER_CLUSTER; ++d) {
        runFirst[d] = cursor;
        // Quads past MAX_QUAD_COUNT were dropped by the mesher.
        const uint written = cursor >= MAX_QUAD_COUNT ? 0u : min(counts[d], MAX_QUAD_COUNT - cursor);
        runQuads[d] = facing[d] ? written : 0u;
        cursor += counts[d];
      }
    }
  }
  uint runs = 0u;
  uint meshlets = 0u;
  [unroll]
  for (uint d = 0u; d < RUNS_PER_CLUSTER; ++d) {
    runs += runQuads[d] != 0u ? 1u : 0u;
    meshlets += (runQuads[d] + QUADS_PER_MESHLET - 1u) / QUADS_PER_MESHLET;
  }

  // Exclusive prefix over lanes through shared memory: independent of the subgroup size.
  laneRunCount[lane] = runs;
  laneMeshletCount[lane] = meshlets;
  GroupMemoryBarrierWithGroupSync();
  uint runSlot = 0u;
  uint meshletBase = 0u;
  uint totalRuns = 0u;
  uint totalMeshlets = 0u;
  for (uint i = 0u; i < CLUSTERS_PER_TASK; ++i) {
    if (i < lane) {
      runSlot += laneRunCount[i];
      meshletBase += laneMeshletCount[i];
    }
    totalRuns += laneRunCount[i];
    totalMeshlets += laneMeshletCount[i];
  }
  [unroll]
  for (uint d = 0u; d < RUNS_PER_CLUSTER; ++d) {
    if (runQuads[d] != 0u) {
      taskPayload.firstQuad[runSlot] = runFirst[d];
      taskPayload.quadCount[runSlot] = runQuads[d];
      taskPayload.firstMeshlet[runSlot] = meshletBase;
      meshletBase += (runQuads[d] + QUADS_PER_MESHLET - 1u) / QUADS_PER_MESHLET;
      ++runSlot;
    }
  }
  if (lane == 0u) {
    taskPayload.runCount = totalRuns;
  }
  GroupMemoryBarrierWithGroupSync();
  DispatchMesh(totalMeshlets, 1, 1, taskPayload);
}
