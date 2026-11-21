#pragma once

#include "lod.hpp"
#include "renderer.hpp"
#include "world.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

struct ClipmapView {
  std::array<float, 3> position;
  std::array<float, 3> forward;
  std::array<float, 3> right;
  std::array<float, 3> up;
  std::array<float, MAX_LOD> lodDistanceSquared;
  float verticalFovRadians;
  float aspectRatio;
  // Drop tiles outside the (margin-padded) view frustum. Only the vertex-pulling path sets this: the mesh path
  // meshes the whole ring and culls on the GPU in its task shader instead.
  bool frustumCull;
  // Center chunk column of the CPU-owned square (world::kCpuChunkRadius), or none while the CPU has nothing ready.
  // Those columns stay quadtree leaves (for neighbor LODs) but get no GPU work; tiles within one more chunk of the
  // square are forced to LOD 0 so the seam joins two full-resolution meshes.
  std::optional<std::array<int32_t, 2>> cpuRegionCenter;

  bool operator==(const ClipmapView &) const = default;
};

// GPU-meshed columns only: which of their vertical chunks get meshed is decided on the GPU from the generated heights
// (allocateWork in chunk_compute.slang).
struct ClipmapResult {
  std::vector<world::ChunkLod> chunks;
  // Per entry of `chunks`: LOD range of the tiles across each edge, so seam border heights can be sampled at the
  // neighbor's real resolution (generateHeights). Edge e (-X, +X, -Z, +Z) at bits e*8: min LOD | max LOD << 4.
  std::vector<uint32_t> neighborLods;

  bool operator==(const ClipmapResult &) const = default;
};

inline ClipmapResult buildVisibleClipmap(const ClipmapView &view) {
  constexpr std::uint32_t CLIPMAP_KEEP_ALIVE_MULTIPLIER = 4u;
  const int32_t cameraChunkX = static_cast<int32_t>(std::floor(view.position[0] / CHUNK_SIZE));
  const int32_t cameraChunkZ = static_cast<int32_t>(std::floor(view.position[2] / CHUNK_SIZE));
  const std::uint32_t maxChunkCount = calculateClipmapCapacity(TARGET_SCREEN_PERCENTAGE, MIN_SUPPORTED_TAN_HALF_FOV,
                                                               CHUNK_SIZE, world::kRenderDistance) *
                                      CLIPMAP_KEEP_ALIVE_MULTIPLIER;

  ClipmapResult result;
  result.chunks.reserve(maxChunkCount);

  const float terrainCenterHeight = (world::kMinTerrainHeight + world::kMaxTerrainHeight) * 0.5f;
  // Vertical distance to the nearest point of the terrain height envelope (0 inside it), matching the nearest-point
  // x/z distance below. Measuring to the envelope's center instead would push tiles under a camera near the
  // envelope's top or bottom (sea level, ocean floor, peaks) to coarser LODs.
  const float deltaY = std::max({static_cast<float>(world::kMinTerrainHeight) - view.position[1], 0.0f,
                                 view.position[1] - static_cast<float>(world::kMaxTerrainHeight)});
  const float maxRadius = static_cast<float>(world::kRenderDistance * CHUNK_SIZE * (1u << (MAX_LOD - 1u)));
  constexpr float motionMargin = static_cast<float>(CHUNK_SIZE * 2);
  const float paddedMaxRadius = maxRadius + motionMargin;
  const float maxRadiusSquared = paddedMaxRadius * paddedMaxRadius;
  const float tanHalfVerticalFov = std::tan(view.verticalFovRadians * 0.5f) + 0.1f;
  const float tanHalfHorizontalFov = tanHalfVerticalFov * view.aspectRatio + 0.1f;
  const float horizontalPlaneRadiusScale = std::sqrt(1.0f + tanHalfHorizontalFov * tanHalfHorizontalFov);
  const float verticalPlaneRadiusScale = std::sqrt(1.0f + tanHalfVerticalFov * tanHalfVerticalFov);
  constexpr float nearPlane = 0.1f;
  constexpr float farPlane = 1000000.0f;

  auto floorDivide = [](int32_t value, int32_t divisor) {
    return static_cast<int32_t>(std::floor(static_cast<double>(value) / static_cast<double>(divisor)));
  };

  // Chunk-coordinate overlap of the tile [chunkX, chunkX + scale) x [chunkZ, chunkZ + scale) with the CPU square
  // grown by `margin` chunks.
  // Overlap of the chunk rectangle [x0, x1] x [z0, z1] (inclusive) with the CPU square grown by `margin` chunks.
  auto overlapsCpuRect = [&](int32_t x0, int32_t x1, int32_t z0, int32_t z1, int32_t margin = 0) {
    if (!view.cpuRegionCenter) {
      return false;
    }
    const int32_t radius = world::kCpuChunkRadius + margin;
    const auto [centerX, centerZ] = *view.cpuRegionCenter;
    return x0 <= centerX + radius && x1 >= centerX - radius && z0 <= centerZ + radius && z1 >= centerZ - radius;
  };
  auto overlapsCpuSquare = [&](int32_t chunkX, int32_t chunkZ, int32_t scale, int32_t margin) {
    return overlapsCpuRect(chunkX, chunkX + scale - 1, chunkZ, chunkZ + scale - 1, margin);
  };

  // Every quadtree leaf, including tiles without work or outside the frustum, keyed by tileKey.
  std::unordered_set<uint64_t> leaves;
  auto tileKey = [](int32_t chunkX, int32_t chunkZ, uint32_t lod) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(chunkX) & 0xffffffu)) |
           (static_cast<uint64_t>(static_cast<uint32_t>(chunkZ) & 0xffffffu) << 24u) |
           (static_cast<uint64_t>(lod) << 48u);
  };

  auto appendTile = [&](auto &&self, int32_t chunkX, int32_t chunkZ, uint32_t lod) -> void {
    const int32_t scale = static_cast<int32_t>(1u << lod);
    const float minX = static_cast<float>(chunkX * CHUNK_SIZE);
    const float minZ = static_cast<float>(chunkZ * CHUNK_SIZE);
    const float tileSize = static_cast<float>(CHUNK_SIZE * scale);
    const float maxX = minX + tileSize;
    const float maxZ = minZ + tileSize;
    const float closestX = std::clamp(view.position[0], minX, maxX);
    const float closestZ = std::clamp(view.position[2], minZ, maxZ);
    const float dx = closestX - view.position[0];
    const float dz = closestZ - view.position[2];
    const float distanceSquared = dx * dx + deltaY * deltaY + dz * dz;
    if (distanceSquared >= maxRadiusSquared) {
      return;
    }

    const float centerX = minX + tileSize * 0.5f;
    const float centerZ = minZ + tileSize * 0.5f;
    const float centerY = terrainCenterHeight;
    const float relativeX = centerX - view.position[0];
    const float relativeY = centerY - view.position[1];
    const float relativeZ = centerZ - view.position[2];
    const float viewDepth = relativeX * view.forward[0] + relativeY * view.forward[1] + relativeZ * view.forward[2];
    const float viewX = relativeX * view.right[0] + relativeY * view.right[1] + relativeZ * view.right[2];
    const float viewY = relativeX * view.up[0] + relativeY * view.up[1] + relativeZ * view.up[2];
    const float halfHeightRange = static_cast<float>(world::kMaxTerrainHeight - world::kMinTerrainHeight) * 0.5f;
    const float boundsRadius = std::sqrt(tileSize * tileSize * 0.5f + halfHeightRange * halfHeightRange);
    const bool visible = !(
        viewDepth + boundsRadius < nearPlane - motionMargin || viewDepth - boundsRadius > farPlane + motionMargin ||
        std::abs(viewX) > viewDepth * tanHalfHorizontalFov + boundsRadius * horizontalPlaneRadiusScale + motionMargin ||
        std::abs(viewY) > viewDepth * tanHalfVerticalFov + boundsRadius * verticalPlaneRadiusScale + motionMargin);
    // The bounds enclose every child tile, so an invisible tile needs no subdivision either.
    if (view.frustumCull && !visible) {
      return;
    }

    if (lod > 0 && (selectLodSquared(distanceSquared, view.lodDistanceSquared) < lod ||
                    overlapsCpuSquare(chunkX, chunkZ, scale, 1))) {
      const int32_t childScale = scale / 2;
      self(self, chunkX, chunkZ, lod - 1);
      self(self, chunkX + childScale, chunkZ, lod - 1);
      self(self, chunkX, chunkZ + childScale, lod - 1);
      self(self, chunkX + childScale, chunkZ + childScale, lod - 1);
      return;
    }

    leaves.insert(tileKey(chunkX, chunkZ, lod));
    if (lod == 0 && overlapsCpuSquare(chunkX, chunkZ, 1, 0)) {
      return; // meshed on the CPU (cpu_world.hpp)
    }
    result.chunks.push_back({chunkX, chunkZ, static_cast<int32_t>(lod), scale});
  };

  constexpr uint32_t outerLod = MAX_LOD - 1u;
  const int32_t outerScale = static_cast<int32_t>(1u << outerLod);
  const float outerCellSize = static_cast<float>(CHUNK_SIZE * outerScale);
  const int32_t outerCellRadius = static_cast<int32_t>(std::ceil(maxRadius / outerCellSize)) + 1;
  const int32_t cameraOuterCellX = floorDivide(cameraChunkX, outerScale);
  const int32_t cameraOuterCellZ = floorDivide(cameraChunkZ, outerScale);
  for (int32_t dz = -outerCellRadius; dz <= outerCellRadius; ++dz) {
    for (int32_t dx = -outerCellRadius; dx <= outerCellRadius; ++dx) {
      appendTile(appendTile, (cameraOuterCellX + dx) * outerScale, (cameraOuterCellZ + dz) * outerScale, outerLod);
    }
  }

  // Walk the strip of chunk cells just outside each edge, leaf by leaf, and record the LOD range found there.
  // Points with no leaf (outside the clipmap, or frustum-culled) don't count; an edge with none keeps the tile's own.
  auto leafLodAt = [&](int32_t chunkX, int32_t chunkZ) -> int32_t {
    for (uint32_t lod = 0; lod < MAX_LOD; ++lod) {
      const int32_t scale = static_cast<int32_t>(1u << lod);
      if (leaves.contains(tileKey(floorDivide(chunkX, scale) * scale, floorDivide(chunkZ, scale) * scale, lod))) {
        return static_cast<int32_t>(lod);
      }
    }
    return -1;
  };
  result.neighborLods.reserve(result.chunks.size());
  for (const world::ChunkLod &chunk : result.chunks) {
    uint32_t packed = 0;
    for (uint32_t edge = 0; edge < 4u; ++edge) {
      const bool alongZ = edge < 2u; // -X/+X edges run along z
      const int32_t across = edge == 0u ? chunk.x - 1 : edge == 1u ? chunk.x + chunk.scale
                           : edge == 2u ? chunk.z - 1 : chunk.z + chunk.scale;
      const int32_t start = alongZ ? chunk.z : chunk.x;
      int32_t minLod = chunk.lod;
      int32_t maxLod = chunk.lod;
      bool found = false;
      for (int32_t along = start; along < start + chunk.scale;) {
        const int32_t lod = alongZ ? leafLodAt(across, along) : leafLodAt(along, across);
        if (lod < 0) {
          ++along;
          continue;
        }
        minLod = found ? std::min(minLod, lod) : lod;
        maxLod = found ? std::max(maxLod, lod) : lod;
        found = true;
        const int32_t leafScale = 1 << lod;
        along = floorDivide(along, leafScale) * leafScale + leafScale; // jump to the end of that leaf
      }
      uint32_t edgeBits = static_cast<uint32_t>(minLod) | (static_cast<uint32_t>(maxLod) << 4u);
      // CPU columns across this edge: generateHeights biases the seam border heights (world::kCpuSeamBias).
      // The strip across the edge is one chunk wide and chunk.scale long.
      const bool cpuAcross = alongZ ? overlapsCpuRect(across, across, start, start + chunk.scale - 1)
                                    : overlapsCpuRect(start, start + chunk.scale - 1, across, across);
      if (cpuAcross) {
        edgeBits |= world::kNeighborCpuFlag;
      }
      packed |= edgeBits << (edge * 8u);
    }
    result.neighborLods.push_back(packed);
  }

  if (result.chunks.size() > maxChunkCount) {
    throw std::runtime_error("frustum-culled clipmap exceeds its allocated GPU buffers");
  }
  return result;
}

struct ClipmapCompletion {
  std::uint64_t sequence;
  ClipmapResult result;
  ClipmapView view;
};

class ClipmapCuller {
public:
  ClipmapCuller() : worker_([this] { run(); }) {}
  ClipmapCuller(const ClipmapCuller &) = delete;
  ClipmapCuller &operator=(const ClipmapCuller &) = delete;

  ~ClipmapCuller() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_one();
    worker_.join();
  }

  std::uint64_t submit(ClipmapView view) {
    std::lock_guard lock(mutex_);
    const std::uint64_t sequence = ++nextSequence_;
    pending_ = std::pair{sequence, std::move(view)};
    condition_.notify_one();
    return sequence;
  }

  std::optional<ClipmapCompletion> takeCompletion() {
    std::lock_guard lock(mutex_);
    if (error_) {
      std::rethrow_exception(error_);
    }
    auto completion = std::move(completion_);
    completion_.reset();
    return completion;
  }

private:
  void run() {
    for (;;) {
      std::pair<std::uint64_t, ClipmapView> job;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [this] { return stopping_ || pending_.has_value(); });
        if (stopping_) {
          return;
        }
        job = std::move(*pending_);
        pending_.reset();
      }

      try {
        ClipmapResult result = buildVisibleClipmap(job.second);
        std::lock_guard lock(mutex_);
        if (!completion_ || completion_->sequence < job.first) {
          completion_ = ClipmapCompletion{job.first, std::move(result), std::move(job.second)};
        }
      } catch (...) {
        std::lock_guard lock(mutex_);
        error_ = std::current_exception();
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<std::pair<std::uint64_t, ClipmapView>> pending_;
  std::optional<ClipmapCompletion> completion_;
  std::exception_ptr error_;
  std::uint64_t nextSequence_ = 0;
  bool stopping_ = false;
  std::thread worker_;
};
