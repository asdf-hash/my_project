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

  bool operator==(const ClipmapView &) const = default;
};

struct ClipmapResult {
  std::vector<world::ChunkLod> chunks;
  std::vector<std::array<int32_t, 2>> work;

  bool operator==(const ClipmapResult &) const = default;
};

inline ClipmapResult buildVisibleClipmap(const ClipmapView &view) {
  constexpr std::uint32_t CLIPMAP_KEEP_ALIVE_MULTIPLIER = 4u;
  const int32_t cameraChunkX = static_cast<int32_t>(std::floor(view.position[0] / CHUNK_SIZE));
  const int32_t cameraChunkY = static_cast<int32_t>(std::floor(view.position[1] / CHUNK_SIZE));
  const int32_t cameraChunkZ = static_cast<int32_t>(std::floor(view.position[2] / CHUNK_SIZE));
  const int32_t verticalChunkCount = 2 * world::kVerticalRenderDistance + 1;
  const std::uint32_t maxChunkCount = calculateClipmapCapacity(TARGET_SCREEN_PERCENTAGE, MIN_SUPPORTED_TAN_HALF_FOV,
                                                               CHUNK_SIZE, world::kRenderDistance) *
                                      CLIPMAP_KEEP_ALIVE_MULTIPLIER;

  ClipmapResult result;
  result.chunks.reserve(maxChunkCount);
  result.work.reserve(static_cast<size_t>(maxChunkCount) * verticalChunkCount);

  const float terrainCenterHeight = (world::kMinTerrainHeight + world::kMaxTerrainHeight) * 0.5f;
  const float deltaY = terrainCenterHeight - view.position[1];
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

    if (lod > 0 && selectLodSquared(distanceSquared, view.lodDistanceSquared) < lod) {
      const int32_t childScale = scale / 2;
      self(self, chunkX, chunkZ, lod - 1);
      self(self, chunkX + childScale, chunkZ, lod - 1);
      self(self, chunkX, chunkZ + childScale, lod - 1);
      self(self, chunkX + childScale, chunkZ + childScale, lod - 1);
      return;
    }

    const std::uint32_t columnIndex = static_cast<std::uint32_t>(result.chunks.size());
    const size_t workStart = result.work.size();
    const int32_t minimumHeight =
        static_cast<int32_t>(std::floor(static_cast<float>(world::kMinTerrainHeight) / scale));
    const int32_t maximumHeight = static_cast<int32_t>(std::ceil(static_cast<float>(world::kMaxTerrainHeight) / scale));
    const int32_t firstChunkY = floorDivide(minimumHeight, CHUNK_SIZE);
    const int32_t lastChunkY = floorDivide(maximumHeight - 1, CHUNK_SIZE);
    for (int32_t chunkY = firstChunkY; chunkY <= lastChunkY; ++chunkY) {
      const int32_t chunkBottom = chunkY * CHUNK_SIZE;
      const int32_t chunkTop = chunkBottom + CHUNK_SIZE;
      if (chunkTop <= minimumHeight || chunkBottom >= maximumHeight) {
        continue;
      }
      result.work.push_back({static_cast<int32_t>(columnIndex), chunkY});
    }
    if (!visible) {
      result.chunks.push_back({chunkX, chunkZ, static_cast<int32_t>(lod), scale});
      return;
    }
    if (result.work.size() != workStart) {
      result.chunks.push_back({chunkX, chunkZ, static_cast<int32_t>(lod), scale});
    }
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

  if (result.chunks.size() > maxChunkCount ||
      result.work.size() > static_cast<size_t>(maxChunkCount) * verticalChunkCount) {
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
