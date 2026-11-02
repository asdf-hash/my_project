#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

inline constexpr std::uint32_t MAX_LOD = 8;

struct TargetResolution {
  std::uint32_t width;
  std::uint32_t height;
};

inline std::array<float, MAX_LOD> initializeLodDistanceSquared(float screenPercentage, float verticalFovRadians,
                                                               TargetResolution resolution) {
  if (!std::isfinite(screenPercentage) || screenPercentage <= 0.0f || screenPercentage >= 1.0f ||
      !std::isfinite(verticalFovRadians) || verticalFovRadians <= 0.0f ||
      verticalFovRadians >= 3.14159265358979323846f || resolution.width == 0 || resolution.height == 0) {
    throw std::invalid_argument("invalid LOD projection parameters");
  }

  const float focalLengthPixels = static_cast<float>(resolution.height) / (2.0f * std::tan(verticalFovRadians * 0.5f));
  const float thresholdPixels = screenPercentage * static_cast<float>(resolution.height);
  std::array<float, MAX_LOD> distances{};
  for (std::uint32_t lod = 0; lod < MAX_LOD; ++lod) {
    const float voxelScale = static_cast<float>(1u << lod);
    const float distance = voxelScale * focalLengthPixels / thresholdPixels;
    distances[lod] = distance * distance;
  }
  return distances;
}

inline std::uint32_t selectLodSquared(float distanceSquared,
                                      const std::array<float, MAX_LOD> &lodDistanceSquared) noexcept {
  std::uint32_t lod = 0;
  for (std::uint32_t candidate = 1; candidate < MAX_LOD; ++candidate) {
    if (distanceSquared < lodDistanceSquared[candidate]) {
      break;
    }
    lod = candidate;
  }
  return lod;
}

inline constexpr std::uint32_t calculateClipmapCapacity(float screenPercentage, double minTanHalfFov,
                                                        std::uint32_t chunkSize, std::uint32_t renderDistance) {
  constexpr double distanceEpsilon = 1.0e-6;
  const double maxRadius = static_cast<double>(chunkSize) * renderDistance * (1u << (MAX_LOD - 1u));
  const double distanceDenominator = 2.0 * minTanHalfFov * static_cast<double>(screenPercentage);

  std::uint64_t capacity = 0;
  for (std::uint32_t lod = 0; lod < MAX_LOD; ++lod) {
    const std::uint32_t scale = 1u << lod;
    const double transitionRadius =
        lod + 1u < MAX_LOD ? std::min(maxRadius, static_cast<double>(1u << (lod + 1u)) / distanceDenominator)
                           : maxRadius;
    const double cellsAcross = (2.0 * (transitionRadius + distanceEpsilon)) / (static_cast<double>(chunkSize) * scale);
    const auto wholeCells = static_cast<std::uint64_t>(cellsAcross);
    const std::uint64_t cellsPerAxis = wholeCells + (static_cast<double>(wholeCells) < cellsAcross ? 1u : 0u) + 2u;
    capacity += cellsPerAxis * cellsPerAxis;
  }
  return static_cast<std::uint32_t>(capacity);
}
