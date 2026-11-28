#include "collision.hpp"

#include <algorithm>
#include <cmath>

namespace collision {

namespace {

/** Calls fn(x, y, z) for each cell the box overlaps until it returns true; returns whether any did. */
template <typename Fn> bool anyCell(const Box &box, Fn &&fn) {
  const auto x0 = static_cast<std::int32_t>(std::floor(box.min.x + kEpsilon));
  const auto x1 = static_cast<std::int32_t>(std::floor(box.max.x - kEpsilon));
  const auto y0 = static_cast<std::int32_t>(std::floor(box.min.y + kEpsilon));
  const auto y1 = static_cast<std::int32_t>(std::floor(box.max.y - kEpsilon));
  const auto z0 = static_cast<std::int32_t>(std::floor(box.min.z + kEpsilon));
  const auto z1 = static_cast<std::int32_t>(std::floor(box.max.z - kEpsilon));
  for (std::int32_t y = y0; y <= y1; ++y) {
    for (std::int32_t z = z0; z <= z1; ++z) {
      for (std::int32_t x = x0; x <= x1; ++x) {
        if (fn(x, y, z)) {
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace

bool collides(const Box &box, const CellQuery &world) {
  return anyCell(box, [&](std::int32_t x, std::int32_t y, std::int32_t z) { return world.collidable(x, y, z); });
}

bool inWater(const Box &box, const CellQuery &world) {
  return anyCell(box, [&](std::int32_t x, std::int32_t y, std::int32_t z) { return world.water(x, y, z); });
}

double clipAxis(Box &box, int axis, double delta, const CellQuery &world) {
  if (delta == 0.0) {
    return 0.0;
  }
  Box swept = box;
  if (delta > 0.0) {
    swept.max[axis] += delta;
  } else {
    swept.min[axis] += delta;
  }
  anyCell(swept, [&](std::int32_t x, std::int32_t y, std::int32_t z) {
    if (!world.collidable(x, y, z)) {
      return false;
    }
    const glm::dvec3 cellMin(x, y, z);
    const glm::dvec3 cellMax = cellMin + 1.0;
    if (delta > 0.0 && cellMin[axis] >= box.max[axis] - kEpsilon) {
      delta = std::min(delta, cellMin[axis] - box.max[axis]);
    } else if (delta < 0.0 && cellMax[axis] <= box.min[axis] + kEpsilon) {
      delta = std::max(delta, cellMax[axis] - box.min[axis]);
    }
    return false;
  });
  box.min[axis] += delta;
  box.max[axis] += delta;
  return delta;
}

} // namespace collision
