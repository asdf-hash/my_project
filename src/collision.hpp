#pragma once

// Axis-aligned boxes against the voxel world (Minecraft's Entity.move collision), for item entities. Cells are
// queried through CellQuery so tests can supply a fake world; every block is a full unit cube today.

#include <cstdint>

#include <glm/glm.hpp>

namespace collision {

// Cell queries the physics needs. Implementations must be safe for concurrent const calls (item ticks run on
// several threads while the world is not being edited).
class CellQuery {
public:
  virtual ~CellQuery() = default;
  /** Any block that stops movement (everything but air and fluids). */
  virtual bool collidable(std::int32_t x, std::int32_t y, std::int32_t z) const = 0;
  /** A water cell. */
  virtual bool water(std::int32_t x, std::int32_t y, std::int32_t z) const = 0;
  /** Whether the column holding cell (x, z) is loaded; entities in unloaded columns don't move. */
  virtual bool loaded(std::int32_t x, std::int32_t z) const = 0;
};

struct Box {
  glm::dvec3 min;
  glm::dvec3 max;
};

inline constexpr double kEpsilon = 1.0e-7;

/** Whether any cell the box overlaps (touching a face doesn't count) is collidable. */
bool collides(const Box &box, const CellQuery &world);
/** Whether any cell the box overlaps is water. */
bool inWater(const Box &box, const CellQuery &world);
/**
 * Clip a move along one axis against every collidable cell the swept box touches (AxisAlignedBB.calculate*Offset)
 * and shift the box by the result.
 *
 * @param axis 0 = x, 1 = y, 2 = z.
 * @param delta Requested displacement along `axis`, in blocks.
 * @return the allowed displacement.
 */
double clipAxis(Box &box, int axis, double delta, const CellQuery &world);

} // namespace collision
