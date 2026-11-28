#include "item_entity.hpp"

#include "job_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace {

constexpr double kDragCoefficient = 1.05; // cube face-on
constexpr double kMaxSpeed = 30.0;        // m/s
constexpr double kMaxSubstepTravel = 0.05; // m per substep (well under the 0.125 half size: no tunnelling)
constexpr int kMinSubsteps = 4;
constexpr int kMaxSubsteps = 16;
constexpr int kSolverIterations = 8;
constexpr double kRestitutionThreshold = 1.0; // m/s: slower impacts don't bounce
constexpr double kPenetrationSlop = 0.001;
constexpr double kSleepSpeed = 0.1;         // m/s
constexpr double kSleepAngularSpeed = 0.5;  // rad/s
constexpr double kMergeRadius = 1.5;       // blocks

std::uint32_t xorshift(std::uint32_t &state) {
  state ^= state << 13u;
  state ^= state >> 17u;
  state ^= state << 5u;
  return state;
}

/** Uniform in [0, 1). */
double random01(std::uint32_t &state) { return static_cast<double>(xorshift(state) >> 8u) / 16777216.0; }

std::int32_t cellOf(double v) { return static_cast<std::int32_t>(std::floor(v)); }

std::uint64_t cellKey(std::int32_t x, std::int32_t y, std::int32_t z) {
  constexpr std::uint64_t mask = (1u << 21u) - 1u;
  constexpr std::int32_t bias = 1 << 20;
  return ((static_cast<std::uint64_t>(x + bias) & mask) << 42u) |
         ((static_cast<std::uint64_t>(y + bias) & mask) << 21u) | (static_cast<std::uint64_t>(z + bias) & mask);
}

bool overlaps(const collision::Box &a, const collision::Box &b) {
  return a.min.x < b.max.x && a.max.x > b.min.x && a.min.y < b.max.y && a.max.y > b.min.y && a.min.z < b.max.z &&
         a.max.z > b.min.z;
}

// Contact points in body space (unit half size): 8 corners, 12 edge midpoints, 6 face centers. Edges and faces
// catch block edges poking into a face, which corners alone would let pass.
const std::array<glm::dvec3, 26> kContactPoints = [] {
  std::array<glm::dvec3, 26> points{};
  std::size_t n = 0;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        if (x != 0 || y != 0 || z != 0) {
          points[n++] = glm::dvec3(x, y, z);
        }
      }
    }
  }
  return points;
}();

// Buoyancy samples: a 3 x 3 x 3 grid at the centers of the cube's 27 sub-cells.
const std::array<glm::dvec3, 27> kVolumeSamples = [] {
  std::array<glm::dvec3, 27> points{};
  std::size_t n = 0;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        points[n++] = glm::dvec3(x, y, z) * (2.0 / 3.0);
      }
    }
  }
  return points;
}();

struct Contact {
  glm::dvec3 r;      // contact point relative to the center
  glm::dvec3 normal; // out of the terrain (axis-aligned)
  double depth = 0.0;
  double normalMass = 0.0; // 1 / (inverse mass + inverse inertia * |r x n|^2)
  double target = 0.0;     // desired normal velocity (restitution)
  double normalImpulse = 0.0;
  glm::dvec3 frictionImpulse{0.0};
};

/**
 * Penetration of a world point into the terrain: the cheapest way out through a face whose neighbor is open.
 *
 * @return false when the point is in open space, or buried so deep no neighbor face is open.
 */
bool pointContact(const glm::dvec3 &p, const collision::CellQuery &world, glm::dvec3 &normal, double &depth) {
  const std::int32_t x = cellOf(p.x);
  const std::int32_t y = cellOf(p.y);
  const std::int32_t z = cellOf(p.z);
  if (!world.collidable(x, y, z)) {
    return false;
  }
  struct Exit {
    glm::ivec3 step;
    double depth;
  };
  const Exit exits[] = {{{1, 0, 0}, x + 1.0 - p.x}, {{-1, 0, 0}, p.x - x},     {{0, 1, 0}, y + 1.0 - p.y},
                        {{0, -1, 0}, p.y - y},     {{0, 0, 1}, z + 1.0 - p.z}, {{0, 0, -1}, p.z - z}};
  depth = std::numeric_limits<double>::max();
  for (const Exit &exit : exits) {
    if (exit.depth < depth && !world.collidable(x + exit.step.x, y + exit.step.y, z + exit.step.z)) {
      depth = exit.depth;
      normal = glm::dvec3(exit.step);
    }
  }
  return depth != std::numeric_limits<double>::max();
}

} // namespace

collision::Box ItemEntityWorld::bounds(std::size_t i) const {
  // Rotated cube: half extent per axis = sum of |R| row entries * half size.
  const glm::dmat3 r = glm::mat3_cast(orientation[i]);
  glm::dvec3 half(0.0);
  for (int axis = 0; axis < 3; ++axis) {
    half[axis] = kHalfSize * (std::abs(r[0][axis]) + std::abs(r[1][axis]) + std::abs(r[2][axis]));
  }
  return {position[i] - half, position[i] + half};
}

void ItemEntityWorld::spawn(const glm::dvec3 &center, const glm::dvec3 &initialVelocity, const glm::dvec3 &spin,
                            double blockDensity, ItemStack item, int initialPickupDelay, std::uint32_t seed,
                            int initialMergeDelay) {
  if (item.empty()) {
    return;
  }
  item.count = std::min(item.count, kMaxEntityStack);
  if (size() >= kMaxEntities) {
    eraseAt(static_cast<std::size_t>(std::ranges::max_element(age) - age.begin()));
  }
  std::uint32_t state = seed * 0x9e3779b9u + static_cast<std::uint32_t>(tickCount) * 0x85ebca6bu + 0x6a09e667u;
  state = state == 0 ? 1u : state;
  xorshift(state);
  // Random yaw, slight tilt: drops don't all land the same way.
  const double yaw = random01(state) * 2.0 * std::numbers::pi;
  const glm::dvec3 tiltAxis = glm::normalize(glm::dvec3(random01(state) - 0.5, 0.0, random01(state) - 0.5) + 1e-9);
  const glm::dquat rotation = glm::angleAxis(random01(state) * 0.4, tiltAxis) * glm::angleAxis(yaw, glm::dvec3(0, 1, 0));
  position.push_back(center);
  previousPosition.push_back(center);
  orientation.push_back(rotation);
  previousOrientation.push_back(rotation);
  velocity.push_back(initialVelocity);
  angularVelocity.push_back(spin);
  density.push_back(std::max(blockDensity, 10.0));
  stack.push_back(item);
  age.push_back(0);
  pickupDelay.push_back(initialPickupDelay);
  mergeDelay.push_back(std::max(initialMergeDelay, 0));
  flags.push_back(0);
  restTicks.push_back(0);
  rng.push_back(state);
}

bool ItemEntityWorld::substep(std::size_t i, double dt, const collision::CellQuery &world, bool &onGround,
                              bool &inWater) {
  glm::dvec3 &c = position[i];
  glm::dquat &q = orientation[i];
  glm::dvec3 &v = velocity[i];
  glm::dvec3 &w = angularVelocity[i];
  const double mass = density[i] * kVolume;
  const double inverseMass = 1.0 / mass;
  const double inverseInertia = 6.0 / (mass * kSize * kSize); // solid cube: I = m s^2 / 6, the same on every axis
  const glm::dmat3 rotation = glm::mat3_cast(q);

  // Forces: gravity, buoyancy of the submerged volume (applied at its centroid: floating cubes right themselves),
  // quadratic drag in the surrounding fluid.
  int submerged = 0;
  glm::dvec3 submergedCentroid(0.0);
  for (const glm::dvec3 &sample : kVolumeSamples) {
    const glm::dvec3 p = c + rotation * (sample * kHalfSize);
    if (world.water(cellOf(p.x), cellOf(p.y), cellOf(p.z))) {
      ++submerged;
      submergedCentroid += p;
    }
  }
  const double fraction = submerged / static_cast<double>(kVolumeSamples.size());
  inWater = inWater || submerged > 0;
  glm::dvec3 force(0.0, -mass * kGravity, 0.0);
  glm::dvec3 torque(0.0);
  if (submerged > 0) {
    submergedCentroid /= static_cast<double>(submerged);
    const glm::dvec3 buoyancy(0.0, kWaterDensity * kVolume * fraction * kGravity, 0.0);
    force += buoyancy;
    torque += glm::cross(submergedCentroid - c, buoyancy);
  }
  const double fluidDensity = kWaterDensity * fraction + kAirDensity * (1.0 - fraction);
  force += -0.5 * fluidDensity * kDragCoefficient * kSize * kSize * glm::length(v) * v;
  v += force * inverseMass * dt;
  w += torque * inverseInertia * dt;
  // Rotational drag: strong in water, slight in air.
  w *= std::exp(-(0.1 + 4.0 * fraction) * dt);
  if (glm::length(v) > kMaxSpeed) {
    v *= kMaxSpeed / glm::length(v);
  }

  // Integrate.
  c += v * dt;
  q = glm::normalize(q + glm::dquat(0.0, w.x, w.y, w.z) * q * (0.5 * dt));

  // Terrain contacts at the new pose.
  const glm::dmat3 moved = glm::mat3_cast(q);
  std::array<Contact, kContactPoints.size()> contacts;
  std::size_t contactCount = 0;
  for (const glm::dvec3 &point : kContactPoints) {
    const glm::dvec3 r = moved * (point * kHalfSize);
    glm::dvec3 normal;
    double depth;
    if (!pointContact(c + r, world, normal, depth)) {
      continue;
    }
    Contact &contact = contacts[contactCount++];
    contact.r = r;
    contact.normal = normal;
    contact.depth = depth;
    const glm::dvec3 rn = glm::cross(r, normal);
    contact.normalMass = 1.0 / (inverseMass + inverseInertia * glm::dot(rn, rn));
    const double approach = glm::dot(v + glm::cross(w, r), normal);
    contact.target = approach < -kRestitutionThreshold ? -kRestitution * approach : 0.0;
    onGround = onGround || normal.y > 0.5;
  }
  if (contactCount == 0) {
    return false;
  }

  // Sequential impulses: non-penetration (accumulated, clamped >= 0), then Coulomb friction within mu * normal.
  for (int iteration = 0; iteration < kSolverIterations; ++iteration) {
    for (std::size_t k = 0; k < contactCount; ++k) {
      Contact &contact = contacts[k];
      const double vn = glm::dot(v + glm::cross(w, contact.r), contact.normal);
      const double total = std::max(contact.normalImpulse + (contact.target - vn) * contact.normalMass, 0.0);
      const glm::dvec3 impulse = contact.normal * (total - contact.normalImpulse);
      contact.normalImpulse = total;
      v += impulse * inverseMass;
      w += glm::cross(contact.r, impulse) * inverseInertia;

      const glm::dvec3 relative = v + glm::cross(w, contact.r);
      const glm::dvec3 tangential = relative - contact.normal * glm::dot(relative, contact.normal);
      const double slip = glm::length(tangential);
      if (slip < 1e-9) {
        continue;
      }
      const glm::dvec3 direction = tangential / slip;
      const glm::dvec3 rt = glm::cross(contact.r, direction);
      const double tangentMass = 1.0 / (inverseMass + inverseInertia * glm::dot(rt, rt));
      glm::dvec3 accumulated = contact.frictionImpulse - direction * (slip * tangentMass);
      const double limit = kFriction * contact.normalImpulse;
      if (glm::length(accumulated) > limit) {
        accumulated *= limit / glm::length(accumulated);
      }
      const glm::dvec3 delta = accumulated - contact.frictionImpulse;
      contact.frictionImpulse = accumulated;
      v += delta * inverseMass;
      w += glm::cross(contact.r, delta) * inverseInertia;
    }
  }

  // Position correction: normals are axis-aligned, so push out by the deepest penetration per direction.
  glm::dvec3 positive(0.0);
  glm::dvec3 negative(0.0);
  for (std::size_t k = 0; k < contactCount; ++k) {
    const Contact &contact = contacts[k];
    const double depth = std::max(contact.depth - kPenetrationSlop, 0.0);
    for (int axis = 0; axis < 3; ++axis) {
      if (contact.normal[axis] > 0.5) {
        positive[axis] = std::max(positive[axis], depth);
      } else if (contact.normal[axis] < -0.5) {
        negative[axis] = std::max(negative[axis], depth);
      }
    }
  }
  c += positive - negative;
  // Rolling resistance: contacts bleed off spin so resting cubes settle.
  w *= std::exp(-2.0 * dt);
  return true;
}

bool ItemEntityWorld::supported(std::size_t i, const collision::CellQuery &world) const {
  const glm::dmat3 rotation = glm::mat3_cast(orientation[i]);
  const glm::dvec3 &c = position[i];
  if (world.water(cellOf(c.x), cellOf(c.y), cellOf(c.z))) {
    return false;
  }
  for (const glm::dvec3 &point : kContactPoints) {
    const glm::dvec3 p = c + rotation * (point * kHalfSize) - glm::dvec3(0.0, 0.02, 0.0);
    if (world.collidable(cellOf(p.x), cellOf(p.y), cellOf(p.z))) {
      return true;
    }
  }
  return false;
}

void ItemEntityWorld::physics(std::size_t i, const collision::CellQuery &world) {
  glm::dvec3 &c = position[i];
  glm::dvec3 &v = velocity[i];
  glm::dvec3 &w = angularVelocity[i];
  std::uint8_t &state = flags[i];
  previousPosition[i] = c;
  previousOrientation[i] = orientation[i];
  if ((state & kDead) != 0) {
    return;
  }
  ++age[i];
  if (pickupDelay[i] > 0) {
    --pickupDelay[i];
  }
  if (mergeDelay[i] > 0) {
    --mergeDelay[i];
  }
  if (age[i] >= kLifetimeTicks) {
    state |= kDead;
    return;
  }
  if (!world.loaded(cellOf(c.x), cellOf(c.z))) {
    return; // frozen until its column loads
  }
  if ((state & kAsleep) != 0) {
    // Re-check now and then: still resting on terrain, out of water?
    if ((age[i] + static_cast<std::int32_t>(i)) % 40 != 0 || supported(i, world)) {
      return;
    }
    state &= static_cast<std::uint8_t>(~kAsleep);
    restTicks[i] = 0;
  }

  // Center inside a block (one was placed on it): head for the nearest open side (vanilla
  // moveTowardsClosestSpace: north, south, west, east or up) without collision this tick.
  const glm::ivec3 cell(cellOf(c.x), cellOf(c.y), cellOf(c.z));
  if (world.collidable(cell.x, cell.y, cell.z)) {
    const glm::dvec3 frac = c - glm::dvec3(cell);
    struct Side {
      glm::ivec3 step;
      int axis;
      double distance;
    };
    const Side sides[] = {{{0, 0, -1}, 2, frac.z},
                          {{0, 0, 1}, 2, 1.0 - frac.z},
                          {{-1, 0, 0}, 0, frac.x},
                          {{1, 0, 0}, 0, 1.0 - frac.x},
                          {{0, 1, 0}, 1, 1.0 - frac.y}};
    const Side *best = &sides[4];
    double bestDistance = std::numeric_limits<double>::max();
    for (const Side &side : sides) {
      const glm::ivec3 next = cell + side.step;
      if (!world.collidable(next.x, next.y, next.z) && side.distance < bestDistance) {
        best = &side;
        bestDistance = side.distance;
      }
    }
    v *= 0.75;
    v[best->axis] = (random01(rng[i]) * 0.2 + 0.1) / kTickSeconds * static_cast<double>(best->step[best->axis]);
    c += v * kTickSeconds;
    state &= static_cast<std::uint8_t>(~(kOnGround | kAsleep));
    restTicks[i] = 0;
    return;
  }

  // Substeps sized so no substep travels more than kMaxSubstepTravel.
  const double travel = glm::length(v) * kTickSeconds;
  const int substeps = std::clamp(static_cast<int>(std::ceil(travel / kMaxSubstepTravel)), kMinSubsteps, kMaxSubsteps);
  const double dt = kTickSeconds / substeps;
  bool onGround = false;
  bool inWater = false;
  for (int s = 0; s < substeps; ++s) {
    substep(i, dt, world, onGround, inWater);
  }
  state = static_cast<std::uint8_t>((state & ~(kOnGround | kInWater)) | (onGround ? kOnGround : 0) |
                                    (inWater ? kInWater : 0));

  if (onGround && !inWater && glm::length(v) < kSleepSpeed && glm::length(w) < kSleepAngularSpeed) {
    if (++restTicks[i] >= kSleepTicks) {
      state |= kAsleep;
      v = glm::dvec3(0.0);
      w = glm::dvec3(0.0);
    }
  } else {
    restTicks[i] = 0;
  }
}

void ItemEntityWorld::rebuildGrid() {
  gridEntries.clear();
  for (std::size_t i = 0; i < size(); ++i) {
    if ((flags[i] & kDead) == 0) {
      gridEntries.emplace_back(cellKey(cellOf(position[i].x), cellOf(position[i].y), cellOf(position[i].z)),
                               static_cast<std::uint32_t>(i));
    }
  }
  std::ranges::sort(gridEntries);
}

std::pair<std::size_t, std::size_t> ItemEntityWorld::gridCell(std::int32_t x, std::int32_t y, std::int32_t z) const {
  const std::uint64_t key = cellKey(x, y, z);
  const auto lower = std::ranges::lower_bound(gridEntries, std::pair{key, std::uint32_t{0}});
  auto upper = lower;
  while (upper != gridEntries.end() && upper->first == key) {
    ++upper;
  }
  return {static_cast<std::size_t>(lower - gridEntries.begin()), static_cast<std::size_t>(upper - gridEntries.begin())};
}

void ItemEntityWorld::mergeNearby() {
  const auto mergeable = [&](std::size_t i) {
    return (flags[i] & kDead) == 0 && age[i] < kLifetimeTicks && mergeDelay[i] == 0 &&
           stack[i].count < kMaxEntityStack;
  };
  // Center boxes of the unrotated cube, as vanilla's item boxes.
  const auto boxOf = [&](std::size_t i) {
    return collision::Box{position[i] - kHalfSize, position[i] + kHalfSize};
  };
  for (std::size_t i = 0; i < size(); ++i) {
    if (!mergeable(i)) {
      continue;
    }
    // Vanilla: every 2 ticks after changing block cell, else every 40.
    const bool changedCell = cellOf(position[i].x) != cellOf(previousPosition[i].x) ||
                             cellOf(position[i].y) != cellOf(previousPosition[i].y) ||
                             cellOf(position[i].z) != cellOf(previousPosition[i].z);
    if (age[i] % (changedCell ? 2 : 40) != 0) {
      continue;
    }
    collision::Box reach = boxOf(i);
    reach.min -= glm::dvec3(kMergeRadius);
    reach.max += glm::dvec3(kMergeRadius);
    const glm::ivec3 low(cellOf(reach.min.x - kHalfSize), cellOf(reach.min.y - kHalfSize), cellOf(reach.min.z - kHalfSize));
    const glm::ivec3 high(cellOf(reach.max.x + kHalfSize), cellOf(reach.max.y + kHalfSize), cellOf(reach.max.z + kHalfSize));
    for (std::int32_t y = low.y; y <= high.y && mergeable(i); ++y) {
      for (std::int32_t z = low.z; z <= high.z && mergeable(i); ++z) {
        for (std::int32_t x = low.x; x <= high.x && mergeable(i); ++x) {
          const auto [begin, end] = gridCell(x, y, z);
          for (std::size_t e = begin; e < end && mergeable(i); ++e) {
            const std::size_t j = gridEntries[e].second;
            if (j == i || !mergeable(j) || stack[j].block != stack[i].block ||
                glm::dot(position[j] - position[i], position[j] - position[i]) > kMergeRadius * kMergeRadius) {
              continue;
            }
            // The larger stack absorbs the smaller (vanilla tryToMerge; ties go to the other entity).
            const std::size_t into = stack[j].count < stack[i].count ? i : j;
            const std::size_t from = into == i ? j : i;
            const int moved = std::min<int>(stack[from].count, kMaxEntityStack - stack[into].count);
            stack[into].count = static_cast<std::uint16_t>(stack[into].count + moved);
            stack[from].count = static_cast<std::uint16_t>(stack[from].count - moved);
            pickupDelay[into] = std::max(pickupDelay[into], pickupDelay[from]);
            mergeDelay[into] = std::max(mergeDelay[into], mergeDelay[from]);
            age[into] = std::min(age[into], age[from]);
            if (stack[from].count == 0) {
              flags[from] |= kDead;
              flights.push_back(Flight{.from = position[from], .orientation = orientation[from],
                                       .stack = ItemStack{.block = stack[into].block, .count = static_cast<std::uint16_t>(moved)},
                                       .toCollector = false, .to = position[into]});
            }
          }
        }
      }
    }
  }
}

void ItemEntityWorld::collect(const Collector &collector) {
  collision::Box reach = collector.box;
  reach.min -= glm::dvec3(1.0, 0.5, 1.0);
  reach.max += glm::dvec3(1.0, 0.5, 1.0);
  const glm::ivec3 low(cellOf(reach.min.x - kSize), cellOf(reach.min.y - kSize), cellOf(reach.min.z - kSize));
  const glm::ivec3 high(cellOf(reach.max.x + kSize), cellOf(reach.max.y + kSize), cellOf(reach.max.z + kSize));
  for (std::int32_t y = low.y; y <= high.y; ++y) {
    for (std::int32_t z = low.z; z <= high.z; ++z) {
      for (std::int32_t x = low.x; x <= high.x; ++x) {
        const auto [begin, end] = gridCell(x, y, z);
        for (std::size_t e = begin; e < end; ++e) {
          const std::size_t i = gridEntries[e].second;
          if ((flags[i] & kDead) != 0 || pickupDelay[i] > 0 || !overlaps(reach, bounds(i))) {
            continue;
          }
          const int taken = std::clamp(collector.take(stack[i]), 0, static_cast<int>(stack[i].count));
          if (taken > 0) {
            flights.push_back(Flight{.from = position[i], .orientation = orientation[i],
                                     .stack = ItemStack{.block = stack[i].block, .count = static_cast<std::uint16_t>(taken)},
                                     .toCollector = true});
          }
          stack[i].count = static_cast<std::uint16_t>(stack[i].count - taken);
          if (stack[i].count == 0) {
            flags[i] |= kDead;
          }
        }
      }
    }
  }
}

void ItemEntityWorld::eraseAt(std::size_t i) {
  const std::size_t last = size() - 1;
  if (i != last) {
    position[i] = position[last];
    previousPosition[i] = previousPosition[last];
    orientation[i] = orientation[last];
    previousOrientation[i] = previousOrientation[last];
    velocity[i] = velocity[last];
    angularVelocity[i] = angularVelocity[last];
    density[i] = density[last];
    stack[i] = stack[last];
    age[i] = age[last];
    pickupDelay[i] = pickupDelay[last];
    mergeDelay[i] = mergeDelay[last];
    flags[i] = flags[last];
    restTicks[i] = restTicks[last];
    rng[i] = rng[last];
  }
  position.pop_back();
  previousPosition.pop_back();
  orientation.pop_back();
  previousOrientation.pop_back();
  velocity.pop_back();
  angularVelocity.pop_back();
  density.pop_back();
  stack.pop_back();
  age.pop_back();
  pickupDelay.pop_back();
  mergeDelay.pop_back();
  flags.pop_back();
  restTicks.pop_back();
  rng.pop_back();
}

void ItemEntityWorld::removeDead() {
  for (std::size_t i = size(); i-- > 0;) {
    if ((flags[i] & kDead) != 0) {
      eraseAt(i);
    }
  }
}

void ItemEntityWorld::tick(const collision::CellQuery &world, JobPool *pool, const Collector *collector) {
  ++tickCount;
  const std::size_t awake =
      static_cast<std::size_t>(std::ranges::count_if(flags, [](std::uint8_t f) { return (f & kAsleep) == 0; }));
  const auto run = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      physics(i, world);
    }
  };
  if (pool != nullptr && awake >= kParallelThreshold) {
    pool->parallelFor(size(), 64, run);
  } else {
    run(0, size());
  }
  rebuildGrid();
  mergeNearby();
  if (collector != nullptr) {
    collect(*collector);
  }
  removeDead();
}

void ItemEntityWorld::wakeAround(std::int32_t x, std::int32_t y, std::int32_t z) {
  for (std::size_t i = 0; i < size(); ++i) {
    if ((flags[i] & kAsleep) == 0) {
      continue;
    }
    const glm::dvec3 &p = position[i];
    if (std::abs(cellOf(p.x) - x) <= 1 && std::abs(cellOf(p.z) - z) <= 1 && cellOf(p.y) >= y - 1 &&
        cellOf(p.y) <= y + 2) {
      flags[i] &= static_cast<std::uint8_t>(~kAsleep);
      restTicks[i] = 0;
    }
  }
}
