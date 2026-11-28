#pragma once

// Dropped item entities as rigid bodies: 0.25 m cubes with mass from their block's density, tumbling under gravity,
// buoyancy (submerged volume) and drag, colliding with the voxel terrain through contact points on the cube
// (corners, edge midpoints, face centers) solved with sequential impulses (restitution, Coulomb friction). Items
// don't collide with each other; they merge into stacks of up to kMaxEntityStack like vanilla, get picked up and
// despawn. Units: blocks = metres, seconds. Stored as parallel arrays; the per-body physics runs on a JobPool,
// merging / pickup / removal serially after it.

#include "collision.hpp"
#include "item_stack.hpp"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class JobPool;

class ItemEntityWorld {
public:
  static constexpr std::size_t kMaxEntities = 8192;
  static constexpr int kLifetimeTicks = 6000;  // 5 minutes
  static constexpr int kDropPickupDelay = 10;  // broken blocks (vanilla setDefaultPickUpDelay)
  static constexpr int kThrowPickupDelay = 40; // thrown by the player
  static constexpr int kThrowMergeDelay = 40;  // thrown items merge after 2 seconds
  static constexpr double kTickSeconds = 1.0 / 20.0;
  static constexpr double kHalfSize = 0.125;   // cube edge 0.25 m
  static constexpr double kSize = 2.0 * kHalfSize;
  static constexpr double kVolume = kSize * kSize * kSize;
  static constexpr double kGravity = 16.0;       // m/s^2: vanilla's item gravity (0.04 blocks / tick^2)
  static constexpr double kWaterDensity = 1000.0; // kg/m^3
  static constexpr double kAirDensity = 1.2;
  static constexpr double kRestitution = 0.25;
  static constexpr double kFriction = 0.6;
  static constexpr int kSleepTicks = 10;          // at rest this long: stop simulating until woken
  static constexpr std::size_t kParallelThreshold = 512; // awake bodies below this tick on one thread

  enum Flags : std::uint8_t { kOnGround = 1, kInWater = 2, kAsleep = 4, kDead = 8 };

  // Per-body state (index i across every array).
  std::vector<glm::dvec3> position;         // cube center
  std::vector<glm::dvec3> previousPosition; // last tick's, for render interpolation
  std::vector<glm::dquat> orientation;
  std::vector<glm::dquat> previousOrientation;
  std::vector<glm::dvec3> velocity;        // m/s
  std::vector<glm::dvec3> angularVelocity; // rad/s, world space
  std::vector<double> density;             // kg/m^3 (mass = density * kVolume)
  std::vector<ItemStack> stack;            // count up to kMaxEntityStack
  std::vector<std::int32_t> age;           // ticks
  std::vector<std::int32_t> pickupDelay;   // ticks until the player may collect it
  std::vector<std::int32_t> mergeDelay;    // ticks until this item may merge
  std::vector<std::uint8_t> flags;
  std::vector<std::uint16_t> restTicks; // consecutive ticks nearly at rest
  std::vector<std::uint32_t> rng;       // xorshift32 state (never shared across threads)

  std::size_t size() const { return position.size(); }

  /**
   * Add a body; over kMaxEntities the oldest is replaced.
   *
   * @param center Cube center.
   * @param velocity m/s.
   * @param spin Angular velocity, rad/s.
   * @param density kg/m^3 of the block (decides floating and sinking).
   * @param seed Mixed into the body's random state (orientation, push-out speed).
   */
  void spawn(const glm::dvec3 &center, const glm::dvec3 &velocity, const glm::dvec3 &spin, double density,
             ItemStack stack, int pickupDelay, std::uint32_t seed, int mergeDelay = 0);

  // The player for pickup: its box and a callback that takes as much of a stack as fits and returns that count.
  struct Collector {
    collision::Box box;
    std::function<int(const ItemStack &)> take;
  };

  /**
   * Advance one 20 Hz tick.
   *
   * @param world Cell queries; must not be edited during the call.
   * @param pool Runs the per-body physics in parallel (null: this thread only).
   * @param collector The player, or null (spectator / no pickup).
   */
  void tick(const collision::CellQuery &world, JobPool *pool, const Collector *collector);

  /** Wake sleeping bodies near a changed cell (a block broken or placed). */
  void wakeAround(std::int32_t x, std::int32_t y, std::int32_t z);

  /** Axis-aligned bounds of body i (the rotated cube). */
  collision::Box bounds(std::size_t i) const;

  std::uint64_t ticks() const { return tickCount; }

  // An item that left the world this tick and should be shown flying somewhere: into the collector (pickup) or
  // into the stack it merged with. The renderer animates these; the simulation is already done with them.
  struct Flight {
    glm::dvec3 from;
    glm::dquat orientation;
    ItemStack stack;
    bool toCollector = false; // else toward `to`
    glm::dvec3 to{0.0};
  };
  /** Flights since the last call (cleared by it). */
  std::vector<Flight> takeFlights() { return std::exchange(flights, {}); }

private:
  /** One tick of rigid-body physics for body i (substepped). */
  void physics(std::size_t i, const collision::CellQuery &world);
  /** One substep: forces, integration, terrain contacts. @return whether any contact touched. */
  bool substep(std::size_t i, double dt, const collision::CellQuery &world, bool &onGround, bool &inWater);
  /** Whether a resting body is still held up by terrain (asleep re-check). */
  bool supported(std::size_t i, const collision::CellQuery &world) const;
  void mergeNearby();
  void collect(const Collector &collector);
  void removeDead();
  void rebuildGrid();
  /** Bodies whose 1-block grid cell is (x, y, z), as a range into gridEntries. */
  std::pair<std::size_t, std::size_t> gridCell(std::int32_t x, std::int32_t y, std::int32_t z) const;
  void eraseAt(std::size_t i);

  std::uint64_t tickCount = 0;
  std::vector<Flight> flights;
  // Spatial grid: (cell key, body) sorted by key, rebuilt every tick.
  std::vector<std::pair<std::uint64_t, std::uint32_t>> gridEntries;
};
