#pragma once

// Minecraft-style player controller: a 0.6 x 1.8 box simulated at 20 ticks per second with vanilla movement
// constants, colliding with the CPU voxel world (cpu_world.hpp). Rendering interpolates between ticks.
//
// Walking steps up ledges up to 1 block tall without jumping.
// Controls: WASD move, Space jump (double-tap: toggle flight), Shift sneak (fly: descend), Ctrl or double-tap W
// sprint. Game modes change only through the F3+F4 switcher (renderer.cpp); in spectator, E or the scroll wheel
// cycles the fly speed.
// Game modes follow Minecraft: survival and adventure take damage and can't fly; creative flies (double-tap Space)
// and is invulnerable; spectator flies through blocks. Which blocks can be broken or placed is up to the renderer.
// In water: Space swims up, Shift sinks; sprinting while submerged sprint-swims (fast, follows the look pitch, in a
// 0.6-block box that fits 1-block gaps).
// Health: 20 points (10 hearts). Fall damage (see player.cpp) is harmless under 2.5 blocks and fatal around 15; high
// falls into water hurt too. Flying resets the fall. One point regenerates every 4 seconds; at 0 the player respawns.

#include <glm/glm.hpp>

#include <array>
#include <cstddef>

struct GLFWwindow;
class Camera;

namespace cpuworld {
class CpuWorld;
}

class Player {
public:
  // Movement state; Spectator exactly when the game mode is spectator.
  enum class Mode { Walking, Flying, Spectator };
  // Minecraft's game modes, in the F3+F4 switcher's order.
  enum class GameMode { Creative, Survival, Adventure, Spectator };
  static constexpr int kGameModeCount = 4;

  /** Switch to spectator (testing: ENGINE_SPAWN with a height above the ground). */
  void setSpectator() { setGameMode(GameMode::Spectator); }

  /**
   * Change the game mode like Minecraft's /gamemode: leaving spectator lands creative players still flying and
   * everyone else walking (falling from where they are); entering survival or adventure ends flight.
   */
  void setGameMode(GameMode mode);
  /** Current game mode. */
  GameMode gameMode() const { return currentGameMode; }
  /** The game mode before the last change (where the F3+F4 switcher starts). */
  GameMode previousGameMode() const { return lastGameMode; }
  /** Survival or adventure: takes damage, shows health, hunger and XP. */
  bool survivalLike() const { return currentGameMode == GameMode::Survival || currentGameMode == GameMode::Adventure; }

  /**
   * Scroll the spectator speed (Minecraft's spectator scroll).
   *
   * @param steps Wheel steps, positive = faster.
   */
  void scrollSpectatorSpeed(int steps);

  /**
   * Place the player and make it the respawn point. Physics stays frozen until the spawn column loads.
   *
   * @param feet Feet position (center of the box's bottom face), in blocks.
   */
  void spawn(const glm::dvec3 &feet);

  /**
   * Handle a key event: E spectator speed, Space double-tap flight (creative), W double-tap sprint.
   *
   * @param key GLFW key code.
   * @param action GLFW_PRESS / GLFW_RELEASE / GLFW_REPEAT (only presses matter).
   */
  void handleKey(int key, int action);

  /**
   * Advance the player by one frame: read held keys, run the fixed 20 Hz ticks, smooth eye height and FOV.
   *
   * @param deltaTime Frame time, in seconds.
   * @param camera Supplies the look direction and whether the cursor is captured.
   * @param world Collision and water queries.
   */
  void update(GLFWwindow *window, float deltaTime, const Camera &camera, const cpuworld::CpuWorld &world);

  /** Eye position for rendering, interpolated between ticks. */
  glm::vec3 eyePosition() const;
  /** Field-of-view multiplier (sprinting widens it, as in Minecraft). */
  float fovMultiplier() const { return fovScale; }
  /** Current movement mode. */
  Mode mode() const { return currentMode; }
  /** Whether physics is paused until the terrain under the player loads. */
  bool waitingForTerrain() const { return frozen; }
  /** Whether the player is in the sprint-swim (or crawl) pose. */
  bool swimming() const { return swimmingPose; }
  /** Standing on a block (as of the last tick). */
  bool grounded() const { return onGround; }
  /** Eyes inside water (as of the last tick). */
  bool eyesUnderwater() const { return eyesInWater; }
  /** Feet position (center of the box's bottom face), in blocks. */
  const glm::dvec3 &feetPosition() const { return position; }

  /**
   * Test whether placing a block at (x, y, z) would trap the player.
   *
   * @return true if the player's box overlaps the block (touching a face doesn't count); always false in spectator.
   */
  bool overlapsBlock(int x, int y, int z) const;

  static constexpr float kMaxHealth = 20.0f;
  /** Current health points. */
  float health() const { return hp; }
  /** Just took damage (the hearts flash for half a second). */
  bool hurtFlashing() const { return hurtTicks > 0; }

  /**
   * Take damage, respawning at 0 health. Ignored outside survival and adventure.
   *
   * @param amount Health points to remove.
   */
  void damage(float amount);

private:
  struct Input {
    float forward = 0.0f; // +1 = W
    float strafe = 0.0f;  // +1 = D
    bool jump = false;
    bool sneak = false;
    bool sprint = false;
  };

  /**
   * Run one 20 Hz physics tick: water state, pose, input acceleration, collision, health, drag.
   *
   * @param yawDegrees Camera yaw, in degrees.
   * @param lookY Vertical component of the camera's look direction (steers sprint-swimming).
   */
  void tick(const Input &input, float yawDegrees, float lookY, const cpuworld::CpuWorld &world);

  /**
   * Move by `delta` with collision (Y, then X, then Z, like Minecraft), stepping up ledges when walking.
   * Updates position, onGround, collidedHorizontally and zeroes blocked motion components.
   *
   * @param delta Requested displacement this tick, in blocks.
   * @param sneakGuard Shrink the horizontal move so the player can't walk off an edge.
   */
  void move(glm::dvec3 delta, bool sneakGuard, const cpuworld::CpuWorld &world);

  /**
   * Test an axis-aligned box against collidable blocks.
   *
   * @return true if any cell the box overlaps (touching a face doesn't count) is collidable.
   */
  bool boxCollides(const glm::dvec3 &minCorner, const glm::dvec3 &maxCorner, const cpuworld::CpuWorld &world) const;

  /**
   * Test an axis-aligned box against ocean cells (same cell loop as boxCollides).
   *
   * @return true if any cell the box overlaps is ocean.
   */
  bool boxInWater(const glm::dvec3 &minCorner, const glm::dvec3 &maxCorner, const cpuworld::CpuWorld &world) const;

  /** Current box height for the pose (standing, sneaking or swimming), in blocks. */
  double boxHeight() const;

  /**
   * After a tick's move: accumulate the fall while airborne and turn it into damage on landing.
   *
   * @param deltaY Vertical displacement this tick, in blocks.
   */
  void updateFall(double deltaY);

  /** Reset to the spawn point with full health, waiting for its terrain to load. */
  void respawn();

  Mode currentMode = Mode::Walking;
  GameMode currentGameMode = GameMode::Creative;
  GameMode lastGameMode = GameMode::Survival; // so the first F3+F4 starts on survival, like Minecraft
  glm::dvec3 position{0.5, 100.0, 0.5}; // feet, center of the box's bottom face
  glm::dvec3 previousPosition = position;
  glm::dvec3 motion{0.0};
  bool onGround = false;
  bool collidedHorizontally = false;
  bool sprinting = false;
  bool sneaking = false;
  bool swimmingPose = false; // sprint-swimming (or crawling under a ceiling after it)
  bool inWater = false;
  bool eyesInWater = false;
  bool frozen = true;
  double tickAccumulator = 0.0;
  float fovScale = 1.0f;
  float eyeHeight = 1.62f;
  float stepLift = 0.0f; // blocks the camera still lags below the feet after stepping up a ledge (decays per frame)

  // Health (ticks are 20 per second).
  glm::dvec3 spawnPoint{0.5, 100.0, 0.5};
  float hp = kMaxHealth;
  double fallDistance = 0.0;
  int hurtTicks = 0;
  int regenTicks = 0;

  // Double taps (seconds, glfwGetTime) and the spectator camera.
  double lastJumpPress = -1.0;
  double lastForwardPress = -1.0;
  bool sprintFromDoubleTap = false;
  std::size_t spectatorSpeedIndex = 1;
  std::array<float, 3> spectatorSpeeds{10.f, 100.f, 1000.f};
};
