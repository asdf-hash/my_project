#pragma once

// Minecraft-style player controller: a 0.6 x 1.8 box simulated at 20 ticks per second with vanilla movement
// constants, colliding with the CPU voxel world (cpu_world.hpp). Rendering interpolates between ticks.
//
// Controls: WASD move, Space jump (double-tap: toggle flight), Shift sneak (fly: descend), Ctrl or double-tap W
// sprint, Z toggles freecam (the old free-fly camera; E cycles its speed).
// In water: Space swims up, Shift sinks; sprinting while submerged sprint-swims (fast, follows the look pitch, in a
// 0.6-block box that fits 1-block gaps).

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
  enum class Mode { Walking, Flying, Spectator };

  // Places the player's feet at `feet` (block coordinates; the spawn column must load before physics starts).
  void spawn(const glm::dvec3 &feet);
  void handleKey(int key, int action);
  void update(GLFWwindow *window, float deltaTime, const Camera &camera, const cpuworld::CpuWorld &world);

  glm::vec3 eyePosition() const;
  // Multiplier for the field of view (sprinting widens it, as in Minecraft).
  float fovMultiplier() const { return fovScale; }
  Mode mode() const { return currentMode; }
  bool waitingForTerrain() const { return frozen; }
  bool swimming() const { return swimmingPose; }
  const glm::dvec3 &feetPosition() const { return position; }
  // The player's box overlaps block (x, y, z) (touching a face doesn't count): placing it there would trap them.
  bool overlapsBlock(int x, int y, int z) const;

private:
  struct Input {
    float forward = 0.0f; // +1 = W
    float strafe = 0.0f;  // +1 = D
    bool jump = false;
    bool sneak = false;
    bool sprint = false;
  };

  // lookY: vertical component of the camera's look direction (steers sprint-swimming).
  void tick(const Input &input, float yawDegrees, float lookY, const cpuworld::CpuWorld &world);
  // Moves by `motion` with collision (Y, then X, then Z, like Minecraft); updates onGround / collision flags.
  void move(glm::dvec3 motion, bool sneakGuard, const cpuworld::CpuWorld &world);
  bool boxCollides(const glm::dvec3 &minCorner, const glm::dvec3 &maxCorner, const cpuworld::CpuWorld &world) const;
  // Any cell the box overlaps is ocean (same cell loop as boxCollides).
  bool boxInWater(const glm::dvec3 &minCorner, const glm::dvec3 &maxCorner, const cpuworld::CpuWorld &world) const;
  double boxHeight() const;

  Mode currentMode = Mode::Walking;
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

  // Double taps (seconds, glfwGetTime) and the spectator camera.
  double lastJumpPress = -1.0;
  double lastForwardPress = -1.0;
  bool sprintFromDoubleTap = false;
  std::size_t spectatorSpeedIndex = 1;
  std::array<float, 3> spectatorSpeeds{10.f, 100.f, 1000.f};
};
