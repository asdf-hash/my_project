#pragma once

// Minecraft-style player controller: a 0.6 x 1.8 box simulated at 20 ticks per second with vanilla movement
// constants, colliding with the CPU voxel world (cpu_world.hpp). Rendering interpolates between ticks.
//
// Controls: WASD move, Space jump (double-tap: toggle flight), Shift sneak (fly: descend), Ctrl or double-tap W
// sprint, Z toggles freecam (the old free-fly camera; E cycles its speed).

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
  const glm::dvec3 &feetPosition() const { return position; }

private:
  struct Input {
    float forward = 0.0f; // +1 = W
    float strafe = 0.0f;  // +1 = D
    bool jump = false;
    bool sneak = false;
    bool sprint = false;
  };

  void tick(const Input &input, float yawDegrees, const cpuworld::CpuWorld &world);
  // Moves by `motion` with collision (Y, then X, then Z, like Minecraft); updates onGround / collision flags.
  void move(glm::dvec3 motion, bool sneakGuard, const cpuworld::CpuWorld &world);
  bool boxCollides(const glm::dvec3 &minCorner, const glm::dvec3 &maxCorner, const cpuworld::CpuWorld &world) const;
  double boxHeight() const;

  Mode currentMode = Mode::Walking;
  glm::dvec3 position{0.5, 100.0, 0.5}; // feet, center of the box's bottom face
  glm::dvec3 previousPosition = position;
  glm::dvec3 motion{0.0};
  bool onGround = false;
  bool collidedHorizontally = false;
  bool sprinting = false;
  bool sneaking = false;
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
