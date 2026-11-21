#include "player.hpp"

#include "camera.hpp"
#include "cpu_world.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

namespace {

// Vanilla constants (per tick, blocks).
constexpr double kTickSeconds = 1.0 / 20.0;
constexpr double kHalfWidth = 0.3;
constexpr double kStandingHeight = 1.8;
constexpr double kSneakingHeight = 1.5;
constexpr float kStandingEye = 1.62f;
constexpr float kSneakingEye = 1.27f;
constexpr double kGravity = 0.08;
constexpr double kVerticalDrag = 0.98;
constexpr double kJumpVelocity = 0.42;
constexpr double kSprintJumpBoost = 0.2;
constexpr double kGroundSlipperiness = 0.6;
constexpr double kAirFriction = 0.91;
constexpr double kWalkSpeed = 0.1;
constexpr double kSprintMultiplier = 1.3;
constexpr double kAirSpeed = 0.02;
constexpr double kSneakInputScale = 0.3;
constexpr double kFlySpeed = 0.05;
constexpr double kFlyVertical = kFlySpeed * 3.0;
constexpr double kFlyVerticalDamping = 0.6;
constexpr double kDoubleTapSeconds = 0.35; // 7 ticks
constexpr int kMaxTicksPerFrame = 10;      // after a stall, drop time instead of spiraling

} // namespace

void Player::spawn(const glm::dvec3 &feet) {
  position = previousPosition = feet;
  motion = glm::dvec3(0.0);
  frozen = true;
}

void Player::handleKey(int key, int action) {
  if (action != GLFW_PRESS) {
    return;
  }
  const double now = glfwGetTime();
  if (key == GLFW_KEY_Z) {
    currentMode = currentMode == Mode::Spectator ? Mode::Flying : Mode::Spectator;
    motion = glm::dvec3(0.0);
    previousPosition = position;
  }
  if (key == GLFW_KEY_E && currentMode == Mode::Spectator) {
    spectatorSpeedIndex = (spectatorSpeedIndex + 1) % spectatorSpeeds.size();
  }
  if (key == GLFW_KEY_SPACE && currentMode != Mode::Spectator) {
    if (now - lastJumpPress < kDoubleTapSeconds) {
      currentMode = currentMode == Mode::Flying ? Mode::Walking : Mode::Flying;
      motion.y = 0.0;
      lastJumpPress = -1.0;
    } else {
      lastJumpPress = now;
    }
  }
  if (key == GLFW_KEY_W) {
    if (now - lastForwardPress < kDoubleTapSeconds) {
      sprintFromDoubleTap = true;
    }
    lastForwardPress = now;
  }
}

double Player::boxHeight() const { return sneaking ? kSneakingHeight : kStandingHeight; }

glm::vec3 Player::eyePosition() const {
  const double alpha = std::clamp(tickAccumulator / kTickSeconds, 0.0, 1.0);
  const glm::dvec3 feet = currentMode == Mode::Spectator ? position : glm::mix(previousPosition, position, alpha);
  return glm::vec3(feet) + glm::vec3(0.0f, eyeHeight, 0.0f);
}

void Player::update(GLFWwindow *window, float deltaTime, const Camera &camera, const cpuworld::CpuWorld &world) {
  const bool captured = camera.cursorCaptured();
  auto down = [&](int key) { return captured && glfwGetKey(window, key) == GLFW_PRESS; };

  if (currentMode == Mode::Spectator) {
    // The old free-fly camera: no collision, no ticks.
    const glm::vec3 front = camera.front();
    const glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    const double step = spectatorSpeeds[spectatorSpeedIndex] * static_cast<double>(deltaTime);
    glm::dvec3 delta(0.0);
    if (down(GLFW_KEY_W)) delta += glm::dvec3(front);
    if (down(GLFW_KEY_S)) delta -= glm::dvec3(front);
    if (down(GLFW_KEY_D)) delta += glm::dvec3(right);
    if (down(GLFW_KEY_A)) delta -= glm::dvec3(right);
    if (down(GLFW_KEY_SPACE)) delta.y += 1.0;
    if (down(GLFW_KEY_LEFT_SHIFT)) delta.y -= 1.0;
    position += delta * step;
    previousPosition = position;
    eyeHeight = kStandingEye;
    fovScale = 1.0f;
    frozen = false;
    return;
  }

  // Hold still until the terrain under the player is in RAM (Minecraft's "loading terrain").
  const int32_t columnX = cpuworld::floorDiv(static_cast<int32_t>(std::floor(position.x)), CHUNK_SIZE);
  const int32_t columnZ = cpuworld::floorDiv(static_cast<int32_t>(std::floor(position.z)), CHUNK_SIZE);
  frozen = !world.columnLoaded(columnX, columnZ);
  if (frozen) {
    tickAccumulator = 0.0;
    previousPosition = position;
    return;
  }
  // Spawned inside terrain (or the terrain changed): pop up to the first free spot.
  while (boxCollides(position + glm::dvec3(-kHalfWidth, 0.0, -kHalfWidth),
                     position + glm::dvec3(kHalfWidth, boxHeight(), kHalfWidth), world) &&
         position.y < world::kMaxTerrainHeight + 2) {
    position.y = std::floor(position.y) + 1.0;
    previousPosition = position;
  }

  Input input;
  input.forward = (down(GLFW_KEY_W) ? 1.0f : 0.0f) - (down(GLFW_KEY_S) ? 1.0f : 0.0f);
  input.strafe = (down(GLFW_KEY_D) ? 1.0f : 0.0f) - (down(GLFW_KEY_A) ? 1.0f : 0.0f);
  input.jump = down(GLFW_KEY_SPACE);
  input.sneak = down(GLFW_KEY_LEFT_SHIFT);
  if (!down(GLFW_KEY_W)) {
    sprintFromDoubleTap = false;
  }
  input.sprint = down(GLFW_KEY_LEFT_CONTROL) || sprintFromDoubleTap;

  tickAccumulator += static_cast<double>(deltaTime);
  int ticks = 0;
  while (tickAccumulator >= kTickSeconds) {
    tickAccumulator -= kTickSeconds;
    if (++ticks > kMaxTicksPerFrame) {
      tickAccumulator = 0.0;
      break;
    }
    previousPosition = position;
    tick(input, camera.yaw(), world);
  }

  // Smooth the eye height and FOV toward their targets (per frame, frame-rate independent).
  const float blend = 1.0f - std::exp(-deltaTime * 15.0f);
  const float targetEye = sneaking && currentMode == Mode::Walking ? kSneakingEye : kStandingEye;
  eyeHeight += (targetEye - eyeHeight) * blend;
  const float targetFov = sprinting ? 1.1f : 1.0f;
  fovScale += (targetFov - fovScale) * blend;
}

void Player::tick(const Input &input, float yawDegrees, const cpuworld::CpuWorld &world) {
  const bool flying = currentMode == Mode::Flying;

  // Sneaking shrinks the box; standing back up needs room for the full height.
  if (flying) {
    sneaking = false;
  } else if (input.sneak) {
    sneaking = true;
  } else if (sneaking) {
    const bool blocked = boxCollides(position + glm::dvec3(-kHalfWidth, 0.0, -kHalfWidth),
                                     position + glm::dvec3(kHalfWidth, kStandingHeight, kHalfWidth), world);
    sneaking = blocked;
  }

  double forward = input.forward;
  double strafe = input.strafe;
  if (sneaking) {
    forward *= kSneakInputScale;
    strafe *= kSneakInputScale;
  }

  // Sprint: needs forward input; stops on sneaking or running into a wall.
  if (input.sprint && input.forward > 0.0f && !sneaking) {
    sprinting = true;
  }
  if (input.forward <= 0.0f || sneaking || collidedHorizontally) {
    sprinting = false;
  }

  const double yaw = glm::radians(static_cast<double>(yawDegrees));
  // Camera yaw -90 faces -z; forward is (cos yaw, sin yaw) in x/z, right is (-sin yaw, cos yaw).
  const glm::dvec2 forwardDir(std::cos(yaw), std::sin(yaw));
  const glm::dvec2 rightDir(-std::sin(yaw), std::cos(yaw));

  if (flying) {
    if (input.sneak) motion.y -= kFlyVertical;
    if (input.jump) motion.y += kFlyVertical;
  } else if (input.jump && onGround) {
    motion.y = kJumpVelocity;
    if (sprinting) {
      motion.x += forwardDir.x * kSprintJumpBoost;
      motion.z += forwardDir.y * kSprintJumpBoost;
    }
  }

  // Acceleration from input (Entity.moveRelative): input normalized to at most 1, times the movement factor.
  double speed;
  double horizontalFriction;
  if (flying) {
    speed = sprinting ? kFlySpeed * 2.0 : kFlySpeed;
    horizontalFriction = kAirFriction;
  } else if (onGround) {
    const double slipperiness = kGroundSlipperiness * kAirFriction;
    // 0.16277136 / slipperiness^3 == 1 for normal blocks: walking speed comes out as the base movement speed.
    speed = kWalkSpeed * (sprinting ? kSprintMultiplier : 1.0) * (0.16277136 / (slipperiness * slipperiness * slipperiness));
    horizontalFriction = slipperiness;
  } else {
    speed = kAirSpeed * (sprinting ? kSprintMultiplier : 1.0);
    horizontalFriction = kAirFriction;
  }
  double inputLength = std::sqrt(forward * forward + strafe * strafe);
  if (inputLength >= 1.0e-4) {
    inputLength = std::max(inputLength, 1.0);
    const double scale = speed / inputLength;
    const double f = forward * scale;
    const double s = strafe * scale;
    motion.x += forwardDir.x * f + rightDir.x * s;
    motion.z += forwardDir.y * f + rightDir.y * s;
  }

  move(motion, sneaking && onGround && !flying, world);

  if (flying) {
    motion.y *= kFlyVerticalDamping;
    if (onGround) {
      currentMode = Mode::Walking; // landing ends flight
    }
  } else {
    motion.y = (motion.y - kGravity) * kVerticalDrag;
  }
  motion.x *= horizontalFriction;
  motion.z *= horizontalFriction;
  // Minecraft zeroes tiny velocities so the player comes to a full stop.
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(motion[axis]) < 0.003) {
      motion[axis] = 0.0;
    }
  }
}

bool Player::boxCollides(const glm::dvec3 &minCorner, const glm::dvec3 &maxCorner,
                         const cpuworld::CpuWorld &world) const {
  constexpr double eps = 1.0e-7;
  const int x0 = static_cast<int>(std::floor(minCorner.x + eps)), x1 = static_cast<int>(std::floor(maxCorner.x - eps));
  const int y0 = static_cast<int>(std::floor(minCorner.y + eps)), y1 = static_cast<int>(std::floor(maxCorner.y - eps));
  const int z0 = static_cast<int>(std::floor(minCorner.z + eps)), z1 = static_cast<int>(std::floor(maxCorner.z - eps));
  for (int y = y0; y <= y1; ++y)
    for (int z = z0; z <= z1; ++z)
      for (int x = x0; x <= x1; ++x)
        if (world.isSolid(x, y, z)) return true;
  return false;
}

void Player::move(glm::dvec3 delta, bool sneakGuard, const cpuworld::CpuWorld &world) {
  const double height = boxHeight();
  glm::dvec3 boxMin = position + glm::dvec3(-kHalfWidth, 0.0, -kHalfWidth);
  glm::dvec3 boxMax = position + glm::dvec3(kHalfWidth, height, kHalfWidth);
  const glm::dvec3 wanted = delta;

  // Sneaking on the ground: shrink the horizontal move until the box would still stand on something.
  if (sneakGuard) {
    constexpr double step = 0.05;
    auto supported = [&](double dx, double dz) {
      return boxCollides(boxMin + glm::dvec3(dx, -1.0, dz), boxMax + glm::dvec3(dx, -height, dz), world);
    };
    while (delta.x != 0.0 && !supported(delta.x, 0.0)) {
      delta.x = std::abs(delta.x) < step ? 0.0 : delta.x - std::copysign(step, delta.x);
    }
    while (delta.z != 0.0 && !supported(0.0, delta.z)) {
      delta.z = std::abs(delta.z) < step ? 0.0 : delta.z - std::copysign(step, delta.z);
    }
    while (delta.x != 0.0 && delta.z != 0.0 && !supported(delta.x, delta.z)) {
      delta.x = std::abs(delta.x) < step ? 0.0 : delta.x - std::copysign(step, delta.x);
      delta.z = std::abs(delta.z) < step ? 0.0 : delta.z - std::copysign(step, delta.z);
    }
  }

  // Clip the move along one axis against every solid block the swept box touches (AxisAlignedBB.calculate*Offset).
  auto clipAxis = [&](int axis, double d) {
    if (d == 0.0) {
      return 0.0;
    }
    glm::dvec3 sweptMin = boxMin, sweptMax = boxMax;
    if (d > 0.0) sweptMax[axis] += d;
    else sweptMin[axis] += d;
    constexpr double eps = 1.0e-7;
    const int x0 = static_cast<int>(std::floor(sweptMin.x + eps)), x1 = static_cast<int>(std::floor(sweptMax.x - eps));
    const int y0 = static_cast<int>(std::floor(sweptMin.y + eps)), y1 = static_cast<int>(std::floor(sweptMax.y - eps));
    const int z0 = static_cast<int>(std::floor(sweptMin.z + eps)), z1 = static_cast<int>(std::floor(sweptMax.z - eps));
    for (int y = y0; y <= y1; ++y)
      for (int z = z0; z <= z1; ++z)
        for (int x = x0; x <= x1; ++x) {
          if (!world.isSolid(x, y, z)) continue;
          const glm::dvec3 blockMin(x, y, z);
          const glm::dvec3 blockMax = blockMin + 1.0;
          if (d > 0.0 && blockMin[axis] >= boxMax[axis] - eps) {
            d = std::min(d, blockMin[axis] - boxMax[axis]);
          } else if (d < 0.0 && blockMax[axis] <= boxMin[axis] + eps) {
            d = std::max(d, blockMax[axis] - boxMin[axis]);
          }
        }
    boxMin[axis] += d;
    boxMax[axis] += d;
    return d;
  };
  delta.y = clipAxis(1, delta.y);
  delta.x = clipAxis(0, delta.x);
  delta.z = clipAxis(2, delta.z);

  position = glm::dvec3((boxMin.x + boxMax.x) * 0.5, boxMin.y, (boxMin.z + boxMax.z) * 0.5);
  const bool collidedVertically = delta.y != wanted.y;
  collidedHorizontally = delta.x != wanted.x || delta.z != wanted.z;
  onGround = collidedVertically && wanted.y < 0.0;
  if (delta.x != wanted.x) motion.x = 0.0;
  if (delta.z != wanted.z) motion.z = 0.0;
  if (collidedVertically) motion.y = 0.0;
}
