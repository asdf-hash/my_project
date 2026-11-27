#include "player.hpp"

#include "camera.hpp"
#include "cpu_world.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

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
// Water (vanilla 1.13+ LivingEntity.travel).
constexpr double kWaterSpeed = 0.02;
constexpr double kWaterFriction = 0.8;
constexpr double kSwimFriction = 0.9;
constexpr double kWaterVerticalDrag = 0.8;
constexpr double kWaterGravity = 0.005; // kGravity / 16
constexpr double kWaterVerticalPush = 0.04;
constexpr double kLedgeJump = 0.3;
constexpr double kWaterBoxShrink = 0.4;  // grow(0, -0.4, 0)
constexpr double kEyeWaterOffset = 0.11; // vanilla eye-in-fluid check
constexpr double kSwimmingHeight = 0.6;
constexpr float kSwimmingEye = 0.4f;
constexpr double kDoubleTapSeconds = 0.35; // 7 ticks
constexpr int kMaxTicksPerFrame = 10;      // after a stall, drop time instead of spiraling
// Fall damage follows the real-world injury of the same drop height (1 block = 1 m), not Minecraft's ~32 m/s^2
// gravity: superlinear in height, +-20% per landing. Water only cushions dives under ~12 m.
constexpr double kSafeFallHeight = 2.5;
constexpr double kFatalFallHeight = 15.0;
constexpr double kSafeWaterFallHeight = 12.0;
constexpr double kFatalWaterFallHeight = 50.0;
constexpr double kFallDamageExponent = 1.5;
constexpr double kFallDamageVariation = 0.2;
constexpr int kHurtFlashTicks = 10;
constexpr double kStepHeight = 1.0;      // walking up ledges this tall without jumping (vanilla: 0.6)
constexpr float kStepSmoothing = 12.0f;  // camera catch-up rate after a step (1/s)
constexpr int kRegenIntervalTicks = 80;   // one health point per 4 seconds

} // namespace

void Player::spawn(const glm::dvec3 &feet) {
  stepLift = 0.0f;
  spawnPoint = feet;
  position = previousPosition = feet;
  motion = glm::dvec3(0.0);
  fallDistance = 0.0;
  frozen = true;
}

void Player::setGameMode(GameMode mode) {
  if (mode == currentGameMode) {
    return;
  }
  lastGameMode = currentGameMode;
  currentGameMode = mode;
  fallDistance = 0.0;
  if (mode == GameMode::Spectator) {
    currentMode = Mode::Spectator;
    motion = glm::dvec3(0.0);
    previousPosition = position;
    sprinting = sneaking = swimmingPose = false;
  } else if (mode == GameMode::Creative) {
    if (currentMode == Mode::Spectator) {
      currentMode = Mode::Flying;
    }
  } else {
    currentMode = Mode::Walking;
  }
}

void Player::scrollSpectatorSpeed(int steps) {
  const int count = static_cast<int>(spectatorSpeeds.size());
  const int index = std::clamp(static_cast<int>(spectatorSpeedIndex) + steps, 0, count - 1);
  spectatorSpeedIndex = static_cast<std::size_t>(index);
}

void Player::damage(float amount) {
  if (amount <= 0.0f || !survivalLike()) {
    return;
  }
  hp = std::max(hp - amount, 0.0f);
  hurtTicks = kHurtFlashTicks;
  regenTicks = 0;
  if (hp <= 0.0f) {
    respawn();
  }
}

void Player::respawn() {
  stepLift = 0.0f;
  std::clog << "You died\n";
  currentMode = Mode::Walking; // only survival and adventure die
  position = previousPosition = spawnPoint;
  motion = glm::dvec3(0.0);
  fallDistance = 0.0;
  hp = kMaxHealth;
  sprinting = sneaking = swimmingPose = false;
  frozen = true; // wait for the spawn column, like spawn()
}

namespace {

/**
 * Health points lost from a fall.
 *
 * @param height Fall distance, in blocks.
 * @param safeHeight Largest harmless fall (ground or water constant).
 * @param fatalHeight Fall that costs about kMaxHealth (ground or water constant).
 * @return whole health points, randomized by +-kFallDamageVariation.
 */
float fallDamage(double height, double safeHeight, double fatalHeight) {
  if (height <= safeHeight) {
    return 0.0f;
  }
  static std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<double> variation(1.0 - kFallDamageVariation, 1.0 + kFallDamageVariation);
  const double severity = (height - safeHeight) / (fatalHeight - safeHeight);
  const double points = Player::kMaxHealth * std::pow(severity, kFallDamageExponent) * variation(rng);
  return static_cast<float>(std::round(points)); // whole points; a tiny jolt rounds to nothing
}

} // namespace

void Player::updateFall(double deltaY) {
  if (currentMode != Mode::Walking) {
    fallDistance = 0.0;
    return;
  }
  if (inWater) {
    // Only the tick you hit the water carries a fall; it resets every tick after.
    damage(fallDamage(fallDistance, kSafeWaterFallHeight, kFatalWaterFallHeight));
    fallDistance = 0.0;
    return;
  }
  if (onGround) {
    damage(fallDamage(fallDistance, kSafeFallHeight, kFatalFallHeight));
    fallDistance = 0.0;
  } else if (deltaY < 0.0) {
    fallDistance -= deltaY;
  }
}

void Player::handleKey(int key, int action) {
  if (action != GLFW_PRESS) {
    return;
  }
  const double now = glfwGetTime();
  if (key == GLFW_KEY_E && currentMode == Mode::Spectator) {
    spectatorSpeedIndex = (spectatorSpeedIndex + 1) % spectatorSpeeds.size();
  }
  if (key == GLFW_KEY_SPACE && currentGameMode == GameMode::Creative) {
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

double Player::boxHeight() const {
  if (swimmingPose) return kSwimmingHeight;
  return sneaking ? kSneakingHeight : kStandingHeight;
}

glm::vec3 Player::eyePosition() const {
  const double alpha = std::clamp(tickAccumulator / kTickSeconds, 0.0, 1.0);
  const glm::dvec3 feet = currentMode == Mode::Spectator ? position : glm::mix(previousPosition, position, alpha);
  return glm::vec3(feet) + glm::vec3(0.0f, eyeHeight - stepLift, 0.0f);
}

void Player::update(GLFWwindow *window, float deltaTime, const Camera &camera, const cpuworld::CpuWorld &world) {
  const bool captured = camera.cursorCaptured();
  auto isKeyDown = [&](int key) { return captured && glfwGetKey(window, key) == GLFW_PRESS; };

  if (currentMode == Mode::Spectator) {
    // The old free-fly camera: no collision, no ticks.
    const glm::vec3 front = camera.front();
    const glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    const double step = spectatorSpeeds[spectatorSpeedIndex] * static_cast<double>(deltaTime);
    glm::dvec3 delta(0.0);
    if (isKeyDown(GLFW_KEY_W)) delta += glm::dvec3(front);
    if (isKeyDown(GLFW_KEY_S)) delta -= glm::dvec3(front);
    if (isKeyDown(GLFW_KEY_D)) delta += glm::dvec3(right);
    if (isKeyDown(GLFW_KEY_A)) delta -= glm::dvec3(right);
    if (isKeyDown(GLFW_KEY_SPACE)) delta.y += 1.0;
    if (isKeyDown(GLFW_KEY_LEFT_SHIFT)) delta.y -= 1.0;
    position += delta * step;
    previousPosition = position;
    stepLift = 0.0f;
    eyeHeight = kStandingEye;
    fovScale = 1.0f;
    swimmingPose = false;
    sneaking = false;
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
  input.forward = (isKeyDown(GLFW_KEY_W) ? 1.0f : 0.0f) - (isKeyDown(GLFW_KEY_S) ? 1.0f : 0.0f);
  input.strafe = (isKeyDown(GLFW_KEY_D) ? 1.0f : 0.0f) - (isKeyDown(GLFW_KEY_A) ? 1.0f : 0.0f);
  input.jump = isKeyDown(GLFW_KEY_SPACE);
  input.sneak = isKeyDown(GLFW_KEY_LEFT_SHIFT);
  if (!isKeyDown(GLFW_KEY_W)) {
    sprintFromDoubleTap = false;
  }
  input.sprint = isKeyDown(GLFW_KEY_LEFT_CONTROL) || sprintFromDoubleTap;

  tickAccumulator += static_cast<double>(deltaTime);
  int ticks = 0;
  while (tickAccumulator >= kTickSeconds) {
    tickAccumulator -= kTickSeconds;
    if (++ticks > kMaxTicksPerFrame) {
      tickAccumulator = 0.0;
      break;
    }
    previousPosition = position;
    tick(input, camera.yaw(), camera.front().y, world);
    if (frozen) { // died and respawned: wait for the spawn column before simulating again
      tickAccumulator = 0.0;
      break;
    }
  }

  // Smooth the eye height and FOV toward their targets (per frame, frame-rate independent).
  const float blend = 1.0f - std::exp(-deltaTime * 15.0f);
  stepLift *= std::exp(-deltaTime * kStepSmoothing);
  const float targetEye = swimmingPose                                ? kSwimmingEye
                          : sneaking && currentMode == Mode::Walking ? kSneakingEye
                                                                     : kStandingEye;
  eyeHeight += (targetEye - eyeHeight) * blend;
  const float targetFov = sprinting ? 1.1f : 1.0f;
  fovScale += (targetFov - fovScale) * blend;
}

void Player::tick(const Input &input, float yawDegrees, float lookY, const cpuworld::CpuWorld &world) {
  const bool flying = currentMode == Mode::Flying;
  auto fullBoxFree = [&] {
    return !boxCollides(position + glm::dvec3(-kHalfWidth, 0.0, -kHalfWidth),
                        position + glm::dvec3(kHalfWidth, kStandingHeight, kHalfWidth), world);
  };

  // Water state at the start of the tick: the box shrunk by 0.4 top and bottom (a 0.6 box ends up as [0.2, 0.4]),
  // and the cell just below the eyes.
  {
    double low = position.y + kWaterBoxShrink;
    double high = position.y + boxHeight() - kWaterBoxShrink;
    if (low > high) std::swap(low, high);
    inWater = boxInWater(glm::dvec3(position.x - kHalfWidth, low, position.z - kHalfWidth),
                         glm::dvec3(position.x + kHalfWidth, high, position.z + kHalfWidth), world);
    const double eyeY = position.y + static_cast<double>(eyeHeight) - kEyeWaterOffset;
    eyesInWater = world.isInfiniteSource(static_cast<int>(std::floor(position.x)), static_cast<int>(std::floor(eyeY)),
                                         static_cast<int>(std::floor(position.z)));
  }
  const bool swimPhysics = inWater && !flying;

  // Sneaking shrinks the box; standing back up needs room for the full height.
  if (flying) {
    sneaking = false;
  } else if (input.sneak) {
    sneaking = true;
  } else if (sneaking) {
    sneaking = !fullBoxFree();
  }

  // Sprint: needs forward input; stops on sneaking or running into a wall.
  if (input.sprint && input.forward > 0.0f && !sneaking) {
    sprinting = true;
  }
  if (input.forward <= 0.0f || sneaking || collidedHorizontally) {
    sprinting = false;
  }

  // Sprint-swim pose: starts when sprinting with the eyes underwater; leaving it needs headroom (else crawl).
  if (!swimmingPose && sprinting && eyesInWater && !flying) {
    swimmingPose = true;
  } else if (swimmingPose && (!sprinting || !inWater || flying)) {
    swimmingPose = !fullBoxFree();
  }
  const bool sprintSwimming = swimmingPose && sprinting && swimPhysics;
  const bool crawling = swimmingPose && !sprintSwimming && !swimPhysics && !flying;

  double forward = input.forward;
  double strafe = input.strafe;
  if ((sneaking && !swimmingPose) || crawling) {
    forward *= kSneakInputScale;
    strafe *= kSneakInputScale;
  }

  const double yaw = glm::radians(static_cast<double>(yawDegrees));
  // Camera yaw -90 faces -z; forward is (cos yaw, sin yaw) in x/z, right is (-sin yaw, cos yaw).
  const glm::dvec2 forwardDir(std::cos(yaw), std::sin(yaw));
  const glm::dvec2 rightDir(-std::sin(yaw), std::cos(yaw));

  if (flying) {
    if (input.sneak) motion.y -= kFlyVertical;
    if (input.jump) motion.y += kFlyVertical;
  } else if (swimPhysics) {
    if (input.jump) motion.y += kWaterVerticalPush; // handleFluidJump
    if (input.sneak) motion.y -= kWaterVerticalPush;
  } else if (input.jump && onGround) {
    motion.y = kJumpVelocity;
    if (sprinting) {
      motion.x += forwardDir.x * kSprintJumpBoost;
      motion.z += forwardDir.y * kSprintJumpBoost;
    }
  }

  // Sprint-swimming steers vertically toward the look direction (PlayerEntity.travel). Looking up only pulls you
  // up while holding Space or with water above the head, so it doesn't launch you out at the surface.
  if (sprintSwimming) {
    const double look = static_cast<double>(lookY);
    const double pullRate = look < -0.2 ? 0.085 : 0.06;
    const bool waterAbove =
        world.isInfiniteSource(static_cast<int>(std::floor(position.x)), static_cast<int>(std::floor(position.y + 1.0 - 0.1)),
                               static_cast<int>(std::floor(position.z)));
    if (look <= 0.0 || input.jump || waterAbove) {
      motion.y += (look - motion.y) * pullRate;
    }
  }

  // Acceleration from input (Entity.moveRelative): input normalized to at most 1, times the movement factor.
  double speed;
  double horizontalFriction;
  if (flying) {
    speed = sprinting ? kFlySpeed * 2.0 : kFlySpeed;
    horizontalFriction = kAirFriction;
  } else if (swimPhysics) {
    speed = kWaterSpeed;
    horizontalFriction = sprintSwimming ? kSwimFriction : kWaterFriction;
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
    const double forwardAccel = forward * scale;
    const double strafeAccel = strafe * scale;
    motion.x += forwardDir.x * forwardAccel + rightDir.x * strafeAccel;
    motion.z += forwardDir.y * forwardAccel + rightDir.y * strafeAccel;
  }

  move(motion, (sneaking || crawling) && onGround && !flying && !inWater, world);
  updateFall(position.y - previousPosition.y);
  if (hurtTicks > 0) {
    --hurtTicks;
  }
  if (hp < kMaxHealth && ++regenTicks >= kRegenIntervalTicks) {
    hp = std::min(hp + 1.0f, kMaxHealth);
    regenTicks = 0;
  }

  if (flying) {
    motion.y *= kFlyVerticalDamping;
    if (onGround) {
      currentMode = Mode::Walking; // landing ends flight
    }
  } else if (swimPhysics) {
    motion.y *= kWaterVerticalDrag;
    if (!sprintSwimming) {
      motion.y -= kWaterGravity;
    }
  } else {
    motion.y = (motion.y - kGravity) * kVerticalDrag;
  }
  motion.x *= horizontalFriction;
  motion.z *= horizontalFriction;

  // Swimming into a ledge: hop out if the box raised by 0.6 (plus this tick's motion) is free and dry.
  if (swimPhysics && collidedHorizontally) {
    const glm::dvec3 offset(motion.x, motion.y + 0.6 - (position.y - previousPosition.y), motion.z);
    const glm::dvec3 boxMin = position + offset + glm::dvec3(-kHalfWidth, 0.0, -kHalfWidth);
    const glm::dvec3 boxMax = position + offset + glm::dvec3(kHalfWidth, boxHeight(), kHalfWidth);
    if (!boxCollides(boxMin, boxMax, world) && !boxInWater(boxMin, boxMax, world)) {
      motion.y = kLedgeJump;
    }
  }

  // Minecraft zeroes tiny velocities so the player comes to a full stop.
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(motion[axis]) < 0.003) {
      motion[axis] = 0.0;
    }
  }
}

bool Player::overlapsBlock(int x, int y, int z) const {
  if (currentMode == Mode::Spectator) {
    return false;
  }
  constexpr double eps = 1.0e-7;
  const glm::dvec3 boxMin = position + glm::dvec3(-kHalfWidth, 0.0, -kHalfWidth);
  const glm::dvec3 boxMax = position + glm::dvec3(kHalfWidth, boxHeight(), kHalfWidth);
  return boxMin.x < x + 1 - eps && boxMax.x > x + eps && boxMin.y < y + 1 - eps && boxMax.y > y + eps &&
         boxMin.z < z + 1 - eps && boxMax.z > z + eps;
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
        if (world.isCollidable(x, y, z)) return true;
  return false;
}

bool Player::boxInWater(const glm::dvec3 &minCorner, const glm::dvec3 &maxCorner,
                        const cpuworld::CpuWorld &world) const {
  constexpr double eps = 1.0e-7;
  const int x0 = static_cast<int>(std::floor(minCorner.x + eps)), x1 = static_cast<int>(std::floor(maxCorner.x - eps));
  const int y0 = static_cast<int>(std::floor(minCorner.y + eps)), y1 = static_cast<int>(std::floor(maxCorner.y - eps));
  const int z0 = static_cast<int>(std::floor(minCorner.z + eps)), z1 = static_cast<int>(std::floor(maxCorner.z - eps));
  for (int y = y0; y <= y1; ++y)
    for (int z = z0; z <= z1; ++z)
      for (int x = x0; x <= x1; ++x)
        if (world.isInfiniteSource(x, y, z)) return true;
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
    /** Whether the box shifted by (dx, dz) still has a collidable block within 1 block below it. */
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

  const glm::dvec3 startMin = boxMin, startMax = boxMax;
  const glm::dvec3 guarded = delta; // after the sneak guard

  /**
   * Clip a move along one axis against every solid block the swept box touches (AxisAlignedBB.calculate*Offset),
   * then shift boxMin / boxMax by the result.
   *
   * @param axis 0 = x, 1 = y, 2 = z.
   * @param d Requested displacement along `axis`, in blocks.
   * @return the allowed displacement.
   */
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
          if (!world.isCollidable(x, y, z)) continue;
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
  bool landed = delta.y != guarded.y && guarded.y < 0.0;

  // Step up (Entity.move's stepHeight): walking into a ledge up to kStepHeight tall, retry the move raised by that
  // much, then settle back down onto whatever is there. Kept only if it gets further horizontally.
  const bool blockedHorizontally = delta.x != guarded.x || delta.z != guarded.z;
  if (currentMode == Mode::Walking && blockedHorizontally && (onGround || landed)) {
    const glm::dvec3 normalMin = boxMin, normalMax = boxMax, normalDelta = delta;
    boxMin = startMin;
    boxMax = startMax;
    glm::dvec3 stepped;
    const double stepUp = clipAxis(1, kStepHeight);
    stepped.x = clipAxis(0, guarded.x);
    stepped.z = clipAxis(2, guarded.z);
    const double settleDown = clipAxis(1, -stepUp + std::min(guarded.y, 0.0));
    stepped.y = stepUp + settleDown;
    const double normalDist = normalDelta.x * normalDelta.x + normalDelta.z * normalDelta.z;
    const double steppedDist = stepped.x * stepped.x + stepped.z * stepped.z;
    if (steppedDist > normalDist + 1.0e-9 && stepped.y > 1.0e-9) {
      delta = stepped;
      landed = true;
      // The camera eases up instead of popping a full block: the render interpolation starts from the raised
      // height, and stepLift (decaying per frame) holds the eye back.
      stepLift += static_cast<float>(stepped.y);
      previousPosition.y += stepped.y;
    } else {
      boxMin = normalMin;
      boxMax = normalMax;
    }
  }

  position = glm::dvec3((boxMin.x + boxMax.x) * 0.5, boxMin.y, (boxMin.z + boxMax.z) * 0.5);
  const bool collidedVertically = delta.y != wanted.y;
  collidedHorizontally = delta.x != wanted.x || delta.z != wanted.z;
  onGround = landed;
  if (delta.x != wanted.x) motion.x = 0.0;
  if (delta.z != wanted.z) motion.z = 0.0;
  if (collidedVertically) motion.y = 0.0;
}
