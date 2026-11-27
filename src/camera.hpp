#pragma once

#include <glm/glm.hpp>

struct GLFWwindow;

// Mouse look (yaw / pitch) and the view matrix. Position comes from the player (player.hpp).
class Camera {
public:
  /** Capture the cursor (raw motion when supported) and derive the initial look direction. */
  void initialize(GLFWwindow *window);

  /**
   * Toggle cursor capture on an Escape press; other keys are ignored.
   *
   * @param key GLFW key code.
   * @param action GLFW_PRESS / GLFW_RELEASE / GLFW_REPEAT.
   */
  void handleKey(GLFWwindow *window, int key, int action);

  /** Apply this frame's mouse movement to yaw / pitch (no-op while uncaptured or locked). */
  void updateLook(GLFWwindow *window);

  /**
   * Build the view matrix for the current look direction.
   *
   * @param eye Camera position in world space.
   * @return right-handed look-at matrix with +y up.
   */
  glm::mat4 view(const glm::vec3 &eye) const;

  /** Unit look direction. */
  const glm::vec3 &front() const { return cameraFront; }
  /** Yaw in degrees; -90 looks toward -z. */
  float yaw() const { return cameraYaw; }
  /** Whether the cursor is captured for mouse look. */
  bool cursorCaptured() const { return captured; }

  /**
   * Set the look direction and lock it, so mouse movement is ignored (testing: ENGINE_LOOK).
   *
   * @param yaw Yaw in degrees.
   * @param pitch Pitch in degrees.
   */
  void setLook(float yaw, float pitch) {
    cameraYaw = yaw;
    cameraPitch = pitch;
    lookLocked = true;
    updateFront();
  }

private:
  /** Recompute cameraFront from cameraYaw / cameraPitch. */
  void updateFront();

  bool captured = true;
  bool lookLocked = false;
  glm::vec3 cameraFront{0.0f, 0.0f, -1.0f};
  float cameraYaw = -90.0f;
  float cameraPitch = -20.0f;
  float mouseSensitivity = 0.15f; // degrees per cursor unit
  bool firstMouse = true;
  double lastMouseX = 400.0;
  double lastMouseY = 300.0;
};
