#pragma once

#include <glm/glm.hpp>

struct GLFWwindow;

// Mouse look (yaw / pitch) and the view matrix. Position comes from the player (player.hpp).
class Camera {
public:
  void initialize(GLFWwindow *window);
  void handleKey(GLFWwindow *window, int key, int action);
  // Applies this frame's mouse movement.
  void updateLook(GLFWwindow *window);
  glm::mat4 view(const glm::vec3 &eye) const;

  const glm::vec3 &front() const { return cameraFront; }
  const glm::vec3 &up() const { return cameraUp; }
  // Degrees; yaw -90 looks toward -z.
  float yaw() const { return cameraYaw; }
  bool cursorCaptured() const { return captured; }

private:
  void updateFront();

  bool captured = true;
  glm::vec3 cameraFront{0.0f, 0.0f, -1.0f}; // set from cameraYaw/cameraPitch by updateFront()
  glm::vec3 cameraUp{0.0f, 1.0f, 0.0f};
  float cameraYaw = -90.0f;
  float cameraPitch = -20.0f;
  float mouseSensitivity = 0.15f;
  bool firstMouse = true;
  double lastMouseX = 400.0;
  double lastMouseY = 300.0;
};
