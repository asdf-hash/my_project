#include "camera.hpp"

#include <GLFW/glfw3.h>

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

void Camera::updateFront() {
  glm::vec3 direction;
  direction.x = std::cos(glm::radians(cameraYaw)) * std::cos(glm::radians(cameraPitch));
  direction.y = std::sin(glm::radians(cameraPitch));
  direction.z = std::sin(glm::radians(cameraYaw)) * std::cos(glm::radians(cameraPitch));
  cameraFront = glm::normalize(direction);
}

void Camera::initialize(GLFWwindow *window) {
  updateFront(); // so the first mouse update can't snap the view
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  if (glfwRawMouseMotionSupported()) {
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
  }
}

void Camera::handleKey(GLFWwindow *window, int key, int action) {
  if (key != GLFW_KEY_ESCAPE || action != GLFW_PRESS) {
    return;
  }

  setCaptured(window, !captured);
}

void Camera::setCaptured(GLFWwindow *window, bool capture) {
  captured = capture;
  glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
  if (glfwRawMouseMotionSupported()) {
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, captured ? GLFW_TRUE : GLFW_FALSE);
  }
  if (captured) {
    firstMouse = true;
  } else {
    // Free cursor at the window center, like vanilla opening a screen.
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);
    glfwSetCursorPos(window, width * 0.5, height * 0.5);
  }
}

void Camera::updateLook(GLFWwindow *window) {
  if (!captured || lookLocked) {
    return;
  }
  double cursorX, cursorY;
  glfwGetCursorPos(window, &cursorX, &cursorY);

  if (firstMouse) {
    lastMouseX = cursorX;
    lastMouseY = cursorY;
    firstMouse = false;
  }

  const double yawDelta = (cursorX - lastMouseX) * mouseSensitivity;
  const double pitchDelta = (lastMouseY - cursorY) * mouseSensitivity;
  lastMouseX = cursorX;
  lastMouseY = cursorY;

  cameraYaw += static_cast<float>(yawDelta);
  cameraPitch = glm::clamp(cameraPitch + static_cast<float>(pitchDelta), -89.f, 89.f);

  updateFront();
  // No recentering: GLFW_CURSOR_DISABLED already gives an unbounded virtual cursor, and on Wayland
  // glfwSetCursorPos can't warp (recentering produced a huge first delta).
}

glm::mat4 Camera::view(const glm::vec3 &eye) const {
  return glm::lookAt(eye, eye + cameraFront, glm::vec3{0.0f, 1.0f, 0.0f});
}
