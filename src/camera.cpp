#include "camera.hpp"

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>

void Camera::initialize(GLFWwindow *window) {
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  if (glfwRawMouseMotionSupported()) {
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
  }
}

void Camera::handleKey(GLFWwindow *window, int key, int action) {
  if (key != GLFW_KEY_ESCAPE || action != GLFW_PRESS) {
    return;
  }

  cursorCaptured = !cursorCaptured;
  glfwSetInputMode(window, GLFW_CURSOR, cursorCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
  if (glfwRawMouseMotionSupported()) {
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, cursorCaptured ? GLFW_TRUE : GLFW_FALSE);
  }
  if (cursorCaptured) {
    firstMouse = true;
  }
}

glm::mat4 Camera::update(GLFWwindow *window, float deltaTime) {
  if (cursorCaptured) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (firstMouse) {
      lastMouseX = xpos;
      lastMouseY = ypos;
      firstMouse = false;
    }

    const double xoffset = (xpos - lastMouseX) * mouseSensitivity;
    const double yoffset = (lastMouseY - ypos) * mouseSensitivity;
    lastMouseX = xpos;
    lastMouseY = ypos;

    cameraYaw += static_cast<float>(xoffset);
    cameraPitch = glm::clamp(cameraPitch + static_cast<float>(yoffset), -89.0f, 89.0f);

    glm::vec3 direction;
    direction.x = std::cos(glm::radians(cameraYaw)) * std::cos(glm::radians(cameraPitch));
    direction.y = std::sin(glm::radians(cameraPitch));
    direction.z = std::sin(glm::radians(cameraYaw)) * std::cos(glm::radians(cameraPitch));
    cameraFront = glm::normalize(direction);

    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    const double centerX = windowWidth / 2.0;
    const double centerY = windowHeight / 2.0;
    glfwSetCursorPos(window, centerX, centerY);
    lastMouseX = centerX;
    lastMouseY = centerY;
  }

  const glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));
  const glm::vec3 worldUp{0.0f, 1.0f, 0.0f};
  const glm::vec3 forward = glm::normalize(glm::cross(right, worldUp));
  const float velocity = cameraSpeed * deltaTime;

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    cameraPos += cameraFront * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    cameraPos -= cameraFront * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    cameraPos -= right * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    cameraPos += right * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    cameraPos += worldUp * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
    cameraPos -= worldUp * velocity;
  }

  static bool eWasPressed = false;
  bool eIsPressed = (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS);

  if (!eIsPressed && eWasPressed) {
    static size_t index = 1;
    index++;
    cameraSpeed = cameraSpeeds[index % 3];
  }

  eWasPressed = eIsPressed;

  return glm::lookAt(cameraPos, cameraPos + cameraFront, worldUp);
}

const glm::vec3 &Camera::position() const { return cameraPos; }

const glm::vec3 &Camera::front() const { return cameraFront; }

const glm::vec3 &Camera::up() const { return cameraUp; }
