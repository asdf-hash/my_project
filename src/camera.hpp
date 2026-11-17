#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>

struct GLFWwindow;

class Camera {
public:
  void initialize(GLFWwindow *window);
  void handleKey(GLFWwindow *window, int key, int action);
  glm::mat4 update(GLFWwindow *window, float deltaTime);

  const glm::vec3 &position() const;
  const glm::vec3 &front() const;
  const glm::vec3 &up() const;

private:
  bool cursorCaptured = true;
  bool speedKeyWasPressed = false;
  glm::vec3 cameraPos{0.5f, 0.5f, 2.5f};
  glm::vec3 cameraFront = glm::normalize(glm::vec3(0.5f, 0.5f, 0.0f) - cameraPos);
  glm::vec3 cameraUp{0.0f, 1.0f, 0.0f};
  float cameraYaw = -90.0f;
  float cameraPitch = 0.0f;
  float cameraSpeed = 100.0f;
  float mouseSensitivity = 0.15f;
  std::size_t cameraSpeedIndex = 1;
  bool firstMouse = true;
  double lastMouseX = 400.0;
  double lastMouseY = 300.0;
  std::array<float, 3> cameraSpeeds{10.f, 100.f, 1000.f};
};
