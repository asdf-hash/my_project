// Precompiled header for the engine target (set_pcxxheader in xmake.lua).
// Holds the heavy third-party headers so they are parsed once, not on every renderer.cpp edit.
// Do not put GLM_FORCE_* defines here: they would change glm::perspective's depth range
// (renderer.cpp currently gets glm's default -1..1 clip depth).
#pragma once

#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
