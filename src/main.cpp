#include "renderer.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
  try {
    if (argc > 0) {
      const auto executablePath = std::filesystem::absolute(argv[0]);
      std::filesystem::current_path(executablePath.parent_path());
    }

    Renderer renderer;
    renderer.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
