#include "renderer.hpp"

#include <cstdlib>
#include <exception>
<<<<<<< HEAD
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
  try {
    if (argc > 0) {
      const auto executablePath = std::filesystem::absolute(argv[0]);
      std::filesystem::current_path(executablePath.parent_path());
    }

=======
#include <iostream>

int main() {
  try {
>>>>>>> a2baaa3b1b2b3ac2e9cea20c5f97a8c336887dda
    Renderer renderer;
    renderer.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
