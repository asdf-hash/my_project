#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Block definitions loaded from blocks.txt. A block's ID is its position in the file; everything outside the
// registry refers to blocks by name.
namespace blocks {

// Face order matches the shaders' normalIndex: +x, -x, +y, -y, +z, -z.
inline constexpr std::size_t kFaceCount = 6;
// Quads store the block ID in 10 bits.
inline constexpr std::size_t kMaxBlockCount = 1024;

struct Face {
  std::uint32_t texture = 0;
  // Texture used instead of `texture` on coarse LOD chunks (LOD > 0), tinted by Block::lodTint.
  std::uint32_t lodTexture = 0;
  bool hasLodTexture = false;
  // Rotate the texture by a per-face hash in 90 degree steps to break up tiling.
  bool randomRotation = false;
};

struct Block {
  std::string name;
  std::array<Face, kFaceCount> faces{};
  std::array<float, 3> lodTint{1.0f, 1.0f, 1.0f};
};

struct Registry {
  std::vector<Block> blocks;
  // Indexed by Face::texture / Face::lodTexture; each path appears once.
  std::vector<std::string> texturePaths;

  // Throws if no block has this name.
  std::uint32_t idOf(std::string_view name) const;
};

// Parses a block definition file. Texture names are resolved relative to textureDirectory.
Registry loadRegistry(const std::string &path, const std::string &textureDirectory);

} // namespace blocks
