#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Block definitions loaded from blocks.txt. A block's ID is its position among the file's blocks of the same opacity
// (non-opaque ones offset by kNonOpaqueBit); everything outside the registry refers to blocks by name.
namespace blocks {

// Face order matches the shaders' normalIndex: +x, -x, +y, -y, +z, -z.
inline constexpr std::size_t kFaceCount = 6;
// Quads store the block ID in 10 bits. Bit 9 marks a non-opaque block (glass, leaves, ...): the binary mesher
// only handles full opaque cubes, so opaque blocks use IDs 0-511 and non-opaque ones 512-1023.
inline constexpr std::size_t kMaxBlockCount = 1024;
inline constexpr std::uint16_t kNonOpaqueBit = 1u << 9;
inline constexpr std::size_t kMaxOpaqueBlocks = kNonOpaqueBit;
inline constexpr std::size_t kMaxNonOpaqueBlocks = kMaxBlockCount - kNonOpaqueBit;
inline constexpr bool isOpaque(std::uint32_t id) { return (id & kNonOpaqueBit) == 0u; }

struct Face {
  std::uint32_t texture = 0;
  // Texture used instead of `texture` on coarse LOD chunks (LOD > 0), tinted by Block::lodTint.
  std::uint32_t lodTexture = 0;
  bool hasLodTexture = false;
  // Rotate the texture by a per-face hash in 90 degree steps to break up tiling.
  bool randomRotation = false;
};

// Liquid behavior of a block (blocks.txt `fluid`).
enum class Fluid {
  None,
  Ocean, // infinite source below sea level: never changes, only placing a block removes it; drawn translucent
};

struct Block {
  std::string name;
  Fluid fluid = Fluid::None;
  // Color multiplier applied to every face at every LOD (blocks.txt `tint`), e.g. to color the shared grayscale
  // fluid texture per liquid.
  std::array<float, 3> tint{1.0f, 1.0f, 1.0f};
  // Seconds per animation frame for textures that are frame strips (blocks.txt `frame_time`).
  float frameTime = 0.1f;
  // Opaque blocks are numbered 0, 1, ... and non-opaque ones 512, 513, ..., each in file order.
  std::uint32_t id = 0;
  bool opaque = true;
  std::array<Face, kFaceCount> faces{};
  std::array<float, 3> lodTint{1.0f, 1.0f, 1.0f};
};

struct Registry {
  std::vector<Block> blocks;
  // Indexed by Face::texture / Face::lodTexture; each path appears once.
  std::vector<std::string> texturePaths;

  // Throws if no block has this name.
  std::uint32_t idOf(std::string_view name) const;
  // Largest block ID in use (size per-ID tables to maxId() + 1).
  std::uint32_t maxId() const;
};

// Parses a block definition file. Texture names are resolved relative to textureDirectory.
Registry loadRegistry(const std::string &path, const std::string &textureDirectory);

} // namespace blocks
