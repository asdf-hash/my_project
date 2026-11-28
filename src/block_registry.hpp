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
/** Whether block `id` is an opaque full cube (kNonOpaqueBit clear). */
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
  // Minecraft's destroy time (blocks.txt `hardness`): survival break time by hand is hardness * 1.5 seconds, or
  // hardness * 5 when `requires_tool` (the block needs a tool to drop, e.g. stone).
  float hardness = 0.0f;
  bool requiresTool = false;
  std::array<Face, kFaceCount> faces{};
  std::array<float, 3> lodTint{1.0f, 1.0f, 1.0f};
  // Color multiplier faded in with distance around LOD 3 (blocks.txt `far_tint`), where a tree is one leaf block.
  std::array<float, 3> farTint{1.0f, 1.0f, 1.0f};
  // Inventory: name shown in tooltips and creative search (blocks.txt `display_name`, default: the name title-cased),
  // creative tab (`category`), whether the creative tabs list it (`creative`).
  std::string displayName;
  std::string category = "building_blocks";
  bool creative = true;
  // What survival breaking drops (blocks.txt `drops = <block> | none`, default the block itself).
  bool dropsNothing = false;
  // Dropped-item physics: kg/m^3 (blocks.txt `density`); below water's 1000 floats.
  float density = 1500.0f;
  std::uint32_t dropId = 0;
};

// Creative tab ids accepted by blocks.txt `category`, in vanilla's tab order.
inline constexpr std::array<std::string_view, 10> kCategories{
    "building_blocks", "colored_blocks", "natural_blocks", "functional_blocks", "redstone_blocks",
    "tools",           "combat",         "food",           "ingredients",       "spawn_eggs"};

struct Registry {
  std::vector<Block> blocks;
  // Indexed by Face::texture / Face::lodTexture; each path appears once.
  std::vector<std::string> texturePaths;

  /**
   * Look up a block's ID by name.
   *
   * @return ID of the block named `name`; throws std::runtime_error if there is none.
   */
  std::uint32_t idOf(std::string_view name) const;

  /**
   * Largest block ID in use (size per-ID tables to maxId() + 1).
   *
   * @return the maximum Block::id, or 0 for an empty registry.
   */
  std::uint32_t maxId() const;
};

/**
 * Parse a block definition file.
 *
 * @param path Path of the blocks.txt-format file.
 * @param textureDirectory Prefix prepended to every texture name (include the trailing slash).
 * @return the registry; throws std::runtime_error with file:line on any syntax or validation error.
 */
Registry loadRegistry(const std::string &path, const std::string &textureDirectory);

} // namespace blocks
