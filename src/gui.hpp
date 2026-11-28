#pragma once

// 2D GUI on the CPU, Vulkan-free: a batch of quads (drawn by shaders/gui.slang in one instanced draw), the vanilla
// sprites from textures/gui, the bitmap font from textures/font/ascii.png, and the HUD layout. Coordinates are GUI
// pixels (screen pixels / guiScale), y down, like Minecraft's GuiGraphics.

#include "item_stack.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

namespace gui {

// One instance of gui.slang's quad; must match GuiQuad there (instance-rate vertex attributes, 64 bytes).
struct Quad {
  glm::vec4 rect;      // screen pixels: x, y, width, height
  glm::vec4 source;    // texels: x, y, width, height of the source region (Sprite)
  glm::vec4 extent;    // xy: texels the quad spans (> source size tiles the region); zw unused
  std::uint32_t texture; // bindless slot (BlockIcon: top face)
  std::uint32_t color;   // RGBA8 multiplier in sRGB space, R in the low byte (Fill: the color itself)
  std::uint32_t kind;    // QuadKind
  std::uint32_t extra;   // BlockIcon: side face slot
};
static_assert(sizeof(Quad) == 64);

enum QuadKind : std::uint32_t { kSprite = 0, kFill = 1, kBlockIcon = 2 };

// Vanilla sprite scaling (.png.mcmeta "gui.scaling"): stretch (default), tile, or nine_slice.
enum class Scaling { Stretch, Tile, NineSlice };

struct Sprite {
  std::uint32_t texture = 0;
  int width = 0;
  int height = 0;
  Scaling scaling = Scaling::Stretch;
  int border = 0;            // nine_slice: border width on every side, in texels
  bool stretchInner = false; // nine_slice: stretch the edges and center instead of tiling them
};

/** Apply a .png.mcmeta's "gui.scaling" block to `sprite` (minimal parser: type, border, stretch_inner). */
void parseScaling(std::string_view mcmeta, Sprite &sprite);

/** Sprite files the engine loads, relative to textures/gui/ without ".png"; names are the lookup keys. */
std::span<const std::string_view> spriteNames();

struct Sprites {
  std::unordered_map<std::string, Sprite> byName;
  /** Sprite `name`; throws std::runtime_error if it wasn't loaded. */
  const Sprite &operator[](std::string_view name) const;
};

// Minecraft's legacy bitmap font: a 16 x 16 grid of 8 x 8 glyphs (code page 437 layout for ASCII).
struct Font {
  std::uint32_t texture = 0;
  int cellSize = 8;
  std::array<std::uint8_t, 256> advance{}; // per character: glyph width + 1 (space: 4)

  /**
   * Measure every glyph from the font image: width = rightmost column with alpha, advance = width + 1.
   *
   * @param rgba Image pixels, 4 bytes each.
   */
  static Font fromPixels(const std::uint8_t *rgba, int width, int height, std::uint32_t texture);
  /** Advance width of `text`, in GUI pixels. */
  int width(std::string_view text) const;
};

// Block icon faces by block ID (renderer fills it from the registry).
struct BlockIcon {
  std::uint32_t top = 0;  // bindless slot of the +y face
  std::uint32_t side = 0; // bindless slot of the +z face
  std::uint32_t bottom = 0; // bindless slot of the -y face (item entities)
  std::uint32_t tint = 0xffffffffu; // RGBA8, R low
  bool valid = false;
};

struct Context {
  const Sprites *sprites = nullptr;
  const Font *font = nullptr;
  std::vector<BlockIcon> icons; // by block ID
};

/** Pack 0xAARRGGBB (vanilla's color ints) into the shader's RGBA8 with R in the low byte. */
constexpr std::uint32_t packArgb(std::uint32_t argb) {
  return ((argb >> 16u) & 0xffu) | (argb & 0xff00u) | ((argb & 0xffu) << 16u) | (argb & 0xff000000u);
}

inline constexpr std::uint32_t kWhite = 0xffffffffu;
inline constexpr std::uint32_t kTextShadowOffset = 1;

/** Vanilla-like GUI scale for a window height: screen pixels per GUI pixel. */
int guiScale(int screenHeight);

class Batch {
public:
  /** Start a frame: clears the quads and sets the GUI size (ceil(screen / scale), like vanilla). */
  void begin(int screenWidth, int screenHeight, int scale);
  int width() const { return guiWidth; }
  int height() const { return guiHeight; }
  int scale() const { return guiScaleFactor; }

  /** Sprite at its native size. */
  void sprite(const Sprite &sprite, int x, int y, std::uint32_t argb = kWhite);
  /** Sprite drawn into (x, y, w, h) by its scaling (stretch, tile or nine-slice). */
  void sprite(const Sprite &sprite, int x, int y, int w, int h, std::uint32_t argb = kWhite);
  /** Region (u, v, w, h) of a texture at 1:1 (container backgrounds, the switcher panel). */
  void region(const Sprite &sprite, int x, int y, int u, int v, int w, int h, std::uint32_t argb = kWhite);
  /** Solid rectangle [x0, x1) x [y0, y1). */
  void fill(int x0, int y0, int x1, int y1, std::uint32_t argb);
  /** 16 x 16 isometric block icon with its top-left corner at (x, y). */
  void blockIcon(const BlockIcon &icon, int x, int y);
  /**
   * Text in the bitmap font, top-left at (x, y).
   *
   * @return the advance width, in GUI pixels.
   */
  int text(const Font &font, std::string_view text, int x, int y, std::uint32_t argb, bool shadow);
  /** Text centered on x (vanilla drawCenteredString: shadowed). */
  void centeredText(const Font &font, std::string_view text, int x, int y, std::uint32_t argb);

  /**
   * An item in a 16 x 16 slot at (x, y), with its count (when > 1) right-aligned in the bottom-right corner.
   */
  void item(const Context &context, const ItemStack &stack, int x, int y);

  const std::vector<Quad> &quads() const { return quadList; }

private:
  /** Emit one sprite quad: destination (GUI pixels) and source (texels); tile repeats the source over the quad. */
  void emit(const Sprite &sprite, int x, int y, int w, int h, int u, int v, int sw, int sh, bool tile,
            std::uint32_t color);
  void emitGlyphs(const Font &font, std::string_view text, int x, int y, std::uint32_t color);

  std::vector<Quad> quadList;
  int guiWidth = 0;
  int guiHeight = 0;
  int guiScaleFactor = 1;
};

// HUD (in-game overlay) state for one frame.
struct HudState {
  std::array<ItemStack, 9> hotbar{};
  ItemStack offhand{};
  int selected = 0;
  bool survivalLike = false; // health, hunger and XP
  int health = 20;           // half hearts, rounded up
  bool hurtFlash = false;
};

/** Hotbar with items and counts, offhand, and in survival the XP bar, hunger and hearts (vanilla Gui layout). */
void drawHud(Batch &batch, const Context &context, const HudState &state);

/**
 * F3+F4 game mode switcher (vanilla GameModeSwitcherScreen layout).
 *
 * @param selected Selected mode, 0-3 in switcher order.
 * @param iconBlocks Block shown per mode slot.
 */
void drawGameModeSwitcher(Batch &batch, const Context &context, int selected, std::string_view title,
                          const std::array<std::uint16_t, 4> &iconBlocks);

} // namespace gui
