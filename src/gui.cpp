#include "gui.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gui {

namespace {

constexpr std::string_view kSpriteNames[] = {
    // HUD
    "sprites/hud/hotbar",
    "sprites/hud/hotbar_selection",
    "sprites/hud/hotbar_offhand_left",
    "sprites/hud/hotbar_offhand_right",
    "sprites/hud/experience_bar_background",
    "sprites/hud/food_empty",
    "sprites/hud/food_half",
    "sprites/hud/food_full",
    "sprites/hud/heart/container",
    "sprites/hud/heart/container_blinking",
    "sprites/hud/heart/full",
    "sprites/hud/heart/half",
    // Game mode switcher
    "container/gamemode_switcher",
    "sprites/gamemode_switcher/slot",
    "sprites/gamemode_switcher/selection",
    // Inventory screens
    "container/inventory",
    "container/creative_inventory/tab_items",
    "container/creative_inventory/tab_item_search",
    "container/creative_inventory/tab_inventory",
    "sprites/container/creative_inventory/scroller",
    "sprites/container/creative_inventory/scroller_disabled",
    "sprites/container/creative_inventory/tab_top_selected_1",
    "sprites/container/creative_inventory/tab_top_selected_2",
    "sprites/container/creative_inventory/tab_top_selected_3",
    "sprites/container/creative_inventory/tab_top_selected_4",
    "sprites/container/creative_inventory/tab_top_selected_5",
    "sprites/container/creative_inventory/tab_top_selected_6",
    "sprites/container/creative_inventory/tab_top_selected_7",
    "sprites/container/creative_inventory/tab_top_unselected_1",
    "sprites/container/creative_inventory/tab_top_unselected_2",
    "sprites/container/creative_inventory/tab_top_unselected_3",
    "sprites/container/creative_inventory/tab_top_unselected_4",
    "sprites/container/creative_inventory/tab_top_unselected_5",
    "sprites/container/creative_inventory/tab_top_unselected_6",
    "sprites/container/creative_inventory/tab_top_unselected_7",
    "sprites/container/creative_inventory/tab_bottom_selected_1",
    "sprites/container/creative_inventory/tab_bottom_selected_2",
    "sprites/container/creative_inventory/tab_bottom_selected_3",
    "sprites/container/creative_inventory/tab_bottom_selected_4",
    "sprites/container/creative_inventory/tab_bottom_selected_5",
    "sprites/container/creative_inventory/tab_bottom_selected_6",
    "sprites/container/creative_inventory/tab_bottom_selected_7",
    "sprites/container/creative_inventory/tab_bottom_unselected_1",
    "sprites/container/creative_inventory/tab_bottom_unselected_2",
    "sprites/container/creative_inventory/tab_bottom_unselected_3",
    "sprites/container/creative_inventory/tab_bottom_unselected_4",
    "sprites/container/creative_inventory/tab_bottom_unselected_5",
    "sprites/container/creative_inventory/tab_bottom_unselected_6",
    "sprites/container/creative_inventory/tab_bottom_unselected_7",
    "sprites/container/slot_highlight_back",
    "sprites/container/slot_highlight_front",
    "sprites/container/slot/helmet",
    "sprites/container/slot/chestplate",
    "sprites/container/slot/leggings",
    "sprites/container/slot/boots",
    "sprites/container/slot/shield",
    "sprites/tooltip/background",
    "sprites/tooltip/frame",
};

/** Integer value after `"key":` in a JSON text, or `fallback`. */
int jsonInt(std::string_view json, std::string_view key, int fallback) {
  const std::string quoted = "\"" + std::string(key) + "\"";
  size_t at = json.find(quoted);
  if (at == std::string_view::npos) {
    return fallback;
  }
  at = json.find(':', at + quoted.size());
  if (at == std::string_view::npos) {
    return fallback;
  }
  ++at;
  while (at < json.size() && (json[at] == ' ' || json[at] == '\t' || json[at] == '\n' || json[at] == '\r')) {
    ++at;
  }
  int value = 0;
  bool any = false;
  while (at < json.size() && json[at] >= '0' && json[at] <= '9') {
    value = value * 10 + (json[at] - '0');
    any = true;
    ++at;
  }
  return any ? value : fallback;
}

} // namespace

void parseScaling(std::string_view mcmeta, Sprite &sprite) {
  if (mcmeta.find("\"scaling\"") == std::string_view::npos) {
    return;
  }
  if (mcmeta.find("\"nine_slice\"") != std::string_view::npos) {
    sprite.scaling = Scaling::NineSlice;
    sprite.border = jsonInt(mcmeta, "border", 0);
    sprite.stretchInner = mcmeta.find("\"stretch_inner\": true") != std::string_view::npos ||
                          mcmeta.find("\"stretch_inner\":true") != std::string_view::npos;
  } else if (mcmeta.find("\"tile\"") != std::string_view::npos) {
    sprite.scaling = Scaling::Tile;
  }
}

std::span<const std::string_view> spriteNames() { return kSpriteNames; }

const Sprite &Sprites::operator[](std::string_view name) const {
  const auto it = byName.find(std::string(name));
  if (it == byName.end()) {
    throw std::runtime_error("GUI sprite not loaded: " + std::string(name));
  }
  return it->second;
}

Font Font::fromPixels(const std::uint8_t *rgba, int width, int height, std::uint32_t texture) {
  Font font;
  font.texture = texture;
  font.cellSize = width / 16;
  const int cell = font.cellSize;
  for (int c = 0; c < 256; ++c) {
    const int cellX = (c % 16) * cell;
    const int cellY = (c / 16) * cell;
    int glyphWidth = 0;
    for (int x = cell - 1; x >= 0 && glyphWidth == 0; --x) {
      for (int y = 0; y < cell && cellY + y < height; ++y) {
        if (rgba[(static_cast<size_t>(cellY + y) * static_cast<size_t>(width) + static_cast<size_t>(cellX + x)) * 4u +
                 3u] != 0) {
          glyphWidth = x + 1;
          break;
        }
      }
    }
    font.advance[static_cast<size_t>(c)] = static_cast<std::uint8_t>(glyphWidth + 1);
  }
  font.advance[' '] = 4;
  return font;
}

int Font::width(std::string_view text) const {
  int total = 0;
  for (const char c : text) {
    total += advance[static_cast<unsigned char>(c)];
  }
  return total;
}

int guiScale(int screenHeight) { return std::max(1, static_cast<int>(std::lround(screenHeight / 333.0))); }

void Batch::begin(int screenWidth, int screenHeight, int scale) {
  quadList.clear();
  guiScaleFactor = std::max(scale, 1);
  guiWidth = (screenWidth + guiScaleFactor - 1) / guiScaleFactor;
  guiHeight = (screenHeight + guiScaleFactor - 1) / guiScaleFactor;
}

void Batch::emit(const Sprite &sprite, int x, int y, int w, int h, int u, int v, int sw, int sh, bool tile,
                 std::uint32_t color) {
  if (w <= 0 || h <= 0 || sw <= 0 || sh <= 0) {
    return;
  }
  const auto s = static_cast<float>(guiScaleFactor);
  quadList.push_back(Quad{
      .rect = {static_cast<float>(x) * s, static_cast<float>(y) * s, static_cast<float>(w) * s,
               static_cast<float>(h) * s},
      .source = {static_cast<float>(u), static_cast<float>(v), static_cast<float>(sw), static_cast<float>(sh)},
      .extent = {static_cast<float>(tile ? w : sw), static_cast<float>(tile ? h : sh), 0.0f, 0.0f},
      .texture = sprite.texture,
      .color = color,
      .kind = kSprite,
      .extra = 0});
}

void Batch::sprite(const Sprite &sprite, int x, int y, std::uint32_t argb) {
  emit(sprite, x, y, sprite.width, sprite.height, 0, 0, sprite.width, sprite.height, false, packArgb(argb));
}

void Batch::sprite(const Sprite &sprite, int x, int y, int w, int h, std::uint32_t argb) {
  const std::uint32_t color = packArgb(argb);
  if (sprite.scaling == Scaling::Tile) {
    emit(sprite, x, y, w, h, 0, 0, sprite.width, sprite.height, true, color);
    return;
  }
  if (sprite.scaling != Scaling::NineSlice || (w == sprite.width && h == sprite.height)) {
    emit(sprite, x, y, w, h, 0, 0, sprite.width, sprite.height, false, color);
    return;
  }
  // Nine-slice: corners 1:1, edges and center tiled (or stretched with stretch_inner). Borders shrink when the
  // destination is smaller than two borders.
  const int bx = std::min(sprite.border, w / 2);
  const int by = std::min(sprite.border, h / 2);
  const int innerW = w - 2 * bx;
  const int innerH = h - 2 * by;
  const int srcInnerW = sprite.width - 2 * sprite.border;
  const int srcInnerH = sprite.height - 2 * sprite.border;
  const bool tile = !sprite.stretchInner;
  const int sb = sprite.border;
  const int rightU = sprite.width - bx;
  const int bottomV = sprite.height - by;
  // Corners.
  emit(sprite, x, y, bx, by, 0, 0, bx, by, false, color);
  emit(sprite, x + w - bx, y, bx, by, rightU, 0, bx, by, false, color);
  emit(sprite, x, y + h - by, bx, by, 0, bottomV, bx, by, false, color);
  emit(sprite, x + w - bx, y + h - by, bx, by, rightU, bottomV, bx, by, false, color);
  // Edges.
  emit(sprite, x + bx, y, innerW, by, sb, 0, srcInnerW, by, tile, color);
  emit(sprite, x + bx, y + h - by, innerW, by, sb, bottomV, srcInnerW, by, tile, color);
  emit(sprite, x, y + by, bx, innerH, 0, sb, bx, srcInnerH, tile, color);
  emit(sprite, x + w - bx, y + by, bx, innerH, rightU, sb, bx, srcInnerH, tile, color);
  // Center.
  emit(sprite, x + bx, y + by, innerW, innerH, sb, sb, srcInnerW, srcInnerH, tile, color);
}

void Batch::region(const Sprite &sprite, int x, int y, int u, int v, int w, int h, std::uint32_t argb) {
  emit(sprite, x, y, w, h, u, v, w, h, false, packArgb(argb));
}

void Batch::fill(int x0, int y0, int x1, int y1, std::uint32_t argb) {
  if (x1 <= x0 || y1 <= y0) {
    return;
  }
  const auto s = static_cast<float>(guiScaleFactor);
  quadList.push_back(Quad{.rect = {static_cast<float>(x0) * s, static_cast<float>(y0) * s,
                                   static_cast<float>(x1 - x0) * s, static_cast<float>(y1 - y0) * s},
                          .source = {},
                          .extent = {},
                          .texture = 0,
                          .color = packArgb(argb),
                          .kind = kFill,
                          .extra = 0});
}

void Batch::blockIcon(const BlockIcon &icon, int x, int y) {
  if (!icon.valid) {
    return;
  }
  const auto s = static_cast<float>(guiScaleFactor);
  quadList.push_back(Quad{.rect = {static_cast<float>(x) * s, static_cast<float>(y) * s, 16.0f * s, 16.0f * s},
                          .source = {},
                          .extent = {},
                          .texture = icon.top,
                          .color = icon.tint,
                          .kind = kBlockIcon,
                          .extra = icon.side});
}

void Batch::emitGlyphs(const Font &font, std::string_view text, int x, int y, std::uint32_t color) {
  const Sprite fontSprite{.texture = font.texture, .width = font.cellSize * 16, .height = font.cellSize * 16};
  const int cell = font.cellSize;
  for (const char ch : text) {
    const auto c = static_cast<unsigned char>(ch);
    if (c != ' ') {
      emit(fontSprite, x, y, cell, cell, (c % 16) * cell, (c / 16) * cell, cell, cell, false, color);
    }
    x += font.advance[c];
  }
}

int Batch::text(const Font &font, std::string_view text, int x, int y, std::uint32_t argb, bool shadow) {
  if (shadow) {
    // Vanilla's shadow: a quarter of the color, one pixel down-right.
    const std::uint32_t shadowArgb = (argb & 0xff000000u) | ((argb & 0x00fcfcfcu) >> 2u);
    emitGlyphs(font, text, x + 1, y + 1, packArgb(shadowArgb));
  }
  emitGlyphs(font, text, x, y, packArgb(argb));
  return font.width(text);
}

void Batch::centeredText(const Font &font, std::string_view text, int x, int y, std::uint32_t argb) {
  this->text(font, text, x - font.width(text) / 2, y, argb, true);
}

void Batch::item(const Context &context, const ItemStack &stack, int x, int y) {
  if (stack.empty() || stack.block >= context.icons.size()) {
    return;
  }
  blockIcon(context.icons[stack.block], x, y);
  if (stack.count > 1) {
    const std::string count = std::to_string(stack.count);
    text(*context.font, count, x + 19 - 2 - context.font->width(count), y + 6 + 3, kWhite, true);
  }
}

void drawHud(Batch &batch, const Context &context, const HudState &state) {
  const Sprites &sprites = *context.sprites;
  const int centerX = batch.width() / 2;
  const int bottom = batch.height();
  // Hotbar, selection frame and offhand (vanilla Gui.renderItemHotbar).
  const int hotbarX = centerX - 91;
  batch.sprite(sprites["sprites/hud/hotbar"], hotbarX, bottom - 22);
  batch.sprite(sprites["sprites/hud/hotbar_selection"], hotbarX - 1 + state.selected * 20, bottom - 22 - 1);
  if (!state.offhand.empty()) {
    batch.sprite(sprites["sprites/hud/hotbar_offhand_left"], hotbarX - 29, bottom - 23);
  }
  for (int slot = 0; slot < 9; ++slot) {
    batch.item(context, state.hotbar[static_cast<size_t>(slot)], hotbarX + slot * 20 + 3, bottom - 16 - 3);
  }
  if (!state.offhand.empty()) {
    batch.item(context, state.offhand, hotbarX - 26, bottom - 16 - 3);
  }
  if (!state.survivalLike) {
    return;
  }
  // XP bar (no experience yet: background only).
  batch.sprite(sprites["sprites/hud/experience_bar_background"], hotbarX, bottom - 32 + 3);
  // Hearts (left) and hunger (right), row 39 above the bottom (vanilla Gui.renderPlayerHealth).
  const int rowY = bottom - 39;
  const Sprite &container =
      sprites[state.hurtFlash ? "sprites/hud/heart/container_blinking" : "sprites/hud/heart/container"];
  for (int heart = 0; heart < 10; ++heart) {
    const int x = hotbarX + heart * 8;
    batch.sprite(container, x, rowY);
    const int points = heart * 2;
    if (points + 1 < state.health) {
      batch.sprite(sprites["sprites/hud/heart/full"], x, rowY);
    } else if (points + 1 == state.health) {
      batch.sprite(sprites["sprites/hud/heart/half"], x, rowY);
    }
  }
  // Hunger: always full (no food yet), right to left.
  const int rightX = centerX + 91;
  for (int icon = 0; icon < 10; ++icon) {
    const int x = rightX - icon * 8 - 9;
    batch.sprite(sprites["sprites/hud/food_empty"], x, rowY);
    batch.sprite(sprites["sprites/hud/food_full"], x, rowY);
  }
}

void drawGameModeSwitcher(Batch &batch, const Context &context, int selected, std::string_view title,
                          const std::array<std::uint16_t, 4> &iconBlocks) {
  const Sprites &sprites = *context.sprites;
  const int centerX = batch.width() / 2;
  const int centerY = batch.height() / 2;
  batch.region(sprites["container/gamemode_switcher"], centerX - 62, centerY - 31 - 27, 0, 0, 125, 75);
  batch.centeredText(*context.font, title, centerX, centerY - 31 - 20, kWhite);
  constexpr int kSlotSpacing = 31;
  constexpr int kAllSlotsWidth = 4 * kSlotSpacing - 5;
  for (int slot = 0; slot < 4; ++slot) {
    const int x = centerX - kAllSlotsWidth / 2 + slot * kSlotSpacing;
    const int y = centerY - 31;
    batch.sprite(sprites["sprites/gamemode_switcher/slot"], x, y);
    batch.item(context, ItemStack{.block = iconBlocks[static_cast<size_t>(slot)], .count = 1}, x + 5, y + 5);
    if (slot == selected) {
      batch.sprite(sprites["sprites/gamemode_switcher/selection"], x, y);
    }
  }
  // "[ F4 ] Next" with the key in aqua (vanilla debug.gamemodes.select_next).
  constexpr std::string_view key = "[ F4 ]";
  constexpr std::string_view rest = " Next";
  const Font &font = *context.font;
  const int x = centerX - (font.width(key) + font.width(rest)) / 2;
  const int keyWidth = batch.text(font, key, x, centerY + 5, 0xff55ffffu, true);
  batch.text(font, rest, x + keyWidth, centerY + 5, kWhite, true);
}

} // namespace gui
