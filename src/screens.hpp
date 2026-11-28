#pragma once

// Inventory screens: the survival inventory and the creative inventory (vanilla InventoryScreen /
// CreativeModeInventoryScreen layouts and mouse rules). Input arrives in GUI pixels; drawing goes into a gui::Batch.
// GLFW- and Vulkan-free: the renderer translates its input events.

#include "gui.hpp"
#include "inventory.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace gui {

// A block the creative tabs list.
struct CreativeItem {
  std::uint16_t block = 0;
  std::string name;     // display name (tooltip, search)
  std::string category; // blocks::kCategories entry
};

// Saved creative hotbars (C+1..9 saves, X+1..9 loads), shown by the Saved Hotbars tab.
using SavedHotbars = std::array<std::array<ItemStack, inv::kHotbarSlots>, 9>;

class InventoryScreen {
public:
  /**
   * @param creative Creative inventory (tabs) instead of the survival one.
   * @param catalog Blocks for the creative tabs, in display order.
   * @param drop Receives stacks thrown out of the screen.
   */
  InventoryScreen(inv::Inventory &inventory, bool creative, std::vector<CreativeItem> catalog,
                  const SavedHotbars *savedHotbars, inv::Menu::DropFn drop);

  /** Window size changed / new frame: the screen is centered on a GUI of this size. */
  void layout(int guiWidth, int guiHeight);
  void mouseMove(double x, double y);
  /**
   * @param button 0 left, 1 right, 2 middle.
   * @param seconds Time of the press (double-click detection).
   */
  void mousePress(int button, bool shift, double seconds);
  void mouseRelease(int button);
  void scroll(double steps);
  /** Number key 1-9 (slot 0-8) over the hovered slot: swap with that hotbar slot. */
  void hotbarKey(int slot);
  /** F over the hovered slot: swap with the offhand. */
  void offhandKey();
  /** Q over the hovered slot (Ctrl: whole stack). */
  void dropKey(bool wholeStack);
  /** Text input for creative search; returns true when consumed (search focused or T opened it). */
  bool character(unsigned codepoint);
  void backspace();
  /** Whether typed keys belong to the search box (so E doesn't close the screen). */
  bool typing() const;
  /** Put the carried stack back into the inventory (remainder dropped). Call before discarding the screen. */
  void close();

  void draw(Batch &batch, const Context &context) const;

  // Creative tab ids (vanilla CreativeModeTabs order).
  enum Tab {
    kBuilding, kColored, kNatural, kFunctional, kRedstone, kSavedHotbars, kSearch,
    kTools, kCombat, kFood, kIngredients, kSpawnEggs, kOpBlocks, kInventoryTab, kTabCount
  };
  void selectTab(int tab);
  int selectedTab() const { return currentTab; }

private:
  // What is under the cursor.
  struct Hit {
    enum class Kind { None, Slot, Grid, Destroy, Outside } kind = Kind::None;
    int index = -1; // menu slot id or grid cell
  };

  Hit hitTest(double x, double y) const;
  /** Item at a creative grid cell (scroll applied), or empty. */
  ItemStack gridItem(int cell) const;
  int gridRows() const;
  bool itemTab() const { return creative && currentTab != kInventoryTab; }
  bool canScroll() const;
  void rebuildTabItems();
  void rebuildMenu();
  int imageWidth() const { return creative ? 195 : 176; }
  int imageHeight() const { return creative ? 136 : 166; }
  /** Top-left of tab `tab` relative to the image, or nullopt for hidden tabs. */
  std::optional<std::pair<int, int>> tabPosition(int tab) const;
  void drawSlots(Batch &batch, const Context &context, const Hit &hover) const;
  void drawCreativeChrome(Batch &batch, const Context &context, bool selectedLayer) const;
  void drawTooltip(Batch &batch, const Context &context, std::string_view text) const;
  std::string nameOf(std::uint16_t block) const;

  inv::Inventory *inventory;
  bool creative;
  std::vector<CreativeItem> catalog;
  const SavedHotbars *savedHotbars;
  inv::Menu::DropFn drop;
  inv::Menu menu;

  int left = 0;
  int top = 0;
  double mouseX = 0.0;
  double mouseY = 0.0;

  int currentTab = kBuilding;
  std::vector<ItemStack> tabItems; // items of the current item tab (rows of 9; saved hotbars keep empties)
  float scrollOffset = 0.0f;       // 0..1
  bool scrolling = false;
  std::string search;

  // Click tracking: drag in progress, double-click.
  int dragMouseButton = -1;
  Hit pressHit;
  double lastClickTime = -1.0;
  Hit lastClickHit;
  int lastClickButton = -1;
  std::uint16_t lastQuickMoved = 0;
  bool hasLastQuickMoved = false;
};

} // namespace gui
