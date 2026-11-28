#include "screens.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace gui {

namespace {

constexpr std::uint32_t kLabelColor = 0xff404040u; // vanilla container labels, no shadow
constexpr double kDoubleClickSeconds = 0.25;
constexpr int kGridColumns = 9;
constexpr int kGridRows = 5;
constexpr int kGridX = 9;
constexpr int kGridY = 18;
constexpr int kScrollX = 175;
constexpr int kScrollY = 18;
constexpr int kScrollHeight = 112;
constexpr int kDestroyX = 173;
constexpr int kDestroyY = 112;

struct TabInfo {
  const char *title;
  const char *category; // blocks.txt category listed by the tab, or nullptr
  bool top;
  int column;
};

// Vanilla CreativeModeTabs: row, column; columns 5-6 are right-aligned.
constexpr std::array<TabInfo, InventoryScreen::kTabCount> kTabs{{
    {"Building Blocks", "building_blocks", true, 0},
    {"Colored Blocks", "colored_blocks", true, 1},
    {"Natural Blocks", "natural_blocks", true, 2},
    {"Functional Blocks", "functional_blocks", true, 3},
    {"Redstone Blocks", "redstone_blocks", true, 4},
    {"Saved Hotbars", nullptr, true, 5},
    {"Search Items", nullptr, true, 6},
    {"Tools & Utilities", "tools", false, 0},
    {"Combat", "combat", false, 1},
    {"Food & Drinks", "food", false, 2},
    {"Ingredients", "ingredients", false, 3},
    {"Spawn Eggs", "spawn_eggs", false, 4},
    {"Operator Utilities", nullptr, false, 5},
    {"Survival Inventory", nullptr, false, 6},
}};

bool inside(double x, double y, int x0, int y0, int w, int h) { return x >= x0 && y >= y0 && x < x0 + w && y < y0 + h; }

std::string lower(std::string_view text) {
  std::string result(text);
  for (char &c : result) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

} // namespace

InventoryScreen::InventoryScreen(inv::Inventory &inventory, bool creative, std::vector<CreativeItem> catalog,
                                 const SavedHotbars *savedHotbars, inv::Menu::DropFn drop)
    : inventory(&inventory), creative(creative), catalog(std::move(catalog)), savedHotbars(savedHotbars),
      drop(std::move(drop)), menu(inventory, creative ? inv::creativeHotbarSlots() : inv::survivalSlots()) {
  if (creative) {
    selectTab(kBuilding);
  }
}

void InventoryScreen::layout(int guiWidth, int guiHeight) {
  left = (guiWidth - imageWidth()) / 2;
  top = (guiHeight - imageHeight()) / 2;
}

void InventoryScreen::selectTab(int tab) {
  if (tab < 0 || tab >= kTabCount || !tabPosition(tab)) {
    return;
  }
  currentTab = tab;
  scrollOffset = 0.0f;
  scrolling = false;
  if (tab != kSearch) {
    search.clear();
  }
  rebuildMenu();
  rebuildTabItems();
}

void InventoryScreen::rebuildMenu() {
  const ItemStack carried = menu.carried;
  menu = inv::Menu(*inventory, !creative                    ? inv::survivalSlots()
                               : currentTab == kInventoryTab ? inv::creativeInventorySlots()
                                                             : inv::creativeHotbarSlots());
  menu.carried = carried;
}

void InventoryScreen::rebuildTabItems() {
  tabItems.clear();
  if (!creative || currentTab == kInventoryTab) {
    return;
  }
  if (currentTab == kSavedHotbars) {
    if (savedHotbars != nullptr) {
      for (const auto &row : *savedHotbars) {
        tabItems.insert(tabItems.end(), row.begin(), row.end());
      }
    }
    return;
  }
  const std::string query = lower(search);
  for (const CreativeItem &item : catalog) {
    const bool listed = currentTab == kSearch ? lower(item.name).find(query) != std::string::npos
                                              : item.category == kTabs[static_cast<size_t>(currentTab)].category;
    if (listed) {
      tabItems.push_back(ItemStack{.block = item.block, .count = 1});
    }
  }
}

std::optional<std::pair<int, int>> InventoryScreen::tabPosition(int tab) const {
  if (tab == kOpBlocks) {
    return std::nullopt; // vanilla hides it without operator permissions
  }
  const TabInfo &info = kTabs[static_cast<size_t>(tab)];
  const int x = info.column >= 5 ? imageWidth() - 27 * (7 - info.column) + 1 : 27 * info.column;
  const int y = info.top ? -28 : imageHeight() - 4;
  return std::pair{x, y};
}

int InventoryScreen::gridRows() const { return static_cast<int>((tabItems.size() + kGridColumns - 1) / kGridColumns); }

bool InventoryScreen::canScroll() const { return itemTab() && gridRows() > kGridRows; }

ItemStack InventoryScreen::gridItem(int cell) const {
  const int extraRows = std::max(gridRows() - kGridRows, 0);
  const int rowOffset = static_cast<int>(std::lround(scrollOffset * static_cast<float>(extraRows)));
  const size_t index = static_cast<size_t>(cell + rowOffset * kGridColumns);
  return index < tabItems.size() ? tabItems[index] : ItemStack{};
}

InventoryScreen::Hit InventoryScreen::hitTest(double x, double y) const {
  const double lx = x - left;
  const double ly = y - top;
  const auto &slots = menu.slots();
  for (int i = 0; i < static_cast<int>(slots.size()); ++i) {
    const inv::Slot &slot = slots[static_cast<size_t>(i)];
    if (inside(lx, ly, slot.x - 1, slot.y - 1, 18, 18)) {
      return Hit{.kind = Hit::Kind::Slot, .index = i};
    }
  }
  if (itemTab()) {
    for (int cell = 0; cell < kGridColumns * kGridRows; ++cell) {
      if (inside(lx, ly, kGridX + (cell % kGridColumns) * 18 - 1, kGridY + (cell / kGridColumns) * 18 - 1, 18, 18)) {
        return Hit{.kind = Hit::Kind::Grid, .index = cell};
      }
    }
  }
  if (creative && currentTab == kInventoryTab && inside(lx, ly, kDestroyX - 1, kDestroyY - 1, 18, 18)) {
    return Hit{.kind = Hit::Kind::Destroy};
  }
  if (!inside(lx, ly, 0, 0, imageWidth(), imageHeight())) {
    return Hit{.kind = Hit::Kind::Outside};
  }
  return Hit{};
}

void InventoryScreen::mouseMove(double x, double y) {
  mouseX = x;
  mouseY = y;
  if (scrolling) {
    const double scrollTop = top + kScrollY;
    scrollOffset = static_cast<float>(
        std::clamp((y - scrollTop - 7.5) / (kScrollHeight - 15.0), 0.0, 1.0));
    return;
  }
  if (menu.dragging()) {
    const Hit hit = hitTest(x, y);
    if (hit.kind == Hit::Kind::Slot) {
      menu.dragOver(hit.index);
    }
  }
}

void InventoryScreen::mousePress(int button, bool shift, double seconds) {
  if (button < 0 || button > 2 || dragMouseButton >= 0) {
    return;
  }
  if (creative) {
    for (int tab = 0; tab < kTabCount; ++tab) {
      const auto position = tabPosition(tab);
      if (position && inside(mouseX - left, mouseY - top, position->first, position->second, 26, 32)) {
        if (button == 0) {
          selectTab(tab);
        }
        return;
      }
    }
    if (canScroll() && button == 0 && inside(mouseX - left, mouseY - top, kScrollX, kScrollY, 14, kScrollHeight)) {
      scrolling = true;
      mouseMove(mouseX, mouseY);
      return;
    }
  }
  const Hit hit = hitTest(mouseX, mouseY);
  const bool doubleClick = seconds - lastClickTime < kDoubleClickSeconds && lastClickButton == button &&
                           lastClickHit.kind == hit.kind && lastClickHit.index == hit.index;
  lastClickTime = seconds;
  lastClickHit = hit;
  lastClickButton = button;

  switch (hit.kind) {
  case Hit::Kind::Grid: {
    const ItemStack item = gridItem(hit.index);
    inv::creativeGridClick(menu, item, button, button == 2 ? inv::ClickType::Clone : inv::ClickType::Pickup, shift,
                           drop);
    return;
  }
  case Hit::Kind::Destroy:
    if (shift) {
      inventory->clear(); // vanilla: shift-click on the destroy slot clears the inventory
    }
    menu.carried = {};
    return;
  case Hit::Kind::None:
    return;
  default:
    break;
  }
  const int slotId = hit.kind == Hit::Kind::Slot ? hit.index : inv::Menu::kOutside;
  if (menu.carried.empty()) {
    if (hit.kind != Hit::Kind::Slot) {
      return;
    }
    if (doubleClick && shift && button == 0 && hasLastQuickMoved) {
      // Shift double-click: quick-move every stack of the last quick-moved item.
      for (int i = 0; i < static_cast<int>(menu.slots().size()); ++i) {
        const ItemStack &item = menu.slotItem(i);
        if (!item.empty() && item.block == lastQuickMoved) {
          menu.click(i, 0, inv::ClickType::QuickMove, creative, drop);
        }
      }
      return;
    }
    const ItemStack before = menu.slotItem(slotId);
    if (shift && button != 2) {
      lastQuickMoved = before.block;
      hasLastQuickMoved = !before.empty();
      menu.click(slotId, button, inv::ClickType::QuickMove, creative, drop);
    } else if (button == 2) {
      menu.click(slotId, button, inv::ClickType::Clone, creative, drop);
    } else {
      menu.click(slotId, button, inv::ClickType::Pickup, creative, drop);
    }
    return;
  }
  if (doubleClick && button == 0 && hit.kind == Hit::Kind::Slot) {
    menu.click(slotId, 0, inv::ClickType::PickupAll, creative, drop);
    return;
  }
  if (button == 2 && !creative) {
    return;
  }
  // Carried stack: start a drag; a release without spreading is an ordinary click on the pressed slot.
  dragMouseButton = button;
  pressHit = hit;
  menu.beginDrag(button);
  if (hit.kind == Hit::Kind::Slot) {
    menu.dragOver(hit.index);
  }
}

void InventoryScreen::mouseRelease(int button) {
  if (button == 0) {
    scrolling = false;
  }
  if (button != dragMouseButton) {
    return;
  }
  dragMouseButton = -1;
  if (menu.dragSlots().size() >= 1) {
    menu.endDrag(creative, drop);
    return;
  }
  menu.endDrag(creative, drop); // nothing dragged: just resets
  if (button <= 1) {
    const int slotId = pressHit.kind == Hit::Kind::Slot ? pressHit.index
                       : pressHit.kind == Hit::Kind::Outside ? inv::Menu::kOutside
                                                             : -1;
    menu.click(slotId, button, inv::ClickType::Pickup, creative, drop);
  }
}

void InventoryScreen::scroll(double steps) {
  if (!canScroll()) {
    return;
  }
  const int extraRows = gridRows() - kGridRows;
  scrollOffset = std::clamp(scrollOffset - static_cast<float>(steps) / static_cast<float>(extraRows), 0.0f, 1.0f);
}

void InventoryScreen::hotbarKey(int slot) {
  if (!menu.carried.empty()) {
    return;
  }
  const Hit hit = hitTest(mouseX, mouseY);
  if (hit.kind == Hit::Kind::Slot) {
    menu.click(hit.index, slot, inv::ClickType::Swap, creative, drop);
  } else if (hit.kind == Hit::Kind::Grid) {
    inv::creativeGridClick(menu, gridItem(hit.index), slot, inv::ClickType::Swap, false, drop);
  }
}

void InventoryScreen::offhandKey() {
  if (!menu.carried.empty()) {
    return;
  }
  const Hit hit = hitTest(mouseX, mouseY);
  if (hit.kind == Hit::Kind::Slot) {
    menu.click(hit.index, inv::kOffhand, inv::ClickType::Swap, creative, drop);
  }
}

void InventoryScreen::dropKey(bool wholeStack) {
  const Hit hit = hitTest(mouseX, mouseY);
  if (hit.kind == Hit::Kind::Slot) {
    menu.click(hit.index, wholeStack ? 1 : 0, inv::ClickType::Throw, creative, drop);
  } else if (hit.kind == Hit::Kind::Grid) {
    inv::creativeGridClick(menu, gridItem(hit.index), wholeStack ? 1 : 0, inv::ClickType::Throw, false, drop);
  }
}

bool InventoryScreen::typing() const { return creative && currentTab == kSearch; }

bool InventoryScreen::character(unsigned codepoint) {
  if (!creative) {
    return false;
  }
  if (currentTab != kSearch) {
    // T (vanilla's chat key) jumps to search from the item tabs.
    if ((codepoint == 't' || codepoint == 'T') && currentTab != kInventoryTab) {
      selectTab(kSearch);
      return true;
    }
    return false;
  }
  if (codepoint >= 32 && codepoint < 127 && search.size() < 50) {
    search.push_back(static_cast<char>(codepoint));
    scrollOffset = 0.0f;
    rebuildTabItems();
  }
  return true;
}

void InventoryScreen::backspace() {
  if (typing() && !search.empty()) {
    search.pop_back();
    scrollOffset = 0.0f;
    rebuildTabItems();
  }
}

void InventoryScreen::close() { menu.returnCarried(drop); }

std::string InventoryScreen::nameOf(std::uint16_t block) const {
  for (const CreativeItem &item : catalog) {
    if (item.block == block) {
      return item.name;
    }
  }
  return "Block " + std::to_string(block);
}

void InventoryScreen::drawCreativeChrome(Batch &batch, const Context &context, bool selectedLayer) const {
  const Sprites &sprites = *context.sprites;
  for (int tab = 0; tab < kTabCount; ++tab) {
    const bool selected = tab == currentTab;
    if (selected != selectedLayer) {
      continue;
    }
    const auto position = tabPosition(tab);
    if (!position) {
      continue;
    }
    const TabInfo &info = kTabs[static_cast<size_t>(tab)];
    const std::string sprite = std::string("sprites/container/creative_inventory/tab_") +
                               (info.top ? "top_" : "bottom_") + (selected ? "selected_" : "unselected_") +
                               std::to_string(info.column + 1);
    const int x = left + position->first;
    const int y = top + position->second;
    batch.sprite(sprites[sprite], x, y);
    // Icon: the tab's first block (no item icons exist for the special tabs yet).
    if (info.category != nullptr) {
      for (const CreativeItem &item : catalog) {
        if (item.category == info.category) {
          batch.item(context, ItemStack{.block = item.block, .count = 1}, x + 5, y + 8 + (info.top ? 1 : -1));
          break;
        }
      }
    }
  }
}

void InventoryScreen::drawSlots(Batch &batch, const Context &context, const Hit &hover) const {
  const Sprites &sprites = *context.sprites;
  ItemStack carriedPreview;
  const std::vector<ItemStack> preview = menu.dragPreview(carriedPreview);
  const auto &slots = menu.slots();
  static constexpr std::array<const char *, 4> armorIcons{
      "sprites/container/slot/helmet", "sprites/container/slot/chestplate", "sprites/container/slot/leggings",
      "sprites/container/slot/boots"};
  const auto highlightBack = [&](int x, int y) {
    batch.sprite(sprites["sprites/container/slot_highlight_back"], x - 4, y - 4, 24, 24);
  };
  const auto highlightFront = [&](int x, int y) {
    batch.sprite(sprites["sprites/container/slot_highlight_front"], x - 4, y - 4, 24, 24);
  };
  for (int i = 0; i < static_cast<int>(slots.size()); ++i) {
    const inv::Slot &slot = slots[static_cast<size_t>(i)];
    const int x = left + slot.x;
    const int y = top + slot.y;
    const bool hovered = hover.kind == Hit::Kind::Slot && hover.index == i;
    if (hovered) {
      highlightBack(x, y);
    }
    ItemStack item = menu.slotItem(i);
    const auto dragged = std::ranges::find(menu.dragSlots(), i);
    if (dragged != menu.dragSlots().end() && menu.dragSlots().size() > 1) {
      item = preview[static_cast<size_t>(dragged - menu.dragSlots().begin())];
    }
    if (item.empty()) {
      if (slot.kind == inv::SlotKind::Armor) {
        batch.sprite(sprites[armorIcons[static_cast<size_t>(inv::kArmorBase + 3 - slot.index)]], x, y);
      } else if (slot.kind == inv::SlotKind::Offhand) {
        batch.sprite(sprites["sprites/container/slot/shield"], x, y);
      }
    } else {
      batch.item(context, item, x, y);
    }
    if (dragged != menu.dragSlots().end() && menu.dragSlots().size() > 1) {
      batch.fill(x, y, x + 16, y + 16, 0x80ffffffu); // vanilla's quick-craft slot overlay
    }
    if (hovered) {
      highlightFront(x, y);
    }
  }
  if (itemTab()) {
    for (int cell = 0; cell < kGridColumns * kGridRows; ++cell) {
      const int x = left + kGridX + (cell % kGridColumns) * 18;
      const int y = top + kGridY + (cell / kGridColumns) * 18;
      const bool hovered = hover.kind == Hit::Kind::Grid && hover.index == cell;
      if (hovered) {
        highlightBack(x, y);
      }
      batch.item(context, gridItem(cell), x, y);
      if (hovered) {
        highlightFront(x, y);
      }
    }
  }
  if (hover.kind == Hit::Kind::Destroy) {
    const int x = left + kDestroyX;
    const int y = top + kDestroyY;
    highlightBack(x, y);
    highlightFront(x, y);
  }
}

void InventoryScreen::drawTooltip(Batch &batch, const Context &context, std::string_view text) const {
  const int width = context.font->width(text) - 1;
  const int height = 8;
  int x = static_cast<int>(mouseX) + 12;
  int y = static_cast<int>(mouseY) - 12;
  if (x + width + 4 > batch.width()) {
    x = std::max(static_cast<int>(mouseX) - 16 - width, 4);
  }
  y = std::clamp(y, 4, std::max(batch.height() - height - 4, 4));
  // Vanilla TooltipRenderUtil: background and frame sprites around the text with 3 px padding (+9 sprite margin).
  const Sprites &sprites = *context.sprites;
  batch.sprite(sprites["sprites/tooltip/background"], x - 3 - 9, y - 3 - 9, width + 6 + 18, height + 6 + 18);
  batch.sprite(sprites["sprites/tooltip/frame"], x - 3 - 9, y - 3 - 9, width + 6 + 18, height + 6 + 18);
  batch.text(*context.font, text, x, y, kWhite, true);
}

void InventoryScreen::draw(Batch &batch, const Context &context) const {
  const Sprites &sprites = *context.sprites;
  const Font &font = *context.font;
  // Darkened world behind the menu (vanilla renderTransparentBackground).
  batch.fill(0, 0, batch.width(), batch.height(), 0xc8101010u);
  if (creative) {
    drawCreativeChrome(batch, context, false);
    const char *background = currentTab == kInventoryTab ? "container/creative_inventory/tab_inventory"
                             : currentTab == kSearch      ? "container/creative_inventory/tab_item_search"
                                                          : "container/creative_inventory/tab_items";
    batch.region(sprites[background], left, top, 0, 0, imageWidth(), imageHeight());
    drawCreativeChrome(batch, context, true);
    if (currentTab != kInventoryTab && currentTab != kSearch) {
      batch.text(font, kTabs[static_cast<size_t>(currentTab)].title, left + 8, top + 6, kLabelColor, false);
    }
    if (currentTab == kSearch) {
      const int width = batch.text(font, search, left + 82, top + 6, kWhite, true);
      batch.text(font, "_", left + 82 + width, top + 6, kWhite, true);
    }
    if (itemTab()) {
      const int extraRows = std::max(gridRows() - kGridRows, 0);
      (void)extraRows;
      const int scrollerY = top + kScrollY + static_cast<int>(static_cast<float>(kScrollHeight - 17) * scrollOffset);
      batch.sprite(sprites[canScroll() ? "sprites/container/creative_inventory/scroller"
                                       : "sprites/container/creative_inventory/scroller_disabled"],
                   left + kScrollX, scrollerY);
    }
  } else {
    batch.region(sprites["container/inventory"], left, top, 0, 0, imageWidth(), imageHeight());
    batch.text(font, "Crafting", left + 97, top + 8, kLabelColor, false);
  }

  const Hit hover = hitTest(mouseX, mouseY);
  drawSlots(batch, context, hover);

  // Carried stack at the cursor (what remains after a drag preview).
  ItemStack carried;
  menu.dragPreview(carried);
  if (!menu.dragging() || menu.dragSlots().size() <= 1) {
    carried = menu.carried;
  }
  batch.item(context, carried, static_cast<int>(mouseX) - 8, static_cast<int>(mouseY) - 8);

  if (!menu.carried.empty()) {
    return;
  }
  if (hover.kind == Hit::Kind::Slot && !menu.slotItem(hover.index).empty()) {
    drawTooltip(batch, context, nameOf(menu.slotItem(hover.index).block));
  } else if (hover.kind == Hit::Kind::Grid && !gridItem(hover.index).empty()) {
    drawTooltip(batch, context, nameOf(gridItem(hover.index).block));
  } else if (hover.kind == Hit::Kind::Destroy) {
    drawTooltip(batch, context, "Destroy Item");
  } else if (creative) {
    for (int tab = 0; tab < kTabCount; ++tab) {
      const auto position = tabPosition(tab);
      if (position && inside(mouseX - left, mouseY - top, position->first, position->second, 26, 32)) {
        drawTooltip(batch, context, kTabs[static_cast<size_t>(tab)].title);
        break;
      }
    }
  }
}

} // namespace gui
