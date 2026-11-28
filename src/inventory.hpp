#pragma once

// Player inventory and Minecraft's container click rules (AbstractContainerMenu.doClick), Vulkan- and GLFW-free so
// the tests can drive them. A Menu is a list of slots over the Inventory plus the stack carried on the cursor.

#include "item_stack.hpp"

#include <array>
#include <functional>
#include <vector>

namespace inv {

// Inventory indices, as vanilla's Inventory: 0-8 hotbar, 9-35 main storage, 36-39 armor (feet, legs, chest, head),
// 40 offhand.
inline constexpr int kHotbarSlots = 9;
inline constexpr int kMainSlots = 36;
inline constexpr int kArmorBase = 36;
inline constexpr int kOffhand = 40;
inline constexpr int kSlotCount = 41;

struct Inventory {
  std::array<ItemStack, kSlotCount> items{};
  int selected = 0; // hotbar slot 0-8

  ItemStack &selectedStack() { return items[static_cast<size_t>(selected)]; }
  const ItemStack &selectedStack() const { return items[static_cast<size_t>(selected)]; }
  ItemStack &offhand() { return items[kOffhand]; }
  const ItemStack &offhand() const { return items[kOffhand]; }

  /**
   * Add a stack like vanilla's Inventory.add: merge into the selected slot, the offhand, then main slots (0-35) that
   * hold the same block and have room, then fill the first empty main slot (hotbar first).
   *
   * @return what didn't fit (empty when everything was added).
   */
  ItemStack add(ItemStack stack);
  /** How many of `block` would fit (merge room plus empty main slots), capped at `limit`. */
  int room(std::uint16_t block, int limit) const;
  /** First main slot (0-35) holding `block`, or -1. */
  int find(std::uint16_t block) const;
  /** First empty hotbar slot, or -1. */
  int freeHotbarSlot() const;
  void clear() { items.fill(ItemStack{}); }
};

enum class ClickType {
  Pickup,     // left / right click: take, place, swap, merge
  QuickMove,  // shift-click
  Swap,       // number key (button = hotbar 0-8) or F (button = kOffhand) over a slot
  Clone,      // middle click (creative): full stack of the slot's item
  Throw,      // Q over a slot: button 0 = one, 1 = whole stack (Ctrl+Q)
  PickupAll,  // double-click: gather the carried item from every slot
};

// Where a slot sits in a menu and what it accepts.
enum class SlotKind { Storage, Hotbar, Armor, Offhand };

struct Slot {
  int index = 0; // Inventory index
  int x = 0;     // GUI pixels from the screen's top-left (item corner)
  int y = 0;
  SlotKind kind = SlotKind::Storage;
};

// A menu over the player's inventory (the survival inventory screen, the creative inventory tab).
class Menu {
public:
  using DropFn = std::function<void(const ItemStack &)>;

  static constexpr int kOutside = -999; // slot id for clicks outside the window

  Menu(Inventory &inventory, std::vector<Slot> slots) : inventory(&inventory), slotList(std::move(slots)) {}

  /**
   * Handle a click (vanilla doClick).
   *
   * @param slotId Index into slots(), kOutside for a click outside the window, or -1 for none.
   * @param button Mouse button (0 left, 1 right, 2 middle), hotbar slot / kOffhand for Swap, 0/1 for Throw.
   * @param creative Creative mode: enables Clone.
   * @param drop Receives stacks thrown out of the menu (item entities).
   */
  void click(int slotId, int button, ClickType type, bool creative, const DropFn &drop);

  // Mouse drag over slots with a carried stack (vanilla quick craft): left spreads evenly, right places one each,
  // middle (creative) places full stacks.
  void beginDrag(int button);
  /** Add a slot the drag passed over, if it can take the carried item. */
  void dragOver(int slotId);
  /** Finish: distribute over the dragged slots (one slot = an ordinary click). */
  void endDrag(bool creative, const DropFn &drop);
  bool dragging() const { return dragButton >= 0; }
  const std::vector<int> &dragSlots() const { return dragSlotIds; }
  /** Preview of the drag: what each dragged slot would hold and what stays carried. */
  std::vector<ItemStack> dragPreview(ItemStack &remaining) const;

  /** Put the carried stack back into the inventory (closing the screen); the remainder is dropped. */
  void returnCarried(const DropFn &drop);

  const std::vector<Slot> &slots() const { return slotList; }
  ItemStack &slotItem(int slotId) { return inventory->items[static_cast<size_t>(slotList[static_cast<size_t>(slotId)].index)]; }
  const ItemStack &slotItem(int slotId) const {
    return inventory->items[static_cast<size_t>(slotList[static_cast<size_t>(slotId)].index)];
  }
  /** Whether a slot accepts `stack` (armor slots take nothing until armor exists). */
  bool mayPlace(int slotId, const ItemStack &stack) const;

  ItemStack carried{};
  Inventory &inv() { return *inventory; }

private:
  void quickMove(int slotId);
  /** Merge `stack` into slots [begin, end), then fill empty ones (vanilla moveItemStackTo). */
  void moveTo(ItemStack &stack, int begin, int end);
  /** First and one-past-last slot id of a kind (slots of a kind are contiguous). */
  std::pair<int, int> range(SlotKind kind) const;

  Inventory *inventory;
  std::vector<Slot> slotList;
  int dragButton = -1;
  std::vector<int> dragSlotIds;
};

/** Slots of the survival inventory screen (container/inventory.png layout): armor, storage, hotbar, offhand. */
std::vector<Slot> survivalSlots();
/** Slots of the creative inventory tab (creative_inventory/tab_inventory.png layout). */
std::vector<Slot> creativeInventorySlots();
/** Slots of a creative item tab: only the hotbar row under the grid. */
std::vector<Slot> creativeHotbarSlots();

/**
 * Click on a creative tab's item grid (vanilla CreativeModeInventoryScreen.slotClicked): the grid is an infinite
 * source, and clicking it with a carried item deletes or adjusts it.
 *
 * @param item The grid entry under the cursor (count 1), or empty.
 * @param shift Shift held (left click: full stack).
 */
void creativeGridClick(Menu &menu, const ItemStack &item, int button, ClickType type, bool shift,
                       const Menu::DropFn &drop);

} // namespace inv
