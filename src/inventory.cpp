#include "inventory.hpp"

#include <algorithm>

namespace inv {

namespace {

constexpr int kMax = kMaxInventoryStack;

} // namespace

ItemStack Inventory::add(ItemStack stack) {
  if (stack.empty()) {
    return stack;
  }
  const auto mergeInto = [&](int index) {
    ItemStack &slot = items[static_cast<size_t>(index)];
    if (!slot.sameItem(stack) || slot.count >= kMax) {
      return;
    }
    const int moved = std::min<int>(stack.count, kMax - slot.count);
    slot.count = static_cast<std::uint16_t>(slot.count + moved);
    stack.count = static_cast<std::uint16_t>(stack.count - moved);
  };
  // Vanilla getSlotWithRemainingSpace: selected slot, offhand, then main slots in order.
  mergeInto(selected);
  if (!stack.empty()) {
    mergeInto(kOffhand);
  }
  for (int i = 0; i < kMainSlots && !stack.empty(); ++i) {
    mergeInto(i);
  }
  for (int i = 0; i < kMainSlots && !stack.empty(); ++i) {
    ItemStack &slot = items[static_cast<size_t>(i)];
    if (slot.empty()) {
      const int moved = std::min<int>(stack.count, kMax);
      slot = ItemStack{.block = stack.block, .count = static_cast<std::uint16_t>(moved)};
      stack.count = static_cast<std::uint16_t>(stack.count - moved);
    }
  }
  if (stack.empty()) {
    stack = {};
  }
  return stack;
}

int Inventory::room(std::uint16_t block, int limit) const {
  int total = 0;
  const auto count = [&](int index) {
    const ItemStack &slot = items[static_cast<size_t>(index)];
    if (slot.empty()) {
      return index == kOffhand ? 0 : kMax; // add() never fills an empty offhand
    }
    return slot.block == block ? kMax - slot.count : 0;
  };
  for (int i = 0; i < kMainSlots && total < limit; ++i) {
    total += count(i);
  }
  if (total < limit) {
    total += count(kOffhand);
  }
  return std::min(total, limit);
}

int Inventory::find(std::uint16_t block) const {
  for (int i = 0; i < kMainSlots; ++i) {
    if (!items[static_cast<size_t>(i)].empty() && items[static_cast<size_t>(i)].block == block) {
      return i;
    }
  }
  return -1;
}

int Inventory::freeHotbarSlot() const {
  for (int i = 0; i < kHotbarSlots; ++i) {
    if (items[static_cast<size_t>(i)].empty()) {
      return i;
    }
  }
  return -1;
}

bool Menu::mayPlace(int slotId, const ItemStack &stack) const {
  (void)stack;
  return slotList[static_cast<size_t>(slotId)].kind != SlotKind::Armor; // no armor items yet
}

std::pair<int, int> Menu::range(SlotKind kind) const {
  int begin = -1;
  int end = -1;
  for (int i = 0; i < static_cast<int>(slotList.size()); ++i) {
    if (slotList[static_cast<size_t>(i)].kind == kind) {
      if (begin < 0) {
        begin = i;
      }
      end = i + 1;
    }
  }
  return begin < 0 ? std::pair{0, 0} : std::pair{begin, end};
}

void Menu::moveTo(ItemStack &stack, int begin, int end) {
  for (int i = begin; i < end && !stack.empty(); ++i) {
    ItemStack &slot = slotItem(i);
    if (slot.sameItem(stack) && slot.count < kMax && mayPlace(i, stack)) {
      const int moved = std::min<int>(stack.count, kMax - slot.count);
      slot.count = static_cast<std::uint16_t>(slot.count + moved);
      stack.count = static_cast<std::uint16_t>(stack.count - moved);
    }
  }
  for (int i = begin; i < end && !stack.empty(); ++i) {
    ItemStack &slot = slotItem(i);
    if (slot.empty() && mayPlace(i, stack)) {
      slot = stack;
      stack = {};
    }
  }
  if (stack.empty()) {
    stack = {};
  }
}

void Menu::quickMove(int slotId) {
  ItemStack stack = slotItem(slotId);
  if (stack.empty()) {
    return;
  }
  const auto [storageBegin, storageEnd] = range(SlotKind::Storage);
  const auto [hotbarBegin, hotbarEnd] = range(SlotKind::Hotbar);
  switch (slotList[static_cast<size_t>(slotId)].kind) {
  case SlotKind::Storage:
    moveTo(stack, hotbarBegin, hotbarEnd);
    break;
  case SlotKind::Hotbar:
    moveTo(stack, storageBegin, storageEnd);
    break;
  case SlotKind::Armor:
  case SlotKind::Offhand:
    moveTo(stack, storageBegin, storageEnd);
    moveTo(stack, hotbarBegin, hotbarEnd);
    break;
  }
  slotItem(slotId) = stack;
}

void Menu::click(int slotId, int button, ClickType type, bool creative, const DropFn &drop) {
  if (slotId == kOutside) {
    if ((type == ClickType::Pickup || type == ClickType::QuickMove) && !carried.empty()) {
      if (button == 0) {
        drop(carried);
        carried = {};
      } else if (button == 1) {
        drop(ItemStack{.block = carried.block, .count = 1});
        carried.count = static_cast<std::uint16_t>(carried.count - 1);
        if (carried.empty()) {
          carried = {};
        }
      }
    }
    return;
  }
  if (slotId < 0 || slotId >= static_cast<int>(slotList.size())) {
    return;
  }
  ItemStack &slot = slotItem(slotId);
  switch (type) {
  case ClickType::Pickup: {
    if (button != 0 && button != 1) {
      return;
    }
    if (slot.empty()) {
      if (!carried.empty() && mayPlace(slotId, carried)) {
        const int moved = std::min<int>(button == 0 ? carried.count : 1, kMax);
        slot = ItemStack{.block = carried.block, .count = static_cast<std::uint16_t>(moved)};
        carried.count = static_cast<std::uint16_t>(carried.count - moved);
      }
    } else if (carried.empty()) {
      const int taken = button == 0 ? slot.count : (slot.count + 1) / 2;
      carried = ItemStack{.block = slot.block, .count = static_cast<std::uint16_t>(taken)};
      slot.count = static_cast<std::uint16_t>(slot.count - taken);
    } else if (mayPlace(slotId, carried)) {
      if (slot.sameItem(carried)) {
        const int moved = std::min<int>(button == 0 ? carried.count : 1, kMax - slot.count);
        slot.count = static_cast<std::uint16_t>(slot.count + std::max(moved, 0));
        carried.count = static_cast<std::uint16_t>(carried.count - std::max(moved, 0));
      } else if (carried.count <= kMax) {
        std::swap(slot, carried);
      }
    } else if (slot.sameItem(carried)) {
      // Can't place here: pick up into the carried stack instead.
      const int moved = std::min<int>(slot.count, kMax - carried.count);
      carried.count = static_cast<std::uint16_t>(carried.count + moved);
      slot.count = static_cast<std::uint16_t>(slot.count - moved);
    }
    break;
  }
  case ClickType::QuickMove:
    quickMove(slotId);
    break;
  case ClickType::Swap: {
    if (button != kOffhand && (button < 0 || button >= kHotbarSlots)) {
      return;
    }
    ItemStack &source = inventory->items[static_cast<size_t>(button)];
    if (&source == &slot || (source.empty() && slot.empty())) {
      return;
    }
    if (source.empty()) {
      std::swap(source, slot);
    } else if (slot.empty()) {
      if (mayPlace(slotId, source)) {
        std::swap(source, slot);
      }
    } else if (mayPlace(slotId, source) && source.count <= kMax) {
      std::swap(source, slot);
    }
    break;
  }
  case ClickType::Clone:
    if (creative && carried.empty() && !slot.empty()) {
      carried = ItemStack{.block = slot.block, .count = kMax};
    }
    break;
  case ClickType::Throw:
    if (carried.empty() && !slot.empty()) {
      const int thrown = button == 0 ? 1 : slot.count;
      drop(ItemStack{.block = slot.block, .count = static_cast<std::uint16_t>(thrown)});
      slot.count = static_cast<std::uint16_t>(slot.count - thrown);
    }
    break;
  case ClickType::PickupAll: {
    if (carried.empty()) {
      break;
    }
    // Two passes like vanilla: partial stacks first, then full ones; left button scans forward.
    const int count = static_cast<int>(slotList.size());
    for (int pass = 0; pass < 2 && carried.count < kMax; ++pass) {
      for (int k = 0; k < count && carried.count < kMax; ++k) {
        const int i = button == 0 ? k : count - 1 - k;
        ItemStack &other = slotItem(i);
        if (!other.sameItem(carried) || (pass == 0 && other.count == kMax)) {
          continue;
        }
        const int moved = std::min<int>(other.count, kMax - carried.count);
        carried.count = static_cast<std::uint16_t>(carried.count + moved);
        other.count = static_cast<std::uint16_t>(other.count - moved);
      }
    }
    break;
  }
  }
  if (slot.empty()) {
    slot = {};
  }
  if (carried.empty()) {
    carried = {};
  }
}

void Menu::beginDrag(int button) {
  dragSlotIds.clear();
  dragButton = carried.empty() ? -1 : button;
}

void Menu::dragOver(int slotId) {
  if (dragButton < 0 || slotId < 0 || slotId >= static_cast<int>(slotList.size()) ||
      std::ranges::find(dragSlotIds, slotId) != dragSlotIds.end()) {
    return;
  }
  const ItemStack &slot = slotItem(slotId);
  const bool fits = slot.empty() || (slot.sameItem(carried) && slot.count < kMax);
  // Left drag can't split into more slots than items carried.
  const bool enough = dragButton != 0 || static_cast<int>(dragSlotIds.size()) < carried.count;
  const bool enoughOne = dragButton != 1 || static_cast<int>(dragSlotIds.size()) < carried.count;
  if (fits && enough && enoughOne && mayPlace(slotId, carried)) {
    dragSlotIds.push_back(slotId);
  }
}

std::vector<ItemStack> Menu::dragPreview(ItemStack &remaining) const {
  std::vector<ItemStack> result;
  remaining = carried;
  if (dragButton < 0 || dragSlotIds.empty()) {
    return result;
  }
  const int n = static_cast<int>(dragSlotIds.size());
  for (const int id : dragSlotIds) {
    ItemStack slot = slotItem(id);
    const int per = dragButton == 0 ? carried.count / n : (dragButton == 1 ? 1 : kMax);
    const int before = slot.empty() ? 0 : slot.count;
    const int after = std::min(before + per, kMax);
    const int added = std::max(after - before, 0);
    slot = ItemStack{.block = carried.block, .count = static_cast<std::uint16_t>(after)};
    if (dragButton != 2) {
      remaining.count = static_cast<std::uint16_t>(remaining.count - std::min<int>(added, remaining.count));
    }
    result.push_back(slot);
  }
  if (remaining.empty()) {
    remaining = {};
  }
  return result;
}

void Menu::endDrag(bool creative, const DropFn &drop) {
  if (dragButton < 0) {
    return;
  }
  const int button = dragButton;
  if (dragSlotIds.size() == 1) {
    const int id = dragSlotIds.front();
    dragButton = -1;
    dragSlotIds.clear();
    if (button <= 1) {
      click(id, button, ClickType::Pickup, creative, drop);
    } else if (creative) {
      // Middle drag over one slot: a full stack there (creative).
      ItemStack &slot = slotItem(id);
      if (slot.empty()) {
        slot = ItemStack{.block = carried.block, .count = kMax};
      }
    }
    return;
  }
  if (button == 2 && !creative) {
    dragButton = -1;
    dragSlotIds.clear();
    return;
  }
  ItemStack remaining;
  const std::vector<ItemStack> preview = dragPreview(remaining);
  for (size_t i = 0; i < dragSlotIds.size(); ++i) {
    slotItem(dragSlotIds[i]) = preview[i];
  }
  carried = remaining;
  dragButton = -1;
  dragSlotIds.clear();
}

void Menu::returnCarried(const DropFn &drop) {
  if (carried.empty()) {
    return;
  }
  const ItemStack rest = inventory->add(carried);
  carried = {};
  if (!rest.empty()) {
    drop(rest);
  }
}

std::vector<Slot> survivalSlots() {
  std::vector<Slot> slots;
  for (int i = 0; i < 4; ++i) { // head first, like vanilla's InventoryMenu
    slots.push_back(Slot{.index = kArmorBase + 3 - i, .x = 8, .y = 8 + i * 18, .kind = SlotKind::Armor});
  }
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 9; ++col) {
      slots.push_back(Slot{.index = 9 + col + row * 9, .x = 8 + col * 18, .y = 84 + row * 18, .kind = SlotKind::Storage});
    }
  }
  for (int col = 0; col < 9; ++col) {
    slots.push_back(Slot{.index = col, .x = 8 + col * 18, .y = 142, .kind = SlotKind::Hotbar});
  }
  slots.push_back(Slot{.index = kOffhand, .x = 77, .y = 62, .kind = SlotKind::Offhand});
  return slots;
}

std::vector<Slot> creativeInventorySlots() {
  std::vector<Slot> slots;
  for (int i = 0; i < 4; ++i) { // head, chest, legs, feet in two columns (vanilla selectTab)
    slots.push_back(Slot{.index = kArmorBase + 3 - i, .x = 54 + (i / 2) * 54, .y = 6 + (i % 2) * 27,
                         .kind = SlotKind::Armor});
  }
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 9; ++col) {
      slots.push_back(Slot{.index = 9 + col + row * 9, .x = 9 + col * 18, .y = 54 + row * 18, .kind = SlotKind::Storage});
    }
  }
  for (int col = 0; col < 9; ++col) {
    slots.push_back(Slot{.index = col, .x = 9 + col * 18, .y = 112, .kind = SlotKind::Hotbar});
  }
  slots.push_back(Slot{.index = kOffhand, .x = 35, .y = 20, .kind = SlotKind::Offhand});
  return slots;
}

std::vector<Slot> creativeHotbarSlots() {
  std::vector<Slot> slots;
  for (int col = 0; col < 9; ++col) {
    slots.push_back(Slot{.index = col, .x = 9 + col * 18, .y = 112, .kind = SlotKind::Hotbar});
  }
  return slots;
}

void creativeGridClick(Menu &menu, const ItemStack &item, int button, ClickType type, bool shift,
                       const Menu::DropFn &drop) {
  ItemStack &carried = menu.carried;
  switch (type) {
  case ClickType::Swap:
    if (!item.empty() && button >= 0 && button < kHotbarSlots) {
      menu.inv().items[static_cast<size_t>(button)] = ItemStack{.block = item.block, .count = kMax};
    }
    return;
  case ClickType::Clone:
    if (carried.empty() && !item.empty()) {
      carried = ItemStack{.block = item.block, .count = kMax};
    }
    return;
  case ClickType::Throw:
    if (!item.empty()) {
      drop(ItemStack{.block = item.block, .count = static_cast<std::uint16_t>(button == 0 ? 1 : kMax)});
    }
    return;
  default:
    break;
  }
  if (!carried.empty() && !item.empty() && carried.block == item.block) {
    if (button == 0) {
      if (shift) {
        carried.count = kMax;
      } else if (carried.count < kMax) {
        ++carried.count;
      }
    } else {
      --carried.count;
    }
  } else if (!item.empty() && carried.empty()) {
    carried = ItemStack{.block = item.block, .count = static_cast<std::uint16_t>(shift ? kMax : 1)};
  } else if (button == 0) {
    carried = {};
  } else if (!carried.empty()) {
    --carried.count;
  }
  if (carried.empty()) {
    carried = {};
  }
}

} // namespace inv
