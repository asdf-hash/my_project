#pragma once

#include <cstdint>

// A stack of one block type. Inventory slots hold up to kMaxInventoryStack; dropped item entities merge up to
// kMaxEntityStack (fewer entities for big drops). Block ID 0 is a real block, so emptiness is count == 0.
inline constexpr std::uint16_t kMaxInventoryStack = 64;
inline constexpr std::uint16_t kMaxEntityStack = 2048;

struct ItemStack {
  std::uint16_t block = 0;
  std::uint16_t count = 0;

  bool empty() const { return count == 0; }
  /** Same block (both non-empty), so the stacks may merge. */
  bool sameItem(const ItemStack &other) const { return !empty() && !other.empty() && block == other.block; }
  bool operator==(const ItemStack &) const = default;
};
