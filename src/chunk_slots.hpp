#pragma once

// Persistent GPU slots for incremental remeshing. A visible column keeps its column slot while its LOD and neighbor
// LODs stay the same, so a clipmap change only generates new columns and retires departed ones. Per-chunk work slots
// are allocated on the GPU (allocateWork / retireWork in chunk_compute.slang).

#include "world.hpp"

#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

// Lowest-free-first slot allocator, so live slots stay packed near 0 and the task shader can walk [0, highWater).
class SlotAllocator {
public:
  /** Take the lowest free slot, or grow the high-water mark. */
  uint32_t allocate() {
    while (!free_.empty()) {
      const uint32_t slot = free_.top();
      free_.pop();
      if (slot < highWater_ && !used_[slot]) { // skip stale entries (above highWater_, or reused since)
        used_[slot] = true;
        return slot;
      }
    }
    const uint32_t slot = highWater_++;
    if (used_.size() < highWater_) {
      used_.resize(highWater_, false);
    }
    used_[slot] = true;
    return slot;
  }

  /** Free `slot` and lower the high-water mark past any trailing free slots. */
  void release(uint32_t slot) {
    used_[slot] = false;
    free_.push(slot);
    while (highWater_ > 0 && !used_[highWater_ - 1]) {
      --highWater_;
    }
  }

  /** Free every slot. */
  void reset() {
    used_.clear();
    free_ = {};
    highWater_ = 0;
  }

  /** One past the highest slot in use. */
  uint32_t highWater() const { return highWater_; }

private:
  std::vector<bool> used_;
  std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<>> free_; // min-heap; may hold stale entries
  uint32_t highWater_ = 0;
};

// First-fit allocator of [offset, offset + size) ranges in a fixed-capacity buffer, with coalescing frees. Used for
// the CPU-meshed quads (renderer.cpp), whose chunks come and go in arbitrary order.
class RangeAllocator {
public:
  explicit RangeAllocator(uint32_t capacity) : capacity_(capacity) { free_.emplace(0u, capacity); }

  /**
   * Allocate `size` elements from the first free range large enough.
   *
   * @return offset of the range (0 for size 0), or nullopt if no free range fits.
   */
  std::optional<uint32_t> allocate(uint32_t size) {
    if (size == 0u) {
      return 0u;
    }
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second < size) {
        continue;
      }
      const uint32_t offset = it->first;
      const uint32_t remaining = it->second - size;
      free_.erase(it);
      if (remaining != 0u) {
        free_.emplace(offset + size, remaining);
      }
      return offset;
    }
    return std::nullopt;
  }

  /** Return [offset, offset + size) to the free list, merging it with adjacent free ranges. */
  void release(uint32_t offset, uint32_t size) {
    if (size == 0u) {
      return;
    }
    auto next = free_.lower_bound(offset);
    if (next != free_.begin()) {
      auto previous = std::prev(next);
      if (previous->first + previous->second == offset) { // merge into the range before
        offset = previous->first;
        size += previous->second;
        free_.erase(previous);
      }
    }
    if (next != free_.end() && offset + size == next->first) { // and the range after
      size += next->second;
      free_.erase(next);
    }
    free_.emplace(offset, size);
  }

  /** One past the highest allocated element. */
  uint32_t highWater() const {
    if (free_.empty()) {
      return capacity_;
    }
    const auto &last = *free_.rbegin();
    return last.first + last.second == capacity_ ? last.first : capacity_;
  }

private:
  uint32_t capacity_;
  std::map<uint32_t, uint32_t> free_; // offset -> size
};

// What a remesh has to do on the GPU, in this order: retire, generate heights, allocate work, mesh.
struct ChunkRemeshPlan {
  bool fullRebuild = false;          // clear clusters, the quad buffer and the work slot pool first
  std::vector<uint32_t> retiredColumns; // column slots that left the clipmap (empty on a full rebuild)
  std::vector<uint32_t> newColumns;     // column slots that need generateHeights and work
};

class ChunkSlotTable {
public:
  /**
   * Move the column slots to a new clipmap.
   *
   * @param chunks Columns of a ClipmapResult.
   * @param neighborLods Packed neighbor LODs per entry of `chunks` (ClipmapResult::neighborLods).
   * @param fullRebuild Drop every slot first and treat all columns as new.
   * @return the GPU work (retired and new column slots) for the change.
   */
  ChunkRemeshPlan update(const std::vector<world::ChunkLod> &chunks, const std::vector<uint32_t> &neighborLods,
                         bool fullRebuild) {
    ChunkRemeshPlan plan;
    plan.fullRebuild = fullRebuild;
    if (fullRebuild) {
      columns_.clear();
      columnSlots_.reset();
    }

    for (auto &[key, entry] : columns_) {
      entry.seen = false;
    }
    std::vector<size_t> added;
    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto it = columns_.find(keyOf(chunks[i], neighborLods[i]));
      if (it != columns_.end()) {
        it->second.seen = true;
      } else {
        added.push_back(i);
      }
    }

    // Free what left first, so new columns can reuse its slots in the same remesh (retire runs before allocating).
    for (auto it = columns_.begin(); it != columns_.end();) {
      if (it->second.seen) {
        ++it;
        continue;
      }
      columnSlots_.release(it->second.columnSlot);
      plan.retiredColumns.push_back(it->second.columnSlot);
      it = columns_.erase(it);
    }

    for (const size_t i : added) {
      ColumnEntry entry;
      entry.columnSlot = columnSlots_.allocate();
      entry.seen = true;
      if (slotCoords_.size() <= entry.columnSlot) {
        slotCoords_.resize(entry.columnSlot + 1);
        slotNeighborLods_.resize(entry.columnSlot + 1);
      }
      slotCoords_[entry.columnSlot] = chunks[i];
      slotNeighborLods_[entry.columnSlot] = neighborLods[i];
      plan.newColumns.push_back(entry.columnSlot);
      columns_.emplace(keyOf(chunks[i], neighborLods[i]), entry);
    }
    return plan;
  }

  /** Coordinates by column slot; entries at or past columnSlotCount() and of freed slots are stale. */
  const std::vector<world::ChunkLod> &slotCoords() const { return slotCoords_; }
  /** Packed neighbor LODs by column slot; stale like slotCoords(). */
  const std::vector<uint32_t> &slotNeighborLods() const { return slotNeighborLods_; }
  /** One past the highest column slot in use. */
  uint32_t columnSlotCount() const { return columnSlots_.highWater(); }

private:
  struct ColumnKey {
    int32_t x, z, lod;
    uint32_t neighborLods; // border heights depend on the neighbors' LODs, so a change means a new column
    bool operator==(const ColumnKey &) const = default;
  };
  struct ColumnKeyHash {
    size_t operator()(const ColumnKey &key) const {
      uint64_t h = static_cast<uint32_t>(key.x) * 0x9e3779b97f4a7c15ull;
      h ^= (static_cast<uint64_t>(static_cast<uint32_t>(key.z)) << 21) + static_cast<uint32_t>(key.lod);
      h ^= static_cast<uint64_t>(key.neighborLods) * 0xc2b2ae3d27d4eb4full;
      return static_cast<size_t>(h ^ (h >> 29));
    }
  };
  struct ColumnEntry {
    uint32_t columnSlot = 0;
    bool seen = false; // present in the clipmap being applied
  };

  /** Slot-table key of a clipmap column. */
  static ColumnKey keyOf(const world::ChunkLod &chunk, uint32_t neighborLods) {
    return {chunk.x, chunk.z, chunk.lod, neighborLods};
  }

  std::unordered_map<ColumnKey, ColumnEntry, ColumnKeyHash> columns_;
  SlotAllocator columnSlots_;
  std::vector<world::ChunkLod> slotCoords_;
  std::vector<uint32_t> slotNeighborLods_;
};
