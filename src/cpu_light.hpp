#pragma once

// Skylight for CPU chunks, recomputed whenever a chunk is meshed (nothing is stored between meshes).
// Rules (Minecraft's): a non-opaque cell with no opaque cell above it is level 15; light spreads to the 6 neighbors
// through non-opaque cells, losing 1 per step. Computed as bitmask dilation over the chunk plus a 15-cell halo, which
// makes each chunk exact on its own: a seed further than 15 cells away cannot reach it.

#include "cpu_chunk.hpp"

#include <array>
#include <cstdint>

namespace cpuworld {

inline constexpr uint32_t kMaxLight = 15;

// Skylight levels over a chunk's padded 34x34x34 cells (chunk-local -1..32 on each axis).
struct ChunkLight {
  // Every non-opaque padded cell sees open sky: all levels are kMaxLight and `planes` is not written.
  bool allLit = true;
  // Bit-sliced levels: bit k of a cell's level is bit (y + 1) of planes[k][(z + 1) * kPad + (x + 1)].
  std::array<std::array<uint64_t, kPad * kPad>, 4> planes;

  /** Skylight level 0..15 of chunk-local padded cell (x, y, z); opaque cells read 0. */
  uint32_t level(int32_t x, int32_t y, int32_t z) const {
    if (allLit) {
      return kMaxLight;
    }
    const size_t i = static_cast<size_t>((z + 1) * kPad + (x + 1));
    uint32_t result = 0;
    for (uint32_t k = 0; k < 4; ++k) {
      result |= static_cast<uint32_t>((planes[k][i] >> (y + 1)) & 1u) << k;
    }
    return result;
  }
};

// Opaque bits of the padded core the mesher already built: bit y + 1 = chunk-local y (-1..32), index
// (z + 1) * kPad + (x + 1). Reused for the early out, which then costs one mask test per column.
using PaddedOpaque = std::array<uint64_t, kPad * kPad>;

/**
 * Compute skylight for chunk `chunkY` of nb.center(). Unloaded neighbors follow the center's border heights
 * (solid below, open sky above).
 *
 * @param nb Neighborhood of the column being meshed.
 * @param chunkY Chunk y, in chunks.
 * @param opaque The chunk's padded opaque bits (same neighborhood rules).
 */
void computeSkyLight(const Neighborhood &nb, int32_t chunkY, const PaddedOpaque &opaque, ChunkLight &out);

/**
 * Skylight level of one cell of nb.center() (like F3's client light), computing its chunk's light from scratch.
 *
 * @param lx, lz Column-local block coords, 0..31.
 * @param y World block y.
 * @return level 0..15 (0 for an opaque cell).
 */
uint32_t skyLightAt(const Neighborhood &nb, int32_t lx, int32_t y, int32_t lz);

} // namespace cpuworld
