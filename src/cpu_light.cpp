#include "cpu_light.hpp"

#include <algorithm>

namespace cpuworld {
namespace {

constexpr int32_t N = CHUNK_SIZE;
constexpr int32_t kHalo = static_cast<int32_t>(kMaxLight);
// Region width per axis: the chunk plus a full halo on both sides (62 cells, fits a 64-bit column).
constexpr int32_t W = N + 2 * kHalo;
constexpr uint64_t kRegionBits = (uint64_t{1} << W) - 1u;
// Window bit b of a region column is chunk-local y = b - kHalo.
// The padded core (chunk-local -1..32) starts at window bit kCoreLow.
constexpr int32_t kCoreLow = kHalo - 1;

/** Window bits at or above `bit`, clamped to the region. */
uint64_t bitsFrom(int32_t bit) {
  if (bit <= 0) {
    return kRegionBits;
  }
  return bit >= W ? 0u : kRegionBits & (~uint64_t{0} << bit);
}

/**
 * Opaque window bits of one region column from the three chunks it overlaps.
 *
 * @param below Opaque bits of chunk chunkY - 1 (its top kHalo cells are used).
 * @param self Opaque bits of chunk chunkY.
 * @param above Opaque bits of chunk chunkY + 1 (its bottom kHalo cells are used).
 */
uint64_t windowBits(uint32_t below, uint32_t self, uint32_t above) {
  constexpr uint32_t aboveMask = (1u << (W - kHalo - N)) - 1u;
  return (static_cast<uint64_t>(below) >> (N - kHalo)) | (static_cast<uint64_t>(self) << kHalo) |
         (static_cast<uint64_t>(above & aboveMask) << (kHalo + N));
}

// Region columns, index (pz + kHalo) * W + (px + kHalo) for chunk-local px, pz in -kHalo .. N + kHalo - 1.
struct Region {
  std::array<uint64_t, W * W> transparent;
  std::array<uint64_t, W * W> seed;
};

/**
 * Fill one region column's transparent and seed (open sky) bits.
 *
 * @param px, pz Chunk-local column coords, -kHalo .. N + kHalo - 1 (may reach into neighbor columns).
 */
void loadColumn(const Neighborhood &nb, int32_t chunkY, int32_t px, int32_t pz, Region &region) {
  const int32_t dx = floorDiv(px, N); // dx, dz: which neighbor column, -1..1
  const int32_t dz = floorDiv(pz, N);
  const Column *column = nb.columns[dz + 1][dx + 1];
  uint64_t opaque;
  int32_t skyTop; // world y
  if (column != nullptr) {
    const int32_t lx = px - dx * N; // lx, lz: column-local coords in that neighbor
    const int32_t lz = pz - dz * N;
    opaque = windowBits(column->solid32(lx, lz, chunkY - 1), column->solid32(lx, lz, chunkY),
                        column->solid32(lx, lz, chunkY + 1));
    skyTop = column->skyTop[lz * N + lx];
  } else {
    // Unloaded: the nearest border height of the center column, as a plain heightmap.
    const int32_t height = nb.center().borderHeight(std::clamp(px, -1, N), std::clamp(pz, -1, N));
    opaque = windowBits(Column::heightMask(height, chunkY - 1), Column::heightMask(height, chunkY),
                        Column::heightMask(height, chunkY + 1));
    skyTop = height;
  }
  const size_t i = static_cast<size_t>((pz + kHalo) * W + (px + kHalo));
  region.transparent[i] = ~opaque & kRegionBits;
  region.seed[i] = region.transparent[i] & bitsFrom(skyTop - chunkY * N + kHalo);
}

} // namespace

/**
 * skyTop (world y) of chunk-local column (px, pz), reaching into neighbor columns; the center's nearest border height
 * where the neighbor isn't loaded.
 */
int32_t skyTopAt(const Neighborhood &nb, int32_t px, int32_t pz) {
  const int32_t dx = floorDiv(px, N);
  const int32_t dz = floorDiv(pz, N);
  if (const Column *column = nb.columns[dz + 1][dx + 1]) {
    return column->skyTop[(pz - dz * N) * N + (px - dx * N)];
  }
  return nb.center().borderHeight(std::clamp(px, -1, N), std::clamp(pz, -1, N));
}

void computeSkyLight(const Neighborhood &nb, int32_t chunkY, const PaddedOpaque &opaque, ChunkLight &out) {
  // Early out: if no non-opaque padded core cell lies below its column's skyTop, every face reads level 15. First per
  // column from the shadow ranges (unloaded neighbors are heightmaps: never shadowed), then exactly per cell.
  bool nearShadow = false;
  for (const auto &row : nb.columns) {
    for (const Column *column : row) {
      nearShadow = nearShadow || (column != nullptr && column->shadowLow < chunkY * N + N + 1 &&
                                  column->shadowHigh > chunkY * N - 1);
    }
  }
  if (!nearShadow) {
    out.allLit = true;
    return;
  }
  constexpr uint64_t kPadBits = (uint64_t{1} << kPad) - 1u;
  bool shadowed = false;
  const Column &center = nb.center();
  for (int32_t pz = -1; pz <= N && !shadowed; ++pz) {
    for (int32_t px = -1; px <= N; ++px) {
      const bool interior = px >= 0 && px < N && pz >= 0 && pz < N;
      const int32_t skyTop = interior ? center.skyTop[pz * N + px] : skyTopAt(nb, px, pz);
      const int32_t skyBit = skyTop - chunkY * N + 1; // padded bit of the first sky cell
      if (skyBit <= 0) {
        continue;
      }
      const uint64_t belowSky = skyBit >= kPad ? kPadBits : (uint64_t{1} << skyBit) - 1u;
      if ((~opaque[static_cast<size_t>((pz + 1) * kPad + (px + 1))] & belowSky) != 0u) {
        shadowed = true;
        break;
      }
    }
  }
  out.allLit = !shadowed;
  if (!shadowed) {
    return;
  }

  thread_local Region region;
  for (int32_t pz = -kHalo; pz < N + kHalo; ++pz) {
    for (int32_t px = -kHalo; px < N + kHalo; ++px) {
      loadColumn(nb, chunkY, px, pz, region);
    }
  }

  // reached[d] = cells within d steps of a seed through transparent cells. A cell's level is the number of
  // d in 0..14 it is reached at (15 - distance), accumulated in bit-sliced counters over the padded core.
  // Step d only needs cells within 14 - d of the padded core, so the dilated area shrinks every step.
  thread_local std::array<uint64_t, W * W> reached;
  thread_local std::array<uint64_t, W * W> next;
  thread_local std::array<std::array<uint64_t, kPad * kPad>, 4> counter;
  reached = region.seed;
  for (auto &plane : counter) {
    plane.fill(0u);
  }
  for (uint32_t step = 0; step < kMaxLight; ++step) {
    for (int32_t pz = -1; pz <= N; ++pz) {
      for (int32_t px = -1; px <= N; ++px) {
        uint64_t carry = reached[static_cast<size_t>((pz + kHalo) * W + (px + kHalo))];
        std::array<uint64_t, kPad * kPad> *plane = counter.data();
        const size_t c = static_cast<size_t>((pz + 1) * kPad + (px + 1));
        for (int k = 0; k < 4 && carry != 0u; ++k) {
          const uint64_t overflow = plane[k][c] & carry;
          plane[k][c] ^= carry;
          carry = overflow;
        }
      }
    }
    if (step + 1 == kMaxLight) {
      break;
    }
    // Cells needed after this step: within (14 - step - 1) of the padded core.
    const int32_t margin = static_cast<int32_t>(kMaxLight) - 2 - static_cast<int32_t>(step);
    const int32_t low = -1 - margin;
    const int32_t high = N + margin;
    for (int32_t pz = low; pz <= high; ++pz) {
      for (int32_t px = low; px <= high; ++px) {
        const size_t i = static_cast<size_t>((pz + kHalo) * W + (px + kHalo));
        const uint64_t self = reached[i];
        const uint64_t grown = self | (self << 1u) | (self >> 1u) | reached[i - 1] | reached[i + 1] |
                               reached[i - W] | reached[i + W];
        next[i] = grown & region.transparent[i];
      }
    }
    for (int32_t pz = low; pz <= high; ++pz) {
      const size_t row = static_cast<size_t>((pz + kHalo) * W + (low + kHalo));
      std::copy_n(next.begin() + static_cast<std::ptrdiff_t>(row), high - low + 1,
                  reached.begin() + static_cast<std::ptrdiff_t>(row));
    }
  }

  // Window bit kCoreLow (chunk-local y = -1) becomes bit 0, i.e. bit y + 1.
  for (uint32_t k = 0; k < 4; ++k) {
    for (size_t c = 0; c < static_cast<size_t>(kPad * kPad); ++c) {
      out.planes[k][c] = (counter[k][c] >> kCoreLow) & ((uint64_t{1} << kPad) - 1u);
    }
  }
}

uint32_t skyLightAt(const Neighborhood &nb, int32_t lx, int32_t y, int32_t lz) {
  const int32_t chunkY = floorDiv(y, N);
  thread_local PaddedOpaque opaque;
  for (int32_t pz = -1; pz <= N; ++pz) {
    for (int32_t px = -1; px <= N; ++px) {
      const int32_t dx = floorDiv(px, N);
      const int32_t dz = floorDiv(pz, N);
      const Column *column = nb.columns[dz + 1][dx + 1];
      uint32_t below, self, above;
      if (column != nullptr) {
        const int32_t cellX = px - dx * N;
        const int32_t cellZ = pz - dz * N;
        below = column->solid32(cellX, cellZ, chunkY - 1);
        self = column->solid32(cellX, cellZ, chunkY);
        above = column->solid32(cellX, cellZ, chunkY + 1);
      } else {
        const int32_t height = nb.center().borderHeight(std::clamp(px, -1, N), std::clamp(pz, -1, N));
        below = Column::heightMask(height, chunkY - 1);
        self = Column::heightMask(height, chunkY);
        above = Column::heightMask(height, chunkY + 1);
      }
      // Bit y + 1 = chunk-local y, as the mesher's buildMasks.
      opaque[static_cast<size_t>((pz + 1) * kPad + (px + 1))] = static_cast<uint64_t>(below >> 31u) |
                                                               (static_cast<uint64_t>(self) << 1u) |
                                                               (static_cast<uint64_t>(above & 1u) << 33u);
    }
  }
  thread_local ChunkLight light;
  computeSkyLight(nb, chunkY, opaque, light);
  return light.level(lx, y - chunkY * N, lz);
}

} // namespace cpuworld
