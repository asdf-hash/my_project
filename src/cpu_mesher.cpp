#include "cpu_mesher.hpp"

#include "cpu_light.hpp"
#include "simd.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace cpuworld {
namespace {

using simd::I8;

constexpr int32_t N = CHUNK_SIZE; // chunk edge, in blocks
constexpr int32_t TILE = 8;      // cluster edge, in columns

struct Int3 {
  int32_t x, y, z;
};
// Component-wise offset arithmetic.
Int3 operator+(Int3 a, Int3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Int3 operator-(Int3 a, Int3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

// Per normal index (+x, -x, +y, -y, +z, -z): the outside cell and calculateAo's u / v axes (chunk_compute.slang).
constexpr Int3 kNormals[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
constexpr Int3 kU[6] = {{0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
constexpr Int3 kV[6] = {{0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
// Cluster run order: +Y, -X, +X, -Z, +Z, -Y.
constexpr uint32_t kRunNormals[6] = {2, 1, 0, 5, 4, 3};

/**
 * Pack one LOD 0 quad in chunk_compute.slang writeQuad's layout.
 * Low word: x | y << 5 | z << 10 (5 bits each) | normal << 15 (3) | ao << 18 (8) | (length - 1) << 26 (5).
 * High word: block id (10) | corner skylight << 10 (16: 4 bits per corner, AO's corner order) | (width - 1) << 26 (3).
 *
 * @param x Chunk-local block x.
 * @param y Chunk-local block y.
 * @param z Chunk-local block z.
 * @param normalIndex Face direction, 0..5 = +x, -x, +y, -y, +z, -z.
 * @param ao Four 2-bit corner AO values (corner k in bits 2k..2k+1).
 * @param blockType Block id (10 bits).
 * @param length Quad extent along its first axis, in blocks (1..32).
 * @param width Quad extent along its second axis, in blocks (1..8).
 * @param cornerLight Smooth skylight, levels 0..15 per corner (corner k in bits 4k..4k+3).
 * @return the packed 64-bit quad.
 */
uint64_t packQuad(uint32_t x, uint32_t y, uint32_t z, uint32_t normalIndex, uint32_t ao, uint32_t blockType,
                  uint32_t length, uint32_t width, uint32_t cornerLight) {
  uint32_t packed = (x & 0x1fu) | ((y & 0x1fu) << 5u) | ((z & 0x1fu) << 10u) | ((normalIndex & 0x7u) << 15u);
  packed |= ao << 18u;
  packed |= ((length - 1u) & 0x1fu) << 26u;
  const uint32_t metadata = blockType | ((cornerLight & 0xffffu) << 10u) | (((width - 1u) & 0x7u) << 26u); // LOD 0
  return (static_cast<uint64_t>(metadata) << 32u) | packed;
}

/**
 * Pack a LOD 0 chunk's world origin: block x in bits 0..23, block z in 24..47, chunk y in 48..60.
 *
 * @param cx Chunk column x, in chunks.
 * @param chunkY Chunk y, in chunks.
 * @param cz Chunk column z, in chunks.
 * @return the packed offset.
 */
uint64_t packChunkOffset(int32_t cx, int32_t chunkY, int32_t cz) {
  const uint64_t offsetX = static_cast<uint32_t>(cx * CHUNK_SIZE) & 0x00ffffffu;
  const uint64_t offsetZ = static_cast<uint64_t>(static_cast<uint32_t>(cz * CHUNK_SIZE) & 0x00ffffffu) << 24u;
  const uint64_t offsetY = static_cast<uint64_t>(static_cast<uint32_t>(chunkY) & 0x1fffu) << 48u;
  return offsetX | offsetZ | offsetY; // LOD 0
}

/** Whether packed AO matches across the u axis (corners 0 == 1, 3 == 2), so the quad may grow along u. */
bool aoConstantAlongU(uint32_t ao) { return (ao & 3u) == ((ao >> 2u) & 3u) && ((ao >> 6u) & 3u) == ((ao >> 4u) & 3u); }
/** Whether packed AO matches across the v axis (corners 0 == 3, 1 == 2), so the quad may grow along v. */
bool aoConstantAlongV(uint32_t ao) { return (ao & 3u) == ((ao >> 6u) & 3u) && ((ao >> 2u) & 3u) == ((ao >> 4u) & 3u); }
/** Corner k's 4-bit light level from packed corner light. */
constexpr uint32_t cornerLevel(uint32_t light, uint32_t k) { return (light >> (4u * k)) & 0xfu; }
/** Whether packed corner light matches across the u axis (corners 0 == 1, 3 == 2). */
bool lightConstantAlongU(uint32_t light) {
  return cornerLevel(light, 0) == cornerLevel(light, 1) && cornerLevel(light, 3) == cornerLevel(light, 2);
}
/** Whether packed corner light matches across the v axis (corners 0 == 3, 1 == 2). */
bool lightConstantAlongV(uint32_t light) {
  return cornerLevel(light, 0) == cornerLevel(light, 3) && cornerLevel(light, 1) == cornerLevel(light, 2);
}

// Column bitmasks over the padded 34x34 area, pre-shifted: shifted[k][i] = uint32(M >> k), where M has bit (y + 1)
// set for a solid cell at chunk-local y in -1..32. Bit y of shifted[1 + dy] is then the cell at y + dy.
struct PaddedMasks {
  std::array<std::array<uint32_t, kPad * kPad>, 3> shifted;

  /**
   * Load 8 consecutive x columns, offset so bit y of each lane is the cell at chunk-local (x, y, z) + off.
   *
   * @param off Neighbor offset; each component in -1..1.
   */
  I8 load(Int3 off, int32_t x, int32_t z) const {
    return I8::load(reinterpret_cast<const int32_t *>(&shifted[1 + off.y][(z + off.z + 1) * kPad + (x + off.x + 1)]));
  }
};

/**
 * Build PaddedMasks of ocean cells (Column::ocean32; border heights where a neighbor isn't loaded).
 *
 * @param nb Neighborhood of the column being meshed.
 * @param chunkY Chunk y, in chunks.
 */
void buildOceanMasks(const Neighborhood &nb, int32_t chunkY, PaddedMasks &masks) {
  const Column &center = nb.center();
  for (int32_t pz = -1; pz <= N; ++pz) { // px/pz: chunk-local coords over the padded area
    const int32_t dz = pz < 0 ? -1 : (pz >= N ? 1 : 0); // dx/dz: which neighbor column, -1..1
    for (int32_t px = -1; px <= N; ++px) {
      const int32_t dx = px < 0 ? -1 : (px >= N ? 1 : 0);
      const Column *column = nb.columns[dz + 1][dx + 1];
      uint32_t below, self, above;
      if (column != nullptr) {
        const int32_t lx = px - dx * N;
        const int32_t lz = pz - dz * N;
        below = column->ocean32(lx, lz, chunkY - 1);
        self = column->ocean32(lx, lz, chunkY);
        above = column->ocean32(lx, lz, chunkY + 1);
      } else if (center.oceanId != kAir) {
        const int32_t h = center.borderHeight(px, pz);
        below = Column::oceanMask(h, chunkY - 1);
        self = Column::oceanMask(h, chunkY);
        above = Column::oceanMask(h, chunkY + 1);
      } else {
        below = self = above = 0u;
      }
      // y = -1..32 at bits 0..33: top cell of the chunk below, this chunk, bottom cell of the chunk above.
      const uint64_t padded = static_cast<uint64_t>(below >> 31u) | (static_cast<uint64_t>(self) << 1u) |
                              (static_cast<uint64_t>(above & 1u) << 33u);
      const int32_t i = (pz + 1) * kPad + (px + 1);
      masks.shifted[0][i] = static_cast<uint32_t>(padded);
      masks.shifted[1][i] = static_cast<uint32_t>(padded >> 1u);
      masks.shifted[2][i] = static_cast<uint32_t>(padded >> 2u);
    }
  }
}

/**
 * Build PaddedMasks of opaque cells (Column::solid32; border heights where a neighbor isn't loaded).
 *
 * @param nb Neighborhood of the column being meshed.
 * @param chunkY Chunk y, in chunks. * @param opaque Receives the same bits unshifted, for computeSkyLight.
 */
void buildMasks(const Neighborhood &nb, int32_t chunkY, PaddedMasks &masks, PaddedOpaque &opaque) {
  const Column &center = nb.center();
  for (int32_t pz = -1; pz <= N; ++pz) { // px/pz: chunk-local coords over the padded area
    const int32_t dz = pz < 0 ? -1 : (pz >= N ? 1 : 0); // dx/dz: which neighbor column, -1..1
    for (int32_t px = -1; px <= N; ++px) {
      const int32_t dx = px < 0 ? -1 : (px >= N ? 1 : 0);
      const Column *column = nb.columns[dz + 1][dx + 1];
      uint32_t below, self, above;
      if (column != nullptr) {
        const int32_t lx = px - dx * N;
        const int32_t lz = pz - dz * N;
        below = column->solid32(lx, lz, chunkY - 1);
        self = column->solid32(lx, lz, chunkY);
        above = column->solid32(lx, lz, chunkY + 1);
      } else {
        const int32_t h = center.borderHeight(px, pz);
        below = Column::heightMask(h, chunkY - 1);
        self = Column::heightMask(h, chunkY);
        above = Column::heightMask(h, chunkY + 1);
      }
      // y = -1..32 at bits 0..33: top cell of the chunk below, this chunk, bottom cell of the chunk above.
      const uint64_t padded = static_cast<uint64_t>(below >> 31u) | (static_cast<uint64_t>(self) << 1u) |
                              (static_cast<uint64_t>(above & 1u) << 33u);
      const int32_t i = (pz + 1) * kPad + (px + 1);
      masks.shifted[0][i] = static_cast<uint32_t>(padded);
      masks.shifted[1][i] = static_cast<uint32_t>(padded >> 1u);
      masks.shifted[2][i] = static_cast<uint32_t>(padded >> 2u);
      opaque[static_cast<size_t>(i)] = padded;
    }
  }
}

// Face bits and AO bitplanes of one tile in one direction: for column c (z * 8 + x within the tile), bit y of
// face[c] marks a face at cell y; corner k's AO value at that face is aoLo[k] bit y | aoHi[k] bit y << 1.
struct TilePlanes {
  std::array<uint32_t, 64> face;
  std::array<std::array<uint32_t, 64>, 4> aoLo;
  std::array<std::array<uint32_t, 64>, 4> aoHi;
  // Light context of the tile and direction last computed (computePlanes / computeOceanPlanes).
  const ChunkLight *light = nullptr;
  const PaddedMasks *opaque = nullptr;
  uint32_t normal = 0;
  int32_t tileX = 0;
  int32_t tileZ = 0;

  /** Packed corner skylight of the face at cell y of column c (cornerLight). */
  uint32_t lightAt(int32_t c, int32_t y) const;

  /**
   * Gather the packed AO of the face at cell y of column c.
   *
   * @return four 2-bit corner values, corner k in bits 2k..2k+1 (packQuad's ao).
   */
  uint32_t ao(int32_t c, int32_t y) const {
    uint32_t packed = 0;
    for (int k = 0; k < 4; ++k) {
      packed |= (((aoLo[k][c] >> y) & 1u) | (((aoHi[k][c] >> y) & 1u) << 1u)) << (2 * k);
    }
    return packed;
  }
};

/**
 * Bit-sliced vertexAo(a, b, c) = a && b ? 3 : a + b + c over 32 cells per lane.
 *
 * @param a Side neighbor solidity along u.
 * @param b Side neighbor solidity along v.
 * @param c Diagonal corner neighbor solidity.
 * @param lo Receives bit 0 of the AO value.
 * @param hi Receives bit 1 of the AO value.
 */
void vertexAoPlanes(I8 a, I8 b, I8 c, I8 &lo, I8 &hi) {
  const I8 ab = a & b;
  lo = ab | (a ^ b ^ c);
  hi = ab | (c & (a | b));
}

/**
 * Compute face bits and AO bitplanes of opaque cells for one tile and one direction.
 *
 * @param normal Face direction, 0..5 = +x, -x, +y, -y, +z, -z.
 * @param tileX Tile x within the chunk, 0..3.
 * @param tileZ Tile z within the chunk, 0..3.
 */
void computePlanes(const PaddedMasks &masks, const ChunkLight &light, uint32_t normal, int32_t tileX, int32_t tileZ,
                   TilePlanes &planes) {
  planes.light = &light;
  planes.opaque = &masks;
  planes.normal = normal;
  planes.tileX = tileX;
  planes.tileZ = tileZ;
  const Int3 faceDir = kNormals[normal];
  const Int3 u = kU[normal];
  const Int3 v = kV[normal];
  const Int3 zero{0, 0, 0};
  for (int32_t row = 0; row < TILE; ++row) {
    const int32_t x = tileX * TILE;
    const int32_t z = tileZ * TILE + row;
    const I8 solid = masks.load(zero, x, z);
    const I8 face = simd::andnot(solid, masks.load(faceDir, x, z));
    // Solidity around the outside cell: u0/u1 and v0/v1 the sides, c0..c3 the diagonal corners.
    const I8 u0 = masks.load(faceDir - u, x, z);
    const I8 u1 = masks.load(faceDir + u, x, z);
    const I8 v0 = masks.load(faceDir - v, x, z);
    const I8 v1 = masks.load(faceDir + v, x, z);
    const I8 c0 = masks.load(faceDir - u - v, x, z);
    const I8 c1 = masks.load(faceDir + u - v, x, z);
    const I8 c2 = masks.load(faceDir + u + v, x, z);
    const I8 c3 = masks.load(faceDir - u + v, x, z);
    I8 lo[4], hi[4];
    vertexAoPlanes(u0, v0, c0, lo[0], hi[0]);
    vertexAoPlanes(u1, v0, c1, lo[1], hi[1]);
    vertexAoPlanes(u1, v1, c2, lo[2], hi[2]);
    vertexAoPlanes(u0, v1, c3, lo[3], hi[3]);
    face.store(reinterpret_cast<int32_t *>(&planes.face[row * TILE]));
    for (int k = 0; k < 4; ++k) {
      lo[k].store(reinterpret_cast<int32_t *>(&planes.aoLo[k][row * TILE]));
      hi[k].store(reinterpret_cast<int32_t *>(&planes.aoHi[k][row * TILE]));
    }
  }
}

/**
 * Compute ocean face bits for one tile and one direction: an ocean cell beside a cell that is neither opaque nor
 * ocean (air, other non-opaque blocks). AO planes are zeroed: water is lit flat.
 *
 * @param normal Face direction, 0..5 = +x, -x, +y, -y, +z, -z.
 * @param tileX Tile x within the chunk, 0..3.
 * @param tileZ Tile z within the chunk, 0..3.
 */
void computeOceanPlanes(const PaddedMasks &solidMasks, const PaddedMasks &oceanMasks, const ChunkLight &light,
                        uint32_t normal, int32_t tileX, int32_t tileZ, TilePlanes &planes) {
  planes.light = &light;
  planes.opaque = &solidMasks;
  planes.normal = normal;
  planes.tileX = tileX;
  planes.tileZ = tileZ;
  const Int3 faceDir = kNormals[normal];
  const Int3 zero{0, 0, 0};
  for (int32_t row = 0; row < TILE; ++row) {
    const int32_t x = tileX * TILE;
    const int32_t z = tileZ * TILE + row;
    const I8 ocean = oceanMasks.load(zero, x, z);
    const I8 covered = solidMasks.load(faceDir, x, z) | oceanMasks.load(faceDir, x, z);
    simd::andnot(ocean, covered).store(reinterpret_cast<int32_t *>(&planes.face[row * TILE]));
  }
  for (int k = 0; k < 4; ++k) {
    planes.aoLo[k].fill(0u);
    planes.aoHi[k].fill(0u);
  }
}

// Vertical extent of a tile's quads, in chunk-local blocks; low > high while empty.
struct TileBounds {
  int32_t low = N;
  int32_t high = 0;
  /** Grow to include cells [lowCell, highCell). */
  void add(int32_t lowCell, int32_t highCell) {
    low = std::min(low, lowCell);
    high = std::max(high, highCell);
  }
};

/**
 * Emit top / bottom faces: per layer, greedy rectangles of identical (block, AO) within the tile, extended along x
 * only when AO is constant along u and along z only when constant along v (the GPU's top-face rules).
 *
 * @param normal 2 (+y) or 3 (-y).
 * @param tileX Tile x within the chunk, 0..3.
 * @param tileZ Tile z within the chunk, 0..3.
 * @param quads Receives the packed quads.
 * @param bounds Grown to cover the emitted faces.
 * @return number of quads emitted.
 */
uint32_t emitHorizontal(const TilePlanes &planes, const VoxelChunk &voxels, uint32_t normal, int32_t tileX,
                        int32_t tileZ, std::vector<uint64_t> &quads, TileBounds &bounds) {
  uint32_t layers = 0;
  for (const uint32_t f : planes.face) {
    layers |= f;
  }
  uint32_t count = 0;
  while (layers != 0u) {
    const int32_t y = std::countr_zero(layers);
    layers &= layers - 1u;
    // Per column: 0 = no face, else corner light << 32 | 1 << 19 | ao << 10 | block id, so equal keys merge.
    std::array<uint64_t, 64> key{};
    for (int32_t c = 0; c < 64; ++c) {
      if ((planes.face[c] >> y) & 1u) {
        const uint32_t ao = planes.ao(c, y);
        const int32_t x = tileX * TILE + (c & 7);
        const int32_t z = tileZ * TILE + (c >> 3);
        key[c] = (static_cast<uint64_t>(planes.lightAt(c, y)) << 32u) | (1u << 19u) | (ao << 10u) |
                 voxels.ids[VoxelChunk::index(x, y, z)];
      }
    }
    std::array<bool, 64> used{};
    for (int32_t row = 0; row < TILE; ++row) {
      for (int32_t col = 0; col < TILE; ++col) {
        const int32_t c = row * TILE + col;
        if (key[c] == 0u || used[c]) {
          continue;
        }
        const uint64_t cellKey = key[c];
        const uint32_t ao = static_cast<uint32_t>(cellKey >> 10u) & 0xffu;
        const uint32_t light = static_cast<uint32_t>(cellKey >> 32u);
        uint32_t length = 1;
        if (aoConstantAlongU(ao) && lightConstantAlongU(light)) {
          while (col + static_cast<int32_t>(length) < TILE && key[c + length] == cellKey && !used[c + length]) {
            ++length;
          }
        }
        uint32_t width = 1;
        if (aoConstantAlongV(ao) && lightConstantAlongV(light)) {
          for (;;) {
            const int32_t nextRow = row + static_cast<int32_t>(width);
            if (nextRow >= TILE) {
              break;
            }
            bool same = true;
            for (uint32_t i = 0; i < length && same; ++i) {
              const int32_t nextCell = nextRow * TILE + col + static_cast<int32_t>(i);
              same = key[nextCell] == cellKey && !used[nextCell];
            }
            if (!same) {
              break;
            }
            ++width;
          }
        }
        for (uint32_t w = 0; w < width; ++w) {
          for (uint32_t i = 0; i < length; ++i) {
            used[c + w * TILE + i] = true;
          }
        }
        const uint32_t x = static_cast<uint32_t>(tileX * TILE + col);
        // +Y faces extend toward -z from the written cell (faceV), -Y faces toward +z: write the far / near row.
        const uint32_t z = static_cast<uint32_t>(tileZ * TILE + row) + (normal == 2u ? width - 1u : 0u);
        quads.push_back(
            packQuad(x, static_cast<uint32_t>(y), z, normal, ao, static_cast<uint32_t>(cellKey) & 0x3ffu, length, width,
                     light));
        ++count;
        bounds.add(y, y + 1);
      }
    }
  }
  return count;
}

// A vertical run of side faces in one column; start / length in chunk-local blocks.
struct SideRun {
  uint32_t start, length, block, ao, light;
  bool operator==(const SideRun &) const = default;
};

/**
 * Emit side faces: vertical runs of equal (block, AO) per column, merged across the tile along the greedy axis
 * (z for x faces, x for z faces) where the neighbor column has the identical run (sideFaceRuns / sideRun on the GPU).
 *
 * @param normal 0, 1, 4 or 5 (+x, -x, +z, -z).
 * @param tileX Tile x within the chunk, 0..3.
 * @param tileZ Tile z within the chunk, 0..3.
 * @param quads Receives the packed quads.
 * @param bounds Grown to cover the emitted faces.
 * @return number of quads emitted.
 */
uint32_t emitSide(const TilePlanes &planes, const VoxelChunk &voxels, uint32_t normal, int32_t tileX, int32_t tileZ,
                  std::vector<uint64_t> &quads, TileBounds &bounds) {
  // Runs per column; at most 16 per column in practice (alternating faces), bounded by 32.
  std::array<std::array<SideRun, 32>, 64> runs;
  std::array<uint8_t, 64> runCount{};
  for (int32_t c = 0; c < 64; ++c) {
    uint32_t bits = planes.face[c];
    if (bits == 0u) {
      continue;
    }
    const int32_t x = tileX * TILE + (c & 7);
    const int32_t z = tileZ * TILE + (c >> 3);
    bool open = false;
    bool openMergeable = false;
    SideRun run{};
    while (bits != 0u) {
      const uint32_t y = static_cast<uint32_t>(std::countr_zero(bits));
      bits &= bits - 1u;
      const uint32_t ao = planes.ao(c, static_cast<int32_t>(y));
      const uint32_t block = voxels.ids[VoxelChunk::index(x, static_cast<int32_t>(y), z)];
      const uint32_t light = planes.lightAt(c, static_cast<int32_t>(y));
      const bool mergeable = aoConstantAlongV(ao) && lightConstantAlongV(light);
      if (open && run.start + run.length == y && openMergeable && mergeable && block == run.block && ao == run.ao &&
          light == run.light) {
        ++run.length;
        continue;
      }
      if (open) {
        runs[c][runCount[c]++] = run;
      }
      run = {y, 1u, block, ao, light};
      open = true;
      openMergeable = mergeable;
    }
    runs[c][runCount[c]++] = run;
  }

  const bool xFace = normal < 2u;
  /** Tile column `step` columns from c along the greedy axis. */
  auto columnAt = [&](int32_t c, int32_t step) { return xFace ? c + step * TILE : c + step; };
  /** Position of column c along the greedy axis, 0..7. */
  auto positionOf = [&](int32_t c) { return xFace ? c >> 3 : c & 7; };
  /** Whether column c has a run identical to `run`. */
  auto hasRun = [&](int32_t c, const SideRun &run) {
    for (uint32_t i = 0; i < runCount[c]; ++i) {
      if (runs[c][i] == run) {
        return true;
      }
    }
    return false;
  };

  uint32_t count = 0;
  for (int32_t c = 0; c < 64; ++c) {
    const int32_t position = positionOf(c);
    for (uint32_t runIndex = 0; runIndex < runCount[c]; ++runIndex) {
      const SideRun &run = runs[c][runIndex];
      const bool mergeable = aoConstantAlongU(run.ao) && lightConstantAlongU(run.light);
      if (mergeable && position > 0 && hasRun(columnAt(c, -1), run)) {
        continue; // covered by the previous column's quad
      }
      uint32_t width = 1;
      if (mergeable) {
        while (position + static_cast<int32_t>(width) < TILE && hasRun(columnAt(c, static_cast<int32_t>(width)), run)) {
          ++width;
        }
      }
      const uint32_t x = static_cast<uint32_t>(tileX * TILE + (c & 7));
      const uint32_t z = static_cast<uint32_t>(tileZ * TILE + (c >> 3));
      quads.push_back(packQuad(x, run.start, z, normal, run.ao, run.block, run.length, width, run.light));
      ++count;
      bounds.add(static_cast<int32_t>(run.start), static_cast<int32_t>(run.start + run.length));
    }
  }
  return count;
}

/**
 * Read one opaque bit from the padded masks.
 *
 * @return whether chunk-local cell (x, y, z) is opaque; x, y, z each in -1..32.
 */
bool opaqueAt(const PaddedMasks &masks, int32_t x, int32_t y, int32_t z) {
  const int32_t i = (z + 1) * kPad + (x + 1);
  // shifted[0] has the cell at y in bit y + 1, shifted[2] in bit y - 1 (reaches y = 32).
  return y <= 30 ? (masks.shifted[0][i] >> (y + 1)) & 1u : (masks.shifted[2][i] >> (y - 1)) & 1u;
}

/**
 * Smooth skylight of one face, vanilla's blend: per corner, the average of the cell the face looks into (front) and
 * its neighbors toward that corner (two sides, and the diagonal unless both sides are opaque, else the first side
 * again), where any sample reading 0 (opaque or dark) counts as the front's level. Rounded to whole levels.
 *
 * @param front Chunk-local cell the face looks into.
 * @param normal Face direction, 0..5 = +x, -x, +y, -y, +z, -z.
 * @return 4 bits per corner, corner k = (-u,-v), (+u,-v), (+u,+v), (-u,+v) in bits 4k..4k+3 (AO's order).
 */
uint32_t cornerLight(const ChunkLight &light, const PaddedMasks &opaque, Int3 front, uint32_t normal) {
  if (light.allLit) {
    return 0xffffu;
  }
  const Int3 u = kU[normal];
  const Int3 v = kV[normal];
  /** Whether the cell at offset (du * u + dv * v) from front is opaque. */
  const auto isOpaque = [&](int32_t du, int32_t dv) {
    return opaqueAt(opaque, front.x + du * u.x + dv * v.x, front.y + du * u.y + dv * v.y,
                    front.z + du * u.z + dv * v.z);
  };
  /** Level of the cell at offset (du * u + dv * v) from front (0 when opaque). */
  const auto level = [&](int32_t du, int32_t dv) {
    return isOpaque(du, dv) ? 0
                            : static_cast<int32_t>(light.level(front.x + du * u.x + dv * v.x,
                                                               front.y + du * u.y + dv * v.y,
                                                               front.z + du * u.z + dv * v.z));
  };
  static constexpr int32_t kCornerU[4] = {-1, 1, 1, -1};
  static constexpr int32_t kCornerV[4] = {-1, -1, 1, 1};
  const int32_t center = level(0, 0);
  /** A sample, with 0 replaced by the front's level. */
  const auto orCenter = [&](int32_t sample) { return sample == 0 ? center : sample; };
  uint32_t packed = 0;
  for (uint32_t k = 0; k < 4; ++k) {
    const bool sideUOpaque = isOpaque(kCornerU[k], 0);
    const bool sideVOpaque = isOpaque(0, kCornerV[k]);
    const int32_t sideU = level(kCornerU[k], 0);
    const int32_t sideV = level(0, kCornerV[k]);
    const int32_t diagonal = sideUOpaque && sideVOpaque ? sideU : level(kCornerU[k], kCornerV[k]);
    const int32_t sum = center + orCenter(sideU) + orCenter(sideV) + orCenter(diagonal);
    packed |= static_cast<uint32_t>((sum + 2) / 4) << (4u * k);
  }
  return packed;
}

uint32_t TilePlanes::lightAt(int32_t c, int32_t y) const {
  const Int3 dir = kNormals[normal];
  const Int3 front{tileX * TILE + (c & 7) + dir.x, y + dir.y, tileZ * TILE + (c >> 3) + dir.z};
  return cornerLight(*light, *opaque, front, normal);
}

/**
 * Look up the block id of chunk-local cell (x, y, z), reaching into neighbor columns.
 *
 * @param nb Neighborhood of the column being meshed.
 * @param chunkY Chunk y, in chunks.
 * @param x Chunk-local x, -1..32.
 * @param z Chunk-local z, -1..32.
 * @return the id; from border heights (kBuriedTerrain / empty) where the neighbor isn't loaded.
 */
uint16_t neighborId(const Neighborhood &nb, int32_t chunkY, int32_t x, int32_t y, int32_t z) {
  const int32_t dx = x < 0 ? -1 : (x >= N ? 1 : 0);
  const int32_t dz = z < 0 ? -1 : (z >= N ? 1 : 0);
  const int32_t worldY = chunkY * N + y;
  if (const Column *column = nb.columns[dz + 1][dx + 1]) {
    return column->blockId(x - dx * N, worldY, z - dz * N);
  }
  return worldY < nb.center().borderHeight(x, z) ? kBuriedTerrain : nb.center().emptyCell(worldY);
}

/** Corner AO 0..3: 3 when both sides are solid, else the count of solid neighbors. */
uint32_t vertexAo(bool a, bool b, bool corner) { return a && b ? 3u : uint32_t(a) + uint32_t(b) + uint32_t(corner); }

/**
 * Emit faces of non-opaque blocks (id bit 9 set), which the binary masks leave out, in a per-cell pass. Full cubes
 * for now: a face shows unless the neighbor is opaque or the same block, as a 1x1 quad with AO from opaque neighbors.
 *
 * @param nb Neighborhood of the column being meshed.
 * @param nonOpaque Per column (z * N + x), bit y set for a non-opaque cell.
 * @param chunkY Chunk y, in chunks.
 * @param normal Face direction, 0..5 = +x, -x, +y, -y, +z, -z.
 * @param tileX Tile x within the chunk, 0..3.
 * @param tileZ Tile z within the chunk, 0..3.
 * @param quads Receives the packed quads.
 * @param bounds Grown to cover the emitted faces.
 * @return number of quads emitted.
 */
uint32_t emitNonOpaque(const PaddedMasks &masks, const ChunkLight &light, const Neighborhood &nb, const VoxelChunk &voxels,
                       const std::array<uint32_t, N * N> &nonOpaque, int32_t chunkY, uint32_t normal, int32_t tileX,
                       int32_t tileZ, std::vector<uint64_t> &quads, TileBounds &bounds) {
  const Int3 faceDir = kNormals[normal];
  const Int3 u = kU[normal];
  const Int3 v = kV[normal];
  uint32_t count = 0;
  for (int32_t row = 0; row < TILE; ++row) {
    for (int32_t col = 0; col < TILE; ++col) {
      const int32_t x = tileX * TILE + col;
      const int32_t z = tileZ * TILE + row;
      uint32_t bits = nonOpaque[z * N + x];
      while (bits != 0u) {
        const int32_t y = std::countr_zero(bits);
        bits &= bits - 1u;
        const Int3 outsideCell{x + faceDir.x, y + faceDir.y, z + faceDir.z};
        if (opaqueAt(masks, outsideCell.x, outsideCell.y, outsideCell.z)) {
          continue;
        }
        const uint16_t id = voxels.ids[VoxelChunk::index(x, y, z)];
        if (neighborId(nb, chunkY, outsideCell.x, outsideCell.y, outsideCell.z) == id) {
          continue;
        }
        /** Whether the cell at offset d from the outside cell is opaque. */
        const auto at = [&](Int3 d) {
          return opaqueAt(masks, outsideCell.x + d.x, outsideCell.y + d.y, outsideCell.z + d.z);
        };
        const Int3 zero{0, 0, 0};
        const bool u0 = at(zero - u), u1 = at(u), v0 = at(zero - v), v1 = at(v);
        const uint32_t ao = vertexAo(u0, v0, at(zero - u - v)) | (vertexAo(u1, v0, at(u - v)) << 2u) |
                            (vertexAo(u1, v1, at(u + v)) << 4u) | (vertexAo(u0, v1, at(v - u)) << 6u);
        quads.push_back(packQuad(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z), normal,
                                 ao, id & 0x3ffu, 1u, 1u, cornerLight(light, masks, outsideCell, normal)));
        ++count;
        bounds.add(y, y + 1);
      }
    }
  }
  return count;
}

// ---- Smooth terrain (Surface Nets) ----------------------------------------------------------------------------
// Every opaque face becomes one quad whose corners are block-grid lattice points moved onto a smooth surface: a
// corner (shared by 8 cells) goes to the average of the points where the density field crosses zero along the edges
// between those cells' centers. Cells that follow the heightmap take their density from the column's float surface
// height, so unedited terrain follows the noise exactly (no terraces); cells that don't (edits, trees) are a plain
// +-1. Non-opaque blocks (leaves, glass) form a second, binary surface drawn in the cutout pass. Everything is
// computed per cell, so chunks sharing a corner place it identically.

// Density magnitude cap: keeps the heightmap's linear crossings on gentle slopes while edits and cliffs (larger
// jumps) cross halfway, like binary cells.
constexpr float kMaxDensity = 1.0f;
// Smallest density magnitude, so a cell's density sign always matches its opacity.
constexpr float kMinDensity = 1.0f / 64.0f;

// Per padded column (index (z + 1) * kPad + (x + 1)): the float surface height and the column top the voxels follow
// where unedited (Column::height, or the border height where the neighbor isn't loaded).
struct HeightPads {
  std::array<float, kPad * kPad> raw;
  std::array<int32_t, kPad * kPad> top;
};

/** Fill HeightPads from the owning columns (the center's padded border where a neighbor isn't loaded). */
void buildHeightPads(const Neighborhood &nb, HeightPads &pads) {
  const Column &center = nb.center();
  for (int32_t pz = -1; pz <= N; ++pz) {
    const int32_t dz = pz < 0 ? -1 : (pz >= N ? 1 : 0);
    for (int32_t px = -1; px <= N; ++px) {
      const int32_t dx = px < 0 ? -1 : (px >= N ? 1 : 0);
      const size_t i = static_cast<size_t>((pz + 1) * kPad + (px + 1));
      if (const Column *column = nb.columns[dz + 1][dx + 1]) {
        const int32_t lx = px - dx * N;
        const int32_t lz = pz - dz * N;
        pads.raw[i] = column->rawHeights[static_cast<size_t>(Column::padIndex(lx, lz))];
        pads.top[i] = column->height(lx, lz);
      } else {
        pads.raw[i] = center.rawHeights[static_cast<size_t>(Column::padIndex(px, pz))];
        pads.top[i] = center.borderHeight(px, pz);
      }
    }
  }
}

/**
 * Density of padded cell `i` (column index as HeightPads) at chunk-local y (-1..32): positive inside opaque cells.
 * Where the cell follows its column's heightmap, the surface sits at raw - 0.5 (cell centers are at y + 0.5), so
 * crossings interpolate the float height; otherwise +-kMaxDensity.
 *
 * @param baseY World y of the chunk's cell 0.
 */
float cellDensity(const PaddedOpaque &opaque, const HeightPads &pads, size_t i, int32_t y, int32_t baseY) {
  const bool solid = ((opaque[i] >> (y + 1)) & 1u) != 0u;
  const int32_t worldY = baseY + y;
  if (solid != (worldY < pads.top[i])) {
    return solid ? kMaxDensity : -kMaxDensity;
  }
  const float d = std::clamp(pads.raw[i] - 1.0f - static_cast<float>(worldY), -kMaxDensity, kMaxDensity);
  return solid ? std::max(d, kMinDensity) : std::min(d, -kMinDensity);
}

/** Signed byte of a unit-range component, in the low 8 bits. */
uint32_t snormByte(float value) {
  const int32_t scaled = static_cast<int32_t>(std::lround(std::clamp(value, -1.0f, 1.0f) * 127.0f));
  return static_cast<uint32_t>(scaled) & 0xffu;
}

/** Packed signed-byte normal (x | y << 8 | z << 16) of -gradient, or `fallback` when the gradient vanishes. */
uint32_t packNormal(float gx, float gy, float gz, uint32_t fallback) {
  const float length = std::sqrt(gx * gx + gy * gy + gz * gz);
  if (!(length > 1e-6f)) {
    return fallback;
  }
  return snormByte(-gx / length) | (snormByte(-gy / length) << 8u) | (snormByte(-gz / length) << 16u);
}

// A placed corner: offset from its lattice point in 1/kSmoothFixedOne blocks, and its packed normal.
struct SmoothCorner {
  Int3 offset;
  uint32_t normal;
};

/** Fixed-point offset of a position relative to its lattice point (-0.5 .. 0.5 blocks per axis). */
Int3 toFixedOffset(float x, float y, float z) {
  const auto fixed = [](float v) {
    return std::clamp(static_cast<int32_t>(std::lround(v * kSmoothFixedOne)), -kSmoothFixedOne / 2,
                      kSmoothFixedOne / 2);
  };
  return {fixed(x), fixed(y), fixed(z)};
}

/**
 * Place the opaque surface's corner at chunk-local lattice point (lx, ly, lz) (each 0..32) from the densities of its
 * 8 cells.
 *
 * @param faceNormal Packed normal used when the densities give no gradient.
 */
SmoothCorner opaqueCorner(const PaddedOpaque &opaque, const HeightPads &pads, int32_t baseY, int32_t lx, int32_t ly,
                          int32_t lz, uint32_t faceNormal) {
  float d[2][2][2]; // [i][j][k]: cell (lx - 1 + i, ly - 1 + j, lz - 1 + k)
  for (int32_t k = 0; k < 2; ++k) {
    for (int32_t i = 0; i < 2; ++i) {
      const size_t column = static_cast<size_t>((lz + k) * kPad + (lx + i));
      d[i][0][k] = cellDensity(opaque, pads, column, ly - 1, baseY);
      d[i][1][k] = cellDensity(opaque, pads, column, ly, baseY);
    }
  }
  float sx = 0.0f, sy = 0.0f, sz = 0.0f;
  int32_t crossings = 0;
  /** Add the zero crossing between two cell centers (relative to the lattice point), if the sign changes. */
  const auto edge = [&](float d0, float d1, float x0, float y0, float z0, float x1, float y1, float z1) {
    if ((d0 > 0.0f) == (d1 > 0.0f)) {
      return;
    }
    const float t = d0 / (d0 - d1);
    sx += x0 + (x1 - x0) * t;
    sy += y0 + (y1 - y0) * t;
    sz += z0 + (z1 - z0) * t;
    ++crossings;
  };
  for (int32_t a = 0; a < 2; ++a) {
    for (int32_t b = 0; b < 2; ++b) {
      const float fa = static_cast<float>(a) - 0.5f;
      const float fb = static_cast<float>(b) - 0.5f;
      edge(d[0][a][b], d[1][a][b], -0.5f, fa, fb, 0.5f, fa, fb); // along x
      edge(d[a][0][b], d[a][1][b], fa, -0.5f, fb, fa, 0.5f, fb); // along y
      edge(d[a][b][0], d[a][b][1], fa, fb, -0.5f, fa, fb, 0.5f); // along z
    }
  }
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;
  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < 2; ++j) {
      for (int32_t k = 0; k < 2; ++k) {
        gx += d[i][j][k] * static_cast<float>(2 * i - 1);
        gy += d[i][j][k] * static_cast<float>(2 * j - 1);
        gz += d[i][j][k] * static_cast<float>(2 * k - 1);
      }
    }
  }
  SmoothCorner corner{{0, 0, 0}, packNormal(gx, gy, gz, faceNormal)};
  if (crossings != 0) {
    const float inverse = 1.0f / static_cast<float>(crossings);
    corner.offset = toFixedOffset(sx * inverse, sy * inverse, sz * inverse);
  }
  return corner;
}

// Binary Surface Nets corner for one configuration of the 8 cells around a lattice corner (bit i | j << 1 | k << 2
// set when cell (corner - 1 + i, corner - 1 + j, corner - 1 + k) is inside): the foliage surface.
struct CornerShape {
  Int3 offset;     // from the lattice point, in 1/kSmoothFixedOne blocks
  uint32_t normal; // packed signed bytes, pointing away from the inside cells; 0 when the gradient vanishes
};

/** Place the corner of one configuration: the average midpoint of its inside / outside edges. */
CornerShape cornerShape(uint32_t config) {
  const auto inside = [&](int32_t i, int32_t j, int32_t k) { return ((config >> (i | (j << 1) | (k << 2))) & 1u) != 0u; };
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;
  for (int32_t i = 0; i < 2; ++i)
    for (int32_t j = 0; j < 2; ++j)
      for (int32_t k = 0; k < 2; ++k)
        if (inside(i, j, k)) {
          gx += static_cast<float>(2 * i - 1);
          gy += static_cast<float>(2 * j - 1);
          gz += static_cast<float>(2 * k - 1);
        }
  float sx = 0.0f, sy = 0.0f, sz = 0.0f;
  int32_t crossings = 0;
  for (int32_t a = 0; a < 2; ++a) {
    for (int32_t b = 0; b < 2; ++b) {
      const float fa = static_cast<float>(a) - 0.5f;
      const float fb = static_cast<float>(b) - 0.5f;
      if (inside(0, a, b) != inside(1, a, b)) { // edge along x
        sy += fa, sz += fb, ++crossings;
      }
      if (inside(a, 0, b) != inside(a, 1, b)) { // edge along y
        sx += fa, sz += fb, ++crossings;
      }
      if (inside(a, b, 0) != inside(a, b, 1)) { // edge along z
        sx += fa, sy += fb, ++crossings;
      }
    }
  }
  CornerShape shape{{0, 0, 0}, packNormal(gx, gy, gz, 0u)};
  if (crossings != 0) {
    const float inverse = 1.0f / static_cast<float>(crossings);
    shape.offset = toFixedOffset(sx * inverse, sy * inverse, sz * inverse);
  }
  return shape;
}

const std::array<CornerShape, 256> kCornerShapes = [] {
  std::array<CornerShape, 256> shapes{};
  for (uint32_t config = 0; config < 256u; ++config) {
    shapes[config] = cornerShape(config);
  }
  return shapes;
}();

constexpr int32_t kCornerU[4] = {-1, 1, 1, -1}; // corner k's side along u (AO's corner order)
constexpr int32_t kCornerV[4] = {-1, -1, 1, 1}; // corner k's side along v

/** Corner k of the face of cell c in direction `normal`, as a lattice point. */
Int3 faceCorner(Int3 c, uint32_t normal, uint32_t k) {
  const Int3 dir = kNormals[normal], u = kU[normal], v = kV[normal];
  // Doubled coordinates: cell center (2c + 1), out to the face (dir), then to the corner.
  return {(2 * c.x + 1 + dir.x + kCornerU[k] * u.x + kCornerV[k] * v.x) / 2,
          (2 * c.y + 1 + dir.y + kCornerU[k] * u.y + kCornerV[k] * v.y) / 2,
          (2 * c.z + 1 + dir.z + kCornerU[k] * u.z + kCornerV[k] * v.z) / 2};
}

/**
 * The 8 cells around chunk-local lattice point (lx, ly, lz) (each 0..32) as a CornerShape configuration, from padded
 * column masks (bit y + 1 = cell y, as PaddedOpaque).
 */
uint32_t cornerConfig(const PaddedOpaque &masks, int32_t lx, int32_t ly, int32_t lz) {
  // Column (lx - 1 + i, lz - 1 + k) sits at padded index (lz + k) * kPad + (lx + i); its cells ly - 1, ly are bits
  // ly, ly + 1.
  const size_t base = static_cast<size_t>(lz * kPad + lx);
  const auto pair = [&](size_t index) { return static_cast<uint32_t>((masks[index] >> ly) & 3u); };
  const uint32_t c00 = pair(base), c10 = pair(base + 1), c01 = pair(base + kPad), c11 = pair(base + kPad + 1);
  // Bit layout i | j << 1 | k << 2: each pair's two bits are j = 0, 1.
  const auto spread = [](uint32_t bits) { return (bits & 1u) | ((bits & 2u) << 1u); };
  return spread(c00) | (spread(c10) << 1u) | (spread(c01) << 4u) | (spread(c11) << 5u);
}

/**
 * Build padded masks (PaddedOpaque layout) of foliage cells: non-opaque blocks other than air and ocean. Only chunks
 * that hold any (VoxelChunk::nonOpaqueCount) are read.
 *
 * @return whether any foliage cell was found.
 */
bool buildFoliageMasks(const Neighborhood &nb, int32_t chunkY, PaddedOpaque &foliage) {
  bool any = false;
  for (int32_t pz = -1; pz <= N; ++pz) {
    const int32_t dz = pz < 0 ? -1 : (pz >= N ? 1 : 0);
    for (int32_t px = -1; px <= N; ++px) {
      const int32_t dx = px < 0 ? -1 : (px >= N ? 1 : 0);
      const size_t index = static_cast<size_t>((pz + 1) * kPad + (px + 1));
      foliage[index] = 0;
      const Column *column = nb.columns[dz + 1][dx + 1];
      if (column == nullptr) {
        continue;
      }
      const int32_t lx = px - dx * N;
      const int32_t lz = pz - dz * N;
      /** Whether a block id is foliage. */
      const auto isFoliage = [&](uint16_t id) { return id != kAir && id != column->oceanId && !blocks::isOpaque(id); };
      uint64_t bits = 0;
      const VoxelChunk *below = column->chunk(chunkY - 1);
      if (below != nullptr && below->nonOpaqueCount != 0 && isFoliage(below->ids[VoxelChunk::index(lx, N - 1, lz)])) {
        bits |= 1u;
      }
      const VoxelChunk *self = column->chunk(chunkY);
      if (self != nullptr && self->nonOpaqueCount != 0) {
        const uint16_t *ids = &self->ids[VoxelChunk::index(lx, 0, lz)];
        for (int32_t y = 0; y < N; ++y) {
          bits |= static_cast<uint64_t>(isFoliage(ids[y])) << (y + 1);
        }
      }
      const VoxelChunk *above = column->chunk(chunkY + 1);
      if (above != nullptr && above->nonOpaqueCount != 0 && isFoliage(above->ids[VoxelChunk::index(lx, 0, lz)])) {
        bits |= 1ull << 33u;
      }
      foliage[index] = bits;
      any = any || bits != 0u;
    }
  }
  return any;
}

/**
 * Append one smooth quad (4 vertices) in the front-facing order of the shared index pattern, split along the shorter
 * diagonal (the quad is not planar).
 *
 * @param fixedPositions Corner k's chunk-local fixed-point position (AO's corner order).
 * @param words Corner k's normalLight word.
 */
void appendSmoothQuad(const std::array<Int3, 4> &fixedPositions, const std::array<uint32_t, 4> &words,
                      uint16_t material, uint32_t normal, std::vector<SmoothVertex> &out) {
  const auto distance2 = [&](uint32_t a, uint32_t b) {
    const int64_t dx = fixedPositions[a].x - fixedPositions[b].x;
    const int64_t dy = fixedPositions[a].y - fixedPositions[b].y;
    const int64_t dz = fixedPositions[a].z - fixedPositions[b].z;
    return dx * dx + dy * dy + dz * dz;
  };
  // The index pattern splits along output corners 0-2; rotating by one splits along 1-3.
  std::array<uint32_t, 4> order = normal == 0u || normal == 5u // flippedWinding in quad_decode.hlsl
                                      ? std::array<uint32_t, 4>{0, 3, 2, 1}
                                      : std::array<uint32_t, 4>{0, 1, 2, 3};
  if (distance2(1, 3) < distance2(0, 2)) {
    std::rotate(order.begin(), order.begin() + 1, order.end());
  }
  for (const uint32_t k : order) {
    out.push_back({static_cast<uint16_t>(fixedPositions[k].x), static_cast<uint16_t>(fixedPositions[k].y),
                   static_cast<uint16_t>(fixedPositions[k].z), material, words[k]});
  }
}

/** Chunk-local fixed-point position of lattice point `lattice` moved by `offset`. */
Int3 fixedPosition(Int3 lattice, Int3 offset) {
  return {(lattice.x + 1) * kSmoothFixedOne + offset.x, (lattice.y + 1) * kSmoothFixedOne + offset.y,
          (lattice.z + 1) * kSmoothFixedOne + offset.z};
}

// Opaque corners already placed this chunk, by lattice point ((lz * 33 + lx) * 33 + ly): faces sharing a corner
// compute it once. An entry is valid when its stamp is the current chunk's.
struct CornerCache {
  static constexpr int32_t kSide = N + 1;
  std::array<uint32_t, kSide * kSide * kSide> stamp{};
  std::array<SmoothCorner, kSide * kSide * kSide> corner;
  uint32_t current = 0;

  /** Start a chunk: invalidate every entry. */
  void begin() {
    if (++current == 0u) { // wrapped: clear the stale stamps once
      stamp.fill(0u);
      current = 1u;
    }
  }
};

/**
 * Emit the smooth quads of the chunk: first the opaque faces (the faces computePlanes finds) on the density surface,
 * then the foliage faces (foliage next to air or ocean) on the binary foliage surface. 4 vertices per quad, no AO.
 *
 * @param voxels The chunk's voxels (block ids of the faces).
 * @param foliage Padded foliage masks, or null when the neighborhood has none.
 * @param out Receives the vertices.
 * @return vertices of the opaque surface (the rest are foliage).
 */
uint32_t emitSmooth(const PaddedMasks &masks, const PaddedOpaque &opaque, const HeightPads &pads,
                    const PaddedOpaque *foliage, const ChunkLight &light, const VoxelChunk &voxels, int32_t chunkY,
                    std::vector<SmoothVertex> &out) {
  const int32_t baseY = chunkY * N;
  thread_local CornerCache cache;
  cache.begin();
  std::array<Int3, 4> positions;
  std::array<uint32_t, 4> words;
  for (uint32_t normal = 0; normal < 6u; ++normal) {
    const Int3 dir = kNormals[normal];
    const uint32_t faceNormal = snormByte(static_cast<float>(dir.x)) | (snormByte(static_cast<float>(dir.y)) << 8u) |
                                (snormByte(static_cast<float>(dir.z)) << 16u);
    for (int32_t z = 0; z < N; ++z) {
      for (int32_t x = 0; x < N; ++x) {
        const size_t i = static_cast<size_t>((z + 1) * kPad + (x + 1));
        const size_t neighbor = static_cast<size_t>((z + dir.z + 1) * kPad + (x + dir.x + 1));
        uint32_t faces = masks.shifted[1][i] & ~masks.shifted[1 + dir.y][neighbor];
        while (faces != 0u) {
          const int32_t y = std::countr_zero(faces);
          faces &= faces - 1u;
          const uint16_t id = voxels.ids[VoxelChunk::index(x, y, z)];
          const uint32_t sky = cornerLight(light, masks, {x + dir.x, y + dir.y, z + dir.z}, normal);
          for (uint32_t k = 0; k < 4; ++k) {
            const Int3 lattice = faceCorner({x, y, z}, normal, k);
            const size_t key =
                static_cast<size_t>((lattice.z * CornerCache::kSide + lattice.x) * CornerCache::kSide + lattice.y);
            if (cache.stamp[key] != cache.current) {
              // A vanishing gradient falls back to the normal of the face that placed the corner first.
              cache.corner[key] = opaqueCorner(opaque, pads, baseY, lattice.x, lattice.y, lattice.z, faceNormal);
              cache.stamp[key] = cache.current;
            }
            const SmoothCorner &corner = cache.corner[key];
            positions[k] = fixedPosition(lattice, corner.offset);
            words[k] = corner.normal | (cornerLevel(sky, k) << 26u);
          }
          appendSmoothQuad(positions, words, static_cast<uint16_t>((id & 0x3ffu) | (normal << 10u)), normal, out);
        }
      }
    }
  }
  const uint32_t opaqueVertices = static_cast<uint32_t>(out.size());
  if (foliage == nullptr) {
    return opaqueVertices;
  }

  for (uint32_t normal = 0; normal < 6u; ++normal) {
    const Int3 dir = kNormals[normal];
    const uint32_t faceNormal = snormByte(static_cast<float>(dir.x)) | (snormByte(static_cast<float>(dir.y)) << 8u) |
                                (snormByte(static_cast<float>(dir.z)) << 16u);
    for (int32_t z = 0; z < N; ++z) {
      for (int32_t x = 0; x < N; ++x) {
        const size_t i = static_cast<size_t>((z + 1) * kPad + (x + 1));
        const size_t neighbor = static_cast<size_t>((z + dir.z + 1) * kPad + (x + dir.x + 1));
        const uint32_t self = static_cast<uint32_t>((*foliage)[i] >> 1u);
        // Bit y = the cell at y + dir.y in the neighbor column (bits 0..33 hold cells -1..32).
        const uint32_t covered = static_cast<uint32_t>(((*foliage)[neighbor] | opaque[neighbor]) >> (1 + dir.y));
        uint32_t faces = self & ~covered;
        while (faces != 0u) {
          const int32_t y = std::countr_zero(faces);
          faces &= faces - 1u;
          const uint16_t id = voxels.ids[VoxelChunk::index(x, y, z)];
          const uint32_t sky = cornerLight(light, masks, {x + dir.x, y + dir.y, z + dir.z}, normal);
          for (uint32_t k = 0; k < 4; ++k) {
            const Int3 lattice = faceCorner({x, y, z}, normal, k);
            const CornerShape &shape = kCornerShapes[cornerConfig(*foliage, lattice.x, lattice.y, lattice.z)];
            positions[k] = fixedPosition(lattice, shape.offset);
            words[k] = (shape.normal != 0u ? shape.normal : faceNormal) | (cornerLevel(sky, k) << 26u);
          }
          appendSmoothQuad(positions, words, static_cast<uint16_t>((id & 0x3ffu) | (normal << 10u)), normal, out);
        }
      }
    }
  }
  return opaqueVertices;
}

/**
 * Mesh a chunk (see meshChunk) into `quads`, a reused per-thread scratch, so the result can be copied out with one
 * exact allocation. Opaque and cutout quads come first, then ocean quads for the translucent clusters.
 *
 * @param nb Neighborhood of the column being meshed.
 * @param chunkY Chunk y, in chunks.
 * @param out Receives clusters and chunk offset (not quads).
 * @param quads Receives the packed quads.
 * @param smooth Receives the Surface Nets vertices of the opaque faces.
 */
void meshChunkInto(const Neighborhood &nb, int32_t chunkY, ChunkMesh &out, std::vector<uint64_t> &quads,
                   std::vector<SmoothVertex> &smooth) {
  const Column &center = nb.center();
  const VoxelChunk *voxels = center.chunk(chunkY);
  out.cy = chunkY;
  quads.clear();
  smooth.clear();
  out.smoothOpaqueVertices = 0;
  out.chunkOffset = packChunkOffset(center.cx, chunkY, center.cz);
  out.clusters = {};
  out.translucentClusters = {};
  if (voxels == nullptr) {
    return;
  }

  thread_local PaddedMasks masks;
  thread_local TilePlanes planes;
  thread_local PaddedOpaque paddedOpaque;
  buildMasks(nb, chunkY, masks, paddedOpaque);
  thread_local ChunkLight light;
  computeSkyLight(nb, chunkY, paddedOpaque, light);
  thread_local std::array<uint32_t, N * N> nonOpaque; // per column, bit y = non-opaque, non-ocean cell
  const bool hasNonOpaque = voxels->nonOpaqueCount != 0;
  if (hasNonOpaque) {
    for (int32_t column = 0; column < N * N; ++column) {
      uint32_t bits = 0;
      for (int32_t y = 0; y < N; ++y) {
        const uint16_t id = voxels->ids[static_cast<size_t>(column * N + y)];
        bits |= (id != kAir && id != center.oceanId && !blocks::isOpaque(id) ? 1u : 0u) << y;
      }
      nonOpaque[static_cast<size_t>(column)] = bits;
    }
  }

  for (int32_t tileZ = 0; tileZ < N / TILE; ++tileZ) {
    for (int32_t tileX = 0; tileX < N / TILE; ++tileX) {
      QuadCluster &cluster = out.clusters[static_cast<size_t>(tileZ * 4 + tileX)];
      cluster.base = static_cast<uint32_t>(quads.size());
      std::array<uint32_t, 6> counts{};
      TileBounds bounds;
      uint32_t cutoutQuads = 0; // non-opaque (alpha cutout) quads, drawn by the cutout pass
      for (int run = 0; run < 6; ++run) {
        const uint32_t normal = kRunNormals[run];
        computePlanes(masks, light, normal, tileX, tileZ, planes);
        counts[run] = normal == 2u || normal == 3u
                          ? emitHorizontal(planes, *voxels, normal, tileX, tileZ, quads, bounds)
                          : emitSide(planes, *voxels, normal, tileX, tileZ, quads, bounds);
        if (hasNonOpaque) {
          const uint32_t nonOpaqueCount =
              emitNonOpaque(masks, light, nb, *voxels, nonOpaque, chunkY, normal, tileX, tileZ, quads, bounds);
          counts[run] += nonOpaqueCount;
          cutoutQuads += nonOpaqueCount;
        }
      }
      cluster.counts01 = counts[0] | (counts[1] << 16u);
      cluster.counts23 = counts[2] | (counts[3] << 16u);
      cluster.counts45 = counts[4] | (counts[5] << 16u);
      const int32_t faceLow = bounds.low <= bounds.high ? bounds.low : 0;
      const int32_t faceHigh = bounds.low <= bounds.high ? bounds.high : N;
      const float minX = static_cast<float>(center.cx * N + tileX * TILE);
      const float minY = static_cast<float>(chunkY * N + faceLow);
      const float minZ = static_cast<float>(center.cz * N + tileZ * TILE);
      cluster.boundsMin = {minX, minY, minZ, 0.0f};
      // w: 1 when the cluster holds non-opaque quads (geometry_shared.hlsl CLUSTER_HAS_CUTOUT).
      cluster.boundsMax = {minX + TILE, minY + static_cast<float>(faceHigh - faceLow), minZ + TILE,
                           cutoutQuads != 0u ? 1.0f : 0.0f};
    }
  }

  thread_local HeightPads pads;
  buildHeightPads(nb, pads);
  thread_local PaddedOpaque foliage;
  const bool hasFoliage = buildFoliageMasks(nb, chunkY, foliage);
  out.smoothOpaqueVertices =
      emitSmooth(masks, paddedOpaque, pads, hasFoliage ? &foliage : nullptr, light, *voxels, chunkY, smooth);

  // Ocean faces into the translucent clusters, after every opaque quad.
  bool hasOcean = false;
  for (const uint32_t bits : voxels->ocean) {
    hasOcean = hasOcean || bits != 0u;
  }
  if (!hasOcean) {
    return; // translucentClusters stay zero (no quads)
  }
  thread_local PaddedMasks oceanMasks;
  buildOceanMasks(nb, chunkY, oceanMasks);
  for (int32_t tileZ = 0; tileZ < N / TILE; ++tileZ) {
    for (int32_t tileX = 0; tileX < N / TILE; ++tileX) {
      QuadCluster &cluster = out.translucentClusters[static_cast<size_t>(tileZ * 4 + tileX)];
      cluster.base = static_cast<uint32_t>(quads.size());
      std::array<uint32_t, 6> counts{};
      TileBounds bounds;
      for (int run = 0; run < 6; ++run) {
        const uint32_t normal = kRunNormals[run];
        computeOceanPlanes(masks, oceanMasks, light, normal, tileX, tileZ, planes);
        counts[run] = normal == 2u || normal == 3u
                          ? emitHorizontal(planes, *voxels, normal, tileX, tileZ, quads, bounds)
                          : emitSide(planes, *voxels, normal, tileX, tileZ, quads, bounds);
      }
      cluster.counts01 = counts[0] | (counts[1] << 16u);
      cluster.counts23 = counts[2] | (counts[3] << 16u);
      cluster.counts45 = counts[4] | (counts[5] << 16u);
      const int32_t faceLow = bounds.low <= bounds.high ? bounds.low : 0;
      const int32_t faceHigh = bounds.low <= bounds.high ? bounds.high : N;
      const float minX = static_cast<float>(center.cx * N + tileX * TILE);
      const float minY = static_cast<float>(chunkY * N + faceLow);
      const float minZ = static_cast<float>(center.cz * N + tileZ * TILE);
      cluster.boundsMin = {minX, minY, minZ, 0.0f};
      cluster.boundsMax = {minX + TILE, minY + static_cast<float>(faceHigh - faceLow), minZ + TILE, 0.0f};
    }
  }
}

} // namespace

void meshChunk(const Neighborhood &nb, int32_t chunkY, ChunkMesh &out) {
  thread_local std::vector<uint64_t> scratch;
  thread_local std::vector<SmoothVertex> smoothScratch;
  meshChunkInto(nb, chunkY, out, scratch, smoothScratch);
  out.quads.assign(scratch.begin(), scratch.end());
  out.smoothVertices.assign(smoothScratch.begin(), smoothScratch.end());
}

} // namespace cpuworld
