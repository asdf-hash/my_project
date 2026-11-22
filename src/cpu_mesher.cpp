#include "cpu_mesher.hpp"

#include "simd.hpp"

#include <bit>
#include <cstring>

namespace cpuworld {
namespace {

using simd::I8;

constexpr int32_t N = CHUNK_SIZE;
constexpr int32_t TILE = 8;

struct Int3 {
  int32_t x, y, z;
};
Int3 operator+(Int3 a, Int3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Int3 operator-(Int3 a, Int3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

// Per normal index (+x, -x, +y, -y, +z, -z): the outside cell and calculateAo's u / v axes (chunk_compute.slang).
constexpr Int3 kNormals[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
constexpr Int3 kU[6] = {{0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
constexpr Int3 kV[6] = {{0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
// Cluster run order: +Y, -X, +X, -Z, +Z, -Y.
constexpr uint32_t kRunNormals[6] = {2, 1, 0, 5, 4, 3};

bool aoConstantAlongU(uint32_t ao) { return (ao & 3u) == ((ao >> 2u) & 3u) && ((ao >> 6u) & 3u) == ((ao >> 4u) & 3u); }
bool aoConstantAlongV(uint32_t ao) { return (ao & 3u) == ((ao >> 6u) & 3u) && ((ao >> 2u) & 3u) == ((ao >> 4u) & 3u); }

// Column bitmasks over the padded 34x34 area, pre-shifted: shifted[k][i] = uint32(M >> k), where M has bit (y + 1)
// set for a solid cell at chunk-local y in -1..32. Bit y of shifted[1 + dy] is then the cell at y + dy.
struct PaddedMasks {
  std::array<std::array<uint32_t, kPad * kPad>, 3> shifted;

  // 8 consecutive x columns starting at chunk-local (x, z) shifted by `off`.
  I8 load(Int3 off, int32_t x, int32_t z) const {
    return I8::load(reinterpret_cast<const int32_t *>(&shifted[1 + off.y][(z + off.z + 1) * kPad + (x + off.x + 1)]));
  }
};

// The same pre-shifted columns for ocean cells (Column::ocean32; border heights where a neighbor isn't loaded).
void buildOceanMasks(const Neighborhood &nb, int32_t chunkY, PaddedMasks &masks) {
  const Column &center = nb.center();
  for (int32_t pz = -1; pz <= N; ++pz) {
    const int32_t dz = pz < 0 ? -1 : (pz >= N ? 1 : 0);
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
      const uint64_t m = static_cast<uint64_t>(below >> 31u) | (static_cast<uint64_t>(self) << 1u) |
                         (static_cast<uint64_t>(above & 1u) << 33u);
      const int32_t i = (pz + 1) * kPad + (px + 1);
      masks.shifted[0][i] = static_cast<uint32_t>(m);
      masks.shifted[1][i] = static_cast<uint32_t>(m >> 1u);
      masks.shifted[2][i] = static_cast<uint32_t>(m >> 2u);
    }
  }
}

void buildMasks(const Neighborhood &nb, int32_t chunkY, PaddedMasks &masks) {
  const Column &center = nb.center();
  for (int32_t pz = -1; pz <= N; ++pz) {
    const int32_t dz = pz < 0 ? -1 : (pz >= N ? 1 : 0);
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
      const uint64_t m = static_cast<uint64_t>(below >> 31u) | (static_cast<uint64_t>(self) << 1u) |
                         (static_cast<uint64_t>(above & 1u) << 33u);
      const int32_t i = (pz + 1) * kPad + (px + 1);
      masks.shifted[0][i] = static_cast<uint32_t>(m);
      masks.shifted[1][i] = static_cast<uint32_t>(m >> 1u);
      masks.shifted[2][i] = static_cast<uint32_t>(m >> 2u);
    }
  }
}

// Face bits and AO bitplanes of one tile in one direction: for column c (z * 8 + x within the tile), bit y of
// face[c] marks a face at cell y; corner k's AO value at that face is aoLo[k] bit y | aoHi[k] bit y << 1.
struct TilePlanes {
  std::array<uint32_t, 64> face;
  std::array<std::array<uint32_t, 64>, 4> aoLo;
  std::array<std::array<uint32_t, 64>, 4> aoHi;

  uint32_t ao(int32_t c, int32_t y) const {
    uint32_t ao = 0;
    for (int k = 0; k < 4; ++k) {
      ao |= (((aoLo[k][c] >> y) & 1u) | (((aoHi[k][c] >> y) & 1u) << 1u)) << (2 * k);
    }
    return ao;
  }
};

// vertexAo(a, b, c) = a && b ? 3 : a + b + c, as two bitplanes over 32 cells.
void vertexAoPlanes(I8 a, I8 b, I8 c, I8 &lo, I8 &hi) {
  const I8 ab = a & b;
  lo = ab | (a ^ b ^ c);
  hi = ab | (c & (a | b));
}

void computePlanes(const PaddedMasks &masks, uint32_t normal, int32_t tileX, int32_t tileZ, TilePlanes &planes) {
  const Int3 o = kNormals[normal];
  const Int3 u = kU[normal];
  const Int3 v = kV[normal];
  const Int3 zero{0, 0, 0};
  for (int32_t row = 0; row < TILE; ++row) {
    const int32_t x = tileX * TILE;
    const int32_t z = tileZ * TILE + row;
    const I8 solid = masks.load(zero, x, z);
    const I8 face = simd::andnot(solid, masks.load(o, x, z));
    const I8 u0 = masks.load(o - u, x, z);
    const I8 u1 = masks.load(o + u, x, z);
    const I8 v0 = masks.load(o - v, x, z);
    const I8 v1 = masks.load(o + v, x, z);
    const I8 c0 = masks.load(o - u - v, x, z);
    const I8 c1 = masks.load(o + u - v, x, z);
    const I8 c2 = masks.load(o + u + v, x, z);
    const I8 c3 = masks.load(o - u + v, x, z);
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

// Ocean faces of one tile in one direction: an ocean cell beside a cell that is neither opaque nor ocean (air, other
// non-opaque blocks). No AO: water is lit flat.
void computeOceanPlanes(const PaddedMasks &solidMasks, const PaddedMasks &oceanMasks, uint32_t normal, int32_t tileX,
                        int32_t tileZ, TilePlanes &planes) {
  const Int3 o = kNormals[normal];
  const Int3 zero{0, 0, 0};
  for (int32_t row = 0; row < TILE; ++row) {
    const int32_t x = tileX * TILE;
    const int32_t z = tileZ * TILE + row;
    const I8 ocean = oceanMasks.load(zero, x, z);
    const I8 covered = solidMasks.load(o, x, z) | oceanMasks.load(o, x, z);
    // andnot(a, b) = a & ~b: ocean cells whose neighbor is neither opaque nor ocean.
    simd::andnot(ocean, covered).store(reinterpret_cast<int32_t *>(&planes.face[row * TILE]));
  }
  for (int k = 0; k < 4; ++k) {
    planes.aoLo[k].fill(0u);
    planes.aoHi[k].fill(0u);
  }
}

struct TileBounds {
  int32_t low = N;
  int32_t high = 0;
  void add(int32_t lowCell, int32_t highCell) {
    low = std::min(low, lowCell);
    high = std::max(high, highCell);
  }
};

// Top / bottom faces: per layer, rectangles of identical (block, AO) within the tile, extended along x only when AO
// is constant along u and along z only when it is constant along v (the GPU's top-face rules).
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
    std::array<uint32_t, 64> key{};
    for (int32_t c = 0; c < 64; ++c) {
      if ((planes.face[c] >> y) & 1u) {
        const uint32_t ao = planes.ao(c, y);
        const int32_t x = tileX * TILE + (c & 7);
        const int32_t z = tileZ * TILE + (c >> 3);
        key[c] = (1u << 19u) | (ao << 10u) | voxels.ids[VoxelChunk::index(x, y, z)];
      }
    }
    std::array<bool, 64> used{};
    for (int32_t row = 0; row < TILE; ++row) {
      for (int32_t col = 0; col < TILE; ++col) {
        const int32_t c = row * TILE + col;
        if (key[c] == 0u || used[c]) {
          continue;
        }
        const uint32_t k = key[c];
        const uint32_t ao = (k >> 10u) & 0xffu;
        uint32_t length = 1;
        if (aoConstantAlongU(ao)) {
          while (col + static_cast<int32_t>(length) < TILE && key[c + length] == k && !used[c + length]) {
            ++length;
          }
        }
        uint32_t width = 1;
        if (aoConstantAlongV(ao)) {
          for (;;) {
            const int32_t nextRow = row + static_cast<int32_t>(width);
            if (nextRow >= TILE) {
              break;
            }
            bool same = true;
            for (uint32_t i = 0; i < length && same; ++i) {
              const int32_t n = nextRow * TILE + col + static_cast<int32_t>(i);
              same = key[n] == k && !used[n];
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
        quads.push_back(packQuad(x, static_cast<uint32_t>(y), z, normal, ao, k & 0x3ffu, length, width));
        ++count;
        bounds.add(y, y + 1);
      }
    }
  }
  return count;
}

struct SideRun {
  uint32_t start, length, block, ao;
  bool operator==(const SideRun &) const = default;
};

// Side faces: vertical runs of equal (block, AO) per column, merged across the tile along the greedy axis (z for x
// faces, x for z faces) when the neighbor column has the identical run (sideFaceRuns / sideRun on the GPU).
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
      const bool mergeable = aoConstantAlongV(ao);
      if (open && run.start + run.length == y && openMergeable && mergeable && block == run.block && ao == run.ao) {
        ++run.length;
        continue;
      }
      if (open) {
        runs[c][runCount[c]++] = run;
      }
      run = {y, 1u, block, ao};
      open = true;
      openMergeable = mergeable;
    }
    runs[c][runCount[c]++] = run;
  }

  const bool xFace = normal < 2u;
  auto columnAt = [&](int32_t c, int32_t step) { return xFace ? c + step * TILE : c + step; };
  auto positionOf = [&](int32_t c) { return xFace ? c >> 3 : c & 7; };
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
    for (uint32_t r = 0; r < runCount[c]; ++r) {
      const SideRun &run = runs[c][r];
      const bool mergeable = aoConstantAlongU(run.ao);
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
      quads.push_back(packQuad(x, run.start, z, normal, run.ao, run.block, run.length, width));
      ++count;
      bounds.add(static_cast<int32_t>(run.start), static_cast<int32_t>(run.start + run.length));
    }
  }
  return count;
}

// Opaque bit of chunk-local cell (x, y, z), x / z in -1..32, y in -1..32, from the padded masks.
bool opaqueAt(const PaddedMasks &masks, int32_t x, int32_t y, int32_t z) {
  const int32_t i = (z + 1) * kPad + (x + 1);
  // shifted[0] has the cell at y in bit y + 1, shifted[2] in bit y - 1 (reaches y = 32).
  return y <= 30 ? (masks.shifted[0][i] >> (y + 1)) & 1u : (masks.shifted[2][i] >> (y - 1)) & 1u;
}

// Block id of chunk-local cell (x, y, z), x / z in -1..32 (neighbor columns; border heights where not loaded).
uint16_t neighborId(const Neighborhood &nb, int32_t chunkY, int32_t x, int32_t y, int32_t z) {
  const int32_t dx = x < 0 ? -1 : (x >= N ? 1 : 0);
  const int32_t dz = z < 0 ? -1 : (z >= N ? 1 : 0);
  const int32_t worldY = chunkY * N + y;
  if (const Column *column = nb.columns[dz + 1][dx + 1]) {
    return column->blockId(x - dx * N, worldY, z - dz * N);
  }
  return worldY < nb.center().borderHeight(x, z) ? kBuriedTerrain : nb.center().emptyCell(worldY);
}

uint32_t vertexAo(bool a, bool b, bool corner) { return a && b ? 3u : uint32_t(a) + uint32_t(b) + uint32_t(corner); }

// Non-opaque blocks (id bit 9 set): the binary masks leave them out, so they get their own per-cell pass. Full
// cubes for now: a face shows unless the neighbor is opaque or the same block, as a 1x1 quad with the binary
// mesher's AO (from opaque neighbors). nonOpaque holds the chunk's non-opaque cells as column bits.
uint32_t emitNonOpaque(const PaddedMasks &masks, const Neighborhood &nb, const VoxelChunk &voxels,
                       const std::array<uint32_t, N * N> &nonOpaque, int32_t chunkY, uint32_t normal, int32_t tileX,
                       int32_t tileZ, std::vector<uint64_t> &quads, TileBounds &bounds) {
  const Int3 o = kNormals[normal];
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
        const Int3 out{x + o.x, y + o.y, z + o.z};
        if (opaqueAt(masks, out.x, out.y, out.z)) {
          continue;
        }
        const uint16_t id = voxels.ids[VoxelChunk::index(x, y, z)];
        if (neighborId(nb, chunkY, out.x, out.y, out.z) == id) {
          continue;
        }
        const auto at = [&](Int3 d) { return opaqueAt(masks, out.x + d.x, out.y + d.y, out.z + d.z); };
        const Int3 zero{0, 0, 0};
        const bool u0 = at(zero - u), u1 = at(u), v0 = at(zero - v), v1 = at(v);
        const uint32_t ao = vertexAo(u0, v0, at(zero - u - v)) | (vertexAo(u1, v0, at(u - v)) << 2u) |
                            (vertexAo(u1, v1, at(u + v)) << 4u) | (vertexAo(u0, v1, at(v - u)) << 6u);
        quads.push_back(packQuad(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z), normal,
                                 ao, id & 0x3ffu, 1u, 1u));
        ++count;
        bounds.add(y, y + 1);
      }
    }
  }
  return count;
}

} // namespace

uint64_t packQuad(uint32_t x, uint32_t y, uint32_t z, uint32_t normalIndex, uint32_t ao, uint32_t blockType,
                  uint32_t length, uint32_t width) {
  uint32_t packed = (x & 0x1fu) | ((y & 0x1fu) << 5u) | ((z & 0x1fu) << 10u) | ((normalIndex & 0x7u) << 15u);
  packed |= ao << 18u;
  packed |= ((length - 1u) & 0x1fu) << 26u;
  const uint32_t metadata = blockType | (15u << 10u) | (0xfffu << 14u) | (((width - 1u) & 0x7u) << 26u); // LOD 0
  return (static_cast<uint64_t>(metadata) << 32u) | packed;
}

uint64_t packChunkOffset(int32_t cx, int32_t chunkY, int32_t cz) {
  const uint64_t offsetX = static_cast<uint32_t>(cx * CHUNK_SIZE) & 0x00ffffffu;
  const uint64_t offsetZ = static_cast<uint64_t>(static_cast<uint32_t>(cz * CHUNK_SIZE) & 0x00ffffffu) << 24u;
  const uint64_t offsetY = static_cast<uint64_t>(static_cast<uint32_t>(chunkY) & 0x1fffu) << 48u;
  return offsetX | offsetZ | offsetY; // LOD 0
}

void meshChunk(const Neighborhood &nb, int32_t chunkY, ChunkMesh &out) {
  const Column &center = nb.center();
  const VoxelChunk *voxels = center.chunk(chunkY);
  out.cx = center.cx;
  out.cy = chunkY;
  out.cz = center.cz;
  out.quads.clear();
  out.chunkOffset = packChunkOffset(center.cx, chunkY, center.cz);
  out.clusters = {};
  out.translucentClusters = {};
  if (voxels == nullptr) {
    return;
  }

  thread_local PaddedMasks masks;
  thread_local TilePlanes planes;
  buildMasks(nb, chunkY, masks);
  thread_local std::array<uint32_t, N * N> nonOpaque;
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
      cluster.base = static_cast<uint32_t>(out.quads.size());
      std::array<uint32_t, 6> counts{};
      TileBounds bounds;
      for (int run = 0; run < 6; ++run) {
        const uint32_t normal = kRunNormals[run];
        computePlanes(masks, normal, tileX, tileZ, planes);
        counts[run] = normal == 2u || normal == 3u
                          ? emitHorizontal(planes, *voxels, normal, tileX, tileZ, out.quads, bounds)
                          : emitSide(planes, *voxels, normal, tileX, tileZ, out.quads, bounds);
        if (hasNonOpaque) {
          counts[run] += emitNonOpaque(masks, nb, *voxels, nonOpaque, chunkY, normal, tileX, tileZ, out.quads, bounds);
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
      cluster.boundsMax = {minX + TILE, minY + static_cast<float>(faceHigh - faceLow), minZ + TILE, 0.0f};
    }
  }

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
      cluster.base = static_cast<uint32_t>(out.quads.size());
      std::array<uint32_t, 6> counts{};
      TileBounds bounds;
      for (int run = 0; run < 6; ++run) {
        const uint32_t normal = kRunNormals[run];
        computeOceanPlanes(masks, oceanMasks, normal, tileX, tileZ, planes);
        counts[run] = normal == 2u || normal == 3u
                          ? emitHorizontal(planes, *voxels, normal, tileX, tileZ, out.quads, bounds)
                          : emitSide(planes, *voxels, normal, tileX, tileZ, out.quads, bounds);
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

} // namespace cpuworld
