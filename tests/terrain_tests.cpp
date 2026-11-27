// CPU terrain checks: mesher output vs brute force, a height checksum (compare AVX2 and scalar builds), timings.
// Printed output is diffed against a baseline: keep the strings stable.
#include "../src/block_registry.hpp"
#include "../src/cpu_light.hpp"
#include "../src/cpu_mesher.hpp"
#include "../src/cpu_world.hpp"
#include "../src/simd.hpp"
#include "../src/terrain_cpu.hpp"
#include "../src/vegetation.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <thread>
#include <tuple>

using namespace cpuworld;

namespace {

int failures = 0;
// Counts a failure and prints the first 20 (file:line plus a printf-style message).
#define CHECK(cond, ...)                                                                                               \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      if (++failures <= 20) {                                                                                          \
        std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                               \
        std::printf(__VA_ARGS__);                                                                                      \
        std::printf("\n");                                                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)

constexpr uint16_t kOcean = blocks::kNonOpaqueBit + 2; // glass / leaves in the edit tests use + 0 / + 1
const TerrainBlockIds kIds{{0, 1, 2, 3}, {4, 5, 2, 3}, {4, 2, 2, 3}, kOcean};

/** Reference answers for the mesher, computed cell by cell from the neighborhood. Coordinates are chunk-local. */
struct Brute {
  const Neighborhood &nb; // nb = 3x3 columns around the meshed one
  int32_t chunkY;
  /** Whether a cell is opaque; x/z may be -1..32 (reaches into neighbors, or border heights when absent). */
  bool solid(int32_t x, int32_t y, int32_t z) const {
    const int32_t dx = x < 0 ? -1 : (x >= 32 ? 1 : 0);
    const int32_t dz = z < 0 ? -1 : (z >= 32 ? 1 : 0);
    const Column *c = nb.columns[dz + 1][dx + 1];
    const int32_t worldY = chunkY * 32 + y;
    if (c) {
      return c->isSolid(x - dx * 32, worldY, z - dz * 32);
    }
    return worldY < nb.center().borderHeight(x, z);
  }
  /** Block id of a cell; x/z may be -1..32, absent neighbors read as buried terrain or empty. */
  uint16_t id(int32_t x, int32_t y, int32_t z) const {
    const int32_t dx = x < 0 ? -1 : (x >= 32 ? 1 : 0);
    const int32_t dz = z < 0 ? -1 : (z >= 32 ? 1 : 0);
    const Column *c = nb.columns[dz + 1][dx + 1];
    const int32_t worldY = chunkY * 32 + y;
    if (c) {
      return c->blockId(x - dx * 32, worldY, z - dz * 32);
    }
    return worldY < nb.center().borderHeight(x, z) ? kBuriedTerrain : nb.center().emptyCell(worldY);
  }
  /**
   * Whether the cell shows a face toward a neighbor. Opaque cells show one toward any non-opaque cell; non-opaque
   * blocks (not air) toward any non-opaque cell holding a different block.
   *
   * @param ox, oy, oz Unit offset to the neighbor.
   */
  bool face(int32_t x, int32_t y, int32_t z, int32_t ox, int32_t oy, int32_t oz) const {
    if (solid(x + ox, y + oy, z + oz)) {
      return false;
    }
    if (solid(x, y, z)) {
      return true;
    }
    const uint16_t self = id(x, y, z);
    return self != kAir && self != id(x + ox, y + oy, z + oz);
  }
  /** Occlusion level 0-3 of one corner from its two side cells and the diagonal cell (both sides = fully occluded). */
  uint32_t vertexAo(bool a, bool b, bool c) const { return a && b ? 3u : uint32_t(a) + uint32_t(b) + uint32_t(c); }
  /**
   * Packed AO of one face, as the mesher stores it: 2 bits per corner, (-u,-v), (+u,-v), (+u,+v), (-u,+v).
   *
   * @param n Normal index 0-5 (+x, -x, +y, -y, +z, -z).
   */
  uint32_t ao(int32_t x, int32_t y, int32_t z, uint32_t n) const {
    if (id(x, y, z) == kOcean) {
      return 0u; // water is lit flat
    }
    // Per normal: the face's u and v axes and the outward offset.
    static const int u[6][3] = {{0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    static const int v[6][3] = {{0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
    static const int o[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const int ox = x + o[n][0], oy = y + o[n][1], oz = z + o[n][2]; // cell in front of the face
    auto solidInPlane = [&](int su, int sv) {
      return solid(ox + su * u[n][0] + sv * v[n][0], oy + su * u[n][1] + sv * v[n][1], oz + su * u[n][2] + sv * v[n][2]);
    };
    return vertexAo(solidInPlane(-1, 0), solidInPlane(0, -1), solidInPlane(-1, -1)) |
           (vertexAo(solidInPlane(1, 0), solidInPlane(0, -1), solidInPlane(1, -1)) << 2) |
           (vertexAo(solidInPlane(1, 0), solidInPlane(0, 1), solidInPlane(1, 1)) << 4) |
           (vertexAo(solidInPlane(-1, 0), solidInPlane(0, 1), solidInPlane(-1, 1)) << 6);
  }
};

// Quads checked with skylight below 15 / at 0, so the light tests can tell they saw shadow.
size_t gShadowedFaces = 0, gDarkFaces = 0;

/**
 * Skylight by plain BFS (Minecraft rules) over the chunk and a 15-cell halo, independent of cpu_light.cpp: seeds come
 * from scanning each column top down, not from Column::skyTop. Unloaded neighbors read the center's nearest border
 * height as a heightmap, the same convention as computeSkyLight.
 *
 * @return levels of the padded core, index ((z + 1) * 34 + (x + 1)) * 34 + (y + 1) for chunk-local x, y, z in -1..32.
 */
std::vector<uint8_t> bruteSkyLight(const Neighborhood &nb, int32_t chunkY) {
  constexpr int H = 15, W = 32 + 2 * H; // halo and region width
  // Column for chunk-local px, pz (lx, lz receive its column-local coords), or null when not loaded.
  auto columnAt = [&](int px, int pz, int &lx, int &lz) -> const Column * {
    const int dx = floorDiv(px, 32), dz = floorDiv(pz, 32);
    lx = px - dx * 32, lz = pz - dz * 32;
    return nb.columns[dz + 1][dx + 1];
  };
  auto borderAt = [&](int px, int pz) { return nb.center().borderHeight(std::clamp(px, -1, 32), std::clamp(pz, -1, 32)); };
  std::vector<int> skyTop(W * W); // lowest world y with no opaque cell at or above, per region column
  for (int pz = -H; pz < 32 + H; ++pz)
    for (int px = -H; px < 32 + H; ++px) {
      int lx, lz;
      const Column *c = columnAt(px, pz, lx, lz);
      int top;
      if (!c) {
        top = borderAt(px, pz);
      } else {
        top = (c->firstChunkY + int(c->chunks.size())) * 32; // all air above the stored range
        while (top > world::kMinTerrainHeight && !c->isSolid(lx, top - 1, lz)) --top;
      }
      skyTop[(pz + H) * W + (px + H)] = top;
    }
  std::vector<uint8_t> opaque(W * W * W);
  std::vector<int8_t> level(W * W * W, -1);
  auto index = [&](int px, int py, int pz) { return ((pz + H) * W + (px + H)) * W + (py + H); };
  std::vector<std::array<int, 3>> queue;
  for (int pz = -H; pz < 32 + H; ++pz)
    for (int px = -H; px < 32 + H; ++px) {
      int lx, lz;
      const Column *c = columnAt(px, pz, lx, lz);
      for (int py = -H; py < 32 + H; ++py) {
        const int wy = chunkY * 32 + py;
        const bool solid = c ? c->isSolid(lx, wy, lz) : wy < borderAt(px, pz);
        opaque[index(px, py, pz)] = solid;
        if (!solid && wy >= skyTop[(pz + H) * W + (px + H)]) {
          level[index(px, py, pz)] = 15;
          queue.push_back({px, py, pz});
        }
      }
    }
  static const int o[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (size_t head = 0; head < queue.size(); ++head) {
    const auto [px, py, pz] = queue[head];
    const int next = level[index(px, py, pz)] - 1;
    if (next <= 0) continue;
    for (const auto &d : o) {
      const int nx = px + d[0], ny = py + d[1], nz = pz + d[2];
      if (nx < -H || nx >= 32 + H || ny < -H || ny >= 32 + H || nz < -H || nz >= 32 + H) continue;
      const int i = index(nx, ny, nz);
      if (opaque[i] || level[i] >= next) continue;
      level[i] = int8_t(next);
      queue.push_back({nx, ny, nz});
    }
  }
  std::vector<uint8_t> core(34 * 34 * 34);
  for (int z = -1; z <= 32; ++z)
    for (int x = -1; x <= 32; ++x)
      for (int y = -1; y <= 32; ++y) core[((z + 1) * 34 + (x + 1)) * 34 + (y + 1)] = uint8_t(std::max<int>(level[index(x, y, z)], 0));
  return core;
}

/**
 * Smooth skylight of one face written plainly (vanilla's blend): per corner (AO's order), the rounded average of four
 * samples: the cell in front of the face, the two side cells toward the corner, and the diagonal cell (the first
 * side again when both sides are opaque). Opaque cells read 0, and every 0 sample counts as the front's level.
 *
 * @param light bruteSkyLight of the chunk.
 * @param n Normal index 0-5 (+x, -x, +y, -y, +z, -z).
 * @return 4 bits per corner, corner k in bits 4k..4k+3.
 */
uint32_t bruteCornerLight(const Brute &brute, const std::vector<uint8_t> &light, int x, int y, int z, int n) {
  static const int u[6][3] = {{0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
  static const int v[6][3] = {{0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
  static const int o[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  const int fx = x + o[n][0], fy = y + o[n][1], fz = z + o[n][2]; // cell in front of the face
  // Whether the cell at offset (su, sv) in the face plane is opaque, and its level (0 if opaque).
  auto opaqueAt = [&](int su, int sv) {
    return brute.solid(fx + su * u[n][0] + sv * v[n][0], fy + su * u[n][1] + sv * v[n][1],
                       fz + su * u[n][2] + sv * v[n][2]);
  };
  auto level = [&](int su, int sv) {
    if (opaqueAt(su, sv)) return 0;
    const int px = fx + su * u[n][0] + sv * v[n][0], py = fy + su * u[n][1] + sv * v[n][1],
              pz = fz + su * u[n][2] + sv * v[n][2];
    return int(light[((pz + 1) * 34 + (px + 1)) * 34 + (py + 1)]);
  };
  const int corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  const int center = level(0, 0);
  uint32_t packed = 0;
  for (int k = 0; k < 4; ++k) {
    const int a = level(corners[k][0], 0), b = level(0, corners[k][1]);
    const int c = opaqueAt(corners[k][0], 0) && opaqueAt(0, corners[k][1]) ? a : level(corners[k][0], corners[k][1]);
    int sum = center;
    for (const int l : {a, b, c}) sum += l == 0 ? center : l;
    packed |= uint32_t((sum + 2) / 4) << (4 * k);
  }
  return packed;
}

/**
 * Checks one chunk's mesh against Brute: cluster bookkeeping, every quad cell's AO, block and corner skylight
 * (bruteCornerLight), and that every face Brute expects is covered exactly once.
 *
 * @param only If set, limits the AO / face checks to chunk-local cells it accepts.
 */
void checkChunk(const Neighborhood &nb, int32_t chunkY, const ChunkMesh &mesh,
                const std::function<bool(int, int, int)> &only = {}) {
  const Column &column = nb.center();
  const VoxelChunk *voxels = column.chunk(chunkY);
  Brute brute{nb, chunkY};
  const std::vector<uint8_t> light = mesh.quads.empty() ? std::vector<uint8_t>{} : bruteSkyLight(nb, chunkY);
  static const int o[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}; // per normal
  std::map<std::tuple<int, int, int, int>, int> covered;
  // Opaque clusters' quads first, then the translucent (ocean) clusters' quads.
  uint32_t clusterQuads = 0;
  // Adds each cluster's quad count to clusterQuads, checking that clusters are contiguous.
  auto sumClusters = [&](const std::array<QuadCluster, kClustersPerChunk> &clusters) {
    for (const QuadCluster &cluster : clusters) {
      const uint32_t count = (cluster.counts01 & 0xffff) + (cluster.counts01 >> 16) + (cluster.counts23 & 0xffff) +
                             (cluster.counts23 >> 16) + (cluster.counts45 & 0xffff) + (cluster.counts45 >> 16);
      CHECK(count == 0 || cluster.base == clusterQuads, "cluster base %u != %u", cluster.base, clusterQuads);
      clusterQuads += count;
    }
  };
  sumClusters(mesh.clusters);
  const uint32_t opaqueQuads = clusterQuads;
  sumClusters(mesh.translucentClusters);
  CHECK(clusterQuads == mesh.quads.size(), "cluster total %u != %zu", clusterQuads, mesh.quads.size());
  for (size_t quadIndex = 0; quadIndex < mesh.quads.size(); ++quadIndex) {
    const uint64_t packed = mesh.quads[quadIndex];
    // Low word: x, y, z (5 bits each), normal (3), ao (8), length - 1 (5). High word: block (10 bits), width - 1 (3 bits) at 26.
    const uint32_t p = uint32_t(packed), meta = uint32_t(packed >> 32);
    const int x = p & 31, y = (p >> 5) & 31, z = (p >> 10) & 31, n = (p >> 15) & 7;
    const uint32_t ao = (p >> 18) & 0xff, length = ((p >> 26) & 31) + 1, width = ((meta >> 26) & 7) + 1;
    const uint32_t block = meta & 0x3ff;
    const uint32_t cornerLight = (meta >> 10) & 0xffff;
    CHECK((block == kOcean) == (quadIndex >= opaqueQuads), "quad %zu (block %u) in the wrong cluster set", quadIndex, block);
    for (uint32_t a = 0; a < length; ++a) {   // along the greedy length
      for (uint32_t b = 0; b < width; ++b) {  // along the greedy width
        int cellX = x, cellY = y, cellZ = z;
        if (n == 2) {
          cellX += a, cellZ -= b;
        } else if (n == 3) {
          cellX += a, cellZ += b;
        } else if (n < 2) {
          cellY += a, cellZ += b;
        } else {
          cellY += a, cellX += b;
        }
        CHECK(cellX >= 0 && cellX < 32 && cellY >= 0 && cellY < 32 && cellZ >= 0 && cellZ < 32, "quad cell out of chunk");
        if (cellX < 0 || cellX >= 32 || cellY < 0 || cellY >= 32 || cellZ < 0 || cellZ >= 32) continue;
        ++covered[{cellX, cellY, cellZ, n}];
        if (only && !only(cellX, cellY, cellZ)) continue;
        CHECK(brute.ao(cellX, cellY, cellZ, n) == ao, "ao mismatch at %d %d %d n%d: %x vs %x (chunk %d %d %d)", cellX, cellY, cellZ,
              n, brute.ao(cellX, cellY, cellZ, n), ao, column.cx, chunkY, column.cz);
        CHECK(voxels->ids[VoxelChunk::index(cellX, cellY, cellZ)] == block, "block mismatch");
        const uint32_t expected = bruteCornerLight(brute, light, cellX, cellY, cellZ, n);
        CHECK(expected == cornerLight, "corner light mismatch at %d %d %d n%d: brute %04x, mesh %04x (chunk %d %d %d)",
              cellX, cellY, cellZ, n, expected, cornerLight, column.cx, chunkY, column.cz);
        gShadowedFaces += cornerLight != 0xffff;
        gDarkFaces += cornerLight == 0;
      }
    }
  }
  for (int z = 0; z < 32; ++z)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        for (int n = 0; n < 6; ++n) {
          if (only && !only(x, y, z)) continue;
          const bool face = brute.face(x, y, z, o[n][0], o[n][1], o[n][2]);
          const auto it = covered.find({x, y, z, n});
          const int count = it == covered.end() ? 0 : it->second;
          CHECK(count == (face ? 1 : 0), "face %d %d %d n%d covered %d times, expected %d (chunk %d %d %d)", x, y, z,
                n, count, face ? 1 : 0, column.cx, chunkY, column.cz);
        }
}

/** Checks a column's block ids and solid / ocean masks against the fill rule written plainly. */
void checkFill(const Column &column) {
  for (size_t i = 0; i < column.chunks.size(); ++i) {
    const int32_t chunkY = column.firstChunkY + int32_t(i);
    if (column.chunks[i] == nullptr) {
      // Unstored slot: only allowed for chunks that are entirely ocean (no faces inside).
      for (int z = 0; z < 32; ++z)
        for (int x = 0; x < 32; ++x)
          CHECK(column.height(x, z) <= chunkY * 32 && (chunkY + 1) * 32 <= world::kSeaLevel,
                "unstored chunk %d is not fully ocean", chunkY);
      continue;
    }
    for (int z = 0; z < 32; ++z)
      for (int x = 0; x < 32; ++x) {
        const int h = column.height(x, z);
        const float dither = column.dither[z * 32 + x];
        for (int y = 0; y < 32; ++y) {
          const int wy = chunkY * 32 + y;
          uint16_t expected = wy < world::kSeaLevel ? kOcean : kAir;
          if (wy < h) {
            int t[3];
            bool near = false;
            for (int k = 0; k < 3; ++k) {
              t[k] = world::kTransitionHeights[k] + int(std::round(dither * world::kTransitionDither[k]));
              near |= wy >= t[k] - 1 && wy < t[k] + 1;
            }
            const int my = near ? h : wy;
            int layer = 0;
            for (int k = 0; k < 3; ++k) layer += my >= t[k];
            const int depth = h - 1 - wy;
            expected = depth == 0 ? kIds.top[layer] : depth <= 3 ? kIds.filler[layer] : kIds.deep[layer];
          }
          CHECK(column.chunks[i]->ids[VoxelChunk::index(x, y, z)] == expected, "fill mismatch");
          CHECK(((column.chunks[i]->solid[z * 32 + x] >> y) & 1u) == (wy < h ? 1u : 0u), "solid mask mismatch");
          CHECK(((column.chunks[i]->ocean[z * 32 + x] >> y) & 1u) == (wy >= h && wy < world::kSeaLevel ? 1u : 0u),
                "ocean mask mismatch");
        }
      }
  }
}

/** Monotonic time in seconds. */
double nowSeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main() {
  std::printf("SIMD: %s\n", ENGINE_SIMD_AVX2 ? "AVX2+FMA" : "scalar");

  /** Height checksum over a fixed grid: AVX2 and scalar builds must print the same value. */
  uint64_t checksum = 1469598103934665603ull;
  double minH = 1e9, maxH = -1e9;
  for (int32_t i = 0; i < 4096; ++i) {
    int32_t xs[8], zs[8];
    float h[8];
    for (int l = 0; l < 8; ++l) {
      xs[l] = (i * 8 + l) * 977 - 2000000;
      zs[l] = (i * 8 + l) * 613 - 1000000;
    }
    terrain::sampleHeights8(xs, zs, h);
    for (float v : h) {
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      checksum = (checksum ^ bits) * 1099511628211ull;
      minH = std::min(minH, double(v));
      maxH = std::max(maxH, double(v));
    }
  }
  std::printf("height checksum %016llx (range %.1f .. %.1f)\n", (unsigned long long)checksum, minH, maxH);

  /** Land floor: wherever the continent layer says land (>= kCoastBand), the surface stays above sea level. */
  {
    size_t land = 0, ocean = 0;
    float lowestLand = 1e9f;
    for (int32_t i = 0; i < 20000; ++i) {
      int32_t xs[8], zs[8];
      float h[8], c[8];
      for (int l = 0; l < 8; ++l) {
        xs[l] = (i * 8 + l) * 389 % 400000 - 200000;
        zs[l] = (i * 8 + l) * 7919 % 400000 - 200000;
      }
      terrain::sampleHeights8(xs, zs, h);
      terrain::sampleContinent8(xs, zs, c);
      for (int l = 0; l < 8; ++l) {
        if (c[l] >= terrain::kCoastBand) {
          ++land;
          lowestLand = std::min(lowestLand, h[l]);
          CHECK(h[l] >= 1.0f, "land at %d %d below the floor: %.2f (continent %.1f)", xs[l], zs[l], h[l], c[l]);
        } else if (c[l] < 0.0f) {
          ++ocean;
        }
      }
    }
    std::printf("land floor: %zu land / %zu ocean samples, lowest land %.2f\n", land, ocean, lowestLand);
  }

  /** Mesher vs brute force on a spread of columns (plains, coasts, mountains), with and without neighbors. */
  std::vector<std::array<int32_t, 2>> samples = {{0, 0},      {3, -7},     {40, 12},    {-120, 75},
                                                 {300, -260}, {-900, 410}, {1500, 1500}, {77, 333}};
  // Plus ocean columns: a deep one (unstored fully-ocean chunks between seabed and surface) and a coast.
  {
    bool deep = false, coast = false;
    for (int32_t i = 0; i < 4000 && !(deep && coast); ++i) {
      const int32_t cx = i * 7 - 9000, cz = i * 3 - 5000; // chunk column
      float lo = 1e9f, hi = -1e9f;
      for (const int32_t d : {0, 31}) {
        for (const int32_t e : {0, 31}) {
          const float h = terrain::sampleHeight(cx * 32 + d, cz * 32 + e);
          lo = std::min(lo, h);
          hi = std::max(hi, h);
        }
      }
      if (!deep && hi < -70.0f) {
        samples.push_back({cx, cz});
        deep = true;
      } else if (!coast && lo < -2.0f && hi > 2.0f) {
        samples.push_back({cx, cz});
        coast = true;
      }
    }
    CHECK(deep && coast, "no deep ocean / coast sample found");
  }
  size_t chunksChecked = 0, quads = 0;
  for (const auto &s : samples) {
    std::shared_ptr<Column> cols[3][3];
    for (int dz = -1; dz <= 1; ++dz)
      for (int dx = -1; dx <= 1; ++dx) cols[dz + 1][dx + 1] = generateColumn(s[0] + dx, s[1] + dz, kIds);
    checkFill(*cols[1][1]);
    for (int withNeighbors = 0; withNeighbors < 2; ++withNeighbors) {
      Neighborhood nb;
      for (int dz = 0; dz < 3; ++dz)
        for (int dx = 0; dx < 3; ++dx)
          nb.columns[dz][dx] = (withNeighbors || (dz == 1 && dx == 1)) ? cols[dz][dx].get() : nullptr;
      for (size_t i = 0; i < cols[1][1]->chunks.size(); ++i) {
        ChunkMesh mesh;
        const int32_t chunkY = cols[1][1]->firstChunkY + int32_t(i);
        meshChunk(nb, chunkY, mesh);
        checkChunk(nb, chunkY, mesh);
        ++chunksChecked;
        quads += mesh.quads.size();
      }
    }
  }
  std::printf("mesher: %zu chunks checked, %zu quads\n", chunksChecked, quads);

  /**
   * Edits through CpuWorld: a shaft broken down a column corner (x = 31 | 32, z = -1 | 0) across chunk borders and
   * below the stored range, a half unload / reload, then placed pillar, glass and leaves. Meshes around the edits
   * must match brute force after each step.
   */
  {
    CpuWorld world(kIds, 4);
    std::map<std::tuple<int32_t, int32_t, int32_t>, ChunkMesh> meshes;
    int32_t top = 0;
    int32_t regionLow = 0, regionHigh = 0; // world y range checkAround compares against brute force
    int32_t areaXHigh = 34;                // east edge (world x) of the area checkAround compares
    // Updates the world around `center` until its square is loaded, collecting mesh events into `meshes`.
    auto pump = [&](ChunkXZ center) {
      for (int i = 0; i < 20000 && !(world.squareLoaded(center) && world.readyCenter() == center); ++i) {
        world.update(center, {});
        for (ColumnMeshEvent &e : world.takeMeshEvents()) {
          if (!e.partial) {
            std::erase_if(meshes, [&](const auto &m) { return std::get<0>(m.first) == e.cx && std::get<2>(m.first) == e.cz; });
          }
          for (ChunkMesh &m : e.chunks) meshes[{e.cx, m.cy, e.cz}] = std::move(m);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      world.update(center, {});
      for (ColumnMeshEvent &e : world.takeMeshEvents()) {
        for (ChunkMesh &m : e.chunks) meshes[{e.cx, m.cy, e.cz}] = std::move(m);
      }
    };
    // checkChunk on every chunk around the shaft, limited to cells near it; prints a summary line.
    auto checkAround = [&](const char *when) {
      const int before = failures;
      size_t checked = 0;
      for (int32_t cz = -2; cz <= 1; ++cz)
        for (int32_t cx = -1; cx <= 2; ++cx) {
          std::shared_ptr<const Column> keep[3][3];
          Neighborhood nb;
          for (int dz = -1; dz <= 1; ++dz)
            for (int dx = -1; dx <= 1; ++dx) {
              keep[dz + 1][dx + 1] = world.column(cx + dx, cz + dz);
              nb.columns[dz + 1][dx + 1] = keep[dz + 1][dx + 1].get();
            }
          CHECK(nb.columns[1][1] != nullptr, "column %d %d not loaded", cx, cz);
          if (!nb.columns[1][1]) continue;
          const Column &c = *nb.columns[1][1];
          for (int32_t cy = c.firstChunkY - 2; cy < c.firstChunkY + int32_t(c.chunks.size()) + 1; ++cy) {
            const auto it = meshes.find({cx, cy, cz});
            // Near the shaft only: borders elsewhere may have been meshed before a neighbor loaded (border heights).
            const auto nearShaft = [&](int x, int y, int z) {
              const int32_t wx = cx * 32 + x, wy = cy * 32 + y, wz = cz * 32 + z;
              return wx >= 29 && wx <= areaXHigh && wz >= -3 && wz <= 2 && wy >= regionLow && wy <= regionHigh;
            };
            checkChunk(nb, cy, it != meshes.end() ? it->second : ChunkMesh{}, nearShaft);
            ++checked;
          }
        }
      std::printf("edits (%s): %zu chunks checked%s\n", when, checked, failures == before ? "" : " FAILED");
    };

    pump({0, 0});
    top = world.column(0, -1)->height(31, 31);
    regionLow = top - 83;
    regionHigh = top + 5;
    const int32_t firstBefore = world.column(0, -1)->firstChunkY;
    int broken = 0;
    for (int32_t y = top + 2; y > top - 80; --y)
      for (const int32_t x : {31, 32})
        for (const int32_t z : {-1, 0})
          if (world.raycast({x + 0.5f, y + 0.5f, z + 0.5f}, {0.0f, -1.0f, 0.0f}, 0.4f)) broken += world.breakBlock(x, y, z);
    CHECK(broken > 200, "only %d blocks broken", broken);
    CHECK(!world.breakBlock(31, top - 10, -1), "broke air");
    pump({0, 0});
    std::printf("edits: shaft y %d..%d; column 0 -1 stored chunks from %d, now %d\n", top - 79, top + 2,
                firstBefore, world.column(0, -1)->firstChunkY);
    CHECK(world.column(0, -1)->firstChunkY < firstBefore, "shaft did not reach below the stored chunks");
    checkAround("after breaking");
    bool open = true;
    for (int32_t y = top - 79; y < top; ++y) open = open && !world.isSolid(31, y, -1) && !world.isSolid(32, y, 0);
    CHECK(open, "shaft not open");

    // Unload only the x <= 31 side (center cx 9 keeps cx 1..17), then come back: that side regenerates unedited,
    // while the loaded x = 32 side keeps its half of the shaft, whose walls must reappear on the new side.
    pump({9, 0});
    CHECK(!world.columnLoaded(0, 0) && !world.columnLoaded(0, -1), "x <= 31 side still loaded");
    CHECK(world.columnLoaded(1, 0) && world.columnLoaded(1, -1), "x >= 32 side unloaded");
    pump({0, 0});
    bool regenerated = true;
    open = true;
    for (int32_t y = top - 79; y < top - 2; ++y) {
      regenerated = regenerated && world.isSolid(31, y, -1) && world.isSolid(31, y, 0);
      open = open && !world.isSolid(32, y, -1) && !world.isSolid(32, y, 0);
    }
    CHECK(regenerated, "unloaded side not regenerated");
    CHECK(open, "loaded side lost its edits");
    checkAround("after reload");

    // Block placing on top of the same corner: an opaque pillar from the surface past the stored chunks (new chunks
    // above), non-opaque blocks beside it (bit 9 ids), same-id glass touching (shared face culled) and two different
    // non-opaque ids touching (both faces shown).
    constexpr uint16_t kStone = 2, kGlass = blocks::kNonOpaqueBit, kLeaves = blocks::kNonOpaqueBit + 1;
    int32_t base = std::numeric_limits<int32_t>::min();
    for (int32_t x = 30; x <= 33; ++x)
      for (int32_t z = -2; z <= 1; ++z) {
        int32_t y = top + 120;
        while (y > top - 200 && !world.isCollidable(x, y, z)) --y;
        base = std::max(base, y + 1);
      }
    const int32_t storedTop = (world.column(0, -1)->firstChunkY + int32_t(world.column(0, -1)->chunks.size())) * 32;
    const int32_t pillarTop = std::max(base + 40, storedTop + 8);
    bool placed = true;
    for (int32_t y = base; y < pillarTop; ++y) placed = placed && world.placeBlock(31, y, -1, kStone);
    for (int32_t y = base; y < base + 6; ++y) {
      placed = placed && world.placeBlock(32, y, -1, kGlass) && world.placeBlock(33, y, -1, kGlass) &&
               world.placeBlock(32, y, 0, kLeaves);
    }
    CHECK(placed, "a placement into air failed");
    CHECK(!world.placeBlock(31, base, -1, kStone), "placed into an occupied cell");
    CHECK(!world.placeBlock(31, pillarTop, -1, kAir), "placed air");
    CHECK(world.isCollidable(32, base, -1) && !world.isSolid(32, base, -1), "glass should collide but not be opaque");
    CHECK(world.blockId(32, base, 0) == kLeaves, "leaves id not stored");
    const auto hit = world.raycast({32.5f, base + 10.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, 20.0f);
    CHECK(hit && hit->block == (BlockXYZ{32, base + 5, -1}), "raycast should stop at the top glass block");
    pump({0, 0});
    regionLow = base - 3;
    regionHigh = pillarTop + 3;
    checkAround("after placing");
    CHECK(world.breakBlock(32, base + 2, -1), "could not break glass");
    CHECK(world.blockId(32, base + 2, -1) == kAir, "broken glass not air");
    pump({0, 0});
    checkAround("after breaking placed glass");

    // Skylight: cap one shaft cell 10 below the surface (light still pours down the other) and dig a 2-tall tunnel
    // east from beside it, so light fades along the tunnel to level 0. Every quad around must match the BFS reference.
    const size_t shadowedBefore = gShadowedFaces, darkBefore = gDarkFaces;
    const int32_t capY = top - 10, tunnelY = top - 40;
    bool capped = true;
    // Only the x = 32 half of the shaft is still open (the reload regenerated x = 31); (32, -1) stays open.
    capped = world.placeBlock(32, capY, 0, kStone);
    CHECK(capped, "could not cap the shaft");
    for (int32_t x = 33; x <= 50; ++x)
      for (const int32_t y : {tunnelY, tunnelY + 1}) world.breakBlock(x, y, 0);
    CHECK(!world.isSolid(50, tunnelY, 0), "tunnel not dug");
    pump({0, 0});
    regionLow = top - 50;
    regionHigh = capY + 2;
    areaXHigh = 53;
    checkAround("skylight under a capped shaft");
    std::printf("skylight: %zu shadowed faces checked, %zu at level 0\n", gShadowedFaces - shadowedBefore,
                gDarkFaces - darkBefore);
    CHECK(gShadowedFaces - gDarkFaces - (shadowedBefore - darkBefore) > 50, "no partly lit faces in the tunnel");
    CHECK(gDarkFaces - darkBefore > 10, "the tunnel's far end is not dark");
  }

  /**
   * Ocean edits: placing into ocean, breaking it back (ocean flows in on the next flow step), the raycast passing
   * through water, and a dry tunnel under the seabed flooding one block per step once a shaft connects it.
   */
  {
    const std::array<int32_t, 2> coast = samples.back();
    const ChunkXZ center{coast[0], coast[1]};
    CpuWorld world(kIds, 4);
    std::map<std::tuple<int32_t, int32_t, int32_t>, ChunkMesh> meshes;
    // Same as the edit test's pump, around the coast column.
    auto pump = [&] {
      for (int i = 0; i < 20000 && !(world.squareLoaded(center) && world.readyCenter() == center); ++i) {
        world.update(center, {});
        for (ColumnMeshEvent &e : world.takeMeshEvents())
          for (ChunkMesh &m : e.chunks) meshes[{e.cx, m.cy, e.cz}] = std::move(m);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      world.update(center, {});
      for (ColumnMeshEvent &e : world.takeMeshEvents())
        for (ChunkMesh &m : e.chunks) meshes[{e.cx, m.cy, e.cz}] = std::move(m);
    };
    pump();
    // Terrain height at a world block column.
    auto heightAt = [&](int32_t x, int32_t z) {
      const int32_t cx = floorDiv(x, 32), cz = floorDiv(z, 32);
      return world.column(cx, cz)->height(x - cx * 32, z - cz * 32);
    };
    // A seabed cell no lower than its 8 neighbors (so a shaft down from it is walled by terrain), with room for a
    // tunnel toward +x.
    int32_t sx = 0, sz = 0, sh = 0; // seabed cell x / z and its height
    bool found = false;
    for (int32_t z = center[1] * 32 - 24; z < center[1] * 32 + 56 && !found; ++z)
      for (int32_t x = center[0] * 32 - 24; x < center[0] * 32 + 48 && !found; ++x) {
        const int32_t h = heightAt(x, z);
        if (h > -6) continue;
        bool walled = true;
        for (int dz = -1; dz <= 1; ++dz)
          for (int dx = -1; dx <= 1; ++dx) walled = walled && heightAt(x + dx, z + dz) >= h;
        if (walled) {
          sx = x, sz = z, sh = h;
          found = true;
        }
      }
    CHECK(found, "no seabed cell for the ocean edit test");
    if (found) {
      double t = 1000.0;
      // One fluid tick, one second later.
      auto step = [&] {
        t += 1.0;
        world.tickFluids(t);
      };
      const int32_t surfaceY = world::kSeaLevel - 1;
      CHECK(world.blockId(sx, surfaceY, sz) == kOcean && world.isInfiniteSource(sx, surfaceY, sz), "no ocean at surface");
      CHECK(!world.isCollidable(sx, surfaceY, sz) && !world.isSolid(sx, surfaceY, sz), "ocean collides");
      const auto hit = world.raycast({sx + 0.5f, 5.5f, sz + 0.5f}, {0.0f, -1.0f, 0.0f}, 200.0f);
      CHECK(hit && hit->block == (BlockXYZ{sx, sh - 1, sz}), "raycast did not pass through the ocean to the seabed");
      CHECK(!world.breakBlock(sx, surfaceY, sz), "broke ocean");
      CHECK(world.placeBlock(sx, surfaceY, sz, 2), "could not place into ocean");
      CHECK(world.blockId(sx, surfaceY, sz) == 2 && world.isCollidable(sx, surfaceY, sz), "placed block missing");
      CHECK(world.breakBlock(sx, surfaceY, sz), "could not break the placed block");
      CHECK(world.blockId(sx, surfaceY, sz) == kAir, "broken cell not air before the flow step");
      step();
      CHECK(world.blockId(sx, surfaceY, sz) == kOcean, "ocean did not flow back in");
      for (int i = 0; i < 100; ++i) CHECK(world.isInfiniteSource(sx, surfaceY, sz), "ocean source changed");

      // Tunnel 3 below the lowest surrounding seabed, 6 long toward +x, dug first (stays dry), then a shaft from
      // the seabed down to it: the ocean floods shaft then tunnel, exactly one cell per step.
      int32_t lowest = sh;
      for (int32_t x = sx - 1; x <= sx + 6; ++x)
        for (int32_t z = sz - 1; z <= sz + 1; ++z) lowest = std::min(lowest, heightAt(x, z));
      const int32_t depth = lowest - 3;
      std::vector<BlockXYZ> path;
      for (int32_t y = sh - 1; y > depth; --y) path.push_back({sx, y, sz});
      for (int32_t x = sx; x < sx + 6; ++x) path.push_back({x, depth, sz});
      for (int32_t x = sx; x < sx + 6; ++x) CHECK(world.breakBlock(x, depth, sz), "tunnel dig failed");
      for (int i = 0; i < 3; ++i) step();
      bool dry = true;
      for (int32_t x = sx; x < sx + 6; ++x) dry = dry && world.blockId(x, depth, sz) == kAir;
      CHECK(dry, "unconnected tunnel flooded");
      for (int32_t y = sh - 1; y > depth; --y) CHECK(world.breakBlock(sx, y, sz), "shaft dig failed");
      for (size_t k = 1; k <= path.size(); ++k) {
        step();
        CHECK(world.blockId(path[k - 1][0], path[k - 1][1], path[k - 1][2]) == kOcean, "path cell %zu not flooded at step %zu", k - 1, k);
        if (k < path.size())
          CHECK(world.blockId(path[k][0], path[k][1], path[k][2]) == kAir, "path cell %zu flooded early at step %zu", k, k);
      }
      std::printf("ocean edits: seabed %d at %d %d, flooded %zu-cell shaft + tunnel one per step\n", sh, sx, sz,
                  path.size());

      // Meshes around the edits against brute force.
      pump();
      const int before = failures;
      for (int32_t cz = floorDiv(sz - 2, 32); cz <= floorDiv(sz + 2, 32); ++cz)
        for (int32_t cx = floorDiv(sx - 2, 32); cx <= floorDiv(sx + 8, 32); ++cx) {
          std::shared_ptr<const Column> keep[3][3];
          Neighborhood nb;
          for (int dz = -1; dz <= 1; ++dz)
            for (int dx = -1; dx <= 1; ++dx) {
              keep[dz + 1][dx + 1] = world.column(cx + dx, cz + dz);
              nb.columns[dz + 1][dx + 1] = keep[dz + 1][dx + 1].get();
            }
          for (int32_t cy = floorDiv(depth - 2, 32); cy <= floorDiv(surfaceY + 1, 32); ++cy) {
            const auto it = meshes.find({cx, cy, cz});
            const auto near = [&](int x, int y, int z) {
              const int32_t wx = cx * 32 + x, wy = cy * 32 + y, wz = cz * 32 + z;
              return wx >= sx - 2 && wx <= sx + 7 && wz >= sz - 2 && wz <= sz + 2 && wy >= depth - 2 && wy <= surfaceY + 1;
            };
            checkChunk(nb, cy, it != meshes.end() ? it->second : ChunkMesh{}, near);
          }
        }
      std::printf("ocean edits: meshes%s\n", failures == before ? " match" : " FAILED");
    }
  }

  /** Registry: opaque blocks numbered from 0, non-opaque ones from 512 (bit 9), each in file order. */
  {
    const char *path = "terrain_tests_blocks.txt";
    {
      std::ofstream file(path);
      file << "[a]\nall = a.png\n[glass]\nall = g.png\nopaque = false\n[b]\nall = b.png\n[leaves]\nopaque = false\n"
              "all = l.png\n";
    }
    const blocks::Registry registry = blocks::loadRegistry(path, "");
    CHECK(registry.idOf("a") == 0 && registry.idOf("b") == 1, "opaque ids not 0, 1");
    CHECK(registry.idOf("glass") == 512 && registry.idOf("leaves") == 513, "non-opaque ids not 512, 513");
    CHECK(registry.maxId() == 513, "maxId %u", registry.maxId());
    CHECK(blocks::isOpaque(registry.idOf("b")) && !blocks::isOpaque(registry.idOf("glass")) && !blocks::isOpaque(kAir),
          "isOpaque");
    {
      std::ofstream file(path);
      file << "[a]\nall = a.png\nopaque = maybe\n";
    }
    bool threw = false;
    try {
      blocks::loadRegistry(path, "");
    } catch (const std::exception &) {
      threw = true;
    }
    CHECK(threw, "opaque = maybe accepted");
    std::remove(path);
    std::printf("registry: id ranges checked\n");
  }

  /** Trees (shaders/vegetation_shared.hlsl, shared with the GPU): bounds, far-LOD agreement, stamped voxels. */
  {
    using namespace vegetation;
    // Every tree block lies inside its region (chunk column) and within TREE_MAX_ABOVE of the ground, for every hash
    // outcome.
    uint32_t state = 12345u;
    auto next = [&] { return state = state * 1664525u + 1013904223u; };
    uint64_t treeChecksum = 1469598103934665603ull;
    size_t accepted = 0;
    for (int i = 0; i < 20000; ++i) {
      const int2 cell(int32_t(next() >> 12) - (1 << 19), int32_t(next() >> 12) - (1 << 19)); // region
      TreeInstance tree = treeCandidate(cell, next() % uint32_t(TREE_SLOTS));
      accepted += tree.trunkHeight != 0;
      treeChecksum = (treeChecksum ^ uint32_t(tree.trunkX) ^ (uint64_t(uint32_t(tree.trunkHeight)) << 32)) * 1099511628211ull;
      // Check every style at its shortest and tallest, rejected candidates included.
      const uint32_t style = uint32_t(i) % 4u;
      tree.shape = (tree.shape & ~(3u << 12u)) | (style << 12u);
      tree.trunkHeight = treeTrunkHeightFor(style, (i / 4) % 2 ? 255u : 0u);
      tree.ground = int32_t(next() % 200u) + 3;
      const int32_t x0 = cell.x * 32, z0 = cell.y * 32;
      for (int32_t z = z0 - 4; z < z0 + 36; ++z)
        for (int32_t x = x0 - 4; x < x0 + 36; ++x)
          for (int32_t y = tree.ground - 2; y < tree.ground + TREE_MAX_ABOVE + 2; ++y) {
            if (treeBlockAt(tree, x, y, z) == TREE_NONE) continue;
            CHECK(x >= x0 && x < x0 + 32 && z >= z0 && z < z0 + 32, "tree block %d %d outside region %d %d", x, z,
                  cell.x, cell.y);
            CHECK(y >= tree.ground && y < tree.ground + TREE_MAX_ABOVE, "tree block y %d outside [%d, +%d)", y,
                  tree.ground, TREE_MAX_ABOVE);
          }
    }
    std::printf("trees: candidate checksum %016llx (%zu of 20000 pass the density)\n",
                (unsigned long long)treeChecksum, accepted);

    // Far LODs show the same trees: a scale-16 / scale-64 column (computeTreeColumn's LOD 4+ rule: candidates whose
    // trunk is inside, density, slope and band from the column's centre) is filled where the exact trees have a trunk.
    {
      constexpr int32_t kRegions = 32; // 1024 x 1024 blocks
      constexpr int32_t kSize = kRegions * 32;
      const int32_t baseX = -6016, baseZ = 1984; // region-aligned (multiples of 64)
      std::vector<TreeInstance> exactTrees;
      for (int32_t rz = 0; rz < kRegions; ++rz)
        for (int32_t rx = 0; rx < kRegions; ++rx)
          for (uint32_t slot = 0; slot < uint32_t(TREE_SLOTS); slot += 8) {
            TreeInstance trees8[8];
            int32_t tx[8], tz[8];
            float th[8], gx[8], gz[8];
            for (uint32_t l = 0; l < 8; ++l) {
              trees8[l] = treeCandidate(int2(baseX / 32 + rx, baseZ / 32 + rz), slot + l);
              tx[l] = trees8[l].trunkX;
              tz[l] = trees8[l].trunkZ;
            }
            terrain::sampleHeightsGrad8(tx, tz, th, gx, gz);
            for (uint32_t l = 0; l < 8; ++l) {
              trees8[l].ground = int32_t(std::floor(th[l]));
              if (trees8[l].trunkHeight != 0 && treeGroundOk(trees8[l].ground, treeSteepness(gx[l], gz[l]), trees8[l].shape))
                exactTrees.push_back(trees8[l]);
            }
          }
      std::vector<uint8_t> canopy(size_t(kSize) * kSize, 0);
      for (const TreeInstance &t : exactTrees)
        for (int32_t dz = -TREE_RADIUS; dz <= TREE_RADIUS; ++dz)
          for (int32_t dx = -TREE_RADIUS; dx <= TREE_RADIUS; ++dx)
            canopy[size_t(t.trunkZ - baseZ + dz) * kSize + size_t(t.trunkX - baseX + dx)] = 1;
      double lod0 = 0;
      for (const uint8_t c : canopy) lod0 += c;
      lod0 /= double(canopy.size());
      std::printf("trees: %zu exact trees, LOD 0 canopy cover %.4f", exactTrees.size(), lod0);
      std::vector<uint8_t> filled8;
      for (const int scale : {8, 16, 64}) {
        const int32_t perSide = kSize / scale;
        std::vector<uint8_t> exact(size_t(perSide) * perSide, 0), filled(size_t(perSide) * perSide, 0);
        for (const TreeInstance &t : exactTrees)
          if (treeShowsAtScale(t.trunkX, t.trunkZ, scale))
            exact[size_t((t.trunkZ - baseZ) / scale) * perSide + size_t((t.trunkX - baseX) / scale)] = 1;
        size_t both = 0, exactOnly = 0, farOnly = 0, count = 0;
        for (int32_t cz = 0; cz < perSide; ++cz)
          for (int32_t cx = 0; cx < perSide; ++cx) {
            const int32_t columnX = baseX + cx * scale, columnZ = baseZ + cz * scale;
            bool f = exact[size_t(cz) * perSide + size_t(cx)] != 0; // scale 8: the staged, exact trees
            if (scale > 8) {
              // computeTreeColumn's LOD 4+ rule: position, roll, density, then slope / band at the column's centre.
              int32_t xs[8], zs[8];
              float h[8], gx[8], gz[8];
              for (int l = 0; l < 8; ++l) {
                xs[l] = columnX + scale / 2;
                zs[l] = columnZ + scale / 2;
              }
              terrain::sampleHeightsGrad8(xs, zs, h, gx, gz);
              const int32_t ground = int32_t(std::floor(h[0]));
              const int steepness = treeSteepness(gx[0], gz[0]);
              f = false;
              for (int32_t rz = columnZ >> 5; rz <= (columnZ + scale - 1) >> 5 && !f; ++rz)
                for (int32_t rx = columnX >> 5; rx <= (columnX + scale - 1) >> 5 && !f; ++rx)
                  for (uint32_t slot = 0; slot < uint32_t(TREE_SLOTS) && !f; ++slot) {
                    const TreeInstance t = treeCandidateUnfiltered(int2(rx, rz), slot);
                    if (t.trunkX < columnX || t.trunkX >= columnX + scale || t.trunkZ < columnZ ||
                        t.trunkZ >= columnZ + scale || !treeShowsAtScale(t.trunkX, t.trunkZ, scale))
                      continue;
                    if (int(treeCandidateHash(int2(rx, rz), slot) >> 24u) >= treeChanceAt(t.trunkX, t.trunkZ)) continue;
                    f = ground >= TREE_MIN_GROUND && ground <= TREE_MAX_GROUND &&
                        treeSlopeOk(steepness, treeSlopeRoll(t.shape));
                  }
            }
            filled[size_t(cz) * perSide + size_t(cx)] = f;
            const bool e = exact[size_t(cz) * perSide + size_t(cx)] != 0;
            both += e && f;
            exactOnly += e && !f;
            farOnly += !e && f;
            count += f;
          }
        const double cover = double(count) / double(perSide * perSide);
        std::printf(", scale %d cover %.4f (%zu on exact trunks, %zu not, %zu missed)", scale, cover, both, farOnly,
                    exactOnly);
        CHECK(std::abs(cover - lod0) < 0.3 * lod0, "scale %d cover %.4f vs LOD 0 %.4f", scale, cover, lod0);
        CHECK(double(farOnly) <= 0.1 * double(std::max<size_t>(1, count)), "scale %d: cells off the real trunks", scale);
        if (scale == 8) filled8 = filled;
        if (scale == 16) {
          // Nested: a filled scale-16 column (from an exact trunk) has a filled scale-8 child.
          size_t orphans = 0;
          for (int32_t cz = 0; cz < perSide; ++cz)
            for (int32_t cx = 0; cx < perSide; ++cx) {
              if (!filled[size_t(cz) * perSide + size_t(cx)] || !exact[size_t(cz) * perSide + size_t(cx)]) continue;
              bool child = false;
              for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b) child = child || filled8[size_t(cz * 2 + a) * (kSize / 8) + size_t(cx * 2 + b)];
              orphans += !child;
            }
          CHECK(orphans == 0, "%zu scale-16 cells without a scale-8 child", orphans);
        }
      }
      std::printf("\n");
    }

    // Stamped voxels against the placement rule written plainly, on columns that have trees; then the mesher.
    // Leaves non-opaque (cutout) and opaque (blocks.txt without `opaque = false`).
    size_t treeColumns = 0, treesSeen = 0, treeChunks = 0;
    for (const uint16_t leavesId : {uint16_t(blocks::kNonOpaqueBit + 1), uint16_t(7)}) {
    TerrainBlockIds treeIds = kIds;
    treeIds.log = 6;
    treeIds.leaves = leavesId;
    const size_t columnLimit = treeColumns + 6;
    for (int32_t i = 0; i < 400 && treeColumns < columnLimit; ++i) {
      const int32_t cx = i * 13 - 2600, cz = i * 7 + 900;
      auto column = generateColumn(cx, cz, treeIds);
      size_t trees = 0;
      for (const TreeInstance &t : column->trees) trees += t.trunkHeight != 0;
      if (trees < 2) continue;
      ++treeColumns;
      treesSeen += trees;
      const auto bare = generateColumn(cx, cz, kIds);
      for (int32_t z = 0; z < 32; ++z)
        for (int32_t x = 0; x < 32; ++x) {
          const int32_t wx = cx * 32 + x, wz = cz * 32 + z;
          // Every slot's tree, by the rule written plainly.
          TreeInstance slotTrees[TREE_SLOTS];
          for (uint32_t slot = 0; slot < uint32_t(TREE_SLOTS); ++slot) {
            TreeInstance &tree = slotTrees[slot];
            tree = treeCandidate(int2(cx, cz), slot);
            if (tree.trunkHeight == 0) continue;
            const int32_t lx = tree.trunkX - cx * 32, lz = tree.trunkZ - cz * 32;
            tree.ground = column->height(lx, lz);
            int32_t tx[8], tz[8];
            float th[8], gx[8], gz[8];
            for (int l = 0; l < 8; ++l) {
              tx[l] = tree.trunkX;
              tz[l] = tree.trunkZ;
            }
            terrain::sampleHeightsGrad8(tx, tz, th, gx, gz);
            if (!treeGroundOk(tree.ground, treeSteepness(gx[0], gz[0]), tree.shape)) {
              tree.trunkHeight = 0;
            }
          }
          // Expected tree block at world y in this cell; overlapping trees: log wins over leaves.
          auto blockAt = [&](int32_t wy) {
            uint32_t block = TREE_NONE;
            for (const TreeInstance &tree : slotTrees) {
              const uint32_t b = treeBlockAt(tree, wx, wy, wz);
              block = b == TREE_LOG || block == TREE_LOG ? TREE_LOG : (b != TREE_NONE ? b : block);
            }
            return block;
          };
          for (int32_t wy = column->firstChunkY * 32; wy < (column->firstChunkY + int32_t(column->chunks.size())) * 32;
               ++wy) {
            const uint32_t block = blockAt(wy);
            const bool placed = block != TREE_NONE && wy >= column->height(x, z);
            const uint16_t expected =
                placed ? (block == TREE_LOG ? treeIds.log : treeIds.leaves) : bare->blockId(x, wy, z);
            CHECK(column->blockId(x, wy, z) == expected, "tree fill mismatch at %d %d %d", wx, wy, wz);
            CHECK(column->isSolid(x, wy, z) == (placed ? blocks::isOpaque(expected) : bare->isSolid(x, wy, z)),
                  "tree solid mask mismatch at %d %d %d", wx, wy, wz);
          }
          // Nothing above the stored range: tree chunks must be stored.
          const int32_t top = (column->firstChunkY + int32_t(column->chunks.size())) * 32;
          for (int32_t wy = top; wy < top + TREE_MAX_ABOVE; ++wy) {
            CHECK(blockAt(wy) == TREE_NONE, "tree above stored chunks");
          }
        }
      for (const auto &chunk : column->chunks) {
        if (chunk == nullptr) continue;
        uint16_t nonOpaque = 0;
        for (const uint16_t id : chunk->ids) nonOpaque += id != kAir && id != kOcean && !blocks::isOpaque(id);
        CHECK(chunk->nonOpaqueCount == nonOpaque, "nonOpaqueCount %u != %u", chunk->nonOpaqueCount, nonOpaque);
      }
      std::shared_ptr<Column> cols[3][3];
      for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx) cols[dz + 1][dx + 1] = generateColumn(cx + dx, cz + dz, treeIds);
      for (int withNeighbors = 0; withNeighbors < 2; ++withNeighbors) {
        Neighborhood nb;
        for (int dz = 0; dz < 3; ++dz)
          for (int dx = 0; dx < 3; ++dx)
            nb.columns[dz][dx] = (withNeighbors || (dz == 1 && dx == 1)) ? cols[dz][dx].get() : nullptr;
        for (size_t c = 0; c < cols[1][1]->chunks.size(); ++c) {
          ChunkMesh mesh;
          const int32_t chunkY = cols[1][1]->firstChunkY + int32_t(c);
          meshChunk(nb, chunkY, mesh);
          checkChunk(nb, chunkY, mesh);
          ++treeChunks;
        }
      }
    }
    }
    CHECK(treeColumns >= 6, "only %zu columns with trees found", treeColumns);
    std::printf("trees: %zu columns (%zu trees) checked, %zu chunks meshed\n", treeColumns, treesSeen, treeChunks);
  }

  /** Timings: column generation and meshing. */
  const int columns = 64;
  double t0 = nowSeconds();
  std::vector<std::shared_ptr<Column>> generated;
  for (int i = 0; i < columns; ++i) generated.push_back(generateColumn(i * 5 - 100, i * 3 + 50, kIds));
  double t1 = nowSeconds();
  size_t meshed = 0;
  for (const auto &c : generated) {
    Neighborhood nb;
    nb.columns[1][1] = c.get();
    for (size_t i = 0; i < c->chunks.size(); ++i) {
      ChunkMesh mesh;
      meshChunk(nb, c->firstChunkY + int32_t(i), mesh);
      ++meshed;
    }
  }
  double t2 = nowSeconds();
  std::printf("generate: %.1f us/column; mesh: %.1f us/chunk (%zu chunks)\n", (t1 - t0) * 1e6 / columns,
              (t2 - t1) * 1e6 / double(meshed), meshed);

  std::printf(failures == 0 ? "ALL PASSED\n" : "%d FAILURES\n", failures);
  return failures == 0 ? 0 : 1;
}
