// CPU terrain checks: mesher output vs brute force, a height checksum (compare AVX2 and scalar builds), timings.
#include "../src/block_registry.hpp"
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

struct Brute {
  const Neighborhood &nb;
  int32_t chunkY;
  bool solid(int32_t x, int32_t y, int32_t z) const { // chunk-local, x/z in -1..32
    const int32_t dx = x < 0 ? -1 : (x >= 32 ? 1 : 0);
    const int32_t dz = z < 0 ? -1 : (z >= 32 ? 1 : 0);
    const Column *c = nb.columns[dz + 1][dx + 1];
    const int32_t worldY = chunkY * 32 + y;
    if (c) {
      return c->isSolid(x - dx * 32, worldY, z - dz * 32);
    }
    return worldY < nb.center().borderHeight(x, z);
  }
  uint16_t id(int32_t x, int32_t y, int32_t z) const { // chunk-local, x/z in -1..32
    const int32_t dx = x < 0 ? -1 : (x >= 32 ? 1 : 0);
    const int32_t dz = z < 0 ? -1 : (z >= 32 ? 1 : 0);
    const Column *c = nb.columns[dz + 1][dx + 1];
    const int32_t worldY = chunkY * 32 + y;
    if (c) {
      return c->blockId(x - dx * 32, worldY, z - dz * 32);
    }
    return worldY < nb.center().borderHeight(x, z) ? kBuriedTerrain : nb.center().emptyCell(worldY);
  }
  // Opaque cells show a face toward any non-opaque cell; non-opaque blocks (not air) toward any non-opaque cell
  // holding a different block.
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
  uint32_t vertexAo(bool a, bool b, bool c) const { return a && b ? 3u : uint32_t(a) + uint32_t(b) + uint32_t(c); }
  uint32_t ao(int32_t x, int32_t y, int32_t z, uint32_t n) const {
    if (id(x, y, z) == kOcean) {
      return 0u; // water is lit flat
    }
    static const int u[6][3] = {{0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    static const int v[6][3] = {{0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
    static const int o[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const int ox = x + o[n][0], oy = y + o[n][1], oz = z + o[n][2];
    auto s = [&](int su, int sv) {
      return solid(ox + su * u[n][0] + sv * v[n][0], oy + su * u[n][1] + sv * v[n][1], oz + su * u[n][2] + sv * v[n][2]);
    };
    return vertexAo(s(-1, 0), s(0, -1), s(-1, -1)) | (vertexAo(s(1, 0), s(0, -1), s(1, -1)) << 2) |
           (vertexAo(s(1, 0), s(0, 1), s(1, 1)) << 4) | (vertexAo(s(-1, 0), s(0, 1), s(-1, 1)) << 6);
  }
};

// `only`, if set, limits the AO / face checks to chunk-local cells it accepts.
void checkChunk(const Neighborhood &nb, int32_t chunkY, const ChunkMesh &mesh,
                const std::function<bool(int, int, int)> &only = {}) {
  const Column &column = nb.center();
  const VoxelChunk *voxels = column.chunk(chunkY);
  Brute brute{nb, chunkY};
  static const int o[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  std::map<std::tuple<int, int, int, int>, int> covered;
  // Opaque clusters' quads first, then the translucent (ocean) clusters' quads.
  uint32_t clusterQuads = 0;
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
  for (size_t qi = 0; qi < mesh.quads.size(); ++qi) {
    const uint64_t q = mesh.quads[qi];
    const uint32_t p = uint32_t(q), meta = uint32_t(q >> 32);
    const int x = p & 31, y = (p >> 5) & 31, z = (p >> 10) & 31, n = (p >> 15) & 7;
    const uint32_t ao = (p >> 18) & 0xff, length = ((p >> 26) & 31) + 1, width = ((meta >> 26) & 7) + 1;
    const uint32_t block = meta & 0x3ff;
    CHECK((block == kOcean) == (qi >= opaqueQuads), "quad %zu (block %u) in the wrong cluster set", qi, block);
    for (uint32_t a = 0; a < length; ++a) {
      for (uint32_t b = 0; b < width; ++b) {
        int cx = x, cy = y, cz = z;
        if (n == 2) {
          cx += a, cz -= b;
        } else if (n == 3) {
          cx += a, cz += b;
        } else if (n < 2) {
          cy += a, cz += b;
        } else {
          cy += a, cx += b;
        }
        CHECK(cx >= 0 && cx < 32 && cy >= 0 && cy < 32 && cz >= 0 && cz < 32, "quad cell out of chunk");
        if (cx < 0 || cx >= 32 || cy < 0 || cy >= 32 || cz < 0 || cz >= 32) continue;
        ++covered[{cx, cy, cz, n}];
        if (only && !only(cx, cy, cz)) continue;
        CHECK(brute.ao(cx, cy, cz, n) == ao, "ao mismatch at %d %d %d n%d: %x vs %x (chunk %d %d %d)", cx, cy, cz,
              n, brute.ao(cx, cy, cz, n), ao, column.cx, chunkY, column.cz);
        CHECK(voxels->ids[VoxelChunk::index(cx, cy, cz)] == block, "block mismatch");
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

// Block ids against the fill rule written plainly.
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

double nowSeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main() {
  std::printf("SIMD: %s\n", ENGINE_SIMD_AVX2 ? "AVX2+FMA" : "scalar");

  // Height checksum over a fixed grid: AVX2 and scalar builds must print the same value.
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

  // Land floor: wherever the continent layer says land (>= kCoastBand), the surface stays above sea level.
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

  // Mesher vs brute force on a spread of columns (plains, coasts, mountains), with and without neighbors.
  std::vector<std::array<int32_t, 2>> samples = {{0, 0},      {3, -7},     {40, 12},    {-120, 75},
                                                 {300, -260}, {-900, 410}, {1500, 1500}, {77, 333}};
  // Plus ocean columns: a deep one (unstored fully-ocean chunks between seabed and surface) and a coast.
  {
    bool deep = false, coast = false;
    for (int32_t i = 0; i < 4000 && !(deep && coast); ++i) {
      const int32_t cx = i * 7 - 9000, cz = i * 3 - 5000;
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

  // Block breaking through CpuWorld: a shaft down a column corner (x = 31 | 32, z = -1 | 0), crossing chunk borders
  // and going below the stored chunk range; every mesh around it must match brute force.
  {
    CpuWorld world(kIds, 4);
    std::map<std::tuple<int32_t, int32_t, int32_t>, ChunkMesh> meshes;
    int32_t top = 0;
    int32_t regionLow = 0, regionHigh = 0; // world y range checkAround compares against brute force
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
              return wx >= 29 && wx <= 34 && wz >= -3 && wz <= 2 && wy >= regionLow && wy <= regionHigh;
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
  }

  // Ocean edits: placing into ocean, breaking it back (ocean flows in on the next flow step), the raycast passing
  // through water, and a dry tunnel under the seabed flooding one block per step once a shaft connects it.
  {
    const std::array<int32_t, 2> coast = samples.back();
    const ChunkXZ center{coast[0], coast[1]};
    CpuWorld world(kIds, 4);
    std::map<std::tuple<int32_t, int32_t, int32_t>, ChunkMesh> meshes;
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
    auto heightAt = [&](int32_t x, int32_t z) {
      const int32_t cx = floorDiv(x, 32), cz = floorDiv(z, 32);
      return world.column(cx, cz)->height(x - cx * 32, z - cz * 32);
    };
    // A seabed cell no lower than its 8 neighbors (so a shaft down from it is walled by terrain), with room for a
    // tunnel toward +x.
    int32_t sx = 0, sz = 0, sh = 0;
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

  // Registry: opaque blocks numbered from 0, non-opaque ones from 512 (bit 9), each in file order.
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

  // Trees (shaders/vegetation_shared.hlsl, shared with the GPU).
  {
    using namespace vegetation;
    // Every tree block lies inside its 8x8 cell and within TREE_MAX_ABOVE of the ground, for every hash outcome.
    uint32_t state = 12345u;
    auto next = [&] { return state = state * 1664525u + 1013904223u; };
    uint64_t treeChecksum = 1469598103934665603ull;
    size_t accepted = 0;
    for (int i = 0; i < 20000; ++i) {
      const int2 cell(int32_t(next() >> 8) - (1 << 23), int32_t(next() >> 8) - (1 << 23));
      TreeInstance tree = treeCandidate(cell);
      accepted += tree.trunkHeight != 0;
      treeChecksum = (treeChecksum ^ uint32_t(tree.trunkX) ^ (uint64_t(uint32_t(tree.trunkHeight)) << 32)) * 1099511628211ull;
      tree.trunkHeight = 4 + i % 3; // check the shape of rejected candidates too
      tree.ground = int32_t(next() % 200u) + 3;
      const int32_t x0 = cell.x * 8, z0 = cell.y * 8;
      for (int32_t z = z0 - 4; z < z0 + 12; ++z)
        for (int32_t x = x0 - 4; x < x0 + 12; ++x)
          for (int32_t y = tree.ground - 2; y < tree.ground + TREE_MAX_ABOVE + 2; ++y) {
            if (treeBlockAt(tree, x, y, z) == TREE_NONE) continue;
            CHECK(x >= x0 && x < x0 + 8 && z >= z0 && z < z0 + 8, "tree block %d %d outside cell %d %d", x, z, cell.x,
                  cell.y);
            CHECK(y >= tree.ground && y < tree.ground + TREE_MAX_ABOVE, "tree block y %d outside [%d, +%d)", y,
                  tree.ground, TREE_MAX_ABOVE);
          }
    }
    std::printf("trees: candidate checksum %016llx (%zu of 20000 pass the density)\n",
                (unsigned long long)treeChecksum, accepted);

    // LOD 1-2 table: every variant has a trunk capped by leaves (no log cell with open sky above it).
    {
      const std::vector<uint32_t> lodTable = buildTreeLodTable();
      for (const int scale : {2, 4}) {
        const int perCell = 8 / scale;
        for (uint32_t offset = 0; offset < 16u; ++offset) {
          for (uint32_t heightIndex = 0; heightIndex < 3u; ++heightIndex) {
            const int trunkX = (TREE_RADIUS + int(offset & 3u)) / scale, trunkZ = (TREE_RADIUS + int(offset >> 2u)) / scale;
            int trunks = 0;
            for (int cz = 0; cz < perCell; ++cz) {
              for (int cx = 0; cx < perCell; ++cx) {
                const uint32_t entry = lodTable[treeLodTableIndex(scale, offset, heightIndex, uint32_t(cx), uint32_t(cz))];
                const uint32_t logs = (entry >> 4u) & 0xfu;
                if (logs == 0u) continue;
                ++trunks;
                CHECK(cx == trunkX && cz == trunkZ, "LOD table: log outside the trunk column (scale %d)", scale);
                CHECK(((entry >> logs) & 1u) != 0u, "LOD table: trunk not capped by leaves (scale %d, offset %u, h %u)",
                      scale, offset, 4u + heightIndex);
              }
            }
            CHECK(trunks == 1, "LOD table: %d trunk columns (scale %d)", trunks, scale);
          }
        }
      }
      std::printf("trees: LOD table checked\n");
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
          TreeInstance tree = treeCandidate(int2(wx >> 3, wz >> 3));
          if (tree.trunkHeight != 0) {
            const int32_t lx = tree.trunkX - cx * 32, lz = tree.trunkZ - cz * 32;
            tree.ground = column->height(lx, lz);
            if (!treeGroundOk(tree.ground, column->height(lx - 1, lz), column->height(lx + 1, lz),
                              column->height(lx, lz - 1), column->height(lx, lz + 1))) {
              tree.trunkHeight = 0;
            }
          }
          for (int32_t wy = column->firstChunkY * 32; wy < (column->firstChunkY + int32_t(column->chunks.size())) * 32;
               ++wy) {
            const uint32_t block = treeBlockAt(tree, wx, wy, wz);
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
            CHECK(tree.trunkHeight == 0 || treeBlockAt(tree, wx, wy, wz) == TREE_NONE, "tree above stored chunks");
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

  // Timings.
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
