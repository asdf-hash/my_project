// CPU terrain checks: mesher output vs brute force, a height checksum (compare AVX2 and scalar builds), timings.
#include "../src/cpu_mesher.hpp"
#include "../src/cpu_world.hpp"
#include "../src/simd.hpp"
#include "../src/terrain_cpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

const TerrainBlockIds kIds{{0, 1, 2, 3}, {4, 5, 2, 3}, {4, 2, 2, 3}};

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
  uint32_t vertexAo(bool a, bool b, bool c) const { return a && b ? 3u : uint32_t(a) + uint32_t(b) + uint32_t(c); }
  uint32_t ao(int32_t x, int32_t y, int32_t z, uint32_t n) const {
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
  uint32_t clusterQuads = 0;
  for (const QuadCluster &cluster : mesh.clusters) {
    CHECK(cluster.base == clusterQuads, "cluster base %u != %u", cluster.base, clusterQuads);
    clusterQuads += (cluster.counts01 & 0xffff) + (cluster.counts01 >> 16) + (cluster.counts23 & 0xffff) +
                    (cluster.counts23 >> 16) + (cluster.counts45 & 0xffff) + (cluster.counts45 >> 16);
  }
  CHECK(clusterQuads == mesh.quads.size(), "cluster total %u != %zu", clusterQuads, mesh.quads.size());
  for (const uint64_t q : mesh.quads) {
    const uint32_t p = uint32_t(q), meta = uint32_t(q >> 32);
    const int x = p & 31, y = (p >> 5) & 31, z = (p >> 10) & 31, n = (p >> 15) & 7;
    const uint32_t ao = (p >> 18) & 0xff, length = ((p >> 26) & 31) + 1, width = ((meta >> 26) & 7) + 1;
    const uint32_t block = meta & 0x3ff;
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
          const bool face = brute.solid(x, y, z) && !brute.solid(x + o[n][0], y + o[n][1], z + o[n][2]);
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
    for (int z = 0; z < 32; ++z)
      for (int x = 0; x < 32; ++x) {
        const int h = column.height(x, z);
        const float dither = column.dither[z * 32 + x];
        for (int y = 0; y < 32; ++y) {
          const int wy = chunkY * 32 + y;
          uint16_t expected = kAir;
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

  // Mesher vs brute force on a spread of columns (plains, coasts, mountains), with and without neighbors.
  const int32_t samples[][2] = {{0, 0}, {3, -7}, {40, 12}, {-120, 75}, {300, -260}, {-900, 410}, {1500, 1500}, {77, 333}};
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
              return wx >= 29 && wx <= 34 && wz >= -3 && wz <= 2 && wy >= top - 83 && wy <= top + 5;
            };
            checkChunk(nb, cy, it != meshes.end() ? it->second : ChunkMesh{}, nearShaft);
            ++checked;
          }
        }
      std::printf("edits (%s): %zu chunks checked%s\n", when, checked, failures == before ? "" : " FAILED");
    };

    pump({0, 0});
    top = world.column(0, -1)->height(31, 31);
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
