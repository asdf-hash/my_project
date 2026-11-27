#include "cpu_world.hpp"

#include "cpu_light.hpp"
#include "cpu_mesher.hpp"
#include "simd.hpp"
#include "terrain_cpu.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace cpuworld {
namespace {

using simd::I8;
using simd::M8;

constexpr int32_t N = CHUNK_SIZE; // short alias: chunk edge, in blocks

/**
 * Write the column's trees into chunk `chunkY`, only into cells the terrain leaves empty; where trees overlap, log
 * wins over leaves. Same rule as treeCellBlock in chunk_compute.slang.
 *
 * @param chunkY Chunk y, in chunks.
 * @param voxels The chunk's voxels, already filled from the heightmap.
 */
void stampTrees(const Column &column, int32_t chunkY, const TerrainBlockIds &ids, VoxelChunk &voxels) {
  if (ids.log == kAir || ids.leaves == kAir) {
    return;
  }
  const int32_t bottom = chunkY * N;
  const int32_t originX = column.cx * N;
  const int32_t originZ = column.cz * N;
  for (const vegetation::TreeInstance &tree : column.trees) {
    if (tree.trunkHeight == 0) {
      continue;
    }
    const int32_t yBegin = std::max(tree.ground, bottom);
    const int32_t yEnd = std::min(tree.ground + vegetation::TREE_MAX_ABOVE, bottom + N);
    for (int32_t y = yBegin; y < yEnd; ++y) {
      for (int32_t dz = -vegetation::TREE_RADIUS; dz <= vegetation::TREE_RADIUS; ++dz) {
        for (int32_t dx = -vegetation::TREE_RADIUS; dx <= vegetation::TREE_RADIUS; ++dx) {
          const int32_t wx = tree.trunkX + dx; // world block x
          const int32_t wz = tree.trunkZ + dz;
          const uint32_t block = vegetation::treeBlockAt(tree, wx, y, wz);
          const int32_t lx = wx - originX; // column-local x
          const int32_t lz = wz - originZ;
          if (block == vegetation::TREE_NONE || y < column.height(lx, lz)) {
            continue;
          }
          const int32_t ly = y - bottom; // chunk-local y
          uint16_t &cell = voxels.ids[VoxelChunk::index(lx, ly, lz)];
          if (block == vegetation::TREE_LEAVES && cell == ids.log) {
            continue;
          }
          const bool wasNonOpaque = cell != kAir && cell != column.oceanId && (cell & blocks::kNonOpaqueBit) != 0;
          cell = block == vegetation::TREE_LOG ? ids.log : ids.leaves;
          const uint32_t bit = 1u << ly;
          voxels.ocean[lz * N + lx] &= ~bit;
          if (blocks::isOpaque(cell)) {
            voxels.solid[lz * N + lx] |= bit;
            voxels.nonOpaqueCount = static_cast<uint16_t>(voxels.nonOpaqueCount - (wasNonOpaque ? 1 : 0));
          } else if (!wasNonOpaque) {
            voxels.nonOpaqueCount = static_cast<uint16_t>(voxels.nonOpaqueCount + 1);
          }
        }
      }
    }
  }
}

/**
 * Fill one voxel chunk from the heightmap, then stamp trees. CPU twin of faceBlockType / selectBlockType at LOD 0:
 * cells within one block of a dithered layer boundary take their material from the column top, not their own height.
 *
 * @param chunkY Chunk y, in chunks.
 * @param voxels Overwritten completely.
 */
void fillChunk(const Column &column, int32_t chunkY, const TerrainBlockIds &ids, VoxelChunk &voxels) {
  const int32_t bottom = chunkY * N;
  // Solid masks, 8 columns at a time: (1 << clamp(h - bottom, 0, 32)) - 1, where a shift of 32 gives 0 - 1 = ~0.
  for (int32_t i = 0; i < N * N; i += 8) {
    const I8 h = I8::load(&column.heights[i]); // column tops
    const I8 filled = simd::max(simd::min(h - I8(bottom), I8(N)), I8(0));
    (simd::sllv(I8(1), filled) - I8(1)).store(reinterpret_cast<int32_t *>(&voxels.solid[i]));
  }
  const uint32_t sea = column.oceanId == kAir ? 0u : Column::heightMask(world::kSeaLevel, chunkY);
  for (int32_t i = 0; i < N * N; ++i) {
    voxels.ocean[i] = sea & ~voxels.solid[i];
  }

  /** Per lane, table[layer] for layer in 0..3. */
  auto layerSelect = [](I8 layer, const std::array<uint16_t, 4> &table) {
    return simd::select(layer == I8(0), I8(table[0]),
                        simd::select(layer == I8(1), I8(table[1]),
                                     simd::select(layer == I8(2), I8(table[2]), I8(table[3]))));
  };

  for (int32_t z = 0; z < N; ++z) {
    for (int32_t x = 0; x < N; ++x) {
      const int32_t height = column.height(x, z);
      uint16_t *cells = &voxels.ids[VoxelChunk::index(x, 0, z)];
      if (height <= bottom) {
        for (int32_t y = 0; y < N; ++y) {
          cells[y] = column.emptyCell(bottom + y);
        }
        continue;
      }
      const float dither = column.dither[z * N + x];
      std::array<int32_t, 3> transition{};
      for (size_t t = 0; t < transition.size(); ++t) {
        transition[t] =
            world::kTransitionHeights[t] + static_cast<int32_t>(std::round(dither * world::kTransitionDither[t]));
      }
      const I8 h(height);
      for (int32_t y0 = 0; y0 < N; y0 += 8) {
        const I8 y = I8(bottom + y0) + I8::iota(); // world y of 8 consecutive cells
        // Cells next to a boundary use the column top for their material (the near test covers t - 1 and t).
        M8 near = y == I8(transition[0] - 1);
        for (const int32_t t : transition) {
          near = near | (y == I8(t - 1)) | (y == I8(t));
        }
        const I8 materialY = simd::select(near, h, y);
        I8 layer(0);
        for (const int32_t t : transition) {
          layer = layer - simd::maskBits(materialY > I8(t - 1)); // += materialY >= t
        }
        const I8 depth = h - I8(1) - y; // 0 at the top block, negative above the terrain
        I8 id = simd::select(depth == I8(0), layerSelect(layer, ids.top),
                             simd::select(I8(world::kFillerDepth) > depth - I8(1), layerSelect(layer, ids.filler),
                                          layerSelect(layer, ids.deep)));
        id = simd::select(depth > I8(-1), id,
                          simd::select(I8(world::kSeaLevel) > y, I8(column.oceanId), I8(kAir)));
        alignas(32) int32_t lanes[8];
        id.store(lanes);
        for (int i = 0; i < 8; ++i) {
          cells[y0 + i] = static_cast<uint16_t>(lanes[i]);
        }
      }
    }
  }
  stampTrees(column, chunkY, ids, voxels);
}

/**
 * Give the column voxel storage for chunk `chunkY`, filled from the heightmap so it matches what the unstored chunk
 * stood for. Slots added between it and the stored range stay null.
 *
 * @param chunkY Chunk y, in chunks.
 */
void materialize(Column &column, int32_t chunkY, const TerrainBlockIds &ids) {
  if (column.chunks.empty()) {
    column.firstChunkY = chunkY;
    column.chunks.emplace_back();
  }
  while (chunkY < column.firstChunkY) {
    column.chunks.insert(column.chunks.begin(), nullptr);
    --column.firstChunkY;
  }
  while (chunkY >= column.firstChunkY + static_cast<int32_t>(column.chunks.size())) {
    column.chunks.emplace_back();
  }
  std::unique_ptr<VoxelChunk> &slot = column.chunks[static_cast<size_t>(chunkY - column.firstChunkY)];
  if (slot == nullptr) {
    slot = std::make_unique_for_overwrite<VoxelChunk>(); // fillChunk writes every cell: skip the zero-fill
    fillChunk(column, chunkY, ids, *slot);
  }
}

/** Deep copy of a column, voxel chunks included. */
std::shared_ptr<Column> cloneColumn(const Column &source) {
  auto column = std::make_shared<Column>();
  column->cx = source.cx;
  column->cz = source.cz;
  column->rawHeights = source.rawHeights;
  column->heights = source.heights;
  column->dither = source.dither;
  column->trees = source.trees;
  column->oceanId = source.oceanId;
  column->firstChunkY = source.firstChunkY;
  column->edited = source.edited;
  column->skyTop = source.skyTop;
  column->shadowLow = source.shadowLow;
  column->shadowHigh = source.shadowHigh;
  column->chunks.reserve(source.chunks.size());
  for (const auto &chunk : source.chunks) {
    column->chunks.push_back(chunk != nullptr ? std::make_unique<VoxelChunk>(*chunk) : nullptr);
  }
  return column;
}

} // namespace

std::shared_ptr<Column> generateColumn(int32_t cx, int32_t cz, const TerrainBlockIds &ids) {
  auto column = std::make_shared<Column>();
  column->cx = cx;
  column->cz = cz;
  column->oceanId = ids.ocean;
  const int32_t originX = cx * N;
  const int32_t originZ = cz * N;

  // Padded heights, 8 cells per call; the last batch is partly unused (lanes clamped to the last cell).
  constexpr int32_t padCells = kPad * kPad;
  alignas(32) int32_t xs[8], zs[8];
  alignas(32) float out[8];
  for (int32_t i = 0; i < padCells; i += 8) {
    for (int32_t l = 0; l < 8; ++l) { // l: SIMD lane
      const int32_t cell = std::min(i + l, padCells - 1);
      xs[l] = originX + cell % kPad - 1;
      zs[l] = originZ + cell / kPad - 1;
    }
    terrain::sampleHeights8(xs, zs, out);
    for (int32_t l = 0; l < 8 && i + l < padCells; ++l) {
      column->rawHeights[static_cast<size_t>(i + l)] = out[l];
    }
  }
  int32_t interiorMax = world::kMinTerrainHeight;
  int32_t interiorMin = world::kMaxTerrainHeight;
  int32_t paddedMin = world::kMaxTerrainHeight;
  for (int32_t lz = -1; lz <= N; ++lz) {
    for (int32_t lx = -1; lx <= N; ++lx) {
      const bool interior = lx >= 0 && lx < N && lz >= 0 && lz < N;
      if (interior) {
        const int32_t h = std::max(static_cast<int32_t>(std::floor(column->rawHeights[Column::padIndex(lx, lz)])),
                                   world::kMinTerrainHeight);
        column->heights[static_cast<size_t>(lz * N + lx)] = h;
        interiorMax = std::max(interiorMax, h);
        interiorMin = std::min(interiorMin, h);
        paddedMin = std::min(paddedMin, h);
      } else {
        paddedMin = std::min(paddedMin, column->borderHeight(lx, lz));
      }
    }
  }

  // Dither per column (chunk_compute.slang samples at cell origin + scale / 2, i.e. the column itself at LOD 0).
  for (int32_t i = 0; i < N * N; i += 8) {
    for (int32_t l = 0; l < 8; ++l) {
      xs[l] = originX + (i + l) % N;
      zs[l] = originZ + (i + l) / N;
    }
    terrain::sampleDither8(xs, zs, &column->dither[static_cast<size_t>(i)]);
  }

  // Trees: TREE_SLOTS candidates (interior trunks), steepness from the noise gradient at each trunk (8 per call).
  int32_t treeTop = world::kMinTerrainHeight;
  if (ids.log != kAir && ids.leaves != kAir) {
    static_assert(vegetation::TREE_SLOTS % 8 == 0);
    for (int32_t i = 0; i < vegetation::TREE_SLOTS; i += 8) {
      int32_t tx[8], tz[8]; // trunk world x / z per lane
      float heights[8], gradX[8], gradZ[8];
      for (int32_t l = 0; l < 8; ++l) {
        column->trees[static_cast<size_t>(i + l)] =
            vegetation::treeCandidate(vegetation::int2(cx, cz), static_cast<uint32_t>(i + l));
        tx[l] = column->trees[static_cast<size_t>(i + l)].trunkX;
        tz[l] = column->trees[static_cast<size_t>(i + l)].trunkZ;
      }
      terrain::sampleHeightsGrad8(tx, tz, heights, gradX, gradZ);
      for (int32_t l = 0; l < 8; ++l) {
        vegetation::TreeInstance &tree = column->trees[static_cast<size_t>(i + l)];
        if (tree.trunkHeight == 0) {
          continue;
        }
        const int32_t lx = tree.trunkX - originX;
        const int32_t lz = tree.trunkZ - originZ;
        tree.ground = column->height(lx, lz);
        if (vegetation::treeGroundOk(tree.ground, vegetation::treeSteepness(gradX[l], gradZ[l]), tree.shape)) {
          treeTop = std::max(treeTop, tree.ground + vegetation::TREE_MAX_ABOVE);
        } else {
          tree.trunkHeight = 0;
        }
      }
    }
  }

  // Faces lie on cells in [paddedMin - 1, interiorMax) (a solid cell needs a non-solid neighbor), plus the ocean
  // surface just below sea level. Fully ocean chunks between the two hold no faces and stay null.
  column->firstChunkY = floorDiv(paddedMin - 1, N);
  const int32_t terrainLastChunkY = floorDiv(interiorMax - 1, N);
  const bool hasSurface = ids.ocean != kAir && interiorMin < world::kSeaLevel;
  const int32_t seaChunkY = floorDiv(world::kSeaLevel - 1, N);
  // Tree chunks above the terrain are stored too, since null slots must follow the heightmap.
  const int32_t treeLastChunkY = floorDiv(treeTop - 1, N);
  const int32_t lastChunkY =
      std::max(hasSurface ? std::max(terrainLastChunkY, seaChunkY) : terrainLastChunkY, treeLastChunkY);
  for (int32_t chunkY = column->firstChunkY; chunkY <= lastChunkY; ++chunkY) {
    if (chunkY > terrainLastChunkY && chunkY != seaChunkY && chunkY > treeLastChunkY) {
      column->chunks.emplace_back();
      continue;
    }
    auto voxels = std::make_unique_for_overwrite<VoxelChunk>(); // fillChunk writes every cell: skip the zero-fill
    fillChunk(*column, chunkY, ids, *voxels);
    column->chunks.push_back(std::move(voxels));
  }
  for (int32_t lz = 0; lz < N; ++lz) {
    for (int32_t lx = 0; lx < N; ++lx) {
      column->rescanSkyTop(lx, lz);
    }
  }
  return column;
}

CpuWorld::CpuWorld(const TerrainBlockIds &ids, unsigned workerCount) : ids_(ids) {
  if (workerCount == 0) {
    const unsigned hardware = std::thread::hardware_concurrency();
    workerCount = hardware > 3 ? hardware - 2 : 1;
  }
  for (unsigned i = 0; i < workerCount; ++i) {
    workers_.emplace_back([this](std::stop_token stop) { workerMain(stop); });
  }
}

CpuWorld::~CpuWorld() {
  for (auto &worker : workers_) {
    worker.request_stop();
  }
  wake_.notify_all();
  workers_.clear(); // joins
}

bool CpuWorld::retained(int32_t cx, int32_t cz) const {
  for (const auto &center : {desiredCenter_, readyCenter_}) {
    if (center && inSquare(*center, cx, cz)) {
      return true;
    }
  }
  for (const ChunkXZ &center : pinnedCenters_) {
    if (inSquare(center, cx, cz)) {
      return true;
    }
  }
  return false;
}

void CpuWorld::update(ChunkXZ desiredCenter, const std::vector<ChunkXZ> &pinnedCenters) {
  pinnedCenters_ = pinnedCenters;
  const bool moved = desiredCenter_ != desiredCenter;
  desiredCenter_ = desiredCenter;

  std::vector<Result> results;
  {
    std::lock_guard lock(mutex_);
    results.swap(results_);
  }
  std::vector<ChunkXZ> arrived;
  for (Result &result : results) {
    const int32_t cx = result.column->cx;
    const int32_t cz = result.column->cz;
    if (!retained(cx, cz)) {
      std::lock_guard lock(mutex_);
      generated_.erase(key(cx, cz));
      continue;
    }
    columns_[key(cx, cz)] = result.column;
    events_.push_back({cx, cz, false, false, std::move(result.meshes)});
    arrived.push_back({cx, cz});
  }
  if (!arrived.empty()) {
    EditBatch batch;
    for (const ChunkXZ &arrivedXZ : arrived) {
      refreshEditedNeighbors(arrivedXZ[0], arrivedXZ[1], batch);
    }
    publish(batch);
  }

  for (auto it = columns_.begin(); it != columns_.end();) {
    const Column &column = *it->second;
    if (retained(column.cx, column.cz)) {
      ++it;
      continue;
    }
    events_.push_back({column.cx, column.cz, true, false, {}});
    {
      std::lock_guard lock(mutex_);
      generated_.erase(it->first);
    }
    it = columns_.erase(it);
  }

  if (moved) {
    // Requeue the desired square's missing columns, nearest first; cancel running ones nothing retains.
    std::vector<ChunkXZ> missing;
    for (int32_t dz = -world::kCpuChunkRadius; dz <= world::kCpuChunkRadius; ++dz) {
      for (int32_t dx = -world::kCpuChunkRadius; dx <= world::kCpuChunkRadius; ++dx) {
        const int32_t cx = desiredCenter[0] + dx;
        const int32_t cz = desiredCenter[1] + dz;
        if (!columns_.contains(key(cx, cz))) {
          missing.push_back({cx, cz});
        }
      }
    }
    std::sort(missing.begin(), missing.end(), [&](const ChunkXZ &a, const ChunkXZ &b) {
      const auto distance = [&](const ChunkXZ &columnXZ) {
        const int32_t dx = columnXZ[0] - desiredCenter[0];
        const int32_t dz = columnXZ[1] - desiredCenter[1];
        return dx * dx + dz * dz;
      };
      return distance(a) < distance(b);
    });
    std::lock_guard lock(mutex_);
    // Drop the old queue (requeued from `missing`). What is left in wanted_ is running; a worker erases its own entry
    // when it finishes, keeping the result only if still wanted.
    for (const ChunkXZ &queuedXZ : queue_) {
      wanted_.erase(key(queuedXZ[0], queuedXZ[1]));
    }
    queue_.clear();
    for (auto &[runningKey, wanted] : wanted_) {
      const ChunkXZ runningXZ{static_cast<int32_t>(runningKey >> 32u),
                              static_cast<int32_t>(static_cast<uint32_t>(runningKey))};
      wanted = retained(runningXZ[0], runningXZ[1]);
    }
    for (const ChunkXZ &missingXZ : missing) {
      auto [it, inserted] = wanted_.try_emplace(key(missingXZ[0], missingXZ[1]), true);
      if (inserted) {
        queue_.push_back(missingXZ);
      } else {
        it->second = true; // still running; its result is wanted again
      }
    }
    wake_.notify_all();
  }

  // A square is ready once all its columns are loaded. A moving player's desired square is rarely complete, so also
  // accept the complete square nearest to it on the line from the current ready center, letting the CPU square
  // follow a fast player.
  if (readyCenter_ != desiredCenter_) {
    const auto complete = [&](const ChunkXZ &center) { return squareLoaded(center); };
    if (!readyCenter_) {
      if (complete(desiredCenter)) {
        readyCenter_ = desiredCenter;
      }
    } else {
      const int32_t steps = std::max(std::abs(desiredCenter[0] - (*readyCenter_)[0]),
                                     std::abs(desiredCenter[1] - (*readyCenter_)[1]));
      for (int32_t step = steps; step > 0; --step) { // desired first, back toward the ready center
        const float t = static_cast<float>(step) / static_cast<float>(steps);
        const ChunkXZ candidate{
            (*readyCenter_)[0] + static_cast<int32_t>(std::lround(t * static_cast<float>(desiredCenter[0] - (*readyCenter_)[0]))),
            (*readyCenter_)[1] + static_cast<int32_t>(std::lround(t * static_cast<float>(desiredCenter[1] - (*readyCenter_)[1])))};
        if (complete(candidate)) {
          readyCenter_ = candidate;
          break;
        }
      }
    }
  }
}

std::vector<ColumnMeshEvent> CpuWorld::takeMeshEvents() {
  std::vector<ColumnMeshEvent> events;
  events.swap(events_);
  return events;
}

bool CpuWorld::isSolid(int32_t x, int32_t y, int32_t z) const {
  const int32_t cx = floorDiv(x, N);
  const int32_t cz = floorDiv(z, N);
  const auto it = columns_.find(key(cx, cz));
  if (it == columns_.end()) {
    return false;
  }
  return it->second->isSolid(x - cx * N, y, z - cz * N);
}

uint16_t CpuWorld::blockId(int32_t x, int32_t y, int32_t z) const {
  const int32_t cx = floorDiv(x, N);
  const int32_t cz = floorDiv(z, N);
  const auto it = columns_.find(key(cx, cz));
  if (it == columns_.end()) {
    return kAir;
  }
  return it->second->blockId(x - cx * N, y, z - cz * N);
}

bool CpuWorld::isCollidable(int32_t x, int32_t y, int32_t z) const {
  const int32_t cx = floorDiv(x, N);
  const int32_t cz = floorDiv(z, N);
  const auto it = columns_.find(key(cx, cz));
  if (it == columns_.end()) {
    return false;
  }
  const Column &column = *it->second;
  const int32_t lx = x - cx * N;
  const int32_t lz = z - cz * N;
  // Opaque cells are in the solid masks; otherwise anything but air and ocean.
  return column.isSolid(lx, y, lz) || !isReplaceable(column.blockId(lx, y, lz));
}

bool CpuWorld::isInfiniteSource(int32_t x, int32_t y, int32_t z) const {
  return ids_.ocean != kAir && blockId(x, y, z) == ids_.ocean;
}

std::optional<BlockHit> CpuWorld::raycast(const std::array<float, 3> &origin, const std::array<float, 3> &direction,
                                          float maxDistance) const {
  const double length = std::sqrt(static_cast<double>(direction[0]) * direction[0] +
                                  static_cast<double>(direction[1]) * direction[1] +
                                  static_cast<double>(direction[2]) * direction[2]);
  if (length == 0.0) {
    return std::nullopt;
  }
  BlockXYZ cell{};
  std::array<int32_t, 3> step{};
  std::array<double, 3> tMax{};   // ray distance at which the next boundary on each axis is crossed
  std::array<double, 3> tDelta{}; // ray distance between boundaries on each axis
  for (int axis = 0; axis < 3; ++axis) {
    const double start = origin[axis];
    const double dir = direction[axis] / length; // normalized direction component
    cell[axis] = static_cast<int32_t>(std::floor(start));
    if (dir > 0.0) {
      step[axis] = 1;
      tDelta[axis] = 1.0 / dir;
      tMax[axis] = (cell[axis] + 1.0 - start) / dir;
    } else if (dir < 0.0) {
      step[axis] = -1;
      tDelta[axis] = -1.0 / dir;
      tMax[axis] = (start - cell[axis]) / -dir;
    } else {
      tDelta[axis] = tMax[axis] = std::numeric_limits<double>::infinity();
    }
  }
  BlockXYZ normal{};
  double t = 0.0;
  while (t <= maxDistance) {
    if (isCollidable(cell[0], cell[1], cell[2])) {
      return BlockHit{cell, normal};
    }
    const int axis = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : (tMax[1] < tMax[2] ? 1 : 2); // nearest boundary
    t = tMax[axis];
    cell[axis] += step[axis];
    tMax[axis] += tDelta[axis];
    normal = {};
    normal[axis] = -step[axis];
  }
  return std::nullopt;
}

bool CpuWorld::breakBlock(int32_t x, int32_t y, int32_t z) {
  const int32_t cx = floorDiv(x, N);
  const int32_t cz = floorDiv(z, N);
  if (!columnLoaded(cx, cz) || !isCollidable(x, y, z)) {
    return false;
  }
  editCell(x, y, z, kAir);
  return true;
}

bool CpuWorld::placeBlock(int32_t x, int32_t y, int32_t z, uint16_t id) {
  if (id == kAir || y < world::kMinTerrainHeight || y >= world::kMaxTerrainHeight) {
    return false;
  }
  if (!columnLoaded(floorDiv(x, N), floorDiv(z, N)) || !isReplaceable(blockId(x, y, z))) {
    return false;
  }
  editCell(x, y, z, id);
  return true;
}

void CpuWorld::editCell(int32_t x, int32_t y, int32_t z, uint16_t id) {
  const int32_t cx = floorDiv(x, N);
  const int32_t cz = floorDiv(z, N);
  EditBatch batch;
  Column &column = editableColumn(cx, cz, batch);
  column.edited = true;
  const int32_t lx = x - cx * N; // lx, lz: column-local block coords
  const int32_t lz = z - cz * N;
  const bool wasOpaque = column.isSolid(lx, y, lz);
  const int32_t oldSkyTop = column.skyTop[lz * N + lx];
  setCell(column, lx, y, lz, id);
  touchCell(x, y, z, batch);
  if (wasOpaque != column.isSolid(lx, y, lz)) {
    const int32_t newSkyTop = column.skyTop[lz * N + lx];
    touchLight(x, z, std::min({y, oldSkyTop, newSkyTop}), std::max({y, oldSkyTop, newSkyTop}), batch);
  }
  publish(batch);
  if (id == kAir && y < world::kSeaLevel && ids_.ocean != kAir) {
    flowCandidates_.push_back({x, y, z});
  }
}

void CpuWorld::tickFluids(double now) {
  if (flowCandidates_.empty()) {
    nextFlowTime_ = now + kOceanFlowSeconds; // the first step after an edit waits a full flow period
    return;
  }
  if (now < nextFlowTime_) {
    return;
  }
  nextFlowTime_ = now + kOceanFlowSeconds;

  std::sort(flowCandidates_.begin(), flowCandidates_.end());
  flowCandidates_.erase(std::unique(flowCandidates_.begin(), flowCandidates_.end()), flowCandidates_.end());
  std::vector<BlockXYZ> candidates;
  if (flowCandidates_.size() > kOceanFlowCellsPerStep) {
    candidates.assign(flowCandidates_.begin(), flowCandidates_.begin() + kOceanFlowCellsPerStep);
    flowCandidates_.erase(flowCandidates_.begin(), flowCandidates_.begin() + kOceanFlowCellsPerStep);
  } else {
    candidates.swap(flowCandidates_);
  }

  // Decide every cell against the pre-step state (columns_ changes only in publish), so ocean advances one block.
  // Water falls first: a cell with air right under it floods that cell instead and waits for a later step, so
  // water reaching a drop fills downward before it spreads.
  static constexpr int32_t kSides[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  std::vector<BlockXYZ> flooded;
  std::vector<BlockXYZ> waiting; // cells that flooded below themselves this step
  for (const BlockXYZ &candidate : candidates) {
    if (candidate[1] >= world::kSeaLevel || !columnLoaded(floorDiv(candidate[0], N), floorDiv(candidate[2], N)) ||
        blockId(candidate[0], candidate[1], candidate[2]) != kAir) {
      continue;
    }
    const bool touchesOcean = std::ranges::any_of(kSides, [&](const auto &side) {
      return blockId(candidate[0] + side[0], candidate[1] + side[1], candidate[2] + side[2]) == ids_.ocean;
    });
    if (!touchesOcean) {
      continue;
    }
    BlockXYZ target = candidate;
    if (target[1] - 1 >= world::kMinTerrainHeight && blockId(target[0], target[1] - 1, target[2]) == kAir) {
      --target[1];
    }
    flooded.push_back(target);
    if (target != candidate) {
      waiting.push_back(candidate);
    }
  }
  if (flooded.empty()) {
    return;
  }
  std::sort(flooded.begin(), flooded.end());
  flooded.erase(std::unique(flooded.begin(), flooded.end()), flooded.end());
  flowCandidates_.insert(flowCandidates_.end(), waiting.begin(), waiting.end());
  EditBatch batch;
  for (const BlockXYZ &cell : flooded) {
    const int32_t cx = floorDiv(cell[0], N);
    const int32_t cz = floorDiv(cell[2], N);
    Column &column = editableColumn(cx, cz, batch);
    column.edited = true;
    setCell(column, cell[0] - cx * N, cell[1], cell[2] - cz * N, ids_.ocean);
    touchCell(cell[0], cell[1], cell[2], batch);
  }
  publish(batch);
  for (const BlockXYZ &cell : flooded) {
    for (const auto &side : kSides) {
      const BlockXYZ neighbor{cell[0] + side[0], cell[1] + side[1], cell[2] + side[2]};
      if (neighbor[1] < world::kSeaLevel && blockId(neighbor[0], neighbor[1], neighbor[2]) == kAir) {
        flowCandidates_.push_back(neighbor);
      }
    }
  }
}

const Column *CpuWorld::loadedColumn(int32_t cx, int32_t cz, const EditBatch &batch) const {
  const uint64_t columnKey = key(cx, cz);
  if (const auto it = batch.copies.find(columnKey); it != batch.copies.end()) {
    return it->second.get();
  }
  const auto it = columns_.find(columnKey);
  return it != columns_.end() ? it->second.get() : nullptr;
}

Column &CpuWorld::editableColumn(int32_t cx, int32_t cz, EditBatch &batch) {
  const uint64_t columnKey = key(cx, cz);
  auto [it, inserted] = batch.copies.try_emplace(columnKey);
  if (inserted) {
    it->second = cloneColumn(*columns_.at(columnKey));
  }
  return *it->second;
}

void CpuWorld::setCell(Column &column, int32_t lx, int32_t y, int32_t lz, uint16_t id) const {
  const int32_t chunkY = floorDiv(y, N);
  materialize(column, chunkY, ids_);
  VoxelChunk &voxels = *column.chunks[static_cast<size_t>(chunkY - column.firstChunkY)];
  const int32_t ly = y - chunkY * N; // chunk-local y
  uint16_t &cell = voxels.ids[VoxelChunk::index(lx, ly, lz)];
  const auto countsAsNonOpaque = [&](uint16_t block) {
    return block != kAir && block != ids_.ocean && !blocks::isOpaque(block);
  };
  voxels.nonOpaqueCount = static_cast<uint16_t>(voxels.nonOpaqueCount - (countsAsNonOpaque(cell) ? 1 : 0) +
                                                (countsAsNonOpaque(id) ? 1 : 0));
  cell = id;
  // Solid masks hold opaque cells only (id bit 9 clear).
  uint32_t &bits = voxels.solid[lz * N + lx];
  const uint32_t oldBits = bits;
  bits = blocks::isOpaque(id) ? bits | (1u << ly) : bits & ~(1u << ly);
  if (bits != oldBits) {
    column.rescanSkyTop(lx, lz);
  }
  uint32_t &oceanBits = voxels.ocean[lz * N + lx];
  oceanBits = id == ids_.ocean && id != kAir ? oceanBits | (1u << ly) : oceanBits & ~(1u << ly);
}

void CpuWorld::touchCell(int32_t x, int32_t y, int32_t z, EditBatch &batch) {
  // Chunks above a column's range are all air and hold no faces, so only an unstored neighbor chunk below the top of
  // the range needs storage.
  static constexpr int32_t kSides[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (const auto &side : kSides) {
    const int32_t nx = x + side[0];
    const int32_t ny = y + side[1];
    const int32_t nz = z + side[2];
    const int32_t cx = floorDiv(nx, N);
    const int32_t cz = floorDiv(nz, N);
    const Column *column = loadedColumn(cx, cz, batch);
    const int32_t chunkY = floorDiv(ny, N);
    if (column != nullptr && column->chunk(chunkY) == nullptr &&
        chunkY < column->firstChunkY + static_cast<int32_t>(column->chunks.size())) {
      materialize(editableColumn(cx, cz, batch), chunkY, ids_);
    }
  }
  // Dirty every chunk holding a cell of the 3x3x3 neighborhood (AO reaches diagonals): per axis the cell's own chunk,
  // plus the neighbor on a chunk edge, so 1 chunk for most cells and at most 8.
  /**
   * Chunks along one axis touched by coordinate v +- 1.
   *
   * @param out Receives the first and last chunk index.
   * @return 1 or 2 (number of distinct chunks).
   */
  const auto chunkRange = [](int32_t v, int32_t (&out)[2]) {
    out[0] = floorDiv(v - 1, N);
    out[1] = floorDiv(v + 1, N);
    return out[0] == out[1] ? 1 : 2; // N >= 3: v-1 and v+1 never both leave v's chunk
  };
  int32_t xs[2], ys[2], zs[2]; // chunk indices per axis
  const int32_t xCount = chunkRange(x, xs), yCount = chunkRange(y, ys), zCount = chunkRange(z, zs);
  for (int32_t iz = 0; iz < zCount; ++iz) {
    for (int32_t ix = 0; ix < xCount; ++ix) {
      if (loadedColumn(xs[ix], zs[iz], batch) == nullptr) {
        continue;
      }
      for (int32_t iy = 0; iy < yCount; ++iy) {
        batch.dirtyChunks.push_back({xs[ix], zs[iz], ys[iy]});
      }
    }
  }
}

void CpuWorld::touchLight(int32_t x, int32_t z, int32_t low, int32_t high, EditBatch &batch) {
  // Levels change up to kMaxLight away; a face averages cells up to 2 further (front cell + corner neighbors).
  const int32_t reach = static_cast<int32_t>(kMaxLight) + 2;
  for (int32_t cz = floorDiv(z - reach, N); cz <= floorDiv(z + reach, N); ++cz) {
    for (int32_t cx = floorDiv(x - reach, N); cx <= floorDiv(x + reach, N); ++cx) {
      if (loadedColumn(cx, cz, batch) == nullptr) {
        continue;
      }
      for (int32_t cy = floorDiv(low - reach, N); cy <= floorDiv(high + reach, N); ++cy) {
        batch.dirtyChunks.push_back({cx, cz, cy});
      }
    }
  }
}

void CpuWorld::refreshEditedNeighbors(int32_t cx, int32_t cz, EditBatch &batch) {
  const int32_t firstChunkY = loadedColumn(cx, cz, batch)->firstChunkY;
  for (int32_t dz = -1; dz <= 1; ++dz) {
    for (int32_t dx = -1; dx <= 1; ++dx) {
      const Column *neighbor = loadedColumn(cx + dx, cz + dz, batch);
      if ((dx == 0 && dz == 0) || neighbor == nullptr || !neighbor->edited) {
        continue;
      }
      for (int32_t cy = neighbor->firstChunkY; cy < neighbor->firstChunkY + static_cast<int32_t>(neighbor->chunks.size());
           ++cy) {
        batch.dirtyChunks.push_back({cx + dx, cz + dz, cy});
      }
      if ((dx != 0 && dz != 0) || neighbor->firstChunkY >= firstChunkY) {
        continue; // only side neighbors share faces; nothing stored below this column's range
      }
      // Air in the neighbor's border cells below this column's stored chunks needs storage (and faces) here.
      // This column's edge facing the neighbor: 0 if the neighbor is on the + side, else 31.
      const int32_t edge = dx + dz > 0 ? 0 : N - 1;
      for (int32_t i = 0; i < N; ++i) {
        const int32_t lx = dx != 0 ? edge : i;
        const int32_t lz = dx != 0 ? i : edge;
        for (int32_t cy = neighbor->firstChunkY; cy < firstChunkY; ++cy) {
          uint32_t air = ~neighbor->solid32(lx, lz, cy);
          while (air != 0u) {
            const int32_t ly = std::countr_zero(air);
            air &= air - 1u;
            touchCell((cx + dx) * N + lx, cy * N + ly, (cz + dz) * N + lz, batch);
          }
        }
      }
    }
  }
}

void CpuWorld::publish(EditBatch &batch) {
  if (!batch.copies.empty()) {
    std::lock_guard lock(mutex_);
    for (auto &[columnKey, column] : batch.copies) {
      columns_[columnKey] = column;
      if (const auto it = generated_.find(columnKey); it != generated_.end()) {
        it->second = column;
      }
    }
  }

  auto &dirty = batch.dirtyChunks;
  std::sort(dirty.begin(), dirty.end());
  dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());
  ColumnMeshEvent *event = nullptr;
  for (const auto &[cx, cz, cy] : dirty) {
    const auto it = columns_.find(key(cx, cz));
    if (it == columns_.end() || it->second->chunk(cy) == nullptr) {
      continue; // chunks without storage never had faces
    }
    Neighborhood nb; // nb: neighborhood
    for (int32_t dz = -1; dz <= 1; ++dz) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        const auto neighbor = columns_.find(key(cx + dx, cz + dz));
        nb.columns[dz + 1][dx + 1] = neighbor != columns_.end() ? neighbor->second.get() : nullptr;
      }
    }
    if (event == nullptr || event->cx != cx || event->cz != cz) {
      event = &events_.emplace_back(ColumnMeshEvent{cx, cz, false, true, {}});
    }
    meshChunk(nb, cy, event->chunks.emplace_back());
  }
}

std::optional<uint32_t> CpuWorld::skyLightAt(int32_t x, int32_t y, int32_t z) const {
  const int32_t cx = floorDiv(x, N);
  const int32_t cz = floorDiv(z, N);
  Neighborhood nb;
  for (int32_t dz = -1; dz <= 1; ++dz) {
    for (int32_t dx = -1; dx <= 1; ++dx) {
      const auto it = columns_.find(key(cx + dx, cz + dz));
      nb.columns[dz + 1][dx + 1] = it != columns_.end() ? it->second.get() : nullptr;
    }
  }
  if (nb.columns[1][1] == nullptr) {
    return std::nullopt;
  }
  return cpuworld::skyLightAt(nb, x - cx * N, y, z - cz * N);
}

bool CpuWorld::columnLoaded(int32_t cx, int32_t cz) const { return columns_.contains(key(cx, cz)); }

bool CpuWorld::squareLoaded(ChunkXZ center) const {
  for (int32_t dz = -world::kCpuChunkRadius; dz <= world::kCpuChunkRadius; ++dz) {
    for (int32_t dx = -world::kCpuChunkRadius; dx <= world::kCpuChunkRadius; ++dx) {
      if (!columns_.contains(key(center[0] + dx, center[1] + dz))) {
        return false;
      }
    }
  }
  return true;
}

int32_t CpuWorld::squareTopY(ChunkXZ center) const {
  int32_t top = world::kSeaLevel;
  for (int32_t dz = -world::kCpuChunkRadius; dz <= world::kCpuChunkRadius; ++dz) {
    for (int32_t dx = -world::kCpuChunkRadius; dx <= world::kCpuChunkRadius; ++dx) {
      const auto it = columns_.find(key(center[0] + dx, center[1] + dz));
      if (it == columns_.end()) {
        continue;
      }
      const Column &column = *it->second;
      for (int32_t i = static_cast<int32_t>(column.chunks.size()) - 1; i >= 0; --i) {
        if (column.chunks[i]) {
          top = std::max(top, (column.firstChunkY + i + 1) * CHUNK_SIZE);
          break;
        }
      }
    }
  }
  return top;
}

void CpuWorld::workerMain(std::stop_token stop) {
  for (;;) {
    ChunkXZ job; // column to generate
    {
      std::unique_lock lock(mutex_);
      if (!wake_.wait(lock, stop, [this] { return !queue_.empty(); })) {
        return; // stop requested
      }
      job = queue_.front();
      queue_.pop_front();
    }

    std::shared_ptr<const Column> column = generateColumn(job[0], job[1], ids_);
    Neighborhood nb; // nb: neighborhood
    std::array<std::shared_ptr<const Column>, 9> keepAlive; // pins the neighbors nb points to after unlocking
    {
      std::lock_guard lock(mutex_);
      generated_[key(job[0], job[1])] = column;
      for (int32_t dz = -1; dz <= 1; ++dz) {
        for (int32_t dx = -1; dx <= 1; ++dx) {
          const auto it = generated_.find(key(job[0] + dx, job[1] + dz));
          if (it != generated_.end()) {
            keepAlive[static_cast<size_t>((dz + 1) * 3 + dx + 1)] = it->second;
            nb.columns[dz + 1][dx + 1] = it->second.get();
          }
        }
      }
    }
    nb.columns[1][1] = column.get();

    // Missing neighbors fall back to this column's biased border heights; unedited terrain meshes the same either
    // way, so a column never needs remeshing when its neighbors arrive.
    std::vector<ChunkMesh> meshes;
    meshes.reserve(column->chunks.size());
    for (size_t i = 0; i < column->chunks.size(); ++i) {
      if (column->chunks[i] != nullptr) { // unstored chunks have no faces
        meshChunk(nb, column->firstChunkY + static_cast<int32_t>(i), meshes.emplace_back());
      }
    }

    std::lock_guard lock(mutex_);
    const auto wanted = wanted_.find(key(job[0], job[1]));
    const bool keep = wanted != wanted_.end() && wanted->second;
    if (wanted != wanted_.end()) {
      wanted_.erase(wanted);
    }
    if (keep) {
      results_.push_back({std::move(column), std::move(meshes)});
    } else {
      generated_.erase(key(job[0], job[1]));
    }
  }
}

} // namespace cpuworld
