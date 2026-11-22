#include "cpu_world.hpp"

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

constexpr int32_t N = CHUNK_SIZE;

// The column's trees (Column::trees) within chunk chunkY, into cells the terrain leaves empty (terrain wins), the
// same rule chunk_compute.slang's treeCellBlock applies.
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
          const int32_t wx = tree.trunkX + dx;
          const int32_t wz = tree.trunkZ + dz;
          const uint32_t block = vegetation::treeBlockAt(tree, wx, y, wz);
          const int32_t lx = wx - originX;
          const int32_t lz = wz - originZ;
          if (block == vegetation::TREE_NONE || y < column.height(lx, lz)) {
            continue;
          }
          const int32_t ly = y - bottom;
          uint16_t &cell = voxels.ids[VoxelChunk::index(lx, ly, lz)];
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

// Fills one voxel chunk from the heightmap: the CPU twin of faceBlockType / selectBlockType at LOD 0. Cells within
// one block of a (dithered) layer boundary take their material from the column's top instead of their own height.
void fillChunk(const Column &column, int32_t chunkY, const TerrainBlockIds &ids, VoxelChunk &voxels) {
  const int32_t bottom = chunkY * N;
  // Solid masks, 8 columns at a time: (1 << clamp(h - bottom, 0, 32)) - 1, where a shift of 32 gives 0 - 1 = ~0.
  for (int32_t i = 0; i < N * N; i += 8) {
    const I8 h = I8::load(&column.heights[i]);
    const I8 n = simd::max(simd::min(h - I8(bottom), I8(N)), I8(0));
    (simd::sllv(I8(1), n) - I8(1)).store(reinterpret_cast<int32_t *>(&voxels.solid[i]));
  }
  // Ocean: empty cells below sea level.
  const uint32_t sea = column.oceanId == kAir ? 0u : Column::heightMask(world::kSeaLevel, chunkY);
  for (int32_t i = 0; i < N * N; ++i) {
    voxels.ocean[i] = sea & ~voxels.solid[i];
  }

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
        const I8 y = I8(bottom + y0) + I8::iota();
        M8 near = y == I8(transition[0] - 1);
        for (const int32_t t : transition) {
          near = near | (y == I8(t - 1)) | (y == I8(t));
        }
        const I8 materialY = simd::select(near, h, y);
        I8 layer(0);
        for (const int32_t t : transition) {
          layer = layer - simd::maskBits(materialY > I8(t - 1)); // += materialY >= t
        }
        const I8 depth = h - I8(1) - y;
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

// Gives the column voxel storage for chunk chunkY, filled from the heightmap (unstored chunks follow the same rule,
// so the new one matches what it stood for). Slots between it and the stored range stay empty (unstored).
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
    slot = std::make_unique<VoxelChunk>();
    fillChunk(column, chunkY, ids, *slot);
  }
}

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

  // Padded heights, 8 cells per call (1156 cells, the last batch partly unused).
  constexpr int32_t padCells = kPad * kPad;
  alignas(32) int32_t xs[8], zs[8];
  alignas(32) float out[8];
  for (int32_t i = 0; i < padCells; i += 8) {
    for (int32_t l = 0; l < 8; ++l) {
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

  // Dither at each column (chunk_compute.slang samples at the cell origin + scale / 2 = the column itself at LOD 0).
  for (int32_t i = 0; i < N * N; i += 8) {
    for (int32_t l = 0; l < 8; ++l) {
      xs[l] = originX + (i + l) % N;
      zs[l] = originZ + (i + l) / N;
    }
    terrain::sampleDither8(xs, zs, &column->dither[static_cast<size_t>(i)]);
  }

  // Trees: one candidate per 8x8 cell. The trunk sits 2..5 blocks into its cell, so it and its four neighbours are
  // interior columns (the same heights generateHeights stores at LOD 0).
  int32_t treeTop = world::kMinTerrainHeight;
  if (ids.log != kAir && ids.leaves != kAir) {
    for (int32_t cellZ = 0; cellZ < Column::kTreeCellsPerSide; ++cellZ) {
      for (int32_t cellX = 0; cellX < Column::kTreeCellsPerSide; ++cellX) {
        vegetation::TreeInstance tree = vegetation::treeCandidate(
            vegetation::int2(cx * Column::kTreeCellsPerSide + cellX, cz * Column::kTreeCellsPerSide + cellZ));
        if (tree.trunkHeight != 0) {
          const int32_t lx = tree.trunkX - originX;
          const int32_t lz = tree.trunkZ - originZ;
          tree.ground = column->height(lx, lz);
          if (vegetation::treeGroundOk(tree.ground, column->height(lx - 1, lz), column->height(lx + 1, lz),
                                       column->height(lx, lz - 1), column->height(lx, lz + 1))) {
            treeTop = std::max(treeTop, tree.ground + vegetation::TREE_MAX_ABOVE);
          } else {
            tree.trunkHeight = 0;
          }
        }
        column->trees[static_cast<size_t>(cellZ * Column::kTreeCellsPerSide + cellX)] = tree;
      }
    }
  }

  // Faces lie on cells in [paddedMin - 1, interiorMax): a solid cell needs a non-solid cell above it or beside it.
  // Ocean surface faces lie on the cell just below sea level, in columns lower than it. Fully ocean chunks between
  // the two hold no faces and stay unstored (null slots).
  column->firstChunkY = floorDiv(paddedMin - 1, N);
  const int32_t terrainLastChunkY = floorDiv(interiorMax - 1, N);
  const bool hasSurface = ids.ocean != kAir && interiorMin < world::kSeaLevel;
  const int32_t seaChunkY = floorDiv(world::kSeaLevel - 1, N);
  // Chunks holding tree blocks above the terrain are stored too (null slots must follow the heightmap).
  const int32_t treeLastChunkY = floorDiv(treeTop - 1, N);
  const int32_t lastChunkY =
      std::max(hasSurface ? std::max(terrainLastChunkY, seaChunkY) : terrainLastChunkY, treeLastChunkY);
  for (int32_t chunkY = column->firstChunkY; chunkY <= lastChunkY; ++chunkY) {
    if (chunkY > terrainLastChunkY && chunkY != seaChunkY && chunkY > treeLastChunkY) {
      column->chunks.emplace_back();
      continue;
    }
    auto voxels = std::make_unique<VoxelChunk>();
    fillChunk(*column, chunkY, ids, *voxels);
    column->chunks.push_back(std::move(voxels));
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
    for (const ChunkXZ &c : arrived) {
      refreshEditedNeighbors(c[0], c[1], batch);
    }
    publish(batch);
  }

  // Unload what no square retains any more.
  for (auto it = columns_.begin(); it != columns_.end();) {
    const Column &column = *it->second;
    if (retained(column.cx, column.cz)) {
      ++it;
      continue;
    }
    events_.push_back({column.cx, column.cz, true, {}});
    {
      std::lock_guard lock(mutex_);
      generated_.erase(it->first);
    }
    it = columns_.erase(it);
  }

  if (moved) {
    // Requeue: missing columns of the desired square, nearest first; cancel queued ones nothing retains.
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
      const auto distance = [&](const ChunkXZ &c) {
        const int32_t dx = c[0] - desiredCenter[0];
        const int32_t dz = c[1] - desiredCenter[1];
        return dx * dx + dz * dz;
      };
      return distance(a) < distance(b);
    });
    std::lock_guard lock(mutex_);
    for (auto &[k, wanted] : wanted_) {
      const ChunkXZ c{static_cast<int32_t>(k >> 32u), static_cast<int32_t>(static_cast<uint32_t>(k))};
      wanted = retained(c[0], c[1]);
    }
    queue_.clear();
    for (const ChunkXZ &c : missing) {
      auto [it, inserted] = wanted_.try_emplace(key(c[0], c[1]), true);
      if (inserted) {
        queue_.push_back(c);
      } else if (!it->second) {
        it->second = true; // still running; its result is wanted again
      }
    }
    // Columns queued before but no longer retained were dropped from queue_ above; forget them.
    for (auto it = wanted_.begin(); it != wanted_.end();) {
      const bool queued = std::any_of(queue_.begin(), queue_.end(),
                                      [&](const ChunkXZ &c) { return key(c[0], c[1]) == it->first; });
      if (!it->second && !queued) {
        // A worker may still be running it; that worker erases nothing, its result is simply discarded.
        it = wanted_.erase(it);
      } else {
        ++it;
      }
    }
    wake_.notify_all();
  }

  // Ready once every column of a square is loaded (its meshes are in events_). While the player keeps moving the
  // desired square is rarely complete, so also accept the complete square closest to it on the way from the
  // current ready center: the CPU square then follows a fast player instead of staying where it last caught up.
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
  for (int a = 0; a < 3; ++a) {
    const double o = origin[a];
    const double d = direction[a] / length;
    cell[a] = static_cast<int32_t>(std::floor(o));
    if (d > 0.0) {
      step[a] = 1;
      tDelta[a] = 1.0 / d;
      tMax[a] = (cell[a] + 1.0 - o) / d;
    } else if (d < 0.0) {
      step[a] = -1;
      tDelta[a] = -1.0 / d;
      tMax[a] = (o - cell[a]) / -d;
    } else {
      tDelta[a] = tMax[a] = std::numeric_limits<double>::infinity();
    }
  }
  BlockXYZ normal{};
  double t = 0.0;
  while (t <= maxDistance) {
    if (isCollidable(cell[0], cell[1], cell[2])) {
      return BlockHit{cell, normal, static_cast<float>(t)};
    }
    const int a = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : (tMax[1] < tMax[2] ? 1 : 2);
    t = tMax[a];
    cell[a] += step[a];
    tMax[a] += tDelta[a];
    normal = {};
    normal[a] = -step[a];
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
  setCell(column, x - cx * N, y, z - cz * N, id);
  touchCell(x, y, z, batch);
  publish(batch);
  if (id == kAir && y < world::kSeaLevel && ids_.ocean != kAir) {
    flowCandidates_.push_back({x, y, z}); // ocean flows in on a later flow step if it is next to ocean
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

  // Decide every cell against the state before this step (columns_ is only replaced in publish), so the ocean
  // advances exactly one block per step.
  static constexpr int32_t kSides[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  std::vector<BlockXYZ> flooded;
  for (const BlockXYZ &c : candidates) {
    if (c[1] >= world::kSeaLevel || !columnLoaded(floorDiv(c[0], N), floorDiv(c[2], N)) ||
        blockId(c[0], c[1], c[2]) != kAir) {
      continue;
    }
    for (const auto &side : kSides) {
      if (blockId(c[0] + side[0], c[1] + side[1], c[2] + side[2]) == ids_.ocean) {
        flooded.push_back(c);
        break;
      }
    }
  }
  if (flooded.empty()) {
    return;
  }
  EditBatch batch;
  for (const BlockXYZ &c : flooded) {
    const int32_t cx = floorDiv(c[0], N);
    const int32_t cz = floorDiv(c[2], N);
    Column &column = editableColumn(cx, cz, batch);
    column.edited = true;
    setCell(column, c[0] - cx * N, c[1], c[2] - cz * N, ids_.ocean);
    touchCell(c[0], c[1], c[2], batch);
  }
  publish(batch);
  for (const BlockXYZ &c : flooded) {
    for (const auto &side : kSides) {
      const BlockXYZ n{c[0] + side[0], c[1] + side[1], c[2] + side[2]};
      if (n[1] < world::kSeaLevel && blockId(n[0], n[1], n[2]) == kAir) {
        flowCandidates_.push_back(n);
      }
    }
  }
}

const Column *CpuWorld::loadedColumn(int32_t cx, int32_t cz, const EditBatch &batch) const {
  const uint64_t k = key(cx, cz);
  if (const auto it = batch.copies.find(k); it != batch.copies.end()) {
    return it->second.get();
  }
  const auto it = columns_.find(k);
  return it != columns_.end() ? it->second.get() : nullptr;
}

Column &CpuWorld::editableColumn(int32_t cx, int32_t cz, EditBatch &batch) {
  const uint64_t k = key(cx, cz);
  auto [it, inserted] = batch.copies.try_emplace(k);
  if (inserted) {
    it->second = cloneColumn(*columns_.at(k));
  }
  return *it->second;
}

void CpuWorld::setCell(Column &column, int32_t lx, int32_t y, int32_t lz, uint16_t id) const {
  const int32_t chunkY = floorDiv(y, N);
  materialize(column, chunkY, ids_);
  VoxelChunk &voxels = *column.chunks[static_cast<size_t>(chunkY - column.firstChunkY)];
  const int32_t ly = y - chunkY * N;
  uint16_t &cell = voxels.ids[VoxelChunk::index(lx, ly, lz)];
  const auto countsAsNonOpaque = [&](uint16_t block) {
    return block != kAir && block != ids_.ocean && !blocks::isOpaque(block);
  };
  voxels.nonOpaqueCount = static_cast<uint16_t>(voxels.nonOpaqueCount - (countsAsNonOpaque(cell) ? 1 : 0) +
                                                (countsAsNonOpaque(id) ? 1 : 0));
  cell = id;
  // The solid masks hold opaque cells only (bit 9 clear); air and non-opaque blocks clear the bit.
  uint32_t &bits = voxels.solid[lz * N + lx];
  bits = blocks::isOpaque(id) ? bits | (1u << ly) : bits & ~(1u << ly);
  uint32_t &oceanBits = voxels.ocean[lz * N + lx];
  oceanBits = id == ids_.ocean && id != kAir ? oceanBits | (1u << ly) : oceanBits & ~(1u << ly);
}

void CpuWorld::touchCell(int32_t x, int32_t y, int32_t z, EditBatch &batch) {
  // Faces only appear on solid or ocean cells beside other cells, and chunks above a column's range are all air, so
  // only a neighbor in an unstored chunk below the top of the range (terrain or ocean) needs storage.
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
  for (int32_t dz = -1; dz <= 1; ++dz) {
    for (int32_t dy = -1; dy <= 1; ++dy) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        const int32_t cx = floorDiv(x + dx, N);
        const int32_t cz = floorDiv(z + dz, N);
        if (loadedColumn(cx, cz, batch) != nullptr) {
          batch.dirtyChunks.push_back({cx, cz, floorDiv(y + dy, N)});
        }
      }
    }
  }
}

void CpuWorld::refreshEditedNeighbors(int32_t cx, int32_t cz, EditBatch &batch) {
  const int32_t first = loadedColumn(cx, cz, batch)->firstChunkY;
  for (int32_t dz = -1; dz <= 1; ++dz) {
    for (int32_t dx = -1; dx <= 1; ++dx) {
      const Column *neighbor = loadedColumn(cx + dx, cz + dz, batch);
      if ((dx == 0 && dz == 0) || neighbor == nullptr || !neighbor->edited) {
        continue;
      }
      // Its meshes saw this column as it was before it unloaded (or as border heights): remesh them all.
      for (int32_t cy = neighbor->firstChunkY; cy < neighbor->firstChunkY + static_cast<int32_t>(neighbor->chunks.size());
           ++cy) {
        batch.dirtyChunks.push_back({cx + dx, cz + dz, cy});
      }
      if ((dx != 0 && dz != 0) || neighbor->firstChunkY >= first) {
        continue; // only side neighbors share faces; nothing stored below this column's range
      }
      // Air in the neighbor's border cells below this column's stored chunks: this column needs storage (and faces)
      // beside them. lx (or lz) = 0 if the neighbor is on the + side, else 31.
      const int32_t edge = dx + dz > 0 ? 0 : N - 1;
      for (int32_t i = 0; i < N; ++i) {
        const int32_t lx = dx != 0 ? edge : i;
        const int32_t lz = dx != 0 ? i : edge;
        for (int32_t cy = neighbor->firstChunkY; cy < first; ++cy) {
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
    for (auto &[k, column] : batch.copies) {
      columns_[k] = column;
      if (const auto it = generated_.find(k); it != generated_.end()) {
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
    Neighborhood nb;
    for (int32_t dz = -1; dz <= 1; ++dz) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        const auto n = columns_.find(key(cx + dx, cz + dz));
        nb.columns[dz + 1][dx + 1] = n != columns_.end() ? n->second.get() : nullptr;
      }
    }
    if (event == nullptr || event->cx != cx || event->cz != cz) {
      event = &events_.emplace_back(ColumnMeshEvent{cx, cz, false, true, {}});
    }
    meshChunk(nb, cy, event->chunks.emplace_back());
  }
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

size_t CpuWorld::voxelChunkCount() const {
  size_t count = 0;
  for (const auto &[k, column] : columns_) {
    count += static_cast<size_t>(std::count_if(column->chunks.begin(), column->chunks.end(),
                                               [](const auto &chunk) { return chunk != nullptr; }));
  }
  return count;
}

void CpuWorld::workerMain(std::stop_token stop) {
  for (;;) {
    ChunkXZ job;
    {
      std::unique_lock lock(mutex_);
      if (!wake_.wait(lock, stop, [this] { return !queue_.empty(); })) {
        return; // stop requested
      }
      job = queue_.front();
      queue_.pop_front();
    }

    std::shared_ptr<const Column> column = generateColumn(job[0], job[1], ids_);
    Neighborhood nb;
    std::array<std::shared_ptr<const Column>, 9> keepAlive;
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

    // Neighbors not generated yet fall back to this column's (biased) border heights. Unedited terrain is the same
    // either way, so a column never needs remeshing when its neighbors arrive.
    std::vector<ChunkMesh> meshes;
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
