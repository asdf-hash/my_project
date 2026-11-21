#pragma once

// CPU-side terrain for the chunk columns within world::kCpuChunkRadius of the player: generated (terrain_cpu.hpp)
// and meshed (cpu_mesher.hpp) on worker threads, kept in RAM for collision, and handed to the renderer as packed
// quads. The GPU clipmap leaves these columns out (ClipmapView::cpuRegionCenter).
//
// Hand-over without holes: the renderer only tells the clipmap about a region center once readyCenter() reports it
// (every column of that square meshed), and columns stay loaded while they belong to the desired, ready or applied
// (clipmap in use) square. So the GPU stops drawing a column only after the CPU draws it, and vice versa.

#include "cpu_chunk.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cpuworld {

using ChunkXZ = std::array<int32_t, 2>;

// Builds a column: padded heights, dither, and voxel chunks for every chunk that can hold a face.
std::shared_ptr<Column> generateColumn(int32_t cx, int32_t cz, const TerrainBlockIds &ids);

// A column's meshes became available (removed = false) or the column was unloaded (removed = true, no chunks).
// partial = true: only the listed chunks were remeshed (after a block edit); they replace the chunks with the same
// cy and every other chunk of the column stays as it is.
struct ColumnMeshEvent {
  int32_t cx = 0;
  int32_t cz = 0;
  bool removed = false;
  bool partial = false;
  std::vector<ChunkMesh> chunks;
};

using BlockXYZ = std::array<int32_t, 3>;

struct BlockHit {
  BlockXYZ block{};  // the solid block hit
  BlockXYZ normal{}; // face entered through (unit axis vector; zero if the ray starts inside the block)
  float distance = 0.0f;
};

class CpuWorld {
public:
  explicit CpuWorld(const TerrainBlockIds &ids, unsigned workerCount = 0);
  ~CpuWorld();
  CpuWorld(const CpuWorld &) = delete;
  CpuWorld &operator=(const CpuWorld &) = delete;

  // Main thread, once per frame: the player's chunk, plus centers whose squares must stay loaded because the GPU
  // relies on them (the center of the clipmap in use, and the one offered to the clipmap builder). Queues
  // generation, collects finished columns, unloads columns no square retains.
  void update(ChunkXZ desiredCenter, const std::vector<ChunkXZ> &pinnedCenters);

  // Latest desired center whose whole square has been meshed and returned by takeMeshEvents.
  std::optional<ChunkXZ> readyCenter() const { return readyCenter_; }
  std::vector<ColumnMeshEvent> takeMeshEvents();

  // Main thread. Unloaded columns count as air.
  bool isSolid(int32_t x, int32_t y, int32_t z) const;
  // First solid block along the ray within maxDistance blocks (voxel DDA). `direction` need not be normalized.
  std::optional<BlockHit> raycast(const std::array<float, 3> &origin, const std::array<float, 3> &direction,
                                  float maxDistance) const;
  // Main thread. Turns a solid block of a loaded column into air and remeshes the touched chunks on this thread (the
  // partial events come out of takeMeshEvents). Edits live only in the loaded column: one that unloads regenerates unedited (no saving yet).
  // Returns false if the block is air or its column isn't loaded.
  bool breakBlock(int32_t x, int32_t y, int32_t z);
  bool columnLoaded(int32_t cx, int32_t cz) const;
  // Main thread; null if not loaded. Edits replace the column, so don't keep the pointer across them.
  std::shared_ptr<const Column> column(int32_t cx, int32_t cz) const {
    const auto it = columns_.find(key(cx, cz));
    return it != columns_.end() ? it->second : nullptr;
  }
  // Every column of the square around `center` is loaded (and its meshes were returned by takeMeshEvents).
  bool squareLoaded(ChunkXZ center) const;
  size_t loadedColumns() const { return columns_.size(); }
  size_t voxelChunkCount() const;

  static bool inSquare(ChunkXZ center, int32_t cx, int32_t cz) {
    return std::abs(cx - center[0]) <= world::kCpuChunkRadius && std::abs(cz - center[1]) <= world::kCpuChunkRadius;
  }

private:
  static uint64_t key(int32_t cx, int32_t cz) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32u) | static_cast<uint32_t>(cz);
  }
  bool retained(int32_t cx, int32_t cz) const;
  void workerMain(std::stop_token stop);

  // One batch of edits on the main thread. Loaded columns are const and shared with the workers (neighbor lookups
  // while meshing), so a column is copied on its first change and the copies replace the originals in publish().
  struct EditBatch {
    std::unordered_map<uint64_t, std::shared_ptr<Column>> copies;
    std::vector<std::array<int32_t, 3>> dirtyChunks; // cx, cz, cy (sorted by column in publish); may repeat
  };
  const Column *loadedColumn(int32_t cx, int32_t cz, const EditBatch &batch) const;
  Column &editableColumn(int32_t cx, int32_t cz, EditBatch &batch);
  void setCell(Column &column, int32_t lx, int32_t y, int32_t lz, uint16_t id) const;
  // After cell (x, y, z) changed: gives its six neighbors voxel storage (a face may now show on them) and marks every
  // chunk whose mesh reads the cell (meshes see one cell beyond their chunk).
  void touchCell(int32_t x, int32_t y, int32_t z, EditBatch &batch);
  // A column just loaded, unedited (edits aren't saved). Loaded neighbors with broken blocks were meshed against its
  // old voxels or border heights: remeshes them, and gives this column storage beside air they dug below its range.
  void refreshEditedNeighbors(int32_t cx, int32_t cz, EditBatch &batch);
  // Installs the copies and queues partial mesh events for the dirty chunks.
  void publish(EditBatch &batch);

  struct Result {
    std::shared_ptr<const Column> column;
    std::vector<ChunkMesh> meshes;
  };

  TerrainBlockIds ids_;
  // Main thread only.
  std::unordered_map<uint64_t, std::shared_ptr<const Column>> columns_;
  std::optional<ChunkXZ> desiredCenter_;
  std::optional<ChunkXZ> readyCenter_;
  std::vector<ChunkXZ> pinnedCenters_;
  std::vector<ColumnMeshEvent> events_;

  // Shared with the workers.
  mutable std::mutex mutex_;
  std::condition_variable_any wake_;
  std::deque<ChunkXZ> queue_;                       // nearest first
  std::unordered_map<uint64_t, bool> wanted_;      // queued or running columns; false once cancelled
  std::unordered_map<uint64_t, std::shared_ptr<const Column>> generated_; // for neighbor lookups while meshing
  std::vector<Result> results_;
  std::vector<std::jthread> workers_;
};

} // namespace cpuworld
