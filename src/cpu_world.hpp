#pragma once

// CPU-side terrain for the columns within world::kCpuChunkRadius of the player: generated and meshed on worker
// threads, kept in RAM for collision, handed to the renderer as packed quads. The GPU clipmap leaves them out.
//
// Hand-over without holes: the clipmap only learns a center once readyCenter() reports it, and columns stay loaded
// while in the desired, ready or applied square, so CPU and GPU coverage always overlap during a switch.

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

using ChunkXZ = std::array<int32_t, 2>; // chunk column (x, z), in chunks

/**
 * Generate a column: padded heights, dither, trees, and voxel chunks for every chunk that can hold a face.
 *
 * @param cx Chunk column x, in chunks.
 * @param cz Chunk column z, in chunks.
 * @param ids Resolved terrain block ids.
 * @return the new, unedited column.
 */
std::shared_ptr<Column> generateColumn(int32_t cx, int32_t cz, const TerrainBlockIds &ids);

// A column's meshes became available, or (removed) it was unloaded. partial: only the listed chunks were remeshed
// after an edit; they replace the chunks with the same cy and the rest stay.
struct ColumnMeshEvent {
  int32_t cx = 0;
  int32_t cz = 0;
  bool removed = false;
  bool partial = false;
  std::vector<ChunkMesh> chunks;
};

using BlockXYZ = std::array<int32_t, 3>; // world block coordinates

struct BlockHit {
  BlockXYZ block{};  // the solid block hit
  BlockXYZ normal{}; // face entered through (unit axis vector; zero if the ray starts inside the block)
};

class CpuWorld {
public:
  explicit CpuWorld(const TerrainBlockIds &ids, unsigned workerCount = 0);
  ~CpuWorld();
  CpuWorld(const CpuWorld &) = delete;
  CpuWorld &operator=(const CpuWorld &) = delete;

  /**
   * Queue generation, collect finished columns, unload columns no square retains, and advance readyCenter().
   * Main thread, once per frame.
   *
   * @param desiredCenter The player's chunk column.
   * @param pinnedCenters Centers the GPU still relies on (the clipmap in use, the one offered to the builder).
   */
  void update(ChunkXZ desiredCenter, const std::vector<ChunkXZ> &pinnedCenters);

  /** Latest center (desired, or on the way to it) whose whole square is meshed and in takeMeshEvents(). */
  std::optional<ChunkXZ> readyCenter() const { return readyCenter_; }
  /** Take the mesh events queued since the last call. Main thread. */
  std::vector<ColumnMeshEvent> takeMeshEvents();

  // Cell queries below: main thread only; unloaded columns count as air.

  /** Opaque block (id bit 9 clear): what meshing and AO see. */
  bool isSolid(int32_t x, int32_t y, int32_t z) const;
  /** Any block but air and ocean: what collision and targeting see. */
  bool isCollidable(int32_t x, int32_t y, int32_t z) const;
  /** Air or ocean: a block may be placed into it. */
  bool isReplaceable(uint16_t id) const { return id == kAir || id == ids_.ocean; }
  /** Whether the cell holds ocean (used for swimming / underwater checks). */
  bool isInfiniteSource(int32_t x, int32_t y, int32_t z) const;
  /** Skylight level 0..15 at a cell (as F3's client light), or nullopt if its column isn't loaded. Main thread. */
  std::optional<uint32_t> skyLightAt(int32_t x, int32_t y, int32_t z) const;
  /** Block id at a cell (kAir when unloaded, kBuriedTerrain for unstored cells below the surface). */
  uint16_t blockId(int32_t x, int32_t y, int32_t z) const;
  /**
   * First collidable block along a ray (voxel DDA).
   *
   * @param direction Need not be normalized.
   * @param maxDistance Ray length, in blocks.
   * @return the hit block and entry face, or nullopt.
   */
  std::optional<BlockHit> raycast(const std::array<float, 3> &origin, const std::array<float, 3> &direction,
                                  float maxDistance) const;
  /**
   * Turn a collidable block of a loaded column into air and remesh the touched chunks (partial events in
   * takeMeshEvents). Main thread. Edits are not saved: an unloaded column regenerates unedited.
   *
   * @return false if the cell isn't collidable or its column isn't loaded.
   */
  bool breakBlock(int32_t x, int32_t y, int32_t z);
  /**
   * Put block `id` into an air or ocean cell of a loaded column, like breakBlock. Main thread.
   *
   * @return false if the cell isn't replaceable, its column isn't loaded, `id` is air, or y is outside the terrain
   *         height envelope.
   */
  bool placeBlock(int32_t x, int32_t y, int32_t z, uint16_t id);
  /**
   * Every kOceanFlowSeconds, flood ocean one block into adjacent air cells below sea level (edited cells, then their
   * air neighbors), remeshing once per step. A cell over air floods the cell right below it instead and fills on
   * a later step. Main thread, once per frame.
   *
   * @param now Current time, in seconds.
   */
  void tickFluids(double now);
  static constexpr double kOceanFlowSeconds = 0.25;
  static constexpr size_t kOceanFlowCellsPerStep = 4096;
  /** Whether chunk column (cx, cz) is loaded. Main thread. */
  bool columnLoaded(int32_t cx, int32_t cz) const;
  /** Loaded column, or null. Main thread. Edits replace the column, so don't keep the pointer across them. */
  std::shared_ptr<const Column> column(int32_t cx, int32_t cz) const {
    const auto it = columns_.find(key(cx, cz));
    return it != columns_.end() ? it->second : nullptr;
  }
  /** Every column of the square around `center` is loaded (its meshes are queued in takeMeshEvents()). */
  bool squareLoaded(ChunkXZ center) const;
  /**
   * Exclusive top, in blocks, of the stored chunks of the loaded columns around `center`, at least sea level. Nothing
   * of the square is drawn above it.
   */
  int32_t squareTopY(ChunkXZ center) const;

  /** Whether chunk column (cx, cz) lies in the CPU square around `center`. */
  static bool inSquare(ChunkXZ center, int32_t cx, int32_t cz) {
    return std::abs(cx - center[0]) <= world::kCpuChunkRadius && std::abs(cz - center[1]) <= world::kCpuChunkRadius;
  }

private:
  /** Map key of a chunk column: cx in the high 32 bits, cz in the low. */
  static uint64_t key(int32_t cx, int32_t cz) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32u) | static_cast<uint32_t>(cz);
  }
  /** Whether any desired, ready or pinned square contains chunk column (cx, cz). */
  bool retained(int32_t cx, int32_t cz) const;
  /** Worker loop: generate and mesh queued columns until stop is requested. */
  void workerMain(std::stop_token stop);

  // One batch of main-thread edits. Loaded columns are const and shared with the workers, so a column is copied on
  // its first change and the copies replace the originals in publish().
  struct EditBatch {
    std::unordered_map<uint64_t, std::shared_ptr<Column>> copies;
    std::vector<std::array<int32_t, 3>> dirtyChunks; // cx, cz, cy (sorted by column in publish); may repeat
  };
  /** Column (cx, cz) as the batch sees it (its copy if edited), or null if not loaded. */
  const Column *loadedColumn(int32_t cx, int32_t cz, const EditBatch &batch) const;
  /** The batch's copy of loaded column (cx, cz), made on first use. */
  Column &editableColumn(int32_t cx, int32_t cz, EditBatch &batch);
  /**
   * Set a cell's block and update its chunk's solid / ocean masks and non-opaque count, materializing the chunk.
   *
   * @param lx Column-local x.
   * @param y World block y.
   * @param lz Column-local z.
   */
  void setCell(Column &column, int32_t lx, int32_t y, int32_t lz, uint16_t id) const;
  /** One-cell edit batch: copy the column, set the cell, remesh what reads it. */
  void editCell(int32_t x, int32_t y, int32_t z, uint16_t id);
  /**
   * After cell (x, y, z) changed: give its six neighbors voxel storage (a face may now show on them) and mark every
   * chunk whose mesh reads the cell (meshes see one cell beyond their chunk).
   */
  void touchCell(int32_t x, int32_t y, int32_t z, EditBatch &batch);
  /**
   * After an opacity change in column (x, z): mark every chunk whose skylight may have changed, i.e. every chunk
   * within kMaxLight + 2 blocks of the column between world y `low` and `high` (the edited cell and the column's old and
   * new skyTop; everything between them switched between open sky and shadow).
   */
  void touchLight(int32_t x, int32_t z, int32_t low, int32_t high, EditBatch &batch);
  /**
   * Column (cx, cz) just loaded unedited. Remesh edited loaded neighbors (meshed against its old voxels or border
   * heights) and give this column storage beside air they dug below its range.
   */
  void refreshEditedNeighbors(int32_t cx, int32_t cz, EditBatch &batch);
  /** Install the batch's copies and queue partial mesh events for its dirty chunks. */
  void publish(EditBatch &batch);

  // A worker's finished column and its meshes.
  struct Result {
    std::shared_ptr<const Column> column;
    std::vector<ChunkMesh> meshes;
  };

  TerrainBlockIds ids_;
  // Main thread only.
  std::unordered_map<uint64_t, std::shared_ptr<const Column>> columns_; // by key()
  std::optional<ChunkXZ> desiredCenter_;
  std::optional<ChunkXZ> readyCenter_;
  std::vector<ChunkXZ> pinnedCenters_;
  std::vector<ColumnMeshEvent> events_;
  std::vector<BlockXYZ> flowCandidates_; // air cells below sea level the ocean may flow into on the next step
  double nextFlowTime_ = 0.0;

  // Shared with the workers; guarded by mutex_.
  mutable std::mutex mutex_;
  std::condition_variable_any wake_;
  std::deque<ChunkXZ> queue_;                 // nearest first
  std::unordered_map<uint64_t, bool> wanted_; // queued or running columns; false once cancelled
  std::unordered_map<uint64_t, std::shared_ptr<const Column>> generated_; // for neighbor lookups while meshing
  std::vector<Result> results_;
  std::vector<std::jthread> workers_;
};

} // namespace cpuworld
