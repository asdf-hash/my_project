// Tree placement, shared verbatim by the GPU (chunk_compute.slang, via terrain_noise.hlsl) and the CPU
// (src/vegetation.hpp, used by cpu_world.cpp). Integer-only on purpose: the two noise implementations may disagree by
// a few ulp, but hash bits and integer column heights don't, so both sides place the same blocks.
//
// The world is split into 8x8-block cells (aligned to 0). Each cell holds at most one tree, and the whole tree
// (trunk jittered within [TREE_RADIUS, 7 - TREE_RADIUS], canopy radius TREE_RADIUS) stays inside its cell. 8 divides
// every tile and chunk size, so a tree never crosses a chunk column, a meshColumns tile or the CPU / GPU border, and
// "which tree covers this block" only ever asks the block's own cell.
#pragma once

#ifdef __cplusplus
#include <cstdint>
namespace vegetation {
using uint = std::uint32_t;
struct int2 {
  std::int32_t x, y;
  constexpr int2(std::int32_t x_, std::int32_t y_) : x(x_), y(y_) {}
};
#define VEG_CONST inline constexpr
#define VEG_FN inline
#else
#define VEG_CONST static const
#define VEG_FN
#endif

VEG_FN uint hashCell(int2 cell, uint seed) {
  uint h = uint(cell.x) * 0x8da6b343u ^ uint(cell.y) * 0xd8163841u ^ seed;
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

VEG_CONST uint SEED_TREE = 0x6a09e667u;
VEG_CONST uint SEED_FOREST = 0xbb67ae85u;
VEG_CONST uint SEED_LEAF = 0x3c6ef372u;

VEG_CONST int TREE_CELL_SHIFT = 3; // 8x8-block cells
VEG_CONST int TREE_RADIUS = 2;     // canopy 5x5
// Highest tree block above the ground (first air cell of the trunk column), exclusive: trunk up to 6, 2 leaf layers
// above it.
VEG_CONST int TREE_MAX_ABOVE = 8;
// Ground band for trunks: beaches (sand below world::kTransitionHeights[0] = 2) excluded, and below the lowest
// dithered grass / stone boundary (320 - 40), so the material dither (a float) never takes part.
VEG_CONST int TREE_MIN_GROUND = 3;
VEG_CONST int TREE_MAX_GROUND = 279;
// Scattered trees everywhere in the band, dense forests in some 64-block patches.
VEG_CONST int TREE_BASE_CHANCE = 6;     // of 256
VEG_CONST int TREE_FOREST_START = 110;  // patch value (0..255) where forests begin
VEG_CONST int TREE_FOREST_SLOPE = 2;    // chance per patch value above it

VEG_CONST uint TREE_NONE = 0u;
VEG_CONST uint TREE_LOG = 1u;
VEG_CONST uint TREE_LEAVES = 2u;

struct TreeInstance {
  int trunkX;      // world block coordinates of the trunk
  int trunkZ;
  int ground;      // first air cell of the trunk column: the trunk's lowest block
  int trunkHeight; // 0: no tree
  uint shape;      // leaf-corner bits (treeBlockAt)
};

VEG_FN int vegAbs(int v) { return v < 0 ? -v : v; }

// The candidate of a cell from hashes alone (ground still 0): position, trunk height (0 if the forest density
// rejects it) and shape. The caller sets `ground` from the trunk column and checks treeGroundOk.
VEG_FN TreeInstance treeCandidate(int2 cell) {
  const uint h = hashCell(cell, SEED_TREE);
  TreeInstance tree;
  tree.trunkX = (cell.x << TREE_CELL_SHIFT) + TREE_RADIUS + int(h & 3u);
  tree.trunkZ = (cell.y << TREE_CELL_SHIFT) + TREE_RADIUS + int((h >> 2u) & 3u);
  tree.ground = 0;
  tree.trunkHeight = 4 + int(((h >> 4u) & 0xffu) % 3u);
  tree.shape = hashCell(cell, SEED_LEAF);

  // Forest patches: 64-block squares, looked up from a point jittered by up to +-32 blocks so edges come out ragged.
  const int jitterX = int((h >> 16u) & 63u) - 32;
  const int jitterZ = int((h >> 22u) & 63u) - 32;
  const int2 patch = int2(((cell.x << TREE_CELL_SHIFT) + 4 + jitterX) >> 6, ((cell.y << TREE_CELL_SHIFT) + 4 + jitterZ) >> 6);
  const int forest = int(hashCell(patch, SEED_FOREST) & 0xffu);
  const int chance = TREE_BASE_CHANCE + (forest > TREE_FOREST_START ? (forest - TREE_FOREST_START) * TREE_FOREST_SLOPE : 0);
  if (int((h >> 8u) & 0xffu) >= chance) {
    tree.trunkHeight = 0;
  }
  return tree;
}

// Trunk column at `ground` in the band, and its four neighbours within one block of it (no trees on cliffs).
VEG_FN bool treeGroundOk(int ground, int negX, int posX, int negZ, int posZ) {
  return ground >= TREE_MIN_GROUND && ground <= TREE_MAX_GROUND && vegAbs(negX - ground) <= 1 &&
         vegAbs(posX - ground) <= 1 && vegAbs(negZ - ground) <= 1 && vegAbs(posZ - ground) <= 1;
}

// Block of the tree at world block (x, y, z): TREE_NONE / TREE_LOG / TREE_LEAVES. Ignores terrain: callers drop
// tree blocks in terrain-solid cells (terrain wins).
//   trunk: [ground, ground + trunkHeight)
//   leaf layers 0-1: 5x5 around the trunk's top two blocks, each corner dropped by a shape bit
//   leaf layer 2: 3x3 above the trunk, corners dropped by shape bits; layer 3: a plus
VEG_FN uint treeBlockAt(TreeInstance tree, int x, int y, int z) {
  if (tree.trunkHeight == 0) {
    return TREE_NONE;
  }
  const int dx = x - tree.trunkX;
  const int dz = z - tree.trunkZ;
  const int dy = y - tree.ground;
  if (dx == 0 && dz == 0 && dy >= 0 && dy < tree.trunkHeight) {
    return TREE_LOG;
  }
  const int layer = dy - (tree.trunkHeight - 2);
  if (layer < 0 || layer > 3) {
    return TREE_NONE;
  }
  const int radius = layer < 2 ? TREE_RADIUS : 1;
  const int adx = vegAbs(dx);
  const int adz = vegAbs(dz);
  if (adx > radius || adz > radius) {
    return TREE_NONE;
  }
  if (adx == radius && adz == radius) {
    if (layer == 3) {
      return TREE_NONE;
    }
    const uint corner = (dx > 0 ? 1u : 0u) | (dz > 0 ? 2u : 0u);
    if (((tree.shape >> (uint(layer) * 4u + corner)) & 1u) != 0u) {
      return TREE_NONE;
    }
  }
  return TREE_LEAVES;
}

// ---- Coarse LODs (chunk_compute.slang computeTreeColumn) ----
// LOD 1-2 (scale 2, 4): the tree downsampled into the LOD's cells, read from a table built once on the CPU
// (vegetation.hpp buildTreeLodTable) from treeBlockAt above. The tree stands on its trunk column's coarse ground,
// so its base is always cell-aligned and only trunk offset, height and the column within the cell matter. Leaf
// corner shape bits are ignored (all corners present).
// LOD 3 (scale 8): the tree is smaller than one cell; its cell is filled for half of the trees (picked by hash).
// The canopy's top-down cover (5x5 of 8x8) would be 0.39; 0.5 keeps distant forests a little denser.
// LOD 4+ (scale 16+): a column spans (scale / 8)^2 cells; it is filled with the fraction of those cells that LOD 3
// would fill (sampling at most TREE_FAR_SAMPLES^2 of them), so distant forests keep LOD 3's density.
VEG_CONST int TREE_MAX_LOD = 7;
VEG_CONST int TREE_FAR_SAMPLES = 4; // per axis
VEG_CONST uint SEED_TREE_LOD = 0xa54ff53au;
VEG_CONST uint TREE_LOD3_FILL_CHANCE = 128u; // of 256: 0.5
// Table entry: bits 0-3 leaf cells (relative to the ground cell), bits 4-7 log cell count (cells [0, n)). One entry per (trunk offset x + 4 z, height - 4, column x + columnsPerCell * z in the cell).
VEG_CONST uint TREE_LOD_TABLE_SCALE2_BASE = 0u;   // 16 offsets x 3 heights x 16 columns
VEG_CONST uint TREE_LOD_TABLE_SCALE4_BASE = 768u; // 16 offsets x 3 heights x 4 columns
VEG_CONST uint TREE_LOD_TABLE_SIZE = 960u;

VEG_FN uint treeLodTableIndex(int scale, uint trunkOffset, uint heightIndex, uint columnX, uint columnZ) {
  const uint perCell = uint(8 / scale);
  const uint base = scale == 2 ? TREE_LOD_TABLE_SCALE2_BASE : TREE_LOD_TABLE_SCALE4_BASE;
  return base + ((trunkOffset * 3u + heightIndex) * perCell + columnZ) * perCell + columnX;
}

#ifdef __cplusplus
} // namespace vegetation
#endif
#undef VEG_CONST
#undef VEG_FN
