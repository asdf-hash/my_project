#pragma once

// CPU side of the tree placement in shaders/vegetation_shared.hlsl (the same source the GPU compiles).

#include "shaders/vegetation_shared.hlsl"
#include "world.hpp"

#include <vector>

static_assert(vegetation::TREE_MIN_GROUND > world::kTransitionHeights[0], "trees must stay off the beach layer");
static_assert(vegetation::TREE_MAX_GROUND <
                  world::kTransitionHeights[1] - static_cast<int32_t>(world::kTransitionDither[1]),
              "trees must stay below the lowest dithered grass / stone boundary");
static_assert((1 << vegetation::TREE_CELL_SHIFT) - 1 - vegetation::TREE_RADIUS >=
                  vegetation::TREE_RADIUS + 3,
              "trunk jitter (h & 3) must keep the canopy inside its cell");
static_assert(CHUNK_SIZE % (1 << vegetation::TREE_CELL_SHIFT) == 0, "tree cells must tile chunk columns");
static_assert(vegetation::TREE_MAX_ABOVE <= 4 * 2, "a scale-2 tree must fit the table's 4 cells");
static_assert(vegetation::TREE_CELL_SHIFT == 3 && vegetation::TREE_RADIUS == 2,
              "the LOD table assumes 8-block cells and 4x4 trunk offsets");

namespace vegetation {
// Downsampled trees for LOD 1-2 (see TREE_LOD_TABLE_* in vegetation_shared.hlsl), shaped to read as trees rather
// than as a voxel average:
// - The 4-block canopy becomes 4 / scale whole cells (at least 1), centred on its real centre height and rounded to
//   cells, so it always caps the trunk (a cell-by-cell average left log cells poking out of the canopy top).
// - Canopy cell j stands for canopy layers [4j / n, 4(j + 1) / n). A column is leaves there when those layers cover
//   more than a quarter of the column from above: the 5x5 rounds to a 3x3 minus a corner at scale 2 and to 1-3
//   columns at scale 4, the 3x3 top to a smaller step above it.
// - The trunk column is log in every cell below the canopy.
inline std::vector<uint32_t> buildTreeLodTable() {
  std::vector<uint32_t> table(TREE_LOD_TABLE_SIZE, 0u);
  for (const int scale : {2, 4}) {
    const int perCell = 8 / scale;
    const int canopyCells = 4 / scale > 0 ? 4 / scale : 1;
    for (uint32_t offset = 0; offset < 16u; ++offset) {
      for (uint32_t heightIndex = 0; heightIndex < 3u; ++heightIndex) {
        const int height = 4 + static_cast<int>(heightIndex);
        const TreeInstance tree{TREE_RADIUS + static_cast<int>(offset & 3u), TREE_RADIUS + static_cast<int>(offset >> 2u),
                                0, height, 0u};
        // Canopy blocks [height - 2, height + 2), centre `height`: first cell round(height / scale - canopyCells / 2).
        const int canopyStart = (2 * height - canopyCells * scale + scale) / (2 * scale);
        for (int columnZ = 0; columnZ < perCell; ++columnZ) {
          for (int columnX = 0; columnX < perCell; ++columnX) {
            uint32_t leafMask = 0u;
            for (int j = 0; j < canopyCells; ++j) {
              int covered = 0;
              for (int z = columnZ * scale; z < (columnZ + 1) * scale; ++z) {
                for (int x = columnX * scale; x < (columnX + 1) * scale; ++x) {
                  bool any = false;
                  for (int layer = j * 4 / canopyCells; layer < (j + 1) * 4 / canopyCells; ++layer) {
                    any = any || treeBlockAt(tree, x, height - 2 + layer, z) != TREE_NONE;
                  }
                  covered += any ? 1 : 0;
                }
              }
              if (4 * covered > scale * scale) {
                leafMask |= 1u << static_cast<uint32_t>(canopyStart + j);
              }
            }
            const bool trunkColumn = tree.trunkX / scale == columnX && tree.trunkZ / scale == columnZ;
            const uint32_t logCells = trunkColumn ? static_cast<uint32_t>(canopyStart) : 0u;
            table[treeLodTableIndex(scale, offset, heightIndex, static_cast<uint>(columnX),
                                    static_cast<uint>(columnZ))] = leafMask | (logCells << 4u);
          }
        }
      }
    }
  }
  return table;
}
} // namespace vegetation
