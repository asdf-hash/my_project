#pragma once

// CPU side of the tree placement in shaders/vegetation_shared.hlsl (the same source the GPU compiles).

#include "shaders/vegetation_shared.hlsl"
#include "world.hpp"

static_assert(vegetation::TREE_MIN_GROUND > world::kTransitionHeights[0], "trees must stay off the beach layer");
static_assert(vegetation::TREE_MAX_GROUND <
                  world::kTransitionHeights[1] - static_cast<int32_t>(world::kTransitionDither[1]),
              "trees must stay below the lowest dithered grass / stone boundary");
static_assert(CHUNK_SIZE == (1 << vegetation::TREE_REGION_SHIFT), "a tree region is one chunk column");
static_assert(vegetation::TREE_SLOTS == 16, "slots are keyed 4 x 4 per region");
static_assert(vegetation::TREE_CELL_SHIFT == 3, "LOD 3 columns (scale 8) are the largest staged per tree");
