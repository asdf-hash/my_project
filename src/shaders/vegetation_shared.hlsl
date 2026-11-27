// Tree placement, shared verbatim by the GPU (chunk_compute.slang via terrain_noise.hlsl) and the CPU
// (src/vegetation.hpp). Integer-only so both sides place the same blocks despite float ulp differences.
// Each chunk column (32x32-block "tree region") has TREE_SLOTS candidates kept TREE_RADIUS inside it, so a tree never
// crosses a chunk column (nor the CPU / GPU border) and a block only asks its own region's slots. Log beats leaves.
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

/**
 * Integer hash of a 2D lattice cell. SIMD twin in terrain_cpu.cpp.
 *
 * @return 32 well-mixed bits.
 */
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
VEG_CONST uint SEED_BIOME = 0x1f83d9abu;
VEG_CONST uint SEED_BIOME_WARP = 0x5be0cd19u;

VEG_CONST int TREE_CELL_SHIFT = 3;   // 8x8-block LOD cells (LOD 3 columns; the LOD 1-2 table's virtual cells)
VEG_CONST int TREE_REGION_SHIFT = 5; // 32x32-block tree regions: one chunk column (CHUNK_SIZE)
VEG_CONST int TREE_SLOTS = 16;       // candidate trees per region (4 x 4 hash keys)
VEG_CONST int TREE_RADIUS = 2;     // widest canopy 5x5 (every style)
// Exclusive top of any tree above its ground: tallest trunk plus leaves (big: 10 + 2; conifer: 11 + 1 tip).
VEG_CONST int TREE_MAX_ABOVE = 12;
// Trunk ground band: above beach sand (world::kTransitionHeights[0] = 2) and below the lowest dithered grass / stone
// boundary (320 - 40), so the float material dither never decides a tree.
VEG_CONST int TREE_MIN_GROUND = 3;
VEG_CONST int TREE_MAX_GROUND = 279;
// Density: Voronoi biomes (forest / woodland / plains) on a warped, jittered 256-block grid; inside one, 32-block
// clump noise blends the keep chance between the biome's min and max (of 256). Then slope (treeGroundOk).
VEG_CONST int TREE_BIOME_SHIFT = 8;  // 256-block Voronoi grid
VEG_CONST int TREE_BIOME_WARP = 48;  // blocks
VEG_CONST int TREE_BIOME_FOREST = 0;
VEG_CONST int TREE_BIOME_WOODLAND = 1;
VEG_CONST int TREE_BIOME_PLAINS = 2;
// Share of regions (of 256): forest below FOREST_SHARE, woodland below WOODLAND_SHARE, plains above.
VEG_CONST int TREE_BIOME_FOREST_SHARE = 100;
VEG_CONST int TREE_BIOME_WOODLAND_SHARE = 150;
/** Lowest keep chance (of 256) in a biome. */
VEG_FN int treeBiomeMinChance(int biome) { return biome == TREE_BIOME_FOREST ? 150 : biome == TREE_BIOME_WOODLAND ? 20 : 0; }
/** Highest keep chance (of 256) in a biome. */
VEG_FN int treeBiomeMaxChance(int biome) { return biome == TREE_BIOME_FOREST ? 256 : biome == TREE_BIOME_WOODLAND ? 130 : 10; }

// Tree styles (shape bits 12-13), rolled per biome from shape bits 24-31. Trunks: round 4-7, big 6-10,
// conifer 6-11 (tiered canopy from 2 blocks up), bush 1-2 (low clump).
VEG_CONST uint TREE_STYLE_ROUND = 0u;
VEG_CONST uint TREE_STYLE_BIG = 1u;
VEG_CONST uint TREE_STYLE_CONIFER = 2u;
VEG_CONST uint TREE_STYLE_BUSH = 3u;
/** Style bits (12-13) of a tree's shape. */
VEG_FN uint treeStyle(uint shape) { return (shape >> 12u) & 3u; }
/**
 * Style of a tree from its biome and a roll; thresholds a < b < c pick ROUND, BIG, CONIFER, else BUSH.
 *
 * @param roll Style roll, 0..255.
 * @return a TREE_STYLE_* value.
 */
VEG_FN uint treeStyleFor(int biome, uint roll) {
  const uint a = biome == TREE_BIOME_FOREST ? 80u : biome == TREE_BIOME_WOODLAND ? 128u : 115u;
  const uint b = biome == TREE_BIOME_FOREST ? 150u : biome == TREE_BIOME_WOODLAND ? 180u : 140u;
  const uint c = biome == TREE_BIOME_FOREST ? 245u : biome == TREE_BIOME_WOODLAND ? 218u : 153u;
  return roll < a ? TREE_STYLE_ROUND : roll < b ? TREE_STYLE_BIG : roll < c ? TREE_STYLE_CONIFER : TREE_STYLE_BUSH;
}
/**
 * Trunk height of a style. Squaring the roll skews toward short trunks.
 *
 * @param roll Height roll, 0..255.
 * @return trunk height, in blocks.
 */
VEG_FN int treeTrunkHeightFor(uint style, uint roll) {
  const uint skewed = roll * roll >> 8u; // 0..254
  const int low = style == TREE_STYLE_BUSH ? 1 : style == TREE_STYLE_ROUND ? 4 : 6;
  const int count = style == TREE_STYLE_BUSH ? 2 : style == TREE_STYLE_ROUND ? 4 : style == TREE_STYLE_BIG ? 5 : 6;
  return low + int((skewed * uint(count)) >> 8u);
}
/** Lowest leaf layer, relative to the ground (canopy is [treeCanopyLow, treeCanopyHigh)). */
VEG_FN int treeCanopyLow(uint style, int trunkHeight) {
  return style == TREE_STYLE_ROUND ? trunkHeight - 2
         : style == TREE_STYLE_BIG ? trunkHeight - 3
         : style == TREE_STYLE_CONIFER ? 2
                                       : trunkHeight - 1;
}
/** Exclusive top leaf layer, relative to the ground. */
VEG_FN int treeCanopyHigh(uint style, int trunkHeight) {
  return style == TREE_STYLE_CONIFER ? trunkHeight + 1 : trunkHeight + 2;
}
// Leaf layer shapes, by layer counted down from the canopy top (0 = top).
VEG_CONST int TREE_LAYER_SINGLE = 0;   // one block above the trunk
VEG_CONST int TREE_LAYER_PLUS = 1;     // the trunk column and its 4 neighbours
VEG_CONST int TREE_LAYER_SMALL = 2;    // 3x3, corners dropped by shape bits
VEG_CONST int TREE_LAYER_DIAMOND = 3;  // |dx| + |dz| <= 2
VEG_CONST int TREE_LAYER_WIDE = 4;     // 5x5, corners dropped by shape bits
VEG_CONST int TREE_LAYER_FULL = 5;     // 5x5, corners dropped only when two shape bits agree (fuller)
/**
 * Leaf layer shape of a style.
 *
 * @param fromTop Layer index counted down from the canopy top (0 = top).
 * @return a TREE_LAYER_* value.
 */
VEG_FN int treeLeafLayerKind(uint style, int fromTop) {
  if (style == TREE_STYLE_CONIFER) {
    // Tip, plus, 3x3, then tiers alternating diamond / 3x3, wide from layer 7 down.
    return fromTop == 0 ? TREE_LAYER_SINGLE
           : fromTop == 1 ? TREE_LAYER_PLUS
           : fromTop == 2 || (fromTop & 1) == 0 ? TREE_LAYER_SMALL
           : fromTop >= 7 ? TREE_LAYER_WIDE
                          : TREE_LAYER_DIAMOND;
  }
  if (style == TREE_STYLE_BUSH) {
    return fromTop == 0 ? TREE_LAYER_SMALL : fromTop == 1 ? TREE_LAYER_WIDE : TREE_LAYER_DIAMOND;
  }
  if (style == TREE_STYLE_BIG) {
    return fromTop == 0 ? TREE_LAYER_PLUS : fromTop == 1 ? TREE_LAYER_SMALL : fromTop == 3 ? TREE_LAYER_FULL : TREE_LAYER_WIDE;
  }
  return fromTop == 0 ? TREE_LAYER_PLUS : fromTop == 1 ? TREE_LAYER_SMALL : TREE_LAYER_WIDE;
}

VEG_CONST uint TREE_NONE = 0u;
VEG_CONST uint TREE_LOG = 1u;
VEG_CONST uint TREE_LEAVES = 2u;

struct TreeInstance {
  int trunkX;      // world block coordinates of the trunk
  int trunkZ;
  int ground;      // first air cell of the trunk column: the trunk's lowest block
  int trunkHeight; // 0: no tree
  // Bits 0-11: leaf-corner drops (treeBlockAt), 12-13: style, 16-23: slope roll (treeSlopeRoll), 24-31: style roll.
  uint shape;
};

/** Integer absolute value (no std / intrinsic, so it compiles on both sides). */
VEG_FN int vegAbs(int v) { return v < 0 ? -v : v; }

/**
 * Integer value noise, smoothstep-interpolated; integer-only so CPU and GPU agree (all terms non-negative).
 *
 * @param x World block x.
 * @param z World block z.
 * @param shift Log2 of the lattice spacing, in blocks.
 * @return noise in 0..255.
 */
VEG_FN int treeValueNoise(int x, int z, int shift, uint seed) {
  const int size = 1 << shift;
  const int cellX = x >> shift;
  const int cellZ = z >> shift;
  const int fracX = x & (size - 1);
  const int fracZ = z & (size - 1);
  // smoothstep(f / size) * 256 = f^2 (3 size - 2 f) * 256 / size^3
  const int weightX = fracX * fracX * (3 * size - 2 * fracX) * 256 / (size * size * size);
  const int weightZ = fracZ * fracZ * (3 * size - 2 * fracZ) * 256 / (size * size * size);
  const int a = int(hashCell(int2(cellX, cellZ), seed) & 0xffu);
  const int b = int(hashCell(int2(cellX + 1, cellZ), seed) & 0xffu);
  const int c = int(hashCell(int2(cellX, cellZ + 1), seed) & 0xffu);
  const int d = int(hashCell(int2(cellX + 1, cellZ + 1), seed) & 0xffu);
  const int top = (a * (256 - weightX) + b * weightX) >> 8;
  const int bottom = (c * (256 - weightX) + d * weightX) >> 8;
  return (top * (256 - weightZ) + bottom * weightZ) >> 8;
}

/**
 * Biome at a world block: the nearest jittered Voronoi point (3x3 search) to the warped position.
 *
 * @return TREE_BIOME_FOREST, TREE_BIOME_WOODLAND or TREE_BIOME_PLAINS.
 */
VEG_FN int treeBiomeAt(int x, int z) {
  const int warpedX = x + (treeValueNoise(x, z, 6, SEED_BIOME_WARP) - 128) * TREE_BIOME_WARP / 128;
  const int warpedZ = z + (treeValueNoise(x, z, 6, SEED_BIOME_WARP ^ 0x9e3779b9u) - 128) * TREE_BIOME_WARP / 128;
  const int gridX = warpedX >> TREE_BIOME_SHIFT;
  const int gridZ = warpedZ >> TREE_BIOME_SHIFT;
  const int mask = (1 << TREE_BIOME_SHIFT) - 1;
  int best = 0x7fffffff;
  uint bestHash = 0u;
  for (int oz = -1; oz <= 1; ++oz) {
    for (int ox = -1; ox <= 1; ++ox) {
      const int2 g = int2(gridX + ox, gridZ + oz);
      const uint h = hashCell(g, SEED_BIOME);
      const int dx = (g.x << TREE_BIOME_SHIFT) + int(h & uint(mask)) - warpedX;
      const int dz = (g.y << TREE_BIOME_SHIFT) + int((h >> 8u) & uint(mask)) - warpedZ;
      const int dist = dx * dx + dz * dz;
      if (dist < best) {
        best = dist;
        bestHash = h;
      }
    }
  }
  const int kind = int((bestHash >> 16u) & 0xffu);
  return kind < TREE_BIOME_FOREST_SHARE     ? TREE_BIOME_FOREST
         : kind < TREE_BIOME_WOODLAND_SHARE ? TREE_BIOME_WOODLAND
                                            : TREE_BIOME_PLAINS;
}

/**
 * Keep chance from the biome and clump noise at a world block.
 *
 * @return chance out of 256.
 */
VEG_FN int treeChanceIn(int biome, int x, int z) {
  const int clump = treeValueNoise(x, z, 5, SEED_FOREST);
  const int minChance = treeBiomeMinChance(biome);
  return minChance + (treeBiomeMaxChance(biome) - minChance) * clump / 255;
}
/** treeChanceIn with the biome looked up at (x, z). */
VEG_FN int treeChanceAt(int x, int z) { return treeChanceIn(treeBiomeAt(x, z), x, z); }

/**
 * Set a tree's style (from the biome and shape's style roll) and trunk height (from hash bits 16-23).
 *
 * @param h The candidate's treeCandidateHash.
 * @return the restyled tree.
 */
VEG_FN TreeInstance treeStyled(TreeInstance tree, uint h, int biome) {
  const uint style = treeStyleFor(biome, tree.shape >> 24u);
  tree.shape = (tree.shape & ~(3u << 12u)) | (style << 12u);
  tree.trunkHeight = treeTrunkHeightFor(style, (h >> 16u) & 0xffu);
  return tree;
}

/** Placement hash of a region's candidate slot (4x4 hash keys per region). */
VEG_FN uint treeCandidateHash(int2 region, uint slot) {
  return hashCell(int2(region.x * 4 + int(slot & 3u), region.y * 4 + int(slot >> 2u)), SEED_TREE);
}

/**
 * A region's candidate before the density check: position, trunk height and shape from hashes alone (ground 0),
 * styled as woodland (treeCandidate restyles it; coarse LODs using this directly ignore style and height). Its
 * density roll is treeCandidateHash >> 24, kept when below treeChanceAt.
 *
 * @param region Tree region (chunk column) coordinates.
 * @param slot Candidate index, 0..TREE_SLOTS-1.
 * @return the unfiltered candidate.
 */
VEG_FN TreeInstance treeCandidateUnfiltered(int2 region, uint slot) {
  const uint h = treeCandidateHash(region, slot);
  const uint span = uint((1 << TREE_REGION_SHIFT) - 2 * TREE_RADIUS); // trunk positions per axis
  TreeInstance tree;
  tree.trunkX = (region.x << TREE_REGION_SHIFT) + TREE_RADIUS + int(((h & 0xffu) * span) >> 8u);
  tree.trunkZ = (region.y << TREE_REGION_SHIFT) + TREE_RADIUS + int((((h >> 8u) & 0xffu) * span) >> 8u);
  tree.ground = 0;
  tree.trunkHeight = 0;
  tree.shape = hashCell(int2(region.x * 4 + int(slot & 3u), region.y * 4 + int(slot >> 2u)), SEED_LEAF);
  return treeStyled(tree, h, TREE_BIOME_WOODLAND);
}

/**
 * A region's candidate with the density applied. The caller sets `ground` from the trunk column and checks
 * treeGroundOk.
 *
 * @param region Tree region (chunk column) coordinates.
 * @param slot Candidate index, 0..TREE_SLOTS-1.
 * @return the candidate, with trunkHeight 0 if rejected.
 */
VEG_FN TreeInstance treeCandidate(int2 region, uint slot) {
  TreeInstance tree = treeCandidateUnfiltered(region, slot);
  const uint h = treeCandidateHash(region, slot);
  const int biome = treeBiomeAt(tree.trunkX, tree.trunkZ);
  if (int(h >> 24u) >= treeChanceIn(biome, tree.trunkX, tree.trunkZ)) {
    tree.trunkHeight = 0;
    return tree;
  }
  return treeStyled(tree, h, biome);
}

// Slope thinning from the pre-erosion terrain gradient at the trunk at lodScale 1, so every LOD (CPU
// terrain::sampleHeightsGrad8, GPU stageTreeSlots) keeps the same trees. The float gradient may differ by an ulp
// between CPU and GPU, flipping steepness only right at a rounding edge.
VEG_CONST int TREE_SLOPE_ZERO = 10;
VEG_CONST int TREE_STEEPNESS_NONE = 1000; // no tree: rejected by every slope check
/**
 * Steepness d = round(7 (|dh/dx| + |dh/dz|)): blocks of rise per 7 blocks, both axes summed.
 *
 * @return d, or TREE_STEEPNESS_NONE for extreme slopes.
 */
VEG_FN int treeSteepness(float gradX, float gradZ) {
  const float slope = (gradX < 0.0f ? -gradX : gradX) + (gradZ < 0.0f ? -gradZ : gradZ);
  return slope >= 100.0f ? TREE_STEEPNESS_NONE : int(slope * 7.0f + 0.5f);
}
/**
 * Keep chance at a steepness: quadratic falloff to 0 at TREE_SLOPE_ZERO.
 *
 * @return chance out of 256.
 */
VEG_FN uint treeSlopeKeep(int steepness) {
  const int left = TREE_SLOPE_ZERO - steepness;
  return left <= 0 ? 0u : uint(left * left * 256 / (TREE_SLOPE_ZERO * TREE_SLOPE_ZERO));
}
/** Slope roll (shape bits 16-23; treeBlockAt only reads bits 0-13, so no extra hash). */
VEG_FN uint treeSlopeRoll(uint shape) { return (shape >> 16u) & 0xffu; }
/** Whether the slope roll passes at a steepness. */
VEG_FN bool treeSlopeOk(int steepness, uint roll) { return roll < treeSlopeKeep(steepness); }

/**
 * Whether a tree's ground is in the trunk band and its slope roll passes. No cliff check: coarse LODs lack exact
 * neighbour heights, and every LOD must accept the same trees.
 *
 * @param ground First air cell of the trunk column.
 * @param steepness treeSteepness at the trunk.
 * @return true if the tree is kept.
 */
VEG_FN bool treeGroundOk(int ground, int steepness, uint shape) {
  return ground >= TREE_MIN_GROUND && ground <= TREE_MAX_GROUND && treeSlopeOk(steepness, treeSlopeRoll(shape));
}

/**
 * Tree block at a world block, ignoring terrain (callers let terrain win). Trunk spans [ground, ground +
 * trunkHeight); leaf corner drops use shape bit (fromTop % 3) * 4 + corner, so stacked layers drop different corners.
 *
 * @return TREE_NONE, TREE_LOG or TREE_LEAVES.
 */
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
  const uint style = treeStyle(tree.shape);
  const int high = treeCanopyHigh(style, tree.trunkHeight);
  if (dy < treeCanopyLow(style, tree.trunkHeight) || dy >= high) {
    return TREE_NONE;
  }
  const int fromTop = high - 1 - dy;
  const int kind = treeLeafLayerKind(style, fromTop);
  const int adx = vegAbs(dx); // |dx|, |dz|
  const int adz = vegAbs(dz);
  const uint corner = (dx > 0 ? 1u : 0u) | (dz > 0 ? 2u : 0u);
  const uint bits = tree.shape >> (uint(fromTop % 3) * 4u);
  const bool dropCorner = ((bits >> corner) & 1u) != 0u;
  if (kind == TREE_LAYER_SINGLE) {
    return adx == 0 && adz == 0 ? TREE_LEAVES : TREE_NONE;
  }
  if (kind == TREE_LAYER_PLUS) {
    return adx + adz <= 1 ? TREE_LEAVES : TREE_NONE;
  }
  if (kind == TREE_LAYER_DIAMOND) {
    return adx + adz <= 2 ? TREE_LEAVES : TREE_NONE;
  }
  const int radius = kind == TREE_LAYER_SMALL ? 1 : TREE_RADIUS;
  if (adx > radius || adz > radius) {
    return TREE_NONE;
  }
  if (adx == radius && adz == radius) {
    const bool drop = kind == TREE_LAYER_FULL ? dropCorner && ((bits >> ((corner + 1u) & 3u)) & 1u) != 0u : dropCorner;
    if (drop) {
      return TREE_NONE;
    }
  }
  return TREE_LEAVES;
}

// ---- Coarse LODs (chunk_compute.slang computeTreeColumn) ----
// Every LOD shows the same trees. LOD 1-2: leaves where a canopy covers at least half a column, at the canopy's real
// height. LOD 3+: one leaf cell per tree whose nested roll passes (chance TREE_CANOPY_AREA / scale^2, so far forests
// keep the same ground cover); LOD 4+ re-derive candidates and judge slope from one sample at the column centre.
VEG_CONST int TREE_MAX_LOD = 7;
VEG_CONST uint SEED_TREE_LOD = 0xa54ff53au;
VEG_CONST int TREE_CANOPY_AREA = 24; // top-down blocks of a canopy: 5x5, corners kept ~3/4 of the time

/**
 * Whether a tree shows as a cell at a coarse scale: its fixed roll from the trunk position against
 * TREE_CANOPY_AREA / scale^2. Nested: shown at 2s implies shown at s.
 *
 * @param scale Voxel size, in blocks.
 * @return true if the tree shows.
 */
VEG_FN bool treeShowsAtScale(int trunkX, int trunkZ, int scale) {
  const uint roll = hashCell(int2(trunkX, trunkZ), SEED_TREE_LOD) & 0xffffu;
  return roll * uint(scale * scale) < uint(TREE_CANOPY_AREA) * 65536u;
}

/**
 * Overlap of a trunk's canopy square with the column [columnX, columnX + scale) x [columnZ, columnZ + scale).
 *
 * @return overlapping blocks, top-down.
 */
VEG_FN int treeCanopyOverlap(int trunkX, int trunkZ, int columnX, int columnZ, int scale) {
  const int lowX = trunkX - TREE_RADIUS > columnX ? trunkX - TREE_RADIUS : columnX;
  const int highX = trunkX + TREE_RADIUS + 1 < columnX + scale ? trunkX + TREE_RADIUS + 1 : columnX + scale;
  const int lowZ = trunkZ - TREE_RADIUS > columnZ ? trunkZ - TREE_RADIUS : columnZ;
  const int highZ = trunkZ + TREE_RADIUS + 1 < columnZ + scale ? trunkZ + TREE_RADIUS + 1 : columnZ + scale;
  return highX > lowX && highZ > lowZ ? (highX - lowX) * (highZ - lowZ) : 0;
}

/** LOD 1-2: whether a canopy covers at least half of a column. */
VEG_FN bool treeCanopyShows(int overlapBlocks, int scale) { return 2 * overlapBlocks >= scale * scale; }

/**
 * Coarse leaf cells of a tree: max(1, round(span / scale)) cells centred on the canopy's middle, lifted to at least
 * columnTop so the terrain never swallows them.
 *
 * @param scale Voxel size, in blocks.
 * @param columnTop The column's first air cell, in cells of this scale.
 * @return (first cell, count).
 */
VEG_FN int2 treeLeafCells(int ground, int trunkHeight, uint shape, int scale, int columnTop) {
  const uint style = treeStyle(shape);
  const int low = treeCanopyLow(style, trunkHeight);
  const int span = treeCanopyHigh(style, trunkHeight) - low;
  const int count = (span + scale / 2) / scale > 0 ? (span + scale / 2) / scale : 1;
  const int center2 = 2 * ground + 2 * low + span; // twice the canopy middle (world)
  const int numerator = center2 - count * scale + scale;
  // Floor division by 2 * scale.
  const int first = numerator >= 0 ? numerator / (2 * scale) : -((-numerator + 2 * scale - 1) / (2 * scale));
  return int2(first > columnTop ? first : columnTop, count);
}

#ifdef __cplusplus
} // namespace vegetation
#endif
#undef VEG_CONST
#undef VEG_FN
