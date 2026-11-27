#pragma once

#include "cpu_chunk.hpp"

namespace cpuworld {

/**
 * Mesh one voxel chunk of the neighborhood's center column into the GPU quad format.
 *
 * Output matches writeQuad in chunk_compute.slang: 16 clusters of 8x8 columns, each with six direction runs
 * (+Y, -X, +X, -Z, +Z, -Y), using the GPU mesher's AO and merge rules. Solidity and AO come from bitmask columns
 * processed 8 at a time (AVX2), so only per-face emission is scalar.
 *
 * @param nb Neighborhood: the center column plus its 8 neighbors (null where not loaded).
 * @param chunkY Vertical chunk index within the column, in chunks.
 * @param out Receives the quads, clusters and chunk offset.
 */
void meshChunk(const Neighborhood &nb, int32_t chunkY, ChunkMesh &out);

} // namespace cpuworld
