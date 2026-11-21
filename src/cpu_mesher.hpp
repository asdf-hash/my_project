#pragma once

#include "cpu_chunk.hpp"

namespace cpuworld {

// Meshes voxel chunk chunkY of nb.center() into the GPU quad format (writeQuad in chunk_compute.slang): 16 clusters
// of 8x8 columns, each with six direction runs (+Y, -X, +X, -Z, +Z, -Y), with the GPU mesher's AO and merge rules.
// Solidity and AO come from bitmask columns processed 8 at a time (AVX2), so per face cell only the emission is
// scalar.
void meshChunk(const Neighborhood &nb, int32_t chunkY, ChunkMesh &out);

// Packing helpers shared with the tests.
uint64_t packQuad(uint32_t x, uint32_t y, uint32_t z, uint32_t normalIndex, uint32_t ao, uint32_t blockType,
                  uint32_t length, uint32_t width);
uint64_t packChunkOffset(int32_t cx, int32_t chunkY, int32_t cz);

} // namespace cpuworld
