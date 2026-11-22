#pragma once

// CPU port of src/shaders/terrain_noise.hlsl (and transitionDither from chunk_compute.slang), evaluated 8 columns at
// a time with AVX2+FMA (scalar lanes without it, see simd.hpp). Always LOD 0: every noise octave runs.
// Keep in sync with the shader: CPU-meshed chunks sit right next to GPU-meshed ones.

#include <cstdint>

namespace terrain {

// Surface height (in blocks, fractional) at 8 world columns.
void sampleHeights8(const int32_t *worldX, const int32_t *worldZ, float *out);
// Per-column material dither in [-1, 1] (chunk_compute.slang transitionDither at scale 1).
void sampleDither8(const int32_t *worldX, const int32_t *worldZ, float *out);

// Continent elevation at 8 world columns: the elevation's coarsest layer plus its bias. Land (never below sea level)
// where it is >= kCoastBand; the land floor fades in over [0, kCoastBand).
void sampleContinent8(const int32_t *worldX, const int32_t *worldZ, float *out);
inline constexpr float kCoastBand = 24.0f;

// Single-column conveniences (broadcast to 8 lanes).
float sampleHeight(int32_t worldX, int32_t worldZ);
float sampleDither(int32_t worldX, int32_t worldZ);

} // namespace terrain
