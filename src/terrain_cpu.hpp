#pragma once

// CPU port of src/shaders/terrain_noise.hlsl (and transitionDither from chunk_compute.slang), evaluated 8 columns at
// a time with AVX2+FMA (scalar lanes without it, see simd.hpp). Always LOD 0: every noise octave runs.
// Keep in sync with the shader: CPU-meshed chunks sit right next to GPU-meshed ones.

#include <cstdint>

namespace terrain {

/**
 * Sample the terrain surface height at 8 world columns.
 *
 * @param worldX 8 world block x coordinates.
 * @param worldZ 8 world block z coordinates.
 * @param out Receives 8 fractional heights, in blocks.
 */
void sampleHeights8(const int32_t *worldX, const int32_t *worldZ, float *out);

/**
 * Sample the surface height plus the pre-erosion slope it is built from (terrain_noise.hlsl
 * sampleTerrainHeightGrad) at 8 world columns.
 *
 * @param worldX 8 world block x coordinates.
 * @param worldZ 8 world block z coordinates.
 * @param outHeight Receives 8 heights, in blocks.
 * @param outGradX Receives 8 values of d height / dx.
 * @param outGradZ Receives 8 values of d height / dz.
 */
void sampleHeightsGrad8(const int32_t *worldX, const int32_t *worldZ, float *outHeight, float *outGradX,
                        float *outGradZ);

/**
 * Sample the per-column material dither (chunk_compute.slang transitionDither at scale 1) at 8 world columns.
 *
 * @param worldX 8 world block x coordinates.
 * @param worldZ 8 world block z coordinates.
 * @param out Receives 8 dither values in [-1, 1].
 */
void sampleDither8(const int32_t *worldX, const int32_t *worldZ, float *out);

/**
 * Sample continent elevation (the elevation's coarsest layer plus its bias) at 8 world columns.
 *
 * Land (never below sea level) where the result is >= kCoastBand; the land floor fades in over [0, kCoastBand).
 *
 * @param worldX 8 world block x coordinates.
 * @param worldZ 8 world block z coordinates.
 * @param out Receives 8 continent elevations.
 */
void sampleContinent8(const int32_t *worldX, const int32_t *worldZ, float *out);
inline constexpr float kCoastBand = 24.0f;

/**
 * Sample the surface height at a single world column (broadcast to 8 lanes).
 *
 * @return fractional surface height, in blocks.
 */
float sampleHeight(int32_t worldX, int32_t worldZ);

} // namespace terrain
