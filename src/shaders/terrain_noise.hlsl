#include "vegetation_shared.hlsl"

// Terrain height field after Big Globe's V6 overworld, without rivers; sea level y = 0. Layers: continental
// elevation (value noise) shaped by baseProfile, Perlin hills, a land floor at y = 1, mountain ranges along the zero
// lines of a noise field r, and branching downhill gully erosion. Also compiled as C++ by terrain_cpu.cpp: keep it
// bit-exact with the CPU port. Layers shorter than two voxels at the LOD scale are skipped.
// Output stays within [kMinTerrainHeight, kMaxTerrainHeight] in world.hpp.

static const float TERRAIN_MIN_HEIGHT = -400.0f;
static const float TERRAIN_MAX_HEIGHT = 1100.0f;

static const float MOUNTAIN_HEIGHT = 900.0f;
static const float RANGE_WAVELENGTH = 9000.0f;   // spacing of range spines
static const float RANGE_HALF_WIDTH = 0.2f;      // in units of r: main belt around the spine
static const float FOOTHILL_HALF_WIDTH = 0.42f;  // in units of r: lower foothills
static const float FOOTHILL_SHARE = 0.18f;       // fraction of mountain height that is foothills
static const float REGION_WAVELENGTH = 16000.0f; // regions with and without ranges
static const float PEAK_WAVELENGTH = 1800.0f;    // peaks and passes along a range

// Land: raw elevation (0..~253) compressed into plains that level off near PLAINS_HEIGHT.
static const float PLAINS_HEIGHT = 70.0f;
static const float PLAINS_SCALE = 180.0f;

// Raw elevation shift: land covers about two thirds of the world, and the extra term lifts the coast BEACH_LIFT
// blocks (land rises PLAINS_HEIGHT / PLAINS_SCALE per unit of elevation there) so beaches sit above sea level.
static const float BEACH_LIFT = 2.0f;
static const float ELEVATION_BIAS = 35.0f + BEACH_LIFT * PLAINS_SCALE / PLAINS_HEIGHT;

// Rolling hills on land (4 Perlin octaves, like Minecraft plains scaled 2x), faded out at the shore.
static const float HILL_WAVELENGTH = 380.0f;
static const float HILL_HEIGHT = 32.0f;
static const float HILL_SHORE_FADE = 48.0f; // base height over which hills fade in above sea level

// Ocean floor steps, in raw elevation below sea level (ELEVATION_MAX is the noise's largest magnitude).
static const float ELEVATION_MAX = 253.0f;
static const float SHELF_EXTENT = 70.0f; // continental shelf: 0 .. -SHELF_DEPTH, leveling out at its edge
static const float SHELF_DEPTH = 30.0f;
static const float SLOPE_EXTENT = 40.0f; // continental slope: steep drop of SLOPE_DEPTH
static const float SLOPE_DEPTH = 200.0f;
static const float ABYSS_DEPTH = 40.0f; // abyssal plain: deepens by this much over the rest of the range

// Where the continent layer (wavelength 4096) says land, the surface is smoothly floored at LAND_FLOOR, faded in
// over COAST_BAND units of continent elevation inland of the coast.
static const float LAND_FLOOR = 1.0f;
static const float LAND_FLOOR_SMOOTHING = 4.0f; // blocks over which the floor blends into the surface above it
static const float COAST_BAND = 24.0f;

static const float EROSION_AMPLITUDE = 128.0f;
static const float EROSION_MAX_WAVELENGTH = 512.0f;
static const float EROSION_MIN_WAVELENGTH = 4.0f;

static const uint SEED_ELEVATION = 0x1b873593u;
static const uint SEED_RANGES = 0xcc9e2d51u;
static const uint SEED_RANGE_REGIONS = 0x85ebca6bu;
static const uint SEED_PEAKS = 0xc2b2ae35u;
static const uint SEED_HILLS = 0xd3a2646cu;
static const uint SEED_EROSION = 0x27d4eb2fu;
static const uint SEED_EROSION_STEP = 0x165667b1u;
static const uint SEED_LAYER_STEP = 0x9e3779b9u;

static const float TAU = 6.28318530718f;

// Deterministic polynomial transcendentals: hardware sin/cos/exp differ across GPU vendors and from the CPU.
// Must match sinCosPoly / tanhPoly in terrain_cpu.cpp.
static const float PIO2_1 = 1.5703125f;
static const float PIO2_2 = 4.837512969970703125e-4f;
static const float PIO2_3 = 7.54978995489188216e-8f;
static const float TWO_OVER_PI = 0.636619772367581343f;

/**
 * Sine and cosine via Cody-Waite reduction to [-pi/4, pi/4] and minimax polynomials.
 *
 * @param s Receives sin(x).
 * @param c Receives cos(x).
 */
void sinCosPoly(float x, out float s, out float c) {
  const float q = floor(x * TWO_OVER_PI + 0.5f);
  const float r = ((x - q * PIO2_1) - q * PIO2_2) - q * PIO2_3; // |r| <= pi/4
  const float z = r * r;
  const float sr = r + r * z * (-1.6666654611e-1f + z * (8.3321608736e-3f + z * -1.9515295891e-4f));
  const float cr = 1.0f - 0.5f * z + z * z * (4.166664568298827e-2f + z * (-1.388731625493765e-3f + z * 2.443315711809948e-5f));
  const int quadrant = int(q) & 3;
  s = (quadrant & 1) != 0 ? cr : sr;
  c = (quadrant & 1) != 0 ? sr : cr;
  if (quadrant == 1 || quadrant == 2) {
    c = -c;
  }
  if (quadrant >= 2) {
    s = -s;
  }
}

/**
 * Polynomial exp(x), with x clamped to [-87, 87].
 *
 * @return e^x.
 */
float expPoly(float x) {
  x = clamp(x, -87.0f, 87.0f);
  const float n = floor(x * 1.44269504088896341f + 0.5f);
  const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
  const float z = r * r;
  const float p =
      ((((((1.9875691500e-4f * r + 1.3981999507e-3f) * r + 8.3334519073e-3f) * r + 4.1665795894e-2f) * r +
         1.6666665459e-1f) * r + 5.0000001201e-1f) * z + r) + 1.0f;
  return p * asfloat(uint(int(n) + 127) << 23u);
}

/**
 * Polynomial tanh(x): direct odd polynomial below |x| = 0.625, else 1 - 2 / (e^2|x| + 1).
 *
 * @return tanh(x).
 */
float tanhPoly(float x) {
  const float a = abs(x);
  float t;
  if (a < 0.625f) {
    const float z = x * x;
    return x + x * z * ((((-5.70498872745e-3f * z + 2.06390887954e-2f) * z - 5.37397155531e-2f) * z +
                         1.33314422036e-1f) * z - 3.33332819422e-1f);
  }
  t = 1.0f - 2.0f / (expPoly(2.0f * a) + 1.0f);
  return x < 0.0f ? -t : t;
}

/**
 * Lattice value of binaryNoise at a cell: +1 or -1 from hashCell (vegetation_shared.hlsl; SIMD twin in
 * terrain_cpu.cpp).
 *
 * @return +1.0 or -1.0.
 */
float latticeSign(int2 cell, uint seed) { return (hashCell(cell, seed) & 1u) != 0u ? 1.0f : -1.0f; }

/**
 * Value noise with lattice values +-1 and smootherstep interpolation.
 *
 * @param worldPos World-space (x, z), in blocks.
 * @param wavelength Lattice spacing, in blocks.
 * @return (value, d/dx, d/dz), derivatives in world units.
 */
float3 binaryNoise(float2 worldPos, float wavelength, uint seed) {
  const float2 g = worldPos / wavelength;
  const float2 cellFloor = floor(g);
  const int2 cell = int2(cellFloor);
  const float2 t = g - cellFloor;
  const float2 u = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
  const float2 du = 30.0f * t * t * (t * (t - 2.0f) + 1.0f);
  const float a = latticeSign(cell, seed);
  const float b = latticeSign(cell + int2(1, 0), seed);
  const float c = latticeSign(cell + int2(0, 1), seed);
  const float d = latticeSign(cell + int2(1, 1), seed);
  const float k = a - b - c + d;
  return float3(a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y, du.x * ((b - a) + k * u.y) / wavelength,
                du.y * ((c - a) + k * u.x) / wavelength);
}

/**
 * Sum of binaryNoise layers; wavelength halves and amplitude scales by 0.4 per layer.
 *
 * @param wavelength First layer's wavelength, in blocks.
 * @param amplitude First layer's amplitude.
 * @param lodScale Voxel size in blocks; layers shorter than two voxels are skipped (the first always runs).
 * @param firstLayer Receives the first layer's value alone.
 * @return (value, d/dx, d/dz).
 */
float3 layeredNoise(float2 worldPos, float wavelength, float amplitude, uint layers, uint seed, int lodScale,
                    out float firstLayer) {
  const float minWavelength = 2.0f * float(lodScale);
  float3 sum = 0.0f;
  firstLayer = 0.0f;
  for (uint i = 0u; i < layers; ++i) {
    if (i > 0u && wavelength < minWavelength) {
      break;
    }
    const float3 layer = amplitude * binaryNoise(worldPos, wavelength, seed + i * SEED_LAYER_STEP);
    if (i == 0u) {
      firstLayer = layer.x;
    }
    sum += layer;
    wavelength *= 0.5f;
    amplitude *= 0.4f;
  }
  return sum;
}

/**
 * 2D Perlin noise with derivative.
 *
 * @param worldPos World-space (x, z), in blocks.
 * @param wavelength Lattice spacing, in blocks.
 * @return (value in about [-1, 1], d/dx, d/dz), derivatives in world units.
 */
float3 perlinNoise(float2 worldPos, float wavelength, uint seed) {
  static const float2 GRADIENTS[8] = { float2(1.0f, 0.0f),        float2(-1.0f, 0.0f),       float2(0.0f, 1.0f),
                                       float2(0.0f, -1.0f),       float2(0.7071f, 0.7071f),  float2(-0.7071f, 0.7071f),
                                       float2(0.7071f, -0.7071f), float2(-0.7071f, -0.7071f) };
  const float2 g = worldPos / wavelength;
  const float2 cellFloor = floor(g);
  const int2 cell = int2(cellFloor);
  const float2 f = g - cellFloor;
  const float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
  const float2 du = 30.0f * f * f * (f * (f - 2.0f) + 1.0f);
  const float2 g00 = GRADIENTS[hashCell(cell, seed) & 7u];
  const float2 g10 = GRADIENTS[hashCell(cell + int2(1, 0), seed) & 7u];
  const float2 g01 = GRADIENTS[hashCell(cell + int2(0, 1), seed) & 7u];
  const float2 g11 = GRADIENTS[hashCell(cell + int2(1, 1), seed) & 7u];
  const float n00 = dot(g00, f);
  const float n10 = dot(g10, f - float2(1.0f, 0.0f));
  const float n01 = dot(g01, f - float2(0.0f, 1.0f));
  const float n11 = dot(g11, f - float2(1.0f, 1.0f));
  const float k = n00 - n10 - n01 + n11;
  const float value = n00 + u.x * (n10 - n00) + u.y * (n01 - n00) + u.x * u.y * k;
  const float2 gradient = g00 + u.x * (g10 - g00) + u.y * (g01 - g00) + u.x * u.y * (g00 - g10 - g01 + g11) +
                          du * float2(n10 - n00 + u.y * k, n01 - n00 + u.x * k);
  return float3(value, gradient / wavelength) * 1.4142f;
}

/**
 * Perlin fbm (amplitude halves per octave), normalized to about [-1, 1].
 *
 * @param wavelength First octave's wavelength, in blocks.
 * @param lodScale Voxel size in blocks; octaves shorter than two voxels are skipped (the first always runs).
 * @return (value, d/dx, d/dz).
 */
float3 perlinFbm(float2 worldPos, float wavelength, uint octaves, uint seed, int lodScale) {
  const float minWavelength = 2.0f * float(lodScale);
  float3 sum = 0.0f;
  float amplitude = 1.0f;
  float totalAmplitude = 0.0f;
  for (uint i = 0u; i < octaves; ++i) {
    if (i > 0u && wavelength < minWavelength) {
      break;
    }
    sum += amplitude * perlinNoise(worldPos, wavelength, seed + i * SEED_LAYER_STEP);
    totalAmplitude += amplitude;
    wavelength *= 0.5f;
    amplitude *= 0.5f;
  }
  return sum / totalAmplitude;
}

/**
 * Gully erosion: sine waves aligned downhill, wavelengths EROSION_MAX_WAVELENGTH down to two voxels. Each octave
 * bends the slope seen by the next and fades out on crests, so gullies branch.
 *
 * @param p World-space (x, z), in blocks.
 * @param y Height before erosion.
 * @param grad Slope (d/dx, d/dz) of y that orients the gullies.
 * @param amplitude Largest octave's amplitude, in blocks.
 * @param lodScale Voxel size, in blocks.
 * @return the eroded height.
 */
float erode(float2 p, float y, float2 grad, float amplitude, int lodScale) {
  float mask = 1.0f - 1.0f / (dot(grad, grad) * 2.0f + 1.0f);
  const float minWavelength = max(EROSION_MIN_WAVELENGTH, 2.0f * float(lodScale));
  uint octave = 0u;
  for (float wavelength = EROSION_MAX_WAVELENGTH; wavelength >= minWavelength; wavelength *= 0.5f, ++octave) {
    const float gradLength = length(grad);
    if (gradLength == 0.0f) {
      break;
    }
    const float2 across = float2(-grad.y, grad.x) / gradLength; // along the contour; waves vary across slope
    const float2 coord = p / wavelength;
    const float2 cellFloor = floor(coord);
    const int2 cell = int2(cellFloor);
    float2 s = coord - cellFloor;
    s = s * s * (3.0f - 2.0f * s);
    const uint seed = SEED_EROSION + octave * SEED_EROSION_STEP;

    float4 sines;
    float4 cosines;
    [unroll]
    for (uint corner = 0u; corner < 4u; ++corner) {
      const int2 cornerCell = cell + int2(corner & 1u, corner >> 1u);
      const float phase = float(hashCell(cornerCell, seed) >> 8u) * (TAU / 16777216.0f);
      float sine, cosine;
      sinCosPoly(dot(across, coord - float2(cornerCell)) * TAU + phase, sine, cosine);
      sines[corner] = sine;
      cosines[corner] = cosine;
    }
    const float wave = lerp(lerp(sines.x, sines.y, s.x), lerp(sines.z, sines.w, s.x), s.y);
    const float waveSlope = lerp(lerp(cosines.x, cosines.y, s.x), lerp(cosines.z, cosines.w, s.x), s.y);

    y += wave * mask * amplitude * wavelength / EROSION_MAX_WAVELENGTH;
    grad += waveSlope * across * mask;
    mask *= sqrt(saturate(1.0f - wave * wave));
  }
  return y;
}

/**
 * smoothstep and its derivative with respect to x.
 *
 * @return (value, d/dx).
 */
float2 smoothstepD(float edge0, float edge1, float x) {
  const float t = saturate((x - edge0) / (edge1 - edge0));
  return float2(t * t * (3.0f - 2.0f * t), 6.0f * t * (1.0f - t) / (edge1 - edge0));
}

/**
 * Base height from raw elevation: land compressed into plains leveling off near PLAINS_HEIGHT; ocean as shelf,
 * steep continental slope, then abyssal plain. Must stay C1: erosion follows the slope, so a kink becomes a fault.
 *
 * @param e Raw elevation (ELEVATION_BIAS applied); >= 0 is land.
 * @return (height, d height / d e, steering slope = d height / d e without the continental-slope and abyss terms).
 */
float3 baseProfile(float e) {
  static const float COAST_SLOPE = PLAINS_HEIGHT / PLAINS_SCALE; // land's d height / d e at the coast
  if (e >= 0.0f) {
    const float t = tanhPoly(e / PLAINS_SCALE);
    const float slope = COAST_SLOPE * (1.0f - t * t);
    return float3(PLAINS_HEIGHT * t, slope, slope);
  }
  const float depthIn = -e;
  // Shelf: cubic Hermite from (0, slope COAST_SLOPE) to (SHELF_EXTENT, SHELF_DEPTH, slope 0), so it leaves the coast
  // at land's slope and levels out where the continental slope begins.
  const float t = saturate(depthIn / SHELF_EXTENT);
  const float startTangent = COAST_SLOPE * SHELF_EXTENT;
  const float shelf = startTangent * (t - 2.0f * t * t + t * t * t) + SHELF_DEPTH * t * t * (3.0f - 2.0f * t);
  const float dShelf =
      (startTangent * (1.0f - 4.0f * t + 3.0f * t * t) + SHELF_DEPTH * 6.0f * t * (1.0f - t)) / SHELF_EXTENT;
  const float2 slope = smoothstepD(SHELF_EXTENT, SHELF_EXTENT + SLOPE_EXTENT, depthIn);
  const float2 abyss = smoothstepD(SHELF_EXTENT + SLOPE_EXTENT, ELEVATION_MAX, depthIn);
  const float depth = shelf + SLOPE_DEPTH * slope.x + ABYSS_DEPTH * abyss.x;
  const float dDepth = dShelf + SLOPE_DEPTH * slope.y + ABYSS_DEPTH * abyss.y;
  return float3(-depth, dDepth, dShelf); // d(-depth)/de = d depth / d depthIn
}

/**
 * Mountain ranges: a belt plus foothills around the zero lines of noise r, masked by region, varied by peak noise.
 *
 * @param p World-space (x, z), in blocks.
 * @param lodScale Voxel size, in blocks.
 * @return (height, d/dx, d/dz).
 */
float3 mountainRanges(float2 p, int lodScale) {
  const float3 regionNoise = perlinFbm(p, REGION_WAVELENGTH, 2u, SEED_RANGE_REGIONS, lodScale);
  const float2 region = smoothstepD(-0.3f, 0.1f, regionNoise.x);
  if (region.x <= 0.0f) {
    return 0.0f;
  }

  // Cross-section around the spine r = 0: main belt plus wider, lower foothills.
  const float3 r = perlinFbm(p, RANGE_WAVELENGTH, 4u, SEED_RANGES, lodScale);
  const float absR = abs(r.x);
  const float2 belt = smoothstepD(RANGE_HALF_WIDTH, 0.0f, absR);
  const float2 foothills = smoothstepD(FOOTHILL_HALF_WIDTH, 0.0f, absR);
  const float profile = lerp(belt.x * belt.x, foothills.x, FOOTHILL_SHARE);
  const float dProfile = lerp(2.0f * belt.x * belt.y, foothills.y, FOOTHILL_SHARE);
  if (profile <= 0.0f) {
    return 0.0f;
  }

  // Peaks and passes along the range.
  const float3 peakNoise = perlinFbm(p, PEAK_WAVELENGTH, 3u, SEED_PEAKS, lodScale);
  // C1 ramp from passes (0) to peaks (1); a hard clamp would kink the slope into an erosion fault.
  const float2 peakRamp = smoothstepD(-1.1f, 0.6f, peakNoise.x);
  const float peaks = peakRamp.x;
  const float dPeaks = peakRamp.y;

  const float height = MOUNTAIN_HEIGHT * region.x * profile * peaks;
  const float2 gradient =
      MOUNTAIN_HEIGHT * (region.y * regionNoise.yz * profile * peaks + region.x * dProfile * sign(r.x) * r.yz * peaks +
                         region.x * profile * dPeaks * peakNoise.yz);
  return float3(height, gradient);
}

/**
 * World-space surface height at a block column.
 *
 * @param lodScale Voxel size of the chunk being generated, in blocks.
 * @param grad Receives the pre-erosion steering slope (d/dx, d/dz). Approximate (no erosion, sea-floor slope terms,
 *        land floor or clamp) and float, so only for GPU-only decisions (coarse tree LODs).
 * @return surface height, clamped to [TERRAIN_MIN_HEIGHT, TERRAIN_MAX_HEIGHT].
 */
float sampleTerrainHeightGrad(int worldX, int worldZ, int lodScale, out float2 grad) {
  const float2 p = float2(float(worldX), float(worldZ));

  float continentLayer;
  const float3 elevation = layeredNoise(p, 4096.0f, 153.6f, 5u, SEED_ELEVATION, lodScale, continentLayer);
  const float3 base = baseProfile(elevation.x + ELEVATION_BIAS);
  const float3 mountains = mountainRanges(p, lodScale);

  const float3 hillNoise = perlinFbm(p, HILL_WAVELENGTH, 4u, SEED_HILLS, lodScale);
  const float2 shore = smoothstepD(0.0f, HILL_SHORE_FADE, base.x);
  const float hills = HILL_HEIGHT * shore.x * hillNoise.x;

  const float height = base.x + hills + mountains.x;
  // Erosion's steering slope: uses base.z (no continental-slope / abyss terms), whose sharp slope changes would
  // become erosion faults.
  const float2 gradHeight = base.z * elevation.yz + mountains.yz +
                            HILL_HEIGHT * (shore.x * hillNoise.yz + hillNoise.x * shore.y * base.z * elevation.yz);

  grad = gradHeight;

  const float erosionAmplitude = EROSION_AMPLITUDE * saturate(mountains.x / MOUNTAIN_HEIGHT * 3.0f);
  // Zero amplitude leaves the height unchanged; skip the octaves (terrain_cpu.cpp does the same).
  const float eroded = erosionAmplitude > 0.0f ? erode(p, height, gradHeight, erosionAmplitude, lodScale) : height;

  // Land floor: polynomial smooth max with LAND_FLOOR, faded in from the coast inland.
  const float landWeight = smoothstepD(0.0f, COAST_BAND, continentLayer + ELEVATION_BIAS).x;
  const float floorBlend = max(LAND_FLOOR_SMOOTHING - abs(eroded - LAND_FLOOR), 0.0f) / LAND_FLOOR_SMOOTHING;
  const float floored = max(eroded, LAND_FLOOR) + floorBlend * floorBlend * LAND_FLOOR_SMOOTHING * 0.25f;
  const float surface = floored * landWeight + eroded * (1.0f - landWeight); // exact floor where landWeight = 1

  return clamp(surface, TERRAIN_MIN_HEIGHT, TERRAIN_MAX_HEIGHT);
}

/** sampleTerrainHeightGrad without the slope output. */
float sampleTerrainHeight(int worldX, int worldZ, int lodScale) {
  float2 grad;
  return sampleTerrainHeightGrad(worldX, worldZ, lodScale, grad);
}
