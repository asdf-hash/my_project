#include "terrain_cpu.hpp"

#include "simd.hpp"
#include "world.hpp"

// Line-by-line port of terrain_noise.hlsl. Operation order follows the shader so both sides round alike; the
// remaining few-ulp differences (FMA contraction, division) are absorbed by world::kCpuSeamBias.

namespace terrain {
namespace {

using simd::F8;
using simd::I8;
using simd::M8;

constexpr float TERRAIN_MIN_HEIGHT = -400.0f;
constexpr float TERRAIN_MAX_HEIGHT = 1100.0f;
constexpr float MOUNTAIN_HEIGHT = 900.0f;
constexpr float RANGE_WAVELENGTH = 9000.0f;
constexpr float RANGE_HALF_WIDTH = 0.2f;
constexpr float FOOTHILL_HALF_WIDTH = 0.42f;
constexpr float FOOTHILL_SHARE = 0.18f;
constexpr float REGION_WAVELENGTH = 16000.0f;
constexpr float PEAK_WAVELENGTH = 1800.0f;
constexpr float PLAINS_HEIGHT = 70.0f;
constexpr float PLAINS_SCALE = 180.0f;
constexpr float BEACH_LIFT = 2.0f; // see terrain_noise.hlsl
constexpr float ELEVATION_BIAS = 35.0f + BEACH_LIFT * PLAINS_SCALE / PLAINS_HEIGHT;
constexpr float LAND_FLOOR = 1.0f;
constexpr float LAND_FLOOR_SMOOTHING = 4.0f;
constexpr float COAST_BAND = kCoastBand;
constexpr float HILL_WAVELENGTH = 380.0f;
constexpr float HILL_HEIGHT = 32.0f;
constexpr float HILL_SHORE_FADE = 48.0f;
constexpr float ELEVATION_MAX = 253.0f;
constexpr float SHELF_EXTENT = 70.0f;
constexpr float SHELF_DEPTH = 30.0f;
constexpr float SLOPE_EXTENT = 40.0f;
constexpr float SLOPE_DEPTH = 200.0f;
constexpr float ABYSS_DEPTH = 40.0f;
constexpr float EROSION_AMPLITUDE = 128.0f;
constexpr float EROSION_MAX_WAVELENGTH = 512.0f;
constexpr float EROSION_MIN_WAVELENGTH = 4.0f;

constexpr uint32_t SEED_ELEVATION = 0x1b873593u;
constexpr uint32_t SEED_RANGES = 0xcc9e2d51u;
constexpr uint32_t SEED_RANGE_REGIONS = 0x85ebca6bu;
constexpr uint32_t SEED_PEAKS = 0xc2b2ae35u;
constexpr uint32_t SEED_HILLS = 0xd3a2646cu;
constexpr uint32_t SEED_EROSION = 0x27d4eb2fu;
constexpr uint32_t SEED_EROSION_STEP = 0x165667b1u;
constexpr uint32_t SEED_LAYER_STEP = 0x9e3779b9u;

constexpr float TAU = 6.28318530718f;

I8 u32(uint32_t v) { return I8(static_cast<int32_t>(v)); }

struct V2 {
  F8 x, z;
};
// (value, d/dx, d/dz)
struct N3 {
  F8 v, dx, dz;
};

N3 operator+(const N3 &a, const N3 &b) { return {a.v + b.v, a.dx + b.dx, a.dz + b.dz}; }
N3 operator*(F8 s, const N3 &a) { return {s * a.v, s * a.dx, s * a.dz}; }

// HLSL lerp (SPIR-V FMix): x * (1 - s) + y * s.
F8 lerp(F8 x, F8 y, F8 s) { return x * (F8(1.0f) - s) + y * s; }

I8 hashCell(I8 cx, I8 cz, uint32_t seed) {
  I8 h = (cx * u32(0x8da6b343u)) ^ (cz * u32(0xd8163841u)) ^ u32(seed);
  h = h ^ simd::srl<16>(h);
  h = h * u32(0x7feb352du);
  h = h ^ simd::srl<15>(h);
  h = h * u32(0x846ca68bu);
  h = h ^ simd::srl<16>(h);
  return h;
}

// (h & 1) ? 1 : -1, by flipping the sign bit of -1.0f.
F8 latticeSign(I8 cx, I8 cz, uint32_t seed) {
  const I8 h = hashCell(cx, cz, seed);
  return simd::bitsToFloat(u32(0xbf800000u) ^ simd::sll<31>(h));
}

N3 binaryNoise(const V2 &p, float wavelength, uint32_t seed) {
  const F8 wl(wavelength);
  const F8 gx = p.x / wl, gz = p.z / wl;
  const F8 fx = simd::floor(gx), fz = simd::floor(gz);
  const I8 cx = simd::toInt(fx), cz = simd::toInt(fz);
  const F8 tx = gx - fx, tz = gz - fz;
  const F8 ux = tx * tx * tx * (tx * (tx * 6.0f - 15.0f) + 10.0f);
  const F8 uz = tz * tz * tz * (tz * (tz * 6.0f - 15.0f) + 10.0f);
  const F8 dux = F8(30.0f) * tx * tx * (tx * (tx - 2.0f) + 1.0f);
  const F8 duz = F8(30.0f) * tz * tz * (tz * (tz - 2.0f) + 1.0f);
  const I8 one(1);
  const F8 a = latticeSign(cx, cz, seed);
  const F8 b = latticeSign(cx + one, cz, seed);
  const F8 c = latticeSign(cx, cz + one, seed);
  const F8 d = latticeSign(cx + one, cz + one, seed);
  const F8 k = a - b - c + d;
  return {a + (b - a) * ux + (c - a) * uz + k * ux * uz, dux * ((b - a) + k * uz) / wl, duz * ((c - a) + k * ux) / wl};
}

// lodScale is always 1 on the CPU: minWavelength = 2.
N3 layeredNoise(const V2 &p, float wavelength, float amplitude, uint32_t layers, uint32_t seed, F8 &firstLayer) {
  N3 sum{0.0f, 0.0f, 0.0f};
  firstLayer = F8(0.0f);
  for (uint32_t i = 0; i < layers; ++i) {
    if (i > 0u && wavelength < 2.0f) {
      break;
    }
    const N3 layer = F8(amplitude) * binaryNoise(p, wavelength, seed + i * SEED_LAYER_STEP);
    if (i == 0u) {
      firstLayer = layer.v;
    }
    sum = sum + layer;
    wavelength *= 0.5f;
    amplitude *= 0.4f;
  }
  return sum;
}

// GRADIENTS[h & 7] from terrain_noise.hlsl, by bit selection instead of a table gather.
void gradient(I8 h, F8 &gx, F8 &gz) {
  const M8 bit0 = ((h & I8(1)) == I8(1));
  const M8 bit1 = ((h & I8(2)) == I8(2));
  const M8 bit2 = ((h & I8(4)) == I8(4));
  constexpr float D = 0.7071f;
  const F8 sign1 = simd::select(bit0, F8(-1.0f), F8(1.0f));
  gx = simd::select(bit2, simd::select(bit0, F8(-D), F8(D)), simd::select(bit1, F8(0.0f), sign1));
  gz = simd::select(bit2, simd::select(bit1, F8(-D), F8(D)), simd::select(bit1, sign1, F8(0.0f)));
}

N3 perlinNoise(const V2 &p, float wavelength, uint32_t seed) {
  const F8 wl(wavelength);
  const F8 gx = p.x / wl, gz = p.z / wl;
  const F8 flx = simd::floor(gx), flz = simd::floor(gz);
  const I8 cx = simd::toInt(flx), cz = simd::toInt(flz);
  const F8 fx = gx - flx, fz = gz - flz;
  const F8 ux = fx * fx * fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);
  const F8 uz = fz * fz * fz * (fz * (fz * 6.0f - 15.0f) + 10.0f);
  const F8 dux = F8(30.0f) * fx * fx * (fx * (fx - 2.0f) + 1.0f);
  const F8 duz = F8(30.0f) * fz * fz * (fz * (fz - 2.0f) + 1.0f);
  const I8 one(1);
  F8 g00x, g00z, g10x, g10z, g01x, g01z, g11x, g11z;
  gradient(hashCell(cx, cz, seed), g00x, g00z);
  gradient(hashCell(cx + one, cz, seed), g10x, g10z);
  gradient(hashCell(cx, cz + one, seed), g01x, g01z);
  gradient(hashCell(cx + one, cz + one, seed), g11x, g11z);
  const F8 fx1 = fx - 1.0f, fz1 = fz - 1.0f;
  const F8 n00 = g00x * fx + g00z * fz;
  const F8 n10 = g10x * fx1 + g10z * fz;
  const F8 n01 = g01x * fx + g01z * fz1;
  const F8 n11 = g11x * fx1 + g11z * fz1;
  const F8 k = n00 - n10 - n01 + n11;
  const F8 value = n00 + ux * (n10 - n00) + uz * (n01 - n00) + ux * uz * k;
  const F8 uxz = ux * uz;
  const F8 gradX =
      g00x + ux * (g10x - g00x) + uz * (g01x - g00x) + uxz * (g00x - g10x - g01x + g11x) + dux * (n10 - n00 + uz * k);
  const F8 gradZ =
      g00z + ux * (g10z - g00z) + uz * (g01z - g00z) + uxz * (g00z - g10z - g01z + g11z) + duz * (n01 - n00 + ux * k);
  const F8 s(1.4142f);
  return {value * s, gradX / wl * s, gradZ / wl * s};
}

N3 perlinFbm(const V2 &p, float wavelength, uint32_t octaves, uint32_t seed) {
  N3 sum{0.0f, 0.0f, 0.0f};
  float amplitude = 1.0f;
  float totalAmplitude = 0.0f;
  for (uint32_t i = 0; i < octaves; ++i) {
    if (i > 0u && wavelength < 2.0f) {
      break;
    }
    sum = sum + F8(amplitude) * perlinNoise(p, wavelength, seed + i * SEED_LAYER_STEP);
    totalAmplitude += amplitude;
    wavelength *= 0.5f;
    amplitude *= 0.5f;
  }
  const F8 t(totalAmplitude);
  return {sum.v / t, sum.dx / t, sum.dz / t};
}

// sinCosPoly / expPoly / tanhPoly in terrain_noise.hlsl.
constexpr float PIO2_1 = 1.5703125f;
constexpr float PIO2_2 = 4.837512969970703125e-4f;
constexpr float PIO2_3 = 7.54978995489188216e-8f;
constexpr float TWO_OVER_PI = 0.636619772367581343f;

void sinCosPoly(F8 x, F8 &s, F8 &c) {
  const F8 q = simd::floor(x * TWO_OVER_PI + 0.5f);
  const F8 r = ((x - q * PIO2_1) - q * PIO2_2) - q * PIO2_3;
  const F8 z = r * r;
  const F8 sr = r + r * z * (F8(-1.6666654611e-1f) + z * (F8(8.3321608736e-3f) + z * -1.9515295891e-4f));
  const F8 cr = F8(1.0f) - F8(0.5f) * z +
                z * z * (F8(4.166664568298827e-2f) + z * (F8(-1.388731625493765e-3f) + z * 2.443315711809948e-5f));
  const I8 quadrant = simd::toInt(q) & I8(3); // q is integral; & 3 also works for negative q (two's complement)
  const M8 odd = ((quadrant & I8(1)) == I8(1));
  const M8 negC = (quadrant == I8(1)) | (quadrant == I8(2));
  const M8 negS = quadrant > I8(1);
  s = simd::select(odd, cr, sr);
  c = simd::select(odd, sr, cr);
  c = simd::select(negC, -c, c);
  s = simd::select(negS, -s, s);
}

F8 expPoly(F8 x) {
  x = simd::clamp(x, -87.0f, 87.0f);
  const F8 n = simd::floor(x * 1.44269504088896341f + 0.5f);
  const F8 r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
  const F8 z = r * r;
  const F8 p = ((((((F8(1.9875691500e-4f) * r + 1.3981999507e-3f) * r + 8.3334519073e-3f) * r + 4.1665795894e-2f) * r +
                  1.6666665459e-1f) *
                     r +
                 5.0000001201e-1f) *
                    z +
                r) +
               1.0f;
  return p * simd::bitsToFloat(simd::sll<23>(simd::toInt(n) + I8(127)));
}

F8 tanhPoly(F8 x) {
  const F8 a = simd::abs(x);
  const F8 z = x * x;
  const F8 small =
      x +
      x * z *
          ((((F8(-5.70498872745e-3f) * z + 2.06390887954e-2f) * z - 5.37397155531e-2f) * z + 1.33314422036e-1f) * z -
           3.33332819422e-1f);
  const F8 t = F8(1.0f) - F8(2.0f) / (expPoly(F8(2.0f) * a) + 1.0f);
  const F8 large = simd::select(x < F8(0.0f), -t, t);
  return simd::select(a < F8(0.625f), small, large);
}

F8 length2(F8 x, F8 z) { return simd::sqrt(x * x + z * z); }

F8 erode(const V2 &p, F8 y, F8 gradX, F8 gradZ, F8 amplitude) {
  F8 mask = F8(1.0f) - F8(1.0f) / ((gradX * gradX + gradZ * gradZ) * 2.0f + 1.0f);
  M8 alive = F8(0.0f) == F8(0.0f); // all lanes
  uint32_t octave = 0;
  for (float wavelength = EROSION_MAX_WAVELENGTH; wavelength >= EROSION_MIN_WAVELENGTH; wavelength *= 0.5f, ++octave) {
    const F8 gradLength = length2(gradX, gradZ);
    alive = alive & !(gradLength == F8(0.0f)); // the shader breaks out of the loop
    if (!simd::any(alive)) {
      break;
    }
    const F8 safeLength = simd::select(alive, gradLength, F8(1.0f));
    const F8 acrossX = -gradZ / safeLength;
    const F8 acrossZ = gradX / safeLength;
    const F8 wl(wavelength);
    const F8 coordX = p.x / wl, coordZ = p.z / wl;
    const F8 flX = simd::floor(coordX), flZ = simd::floor(coordZ);
    const I8 cellX = simd::toInt(flX), cellZ = simd::toInt(flZ);
    F8 sx = coordX - flX, sz = coordZ - flZ;
    sx = sx * sx * (F8(3.0f) - F8(2.0f) * sx);
    sz = sz * sz * (F8(3.0f) - F8(2.0f) * sz);
    const uint32_t seed = SEED_EROSION + octave * SEED_EROSION_STEP;

    F8 sines[4], cosines[4];
    for (int corner = 0; corner < 4; ++corner) {
      const int ox = corner & 1, oz = corner >> 1;
      const I8 h = hashCell(cellX + I8(ox), cellZ + I8(oz), seed);
      const F8 phase = simd::toFloatUnsigned(simd::srl<8>(h)) * (TAU / 16777216.0f);
      const F8 dx = coordX - (flX + static_cast<float>(ox));
      const F8 dz = coordZ - (flZ + static_cast<float>(oz));
      sinCosPoly((acrossX * dx + acrossZ * dz) * TAU + phase, sines[corner], cosines[corner]);
    }
    const F8 wave = lerp(lerp(sines[0], sines[1], sx), lerp(sines[2], sines[3], sx), sz);
    const F8 waveSlope = lerp(lerp(cosines[0], cosines[1], sx), lerp(cosines[2], cosines[3], sx), sz);

    y = simd::select(alive, y + wave * mask * amplitude * wl / EROSION_MAX_WAVELENGTH, y);
    gradX = simd::select(alive, gradX + waveSlope * acrossX * mask, gradX);
    gradZ = simd::select(alive, gradZ + waveSlope * acrossZ * mask, gradZ);
    mask = simd::select(alive, mask * simd::sqrt(simd::saturate(F8(1.0f) - wave * wave)), mask);
  }
  return y;
}

struct SD {
  F8 v, d;
};
SD smoothstepD(float edge0, float edge1, F8 x) {
  const F8 t = simd::saturate((x - edge0) / F8(edge1 - edge0));
  return {t * t * (F8(3.0f) - F8(2.0f) * t), F8(6.0f) * t * (F8(1.0f) - t) / F8(edge1 - edge0)};
}

// (height, d height / d e, steering slope)
N3 baseProfile(F8 e) {
  constexpr float COAST_SLOPE = PLAINS_HEIGHT / PLAINS_SCALE;
  const F8 tl = tanhPoly(e / F8(PLAINS_SCALE));
  const F8 landSlope = F8(COAST_SLOPE) * (F8(1.0f) - tl * tl);
  const N3 land{F8(PLAINS_HEIGHT) * tl, landSlope, landSlope};

  const F8 depthIn = -e;
  const F8 t = simd::saturate(depthIn / F8(SHELF_EXTENT));
  const F8 startTangent(COAST_SLOPE * SHELF_EXTENT);
  const F8 shelf =
      startTangent * (t - F8(2.0f) * t * t + t * t * t) + F8(SHELF_DEPTH) * t * t * (F8(3.0f) - F8(2.0f) * t);
  const F8 dShelf =
      (startTangent * (F8(1.0f) - F8(4.0f) * t + F8(3.0f) * t * t) + F8(SHELF_DEPTH * 6.0f) * t * (F8(1.0f) - t)) /
      F8(SHELF_EXTENT);
  const SD slope = smoothstepD(SHELF_EXTENT, SHELF_EXTENT + SLOPE_EXTENT, depthIn);
  const SD abyss = smoothstepD(SHELF_EXTENT + SLOPE_EXTENT, ELEVATION_MAX, depthIn);
  const F8 depth = shelf + F8(SLOPE_DEPTH) * slope.v + F8(ABYSS_DEPTH) * abyss.v;
  const F8 dDepth = dShelf + F8(SLOPE_DEPTH) * slope.d + F8(ABYSS_DEPTH) * abyss.d;

  const M8 isLand = e >= F8(0.0f);
  return {simd::select(isLand, land.v, -depth), simd::select(isLand, land.dx, dDepth),
          simd::select(isLand, land.dz, dShelf)};
}

F8 signOf(F8 x) { return simd::select(x > F8(0.0f), F8(1.0f), simd::select(x < F8(0.0f), F8(-1.0f), F8(0.0f))); }

N3 mountainRanges(const V2 &p) {
  const N3 zero{0.0f, 0.0f, 0.0f};
  const N3 regionNoise = perlinFbm(p, REGION_WAVELENGTH, 2u, SEED_RANGE_REGIONS);
  const SD region = smoothstepD(-0.3f, 0.1f, regionNoise.v);
  M8 active = region.v > F8(0.0f);
  if (!simd::any(active)) {
    return zero;
  }

  const N3 r = perlinFbm(p, RANGE_WAVELENGTH, 4u, SEED_RANGES);
  const F8 absR = simd::abs(r.v);
  const SD belt = smoothstepD(RANGE_HALF_WIDTH, 0.0f, absR);
  const SD foothills = smoothstepD(FOOTHILL_HALF_WIDTH, 0.0f, absR);
  const F8 profile = lerp(belt.v * belt.v, foothills.v, F8(FOOTHILL_SHARE));
  const F8 dProfile = lerp(F8(2.0f) * belt.v * belt.d, foothills.d, F8(FOOTHILL_SHARE));
  active = active & (profile > F8(0.0f));
  if (!simd::any(active)) {
    return zero;
  }

  const N3 peakNoise = perlinFbm(p, PEAK_WAVELENGTH, 3u, SEED_PEAKS);
  const SD peakRamp = smoothstepD(-1.1f, 0.6f, peakNoise.v);
  const F8 peaks = peakRamp.v;
  const F8 dPeaks = peakRamp.d;

  const F8 height = F8(MOUNTAIN_HEIGHT) * region.v * profile * peaks;
  const F8 sr = signOf(r.v);
  const F8 gx =
      F8(MOUNTAIN_HEIGHT) * (region.d * regionNoise.dx * profile * peaks + region.v * dProfile * sr * r.dx * peaks +
                             region.v * profile * dPeaks * peakNoise.dx);
  const F8 gz =
      F8(MOUNTAIN_HEIGHT) * (region.d * regionNoise.dz * profile * peaks + region.v * dProfile * sr * r.dz * peaks +
                             region.v * profile * dPeaks * peakNoise.dz);
  return {simd::select(active, height, F8(0.0f)), simd::select(active, gx, F8(0.0f)),
          simd::select(active, gz, F8(0.0f))};
}

F8 sampleTerrainHeight(const V2 &p) {
  F8 continentLayer;
  const N3 elevation = layeredNoise(p, 4096.0f, 153.6f, 5u, SEED_ELEVATION, continentLayer);
  const N3 base = baseProfile(elevation.v + F8(ELEVATION_BIAS));
  const N3 mountains = mountainRanges(p);

  const N3 hillNoise = perlinFbm(p, HILL_WAVELENGTH, 4u, SEED_HILLS);
  const SD shore = smoothstepD(0.0f, HILL_SHORE_FADE, base.v);
  const F8 hills = F8(HILL_HEIGHT) * shore.v * hillNoise.v;

  const F8 height = base.v + hills + mountains.v;
  const F8 gradX = base.dz * elevation.dx + mountains.dx +
                   F8(HILL_HEIGHT) * (shore.v * hillNoise.dx + hillNoise.v * shore.d * base.dz * elevation.dx);
  const F8 gradZ = base.dz * elevation.dz + mountains.dz +
                   F8(HILL_HEIGHT) * (shore.v * hillNoise.dz + hillNoise.v * shore.d * base.dz * elevation.dz);

  const F8 erosionAmplitude = F8(EROSION_AMPLITUDE) * simd::saturate(mountains.v / F8(MOUNTAIN_HEIGHT) * 3.0f);
  // Erosion adds wave * mask * amplitude: with zero amplitude in every lane it can't change the height, so skip it
  // (most land has no mountains).
  const F8 eroded = simd::any(erosionAmplitude > F8(0.0f)) ? erode(p, height, gradX, gradZ, erosionAmplitude) : height;

  // Land floor (terrain_noise.hlsl): polynomial smooth max with LAND_FLOOR, faded in from the coast inland.
  const F8 landWeight = smoothstepD(0.0f, COAST_BAND, continentLayer + F8(ELEVATION_BIAS)).v;
  const F8 floorBlend =
      simd::max(F8(LAND_FLOOR_SMOOTHING) - simd::abs(eroded - F8(LAND_FLOOR)), F8(0.0f)) / F8(LAND_FLOOR_SMOOTHING);
  const F8 floored = simd::max(eroded, F8(LAND_FLOOR)) + floorBlend * floorBlend * F8(LAND_FLOOR_SMOOTHING) * F8(0.25f);
  const F8 surface = floored * landWeight + eroded * (F8(1.0f) - landWeight); // exact floor where landWeight = 1
  return simd::clamp(surface, TERRAIN_MIN_HEIGHT, TERRAIN_MAX_HEIGHT);
}

V2 positions(const int32_t *worldX, const int32_t *worldZ) {
  return {simd::toFloat(I8::load(worldX)), simd::toFloat(I8::load(worldZ))};
}

} // namespace

void sampleHeights8(const int32_t *worldX, const int32_t *worldZ, float *out) {
  sampleTerrainHeight(positions(worldX, worldZ)).store(out);
}

void sampleDither8(const int32_t *worldX, const int32_t *worldZ, float *out) {
  const V2 p = positions(worldX, worldZ);
  const F8 wiggle = perlinFbm(p, world::kDitherWavelength, 2u, world::kSeedDither).v;
  const I8 h = hashCell(I8::load(worldX), I8::load(worldZ), world::kSeedDitherWhite);
  const F8 speckle = simd::toFloatUnsigned(simd::srl<8>(h)) * (2.0f / 16777216.0f) - 1.0f;
  simd::clamp(F8(0.65f) * wiggle + F8(0.35f) * speckle, -1.0f, 1.0f).store(out);
}

void sampleContinent8(const int32_t *worldX, const int32_t *worldZ, float *out) {
  (F8(153.6f) * binaryNoise(positions(worldX, worldZ), 4096.0f, SEED_ELEVATION).v + F8(ELEVATION_BIAS)).store(out);
}

float sampleHeight(int32_t worldX, int32_t worldZ) {
  int32_t xs[8], zs[8];
  float out[8];
  for (int i = 0; i < 8; ++i) {
    xs[i] = worldX;
    zs[i] = worldZ;
  }
  sampleHeights8(xs, zs, out);
  return out[0];
}

float sampleDither(int32_t worldX, int32_t worldZ) {
  int32_t xs[8], zs[8];
  float out[8];
  for (int i = 0; i < 8; ++i) {
    xs[i] = worldX;
    zs[i] = worldZ;
  }
  sampleDither8(xs, zs, out);
  return out[0];
}

} // namespace terrain
