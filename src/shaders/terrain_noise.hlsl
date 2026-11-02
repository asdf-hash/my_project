#include "FastNoiseLite.hlsl"

// Helper function for cubic/smoothstep spline interpolation between keypoints
float terrainSplineSegment(float x, float x0, float x1, float y0, float y1) {
  return lerp(y0, y1, smoothstep(0.0f, 1.0f, saturate((x - x0) / (x1 - x0))));
}

// -----------------------------------------------------------------------------
// GREEN INSET: Peaks & Valleys Spline
// Controls valley floor to mountain peak transitions
// -----------------------------------------------------------------------------
float peaksAndValleysSpline(float pv) {
  pv = saturate(pv);
  if (pv < 0.25f) {
    return terrainSplineSegment(pv, 0.0f, 0.25f, 0.0f, 0.12f); // Flat valley floor
  }
  if (pv < 0.45f) {
    return terrainSplineSegment(pv, 0.25f, 0.45f, 0.12f, 0.25f); // Rolling foothills
  }
  if (pv < 0.75f) {
    return terrainSplineSegment(pv, 0.45f, 0.75f, 0.25f, 0.78f); // Steep mountain slope
  }
  return terrainSplineSegment(pv, 0.75f, 1.0f, 0.78f, 1.0f); // High peak ridge
}

// -----------------------------------------------------------------------------
// RED INSET: Erosion Spline
// Low Erosion (0.0)  = Extreme height, jagged peaks, mesa plateaus
// High Erosion (1.0) = Flat, smooth, heavily eroded lowlands
// -----------------------------------------------------------------------------
float erosionSpline(float e, float pvFactor) {
  e = saturate(e);

  if (e < 0.15f) {
    // Un-eroded high mountains (up to +600 blocks)
    const float h0 = lerp(380.0f, 600.0f, pvFactor);
    const float h1 = lerp(240.0f, 440.0f, pvFactor);
    return terrainSplineSegment(e, 0.0f, 0.15f, h0, h1);
  }
  if (e < 0.32f) {
    // High mountain shoulders and highlands
    const float h0 = lerp(240.0f, 440.0f, pvFactor);
    const float h1 = lerp(130.0f, 260.0f, pvFactor);
    return terrainSplineSegment(e, 0.15f, 0.32f, h0, h1);
  }
  if (e < 0.45f) {
    // Mid-erosion ridge bump (red graph inset detail)
    const float h0 = lerp(130.0f, 260.0f, pvFactor);
    const float h1 = lerp(190.0f, 320.0f, pvFactor);
    return terrainSplineSegment(e, 0.32f, 0.45f, h0, h1);
  }
  if (e < 0.65f) {
    // Descent into eroded plains
    const float h0 = lerp(190.0f, 320.0f, pvFactor);
    const float h1 = lerp(25.0f, 75.0f, pvFactor);
    return terrainSplineSegment(e, 0.45f, 0.65f, h0, h1);
  }
  if (e < 0.70f) {
    // Sharp step up to Mesa/Plateau (red graph inset step feature)
    const float h0 = lerp(25.0f, 75.0f, pvFactor);
    const float h1 = lerp(160.0f, 240.0f, pvFactor);
    return terrainSplineSegment(e, 0.65f, 0.70f, h0, h1);
  }
  if (e < 0.80f) {
    // Flat plateau surface
    return lerp(160.0f, 240.0f, pvFactor);
  }
  if (e < 0.85f) {
    // Cliff edge drop-off
    const float h0 = lerp(160.0f, 240.0f, pvFactor);
    const float h1 = lerp(15.0f, 45.0f, pvFactor);
    return terrainSplineSegment(e, 0.80f, 0.85f, h0, h1);
  }
  // Fully eroded lowlands & flats
  return lerp(10.0f, 35.0f, pvFactor);
}

// -----------------------------------------------------------------------------
// MAIN BLUE GRAPH: Continentalness Spline
// Controls macro landmasses: Ocean Trenches -> Coastal Shelf -> Inland Highlands.
// The broad ocean band keeps the large-scale field from producing island chains.
// -----------------------------------------------------------------------------
float continentalnessSpline(float c, float e, float pvFactor, float oceanFloorHeight) {
  c = saturate(c);

  // Deep Ocean (-400 to -250 range): Driven by smooth single-octave ocean noise
  if (c < 0.18f) {
    return oceanFloorHeight;
  }
  // Transition from ocean floor up to continental shelf
  if (c < 0.34f) {
    const float shelfBase = lerp(-120.0f, 2.0f, smoothstep(0.18f, 0.34f, c));
    return lerp(oceanFloorHeight, shelfBase, smoothstep(0.18f, 0.31f, c));
  }
  // Coastal Plains & Lowland Interior
  if (c < 0.52f) {
    return terrainSplineSegment(c, 0.34f, 0.52f, 2.0f, 50.0f);
  }
  // Inland Step up to Mountainous Plateau Bases
  if (c < 0.65f) {
    return terrainSplineSegment(c, 0.52f, 0.65f, 50.0f, 150.0f);
  }

  // High Inland / Mountain Zones: Pass control to Erosion & Peaks/Valleys Splines
  const float inlandBase = terrainSplineSegment(c, 0.65f, 0.78f, 150.0f, 200.0f);
  const float mountainHeight = erosionSpline(e, pvFactor);

  const float mountainBlend = smoothstep(0.65f, 0.80f, c);
  return lerp(inlandBase, mountainHeight, mountainBlend);
}

static const fnl_state WARP_STATE = { 7777,
                                      0.0006f,
                                      FNL_NOISE_OPENSIMPLEX2,
                                      FNL_ROTATION_NONE,
                                      FNL_FRACTAL_NONE,
                                      2,
                                      2.0f,
                                      0.5f,
                                      0.0f,
                                      2.0f,
                                      FNL_CELLULAR_DISTANCE_EUCLIDEANSQ,
                                      FNL_CELLULAR_RETURN_TYPE_DISTANCE,
                                      1.0f,
                                      FNL_DOMAIN_WARP_OPENSIMPLEX2,
                                      30.0f };
static const fnl_state CONTINENTALNESS_STATE = { 1337,
                                                 0.00025f,
                                                 FNL_NOISE_OPENSIMPLEX2,
                                                 FNL_ROTATION_NONE,
                                                 FNL_FRACTAL_FBM,
                                                 5,
                                                 2.0f,
                                                 0.5f,
                                                 0.0f,
                                                 2.0f,
                                                 FNL_CELLULAR_DISTANCE_EUCLIDEANSQ,
                                                 FNL_CELLULAR_RETURN_TYPE_DISTANCE,
                                                 1.0f,
                                                 FNL_DOMAIN_WARP_OPENSIMPLEX2,
                                                 30.0f };
static const fnl_state OCEAN_STATE = { 54321,
                                       0.0005f,
                                       FNL_NOISE_OPENSIMPLEX2,
                                       FNL_ROTATION_NONE,
                                       FNL_FRACTAL_NONE,
                                       1,
                                       2.0f,
                                       0.5f,
                                       0.0f,
                                       2.0f,
                                       FNL_CELLULAR_DISTANCE_EUCLIDEANSQ,
                                       FNL_CELLULAR_RETURN_TYPE_DISTANCE,
                                       1.0f,
                                       FNL_DOMAIN_WARP_OPENSIMPLEX2,
                                       30.0f };
static const fnl_state EROSION_STATE = { 2468,
                                         0.0005f,
                                         FNL_NOISE_OPENSIMPLEX2,
                                         FNL_ROTATION_NONE,
                                         FNL_FRACTAL_FBM,
                                         4,
                                         2.0f,
                                         0.5f,
                                         0.0f,
                                         2.0f,
                                         FNL_CELLULAR_DISTANCE_EUCLIDEANSQ,
                                         FNL_CELLULAR_RETURN_TYPE_DISTANCE,
                                         1.0f,
                                         FNL_DOMAIN_WARP_OPENSIMPLEX2,
                                         30.0f };
static const fnl_state PEAKS_AND_VALLEYS_STATE = { 97531,
                                                   0.0010f,
                                                   FNL_NOISE_OPENSIMPLEX2,
                                                   FNL_ROTATION_NONE,
                                                   FNL_FRACTAL_RIDGED,
                                                   5,
                                                   2.0f,
                                                   0.5f,
                                                   0.0f,
                                                   2.0f,
                                                   FNL_CELLULAR_DISTANCE_EUCLIDEANSQ,
                                                   FNL_CELLULAR_RETURN_TYPE_DISTANCE,
                                                   1.0f,
                                                   FNL_DOMAIN_WARP_OPENSIMPLEX2,
                                                   30.0f };
static const fnl_state DETAIL_STATE = { 9999,
                                        0.0035f,
                                        FNL_NOISE_OPENSIMPLEX2,
                                        FNL_ROTATION_NONE,
                                        FNL_FRACTAL_FBM,
                                        4,
                                        2.0f,
                                        0.5f,
                                        0.0f,
                                        2.0f,
                                        FNL_CELLULAR_DISTANCE_EUCLIDEANSQ,
                                        FNL_CELLULAR_RETURN_TYPE_DISTANCE,
                                        1.0f,
                                        FNL_DOMAIN_WARP_OPENSIMPLEX2,
                                        30.0f };

// -----------------------------------------------------------------------------
// MAIN SAMPLING FUNCTION
// -----------------------------------------------------------------------------
float sampleTerrainHeight(int worldX, int worldZ, uint lodLevel) {
  const float x = float(worldX);
  const float z = float(worldZ);

  // 1. Domain Warping (Adds organic, non-linear distortion to biome borders)
  fnl_state warpState = WARP_STATE;
  warpState.noise_type = FNL_NOISE_OPENSIMPLEX2;
  warpState.octaves = 2;
  warpState.frequency = 0.0006f;
  const float warpX = fnlGetNoise2D(warpState, x + 400.0f, z - 400.0f) * 100.0f;
  const float warpZ = fnlGetNoise2D(warpState, x - 200.0f, z + 200.0f) * 100.0f;

  const float wx = x + warpX;
  const float wz = z + warpZ;

  // 2. Continentalness Noise (Macro scale landscape / landmass placement)
  fnl_state continentalnessState = CONTINENTALNESS_STATE;
  continentalnessState.noise_type = FNL_NOISE_OPENSIMPLEX2;
  continentalnessState.fractal_type = FNL_FRACTAL_FBM;
  // Keep continentalness much broader than terrain detail so continents are
  // separated by genuinely large ocean basins instead of frequent islands.
  continentalnessState.octaves = 4;
  continentalnessState.frequency = 0.00012f;
  const float continentalnessNoise = fnlGetNoise2D(continentalnessState, wx, wz);
  const float continentalness = saturate((continentalnessNoise + 1.0f) * 0.5f);

  float oceanFloorHeight = 0.0f;
  if (continentalness < 0.34f) {
    // 3. Smooth single-octave ocean-floor noise, only used on the ocean/coastal transition.
    fnl_state oceanState = OCEAN_STATE;
    oceanState.noise_type = FNL_NOISE_OPENSIMPLEX2;
    oceanState.octaves = 1;
    oceanState.frequency = 0.00025f;
    const float oceanNoise = fnlGetNoise2D(oceanState, x, z);
    oceanFloorHeight = lerp(-380.0f, -180.0f, (oceanNoise + 1.0f) * 0.5f);
  }

  // 4. Erosion Noise (Regional terrain sharpness and plateau formation)
  fnl_state erosionState = EROSION_STATE;
  erosionState.noise_type = FNL_NOISE_OPENSIMPLEX2;
  erosionState.fractal_type = FNL_FRACTAL_FBM;
  erosionState.octaves = 4;
  erosionState.frequency = 0.0005f;
  const float erosionNoise = fnlGetNoise2D(erosionState, wx, wz);
  const float erosion = saturate((erosionNoise + 1.0f) * 0.5f);

  // 5. Peaks & Valleys Noise (Local ridge density and valley carving)
  fnl_state peaksAndValleysState = PEAKS_AND_VALLEYS_STATE;
  peaksAndValleysState.noise_type = FNL_NOISE_OPENSIMPLEX2;
  peaksAndValleysState.fractal_type = FNL_FRACTAL_RIDGED;
  peaksAndValleysState.octaves = 5;
  peaksAndValleysState.frequency = 0.0010f;
  const float pvNoise = fnlGetNoise2D(peaksAndValleysState, wx, wz);
  const float peaksAndValleys = (pvNoise + 1.0f) * 0.5f;
  const float pvFactor = peaksAndValleysSpline(peaksAndValleys);

  // 6. Primary Spline Evaluation
  float terrainHeight = continentalnessSpline(continentalness, erosion, pvFactor, oceanFloorHeight);

  // 7. Surface High-Frequency Detail Noise (Applies ONLY on land, higher on un-eroded peaks)
  if (terrainHeight > -10.0f) {
    fnl_state detailState = DETAIL_STATE;
    detailState.noise_type = FNL_NOISE_OPENSIMPLEX2;
    detailState.fractal_type = FNL_FRACTAL_FBM;
    detailState.octaves = 4;
    detailState.frequency = 0.0035f;

    const float detailNoise = fnlGetNoise2D(detailState, x, z);

    // Scale micro-detail strength: low erosion & high PV get dramatic rocky roughness
    const float detailMask = (1.0f - erosion) * pvFactor;
    const float landFactor = smoothstep(-10.0f, 15.0f, terrainHeight);

    const float detailAmplitude = lerp(4.0f, 45.0f, detailMask) * landFactor;
    terrainHeight += detailNoise * detailAmplitude;
  }

  // Keep coastal land above sea level after detail is applied. Without this
  // seal, small detail dips near the shoreline become isolated inland ponds.
  if (continentalness >= 0.34f) {
    terrainHeight = max(terrainHeight, 2.0f);
  }

  return clamp(terrainHeight, -400.0f, 600.0f);
}
