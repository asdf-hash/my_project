// Must match OVERLAY_LINES / OVERLAY_COLUMNS in renderer.cpp.
static const uint OVERLAY_LINES = 4u;
static const uint OVERLAY_COLUMNS = 28u;
static const uint OVERLAY_TEXT_WORDS = OVERLAY_LINES * OVERLAY_COLUMNS / 4u;

// GUI drawn into the swapchain image after TAA: debug text (top right), crosshair, hotbar HUD, health bar.
// Text arrives as glyph indices in push constants and uses a built-in 5x7 bitmap font. All passes share OverlayParams;
// some reuse its fields (noted at each entry point).

// Matches OverlayParams in renderer.cpp.
struct OverlayParams {
  float2 screenSize;
  float scale;                   // screen pixels per font pixel
  uint columns;                  // longest line; shorter lines are left-padded with spaces by the renderer
  uint text[OVERLAY_TEXT_WORDS]; // OVERLAY_LINES x OVERLAY_COLUMNS glyph indices, one byte each
};

[[vk::push_constant]]
ConstantBuffer<OverlayParams> params;

// Rows top to bottom, bit 4 = leftmost column; .x = rows 0-3, .y = rows 4-6, 8 bits per row.
// Order: " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:.+-/()" (overlayGlyphIndex in renderer.cpp).
static const uint FONT_GLYPHS = 44u;
static const uint2 FONT[FONT_GLYPHS] = {
  uint2(0x00000000u, 0x00000000u), // ' '
  uint2(0x1513110eu, 0x000e1119u), // '0'
  uint2(0x04040c04u, 0x000e0404u), // '1'
  uint2(0x0201110eu, 0x001f0804u), // '2'
  uint2(0x0204021fu, 0x000e1101u), // '3'
  uint2(0x120a0602u, 0x0002021fu), // '4'
  uint2(0x011e101fu, 0x000e1101u), // '5'
  uint2(0x1e100806u, 0x000e1111u), // '6'
  uint2(0x0402011fu, 0x00080808u), // '7'
  uint2(0x0e11110eu, 0x000e1111u), // '8'
  uint2(0x0f11110eu, 0x000c0201u), // '9'
  uint2(0x1111110eu, 0x0011111fu), // 'A'
  uint2(0x1e11111eu, 0x001e1111u), // 'B'
  uint2(0x1010110eu, 0x000e1110u), // 'C'
  uint2(0x1111121cu, 0x001c1211u), // 'D'
  uint2(0x1e10101fu, 0x001f1010u), // 'E'
  uint2(0x1e10101fu, 0x00101010u), // 'F'
  uint2(0x1710110eu, 0x000f1111u), // 'G'
  uint2(0x1f111111u, 0x00111111u), // 'H'
  uint2(0x0404040eu, 0x000e0404u), // 'I'
  uint2(0x02020207u, 0x000c1202u), // 'J'
  uint2(0x18141211u, 0x00111214u), // 'K'
  uint2(0x10101010u, 0x001f1010u), // 'L'
  uint2(0x15151b11u, 0x00111111u), // 'M'
  uint2(0x15191111u, 0x00111113u), // 'N'
  uint2(0x1111110eu, 0x000e1111u), // 'O'
  uint2(0x1e11111eu, 0x00101010u), // 'P'
  uint2(0x1111110eu, 0x000d1215u), // 'Q'
  uint2(0x1e11111eu, 0x00111214u), // 'R'
  uint2(0x0e10100fu, 0x001e0101u), // 'S'
  uint2(0x0404041fu, 0x00040404u), // 'T'
  uint2(0x11111111u, 0x000e1111u), // 'U'
  uint2(0x11111111u, 0x00040a11u), // 'V'
  uint2(0x15111111u, 0x000a1515u), // 'W'
  uint2(0x040a1111u, 0x0011110au), // 'X'
  uint2(0x0a111111u, 0x00040404u), // 'Y'
  uint2(0x0402011fu, 0x001f1008u), // 'Z'
  uint2(0x000c0c00u, 0x00000c0cu), // ':'
  uint2(0x00000000u, 0x000c0c00u), // '.'
  uint2(0x1f040400u, 0x00000404u), // '+'
  uint2(0x1f000000u, 0x00000000u), // '-'
  uint2(0x04020100u, 0x00001008u), // '/'
  uint2(0x08080402u, 0x00020408u), // '('
  uint2(0x02020408u, 0x00080402u), // ')'
};

static const float2 CELL = float2(6.0f, 9.0f); // 5x7 glyph + spacing, in font pixels
static const float PADDING = 3.0f;             // font pixels around the text
static const float MARGIN = 8.0f;              // screen pixels from the window edge

// Bindless block textures; the HUD atlas follows them (renderer.cpp createTextureImages).
[[vk::binding(11, 0)]]
Texture2D selectedTextures[];

// Vanilla survival HUD below the hearts: hotbar with block icons and selection frame, XP bar, hunger. Sprites come
// from textures/gui/hud.png (atlas: hotbar 182x22 at 0,0; selection 24x24 at 182,0; drumstick 9x9 at 206,0; XP bar
// 182x5 at 0,24). Layout in GUI pixels from the hotbar's top-left corner (centered, 22 above the bottom edge):
// XP bar 7 above it, hunger row 17 above it (right-aligned), selection at (20 * slot - 1, -1).
// HUD params, packed into OverlayParams: columns = selected slot | 0x100 when the XP / hunger bars show;
// text[0..8] = per slot top texture | side texture << 16; text[9..17] = per slot tint RGBA8; text[18] = atlas slot;
// text[19] = slot count. Must match recordOverlay in renderer.cpp.
static const uint HUD_SLOTS = 9u;
static const float HOTBAR_WIDTH = 182.0f;
static const float HOTBAR_HEIGHT = 22.0f;
static const float XP_BAR_TOP = -7.0f;
static const float HUNGER_TOP = -17.0f;
static const float HUD_TOP = HUNGER_TOP;
static const uint HUD_SHOW_BARS = 0x100u;

/** HUD word i of the packed params (text[i]). */
uint hudWord(uint i) { return params.text[i]; }

/** sRGB transfer function, linear to encoded. */
float3 linearToSrgb(float3 c) {
  return select(c <= 0.0031308f, c * 12.92f, 1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f);
}
/** sRGB transfer function, encoded to linear. */
float3 srgbToLinear(float3 c) {
  return select(c <= 0.04045f, c / 12.92f, pow(max((c + 0.055f) / 1.055f, 0.0f), 2.4f));
}

/** Hotbar top-left corner in screen pixels. */
float2 hotbarOrigin() {
  const float s = heartScale();
  return floor(float2(params.screenSize.x * 0.5f - HOTBAR_WIDTH * 0.5f * s, params.screenSize.y - HOTBAR_HEIGHT * s));
}

/** Atlas texel (x, y); transparent outside [0, size). */
float4 atlasTexel(int2 texel, int2 atlasPos, int2 size) {
  if (any(texel < 0) || any(texel >= size)) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  return selectedTextures[NonUniformResourceIndex(hudWord(18u))].Load(int3(atlasPos + texel, 0));
}

/** `top` over `bottom` (straight alpha). */
float4 over(float4 top, float4 bottom) {
  const float alpha = top.a + bottom.a * (1.0f - top.a);
  const float3 color = alpha > 0.0f ? (top.rgb * top.a + bottom.rgb * bottom.a * (1.0f - top.a)) / alpha : 0.0f;
  return float4(color, alpha);
}

/**
 * A block icon like vanilla's GUI item: an orthographic cube seen 30 degrees from above at a 45 degree yaw, 10 GUI
 * pixels per block, its top face full bright and its left / right faces at 0.64 / 0.41 (display space).
 *
 * @param local Position relative to the icon center, in GUI pixels (y down); fractional, so edges stay screen-sharp.
 * @param slot Hotbar slot of the block.
 * @return the icon color (linear), transparent where the cube is missed or the texture is cut out.
 */
float4 blockIcon(float2 local, uint slot) {
  // Camera direction (toward the viewer), screen right and up, in block space.
  const float3 toCamera = float3(0.6123724f, 0.5f, 0.6123724f);
  const float3 right = float3(0.7071068f, 0.0f, -0.7071068f);
  const float3 up = float3(-0.3535534f, 0.8660254f, -0.3535534f);
  const float3 origin = (local.x * right - local.y * up) / 10.0f + 4.0f * toCamera;
  const float3 dir = -toCamera;
  const float3 tLow = (-0.5f - origin) / dir;
  const float3 tHigh = (0.5f - origin) / dir;
  const float3 tNear3 = min(tLow, tHigh);
  const float3 tFar3 = max(tLow, tHigh);
  const float tNear = max(tNear3.x, max(tNear3.y, tNear3.z));
  const float tFar = min(tFar3.x, min(tFar3.y, tFar3.z));
  if (tNear > tFar) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const float3 hit = origin + dir * tNear;
  const uint textures = hudWord(slot);
  float2 uv;
  uint texture;
  float shade;
  if (tNear == tNear3.y) { // top (+y)
    uv = float2(hit.x + 0.5f, hit.z + 0.5f);
    texture = textures & 0xffffu;
    shade = 1.0f;
  } else if (tNear == tNear3.z) { // +z: the left face on screen
    uv = float2(hit.x + 0.5f, 0.5f - hit.y);
    texture = textures >> 16u;
    shade = 0.64f;
  } else { // +x: the right face
    uv = float2(0.5f - hit.z, 0.5f - hit.y);
    texture = textures >> 16u;
    shade = 0.41f;
  }
  const int2 texel = clamp(int2(floor(uv * 16.0f)), int2(0, 0), int2(15, 15));
  const float4 color = selectedTextures[NonUniformResourceIndex(texture)].Load(int3(texel, 0));
  if (color.a < 0.5f) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const uint tintBits = hudWord(9u + slot);
  const float3 tint = float3(float(tintBits & 0xffu), float((tintBits >> 8u) & 0xffu), float((tintBits >> 16u) & 0xffu)) /
                      255.0f;
  // Tint and shade multiply the on-screen color, as in vanilla.
  return float4(srgbToLinear(linearToSrgb(color.rgb) * tint * shade), 1.0f);
}

/** Quad covering the HUD: the hunger row's top to the bottom edge, the selection frame's width. */
[shader("vertex")]
float4 hudVertexMain(uint vertexID: SV_VertexID) : SV_Position {
  const uint corner = vertexID < 3u ? vertexID : vertexID - 2u;
  const float2 unit = float2(float(corner & 1u), float(corner >> 1u));
  const float s = heartScale();
  const float2 low = hotbarOrigin() + float2(-1.0f, HUD_TOP) * s;
  const float2 high = hotbarOrigin() + float2(HOTBAR_WIDTH + 1.0f, HOTBAR_HEIGHT) * s;
  const float2 pixel = lerp(low, high, unit);
  return float4(pixel / params.screenSize * 2.0f - 1.0f, 0.0f, 1.0f);
}

/** Composites, back to front: XP bar, hunger, hotbar, selection frame, block icons. */
[shader("fragment")]
float4 hudFragmentMain(float4 position: SV_Position) : SV_Target0 {
  const float s = heartScale();
  const float2 gui = (position.xy - hotbarOrigin()) / s;
  const int2 texel = int2(floor(gui));
  const uint selected = params.columns & 0xffu;
  const bool showBars = (params.columns & HUD_SHOW_BARS) != 0u;
  float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (showBars) {
    color = over(atlasTexel(texel - int2(0, int(XP_BAR_TOP)), int2(0, 24), int2(182, 5)), color);
    // Hunger: always full, 10 drumsticks right-aligned, spaced 8; drawn right to left like vanilla, so each icon
    // covers the overlapping outline column of the one to its right.
    for (int icon = 0; icon < 10; ++icon) {
      const int2 iconTexel = int2(texel.x - (int(HOTBAR_WIDTH) - 9 - 8 * icon), texel.y - int(HUNGER_TOP));
      color = over(atlasTexel(iconTexel, int2(206, 0), int2(9, 9)), color);
    }
  }
  color = over(atlasTexel(texel, int2(0, 0), int2(182, 22)), color);
  color = over(atlasTexel(texel - int2(20 * int(selected) - 1, -1), int2(182, 0), int2(24, 24)), color);
  const uint slotCount = min(hudWord(19u), HUD_SLOTS);
  for (uint slot = 0u; slot < slotCount; ++slot) {
    const float2 local = gui - float2(3.0f + 20.0f * float(slot) + 8.0f, 3.0f + 8.0f);
    if (all(abs(local) <= 8.0f)) {
      color = over(blockIcon(local, slot), color);
    }
  }
  return color;
}

/** Text box size in screen pixels (last cell's trailing spacing trimmed, padding added). */
float2 boxSize() {
  return (float2(float(params.columns), float(OVERLAY_LINES)) * CELL - float2(1.0f, 2.0f) + 2.0f * PADDING) *
         params.scale;
}

/** Text box top-left corner in screen pixels. */
float2 boxOrigin() { return float2(params.screenSize.x - MARGIN - boxSize().x, MARGIN); }

/** Two triangles covering the text box. */
[shader("vertex")]
float4 overlayVertexMain(uint vertexID: SV_VertexID) : SV_Position {
  const uint corner = vertexID < 3u ? vertexID : vertexID - 2u; // 0 1 2, 1 2 3
  const float2 unit = float2(float(corner & 1u), float(corner >> 1u));
  const float2 pixel = boxOrigin() + unit * boxSize();
  return float4(pixel / params.screenSize * 2.0f - 1.0f, 0.0f, 1.0f);
}

/** Glyph index at a text cell; params.text packs four byte-sized indices per uint, lowest byte first. */
uint glyphAt(uint line, uint column) {
  const uint index = line * OVERLAY_COLUMNS + column;
  return (params.text[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu;
}

/** Whether FONT glyph `glyph` covers pixel `inCell` (x 0..4, y 0..6; anything else is unlit). */
bool glyphLit(uint glyph, uint2 inCell) {
  if (inCell.x >= 5u || inCell.y >= 7u) {
    return false;
  }
  glyph = min(glyph, FONT_GLYPHS - 1u);
  const uint rowBits = ((inCell.y < 4u ? FONT[glyph].x : FONT[glyph].y) >> ((inCell.y & 3u) * 8u)) & 0x1fu;
  return (rowBits >> (4u - inCell.x) & 1u) != 0u;
}

/** Text box: translucent background with white FONT glyphs. */
[shader("fragment")]
float4 overlayFragmentMain(float4 position: SV_Position) : SV_Target0 {
  const float4 background = float4(0.0f, 0.0f, 0.0f, 0.55f);
  const float2 fontPixel = (position.xy - boxOrigin()) / params.scale - PADDING;
  if (any(fontPixel < 0.0f)) {
    return background;
  }
  const uint2 cell = uint2(fontPixel / CELL);
  const uint2 inCell = uint2(fontPixel - float2(cell) * CELL);
  if (cell.x >= params.columns || cell.y >= OVERLAY_LINES || inCell.x >= 5u || inCell.y >= 7u) {
    return background;
  }
  return glyphLit(glyphAt(cell.y, cell.x), inCell) ? float4(1.0f, 1.0f, 1.0f, 1.0f) : background;
}

// Crosshair at the screen center: a 15x15 plus, one GUI pixel (params.scale screen pixels) thick, as three rectangles
// so the center isn't covered twice. Drawn with an inverting blend (1 - destination), so it shows on any background.
static const float CROSSHAIR_SIZE = 15.0f; // GUI pixels
static const float4 CROSSHAIR_RECTS[3] = {
  float4(0.0f, 7.0f, 15.0f, 8.0f), // horizontal bar: min x, min y, max x, max y
  float4(7.0f, 0.0f, 8.0f, 7.0f),  // upper arm
  float4(7.0f, 8.0f, 8.0f, 15.0f), // lower arm
};

/** Three quads (CROSSHAIR_RECTS), 6 vertices each, around the screen center. */
[shader("vertex")]
float4 crosshairVertexMain(uint vertexID: SV_VertexID) : SV_Position {
  const float4 rect = CROSSHAIR_RECTS[vertexID / 6u];
  const uint local = vertexID % 6u;
  const uint corner = local < 3u ? local : local - 2u; // 0 1 2, 1 2 3
  const float2 unit = float2(float(corner & 1u), float(corner >> 1u));
  // Whole screen pixels so the lines stay crisp.
  const float2 origin = floor(params.screenSize * 0.5f - 0.5f * CROSSHAIR_SIZE * params.scale);
  const float2 pixel = origin + lerp(rect.xy, rect.zw, unit) * params.scale;
  return float4(pixel / params.screenSize * 2.0f - 1.0f, 0.0f, 1.0f);
}

/** Solid white; the inverting blend does the rest. */
[shader("fragment")]
float4 crosshairFragmentMain() : SV_Target0 { return float4(1.0f, 1.0f, 1.0f, 1.0f); }

// Health bar in vanilla's survival HUD layout: 10 hearts, 2 health points each (params.columns = health points,
// rounded up; params.text[0] != 0 while the player was just hurt, which flashes the outlines white). Hearts are 9x9
// GUI pixels spaced 8 apart, left-aligned with the (182 wide, centered) hotbar, row top 39 above the bottom edge.
static const uint HEART_COUNT = 10u;
static const float HEART_SPACING = 8.0f;
static const float HOTBAR_HALF_WIDTH = 91.0f; // GUI pixels
static const float HEART_ROW_TOP = 39.0f;     // GUI pixels above the bottom edge
// Window height per vanilla GUI scale step (scale 3 at 1080p, like vanilla's usual choice there).
static const float GUI_SCALE_HEIGHT = 333.0f;
// Vanilla's full heart, 9x9 pixels, rows top to bottom, 3 bits per pixel with column 0 in the low bits:
// 0 transparent, 1 outline, 2 red, 3 highlight, 4 shade.
//   ..KK.KK..  .KRRKRRK.  KRHRRRRRK  KRRRRRRRK  KDRRRRRDK  .KDRRRDK.  ..KDRDK..  ...KDK...  ....K....
static const uint HEART_PIXELS[9] = {0x0048240u, 0x0291488u, 0x14924d1u, 0x1492491u, 0x18924a1u, 0x0312508u, 0x0062840u, 0x000c200u, 0x0001000u};
// Colors (linear; the swapchain is sRGB): outline 000000, red FF1313, highlight FFC8C8, shade BB1313; lost health
// draws the red / highlight / shade pixels in the empty color 2C2C2C.
static const float3 HEART_OUTLINE = float3(0.0f, 0.0f, 0.0f);
static const float3 HEART_RED = float3(1.0f, 0.00651f, 0.00651f);
static const float3 HEART_HIGHLIGHT = float3(1.0f, 0.578f, 0.578f);
static const float3 HEART_SHADE = float3(0.497f, 0.00651f, 0.00651f);
static const float3 HEART_EMPTY = float3(0.0252f, 0.0252f, 0.0252f);

/** Screen pixels per heart pixel: a vanilla-like GUI scale for the window height. */
float heartScale() { return max(1.0f, round(params.screenSize.y / GUI_SCALE_HEIGHT)); }

/** Health bar size in screen pixels. */
float2 healthBarSize() { return float2(HEART_SPACING * float(HEART_COUNT) + 1.0f, 9.0f) * heartScale(); }

/** Health bar top-left corner in screen pixels, snapped to whole pixels. */
float2 healthBarOrigin() {
  return floor(float2(params.screenSize.x * 0.5f - HOTBAR_HALF_WIDTH * heartScale(),
                      params.screenSize.y - HEART_ROW_TOP * heartScale()));
}

/** Pixel code (HEART_PIXELS) of heart pixel (x, y), 0..8 each; 0 (transparent) outside. */
uint heartPixel(int x, int y) {
  if (x < 0 || x > 8 || y < 0 || y > 8) {
    return 0u;
  }
  return (HEART_PIXELS[y] >> (3u * uint(x))) & 0x7u;
}

/** Quad covering the whole health bar. */
[shader("vertex")]
float4 healthVertexMain(uint vertexID: SV_VertexID) : SV_Position {
  const uint corner = vertexID < 3u ? vertexID : vertexID - 2u;
  const float2 unit = float2(float(corner & 1u), float(corner >> 1u));
  const float2 pixel = healthBarOrigin() + unit * healthBarSize();
  return float4(pixel / params.screenSize * 2.0f - 1.0f, 0.0f, 1.0f);
}

/** Hearts: params.columns = health points, params.text[0] != 0 = just hurt (white outlines). */
[shader("fragment")]
float4 healthFragmentMain(float4 position: SV_Position) : SV_Target0 {
  const float2 gui = (position.xy - healthBarOrigin()) / heartScale();
  // Later hearts draw over the overlapping outline column of earlier ones: pick the rightmost heart covering x.
  const int heart = min(int(gui.x / HEART_SPACING), int(HEART_COUNT) - 1);
  const int x = int(gui.x - float(heart) * HEART_SPACING);
  const int y = int(gui.y);
  const uint pixel = heartPixel(x, y);
  if (pixel == 0u) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (pixel == 1u) {
    return params.text[0] != 0u ? float4(1.0f, 1.0f, 1.0f, 1.0f) : float4(HEART_OUTLINE, 1.0f); // white while hurt
  }
  // Left half (columns 1-4) is the heart's first point, right half (5-7) its second.
  const bool full = params.columns >= uint(heart) * 2u + (x <= 4 ? 1u : 2u);
  if (!full) {
    return float4(HEART_EMPTY, 1.0f);
  }
  return float4(pixel == 2u ? HEART_RED : (pixel == 3u ? HEART_HIGHLIGHT : HEART_SHADE), 1.0f);
}

// F3+F4 game mode switcher, Minecraft's layout and sprites (hud.png: panel 125x75 at 0,32 with the four mode icons,
// gold selection ring 26x26 at 128,32). GUI pixels from the screen center: panel at (-62, -58), slots 31 apart from
// (-59, -31), title centered at y -51, "F4 Next" hint centered at y 5 with "F4" in aqua. Text uses MC_FONT with
// Minecraft's drop shadow (1 pixel down-right, a quarter of the color).
// params.columns = selected slot; text line 0 = title, line 1 = hint, as MC_FONT indices with each line's length in
// its last byte. Must match recordGameModeSwitcher in renderer.cpp.
static const int2 SWITCHER_PANEL_LOW = int2(-62, -58);
static const int2 SWITCHER_PANEL_SIZE = int2(125, 75);
static const int2 SWITCHER_PANEL_ATLAS = int2(0, 32);
static const int2 SWITCHER_SELECTION_ATLAS = int2(128, 32);
static const int SWITCHER_SLOT_SIZE = 26;
static const int2 SWITCHER_FIRST_SLOT = int2(3, 27); // panel pixels
static const int SWITCHER_SLOT_SPACING = 31;
static const int SWITCHER_TITLE_Y = 7;  // panel pixels
static const int SWITCHER_HINT_Y = 63;
static const uint SWITCHER_HINT_AQUA_GLYPHS = 2u; // "F4"
static const float3 TEXT_AQUA = float3(0.0908f, 1.0f, 1.0f); // 55FFFF, linear
static const float3 TEXT_AQUA_SHADOW = float3(0.0075f, 0.0497f, 0.0497f); // 153F3F
static const float3 TEXT_WHITE_SHADOW = float3(0.0497f, 0.0497f, 0.0497f); // 3F3F3F

// Minecraft's GUI font for the switcher's text, rows top to bottom (8, the last for descenders), bit 4 = leftmost
// column; .x = rows 0-3, .y = rows 4-7, 8 bits per row; .z = width (advance = width + 1).
// Order: " CSAMNFacdeilnoprtuvx4" (minecraftGlyphIndex in renderer.cpp).
static const uint MC_FONT_GLYPHS = 22u;
static const uint3 MC_FONT[MC_FONT_GLYPHS] = {
  uint3(0x00000000u, 0x00000000u, 3u), // ' '
  uint3(0x1010110eu, 0x000e1110u, 5u), // 'C'
  uint3(0x010e100fu, 0x000e1101u, 5u), // 'S'
  uint3(0x111f110eu, 0x00111111u, 5u), // 'A'
  uint3(0x11151b11u, 0x00111111u, 5u), // 'M'
  uint3(0x13151911u, 0x00111111u, 5u), // 'N'
  uint3(0x101c101fu, 0x00101010u, 5u), // 'F'
  uint3(0x010e0000u, 0x000f110fu, 5u), // 'a'
  uint3(0x110e0000u, 0x000e1110u, 5u), // 'c'
  uint3(0x130d0101u, 0x000f1111u, 5u), // 'd'
  uint3(0x110e0000u, 0x000f101fu, 5u), // 'e'
  uint3(0x10100010u, 0x00101010u, 1u), // 'i'
  uint3(0x10101010u, 0x00081010u, 2u), // 'l'
  uint3(0x111e0000u, 0x00111111u, 5u), // 'n'
  uint3(0x110e0000u, 0x000e1111u, 5u), // 'o'
  uint3(0x19160000u, 0x10101e11u, 5u), // 'p'
  uint3(0x19160000u, 0x00101010u, 5u), // 'r'
  uint3(0x081c0808u, 0x00040808u, 3u), // 't'
  uint3(0x11110000u, 0x000f1111u, 5u), // 'u'
  uint3(0x11110000u, 0x00040a11u, 5u), // 'v'
  uint3(0x0a110000u, 0x00110a04u, 5u), // 'x'
  uint3(0x11090503u, 0x0001011fu, 5u), // '4'
};

/** Whether MC_FONT glyph `glyph` covers pixel (x, y). */
bool minecraftGlyphLit(uint glyph, int x, int y) {
  glyph = min(glyph, MC_FONT_GLYPHS - 1u);
  if (x < 0 || y < 0 || y > 7 || x >= int(MC_FONT[glyph].z)) {
    return false;
  }
  const uint rowBits = ((y < 4 ? MC_FONT[glyph].x : MC_FONT[glyph].y) >> (uint(y & 3) * 8u)) & 0x1fu;
  return ((rowBits >> (4u - uint(x))) & 1u) != 0u;
}

/**
 * Text pixel of switcher line `line`, centered on the panel like Minecraft's drawCenteredString.
 *
 * @param pixel Panel pixel.
 * @param top Line top, in panel pixels.
 * @param glyphIndex Receives the character index under `pixel`.
 * @return 0 = nothing, 1 = shadow, 2 = text.
 */
uint switcherText(int2 pixel, uint line, int top, out uint glyphIndex) {
  const uint length = glyphAt(line, OVERLAY_COLUMNS - 1u);
  int width = 0;
  for (uint i = 0u; i < length; ++i) {
    width += int(MC_FONT[min(glyphAt(line, i), MC_FONT_GLYPHS - 1u)].z) + 1;
  }
  const int2 local = pixel - int2(-SWITCHER_PANEL_LOW.x - width / 2, top);
  uint result = 0u;
  glyphIndex = 0u;
  int x = 0;
  for (uint i = 0u; i < length; ++i) {
    const uint glyph = glyphAt(line, i);
    if (minecraftGlyphLit(glyph, local.x - x, local.y)) {
      glyphIndex = i;
      return 2u;
    }
    if (result == 0u && minecraftGlyphLit(glyph, local.x - x - 1, local.y - 1)) {
      glyphIndex = i;
      result = 1u;
    }
    x += int(MC_FONT[min(glyph, MC_FONT_GLYPHS - 1u)].z) + 1;
  }
  return result;
}

/** Panel top-left corner in screen pixels. */
float2 switcherOrigin() { return floor(params.screenSize * 0.5f) + float2(SWITCHER_PANEL_LOW) * heartScale(); }

/** Quad covering the switcher panel. */
[shader("vertex")]
float4 switcherVertexMain(uint vertexID: SV_VertexID) : SV_Position {
  const uint corner = vertexID < 3u ? vertexID : vertexID - 2u;
  const float2 unit = float2(float(corner & 1u), float(corner >> 1u));
  const float2 pixel = switcherOrigin() + unit * float2(SWITCHER_PANEL_SIZE) * heartScale();
  return float4(pixel / params.screenSize * 2.0f - 1.0f, 0.0f, 1.0f);
}

/** Panel with icons, selection ring, title and hint. */
[shader("fragment")]
float4 switcherFragmentMain(float4 position: SV_Position) : SV_Target0 {
  const int2 pixel = int2(floor((position.xy - switcherOrigin()) / heartScale()));
  float4 color = atlasTexel(pixel, SWITCHER_PANEL_ATLAS, SWITCHER_PANEL_SIZE);
  const int2 slotOrigin = SWITCHER_FIRST_SLOT + int2(SWITCHER_SLOT_SPACING * int(params.columns), 0);
  color = over(atlasTexel(pixel - slotOrigin, SWITCHER_SELECTION_ATLAS, int2(SWITCHER_SLOT_SIZE, SWITCHER_SLOT_SIZE)),
               color);
  uint glyphIndex;
  const uint title = switcherText(pixel, 0u, SWITCHER_TITLE_Y, glyphIndex);
  if (title != 0u) {
    return float4(title == 2u ? float3(1.0f, 1.0f, 1.0f) : TEXT_WHITE_SHADOW, 1.0f);
  }
  const uint hint = switcherText(pixel, 1u, SWITCHER_HINT_Y, glyphIndex);
  if (hint != 0u) {
    const bool aqua = glyphIndex < SWITCHER_HINT_AQUA_GLYPHS;
    return float4(hint == 2u ? (aqua ? TEXT_AQUA : float3(1.0f, 1.0f, 1.0f))
                             : (aqua ? TEXT_AQUA_SHADOW : TEXT_WHITE_SHADOW), 1.0f);
  }
  return color;
}
