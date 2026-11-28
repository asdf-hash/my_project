// Must match OVERLAY_LINES / OVERLAY_COLUMNS in renderer.cpp.
static const uint OVERLAY_LINES = 4u;
static const uint OVERLAY_COLUMNS = 28u;
static const uint OVERLAY_TEXT_WORDS = OVERLAY_LINES * OVERLAY_COLUMNS / 4u;

// Drawn into the swapchain image after TAA: debug text (top right) and the crosshair. Text arrives as glyph indices
// in push constants and uses a built-in 5x7 bitmap font. The HUD and screens are gui.slang.

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
