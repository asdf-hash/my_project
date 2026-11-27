#include "renderer.hpp"
#include "block_registry.hpp"
#include "camera.hpp"
#include "chunk_slots.hpp"
#include "clipmap_culler.hpp"
#include "cpu_world.hpp"
#include "lod.hpp"
#include "player.hpp"
#include "terrain_cpu.hpp"
#include "world.hpp"

#include <algorithm>
#include <array>
#include <assert.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

// GLFW_INCLUDE_VULKAN (for glfwCreateWindowSurface) is defined in pch.hpp, which includes these first. No GLM_FORCE_*
// defines here: they would have no effect after the PCH (see the projection in updateUniformBuffer).
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stb/stb_image.h> // implementation in stb_image_impl.cpp

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr vk::SampleCountFlagBits RENDER_SAMPLE_COUNT = vk::SampleCountFlagBits::e2;
constexpr float BASE_FOV_RADIANS = glm::radians(70.0f);
constexpr vk::Format TAA_HISTORY_FORMAT = vk::Format::eR16G16B16A16Sfloat;
static_assert(sizeof(world::ChunkLod) == sizeof(int32_t) * 4);

const std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct alignas(16) UniformBufferObject {
  glm::mat4 mvp;
  glm::mat4 previousMvp;
  glm::vec2 renderSize;
  uint32_t contourMode; // 1: elevation view (H), see scene_ubo.hlsl
  float farTintStart;   // see scene_ubo.hlsl
  float lodAoLighteningPerLevel;
  float nearPlane; // projection planes, for view depth from the depth buffer (water fog)
  float farPlane;
  alignas(16) glm::vec4 cameraPosition;
  glm::vec4 viewRayScale; // xy: 1 / proj[0][0], 1 / proj[1][1]; view ray per NDC unit, for fog distance. z: underwater
                          // w: border fog end (camera distance where the air fog is opaque)
  glm::vec4 hzbParams;    // xy: Hi-Z mip 0 size, z: mip count, w: 1 when it holds last frame's depth
  glm::mat4 cullMvp;      // task-shader frustum test: mvp, or the frozen one while LOD freeze (G) is on
  glm::vec4 cullCameraPosition; // run / quad facing tests: camera, or the frozen eye
  glm::mat4 cullPreviousMvp;    // Hi-Z test: the matrices the Hi-Z was built with (frozen with it under G)
};
static_assert(offsetof(UniformBufferObject, previousMvp) == 64);
static_assert(offsetof(UniformBufferObject, renderSize) == 128);
static_assert(offsetof(UniformBufferObject, contourMode) == 136);
static_assert(offsetof(UniformBufferObject, lodAoLighteningPerLevel) == 144);
static_assert(offsetof(UniformBufferObject, nearPlane) == 148);
static_assert(offsetof(UniformBufferObject, farPlane) == 152);
static_assert(offsetof(UniformBufferObject, cameraPosition) == 160);
static_assert(offsetof(UniformBufferObject, viewRayScale) == 176);
static_assert(offsetof(UniformBufferObject, hzbParams) == 192);
static_assert(offsetof(UniformBufferObject, cullMvp) == 208);
static_assert(offsetof(UniformBufferObject, cullCameraPosition) == 272);
static_assert(offsetof(UniformBufferObject, cullPreviousMvp) == 288);
static_assert(sizeof(UniformBufferObject) == 352);

struct ChunkComputeParams {
  uint32_t horizontalChunkCount; // new columns this remesh (ColumnList entries)
  uint32_t retireCount;          // retired column slots (RetireList entries)
  uint32_t workPoolCapacity;     // work slots on the GPU (initWorkPool)
  uint32_t oceanBlock;           // block ID of the GPU clipmap's ocean surface quads
  // Block ID per terrain height layer (world::kTerrainLayerBlocks) and below the top block (kTerrainLayerFillerBlocks).
  std::array<uint32_t, world::kTerrainLayerBlocks.size()> layerBlocks;
  std::array<uint32_t, world::kTerrainLayerBlocks.size()> layerFillerBlocks;
  std::array<uint32_t, world::kTerrainLayerBlocks.size()> layerDeepBlocks; // kTerrainLayerDeepBlocks
  // First item of a batched dispatch (see dispatchBatchedZ): item = groupID.z + zBase.
  uint32_t zBase;
  uint32_t logBlock;    // world::kTreeLogBlock (tree placement: shaders/vegetation_shared.hlsl)
  uint32_t leavesBlock; // world::kTreeLeavesBlock
};
static_assert(offsetof(ChunkComputeParams, layerBlocks) == 16);
static_assert(offsetof(ChunkComputeParams, layerFillerBlocks) == 32);
static_assert(offsetof(ChunkComputeParams, layerDeepBlocks) == 48);
static_assert(offsetof(ChunkComputeParams, zBase) == 64);
static_assert(offsetof(ChunkComputeParams, leavesBlock) == 72);

// One entry per (block, face), indexed blockId * 6 + normalIndex. Matches BlockFace in slang.slang.
struct GpuBlockFace {
  uint32_t texture;
  uint32_t lodTexture;
  uint32_t flags; // BLOCK_FACE_* bits
  // Frame strips: frames of texture (bits 0-7) and lodTexture (8-15), ms per frame (16-31). Frame k of a texture is
  // at its slot + k.
  uint32_t animation;
  std::array<float, 4> lodTint;
  std::array<float, 4> tint;    // every LOD (blocks.txt `tint`)
  std::array<float, 4> farTint; // faded in with distance (blocks.txt `far_tint`)
};
static_assert(sizeof(GpuBlockFace) == 64);
constexpr uint32_t BLOCK_FACE_RANDOM_ROTATION = 1u << 0;
constexpr uint32_t BLOCK_FACE_HAS_LOD_TEXTURE = 1u << 1;
constexpr uint32_t BLOCK_FACE_FLUID = 1u << 2; // a liquid (drawn in the translucent pass)

// Push constant of the scene pipelines (geometry_shared.hlsl drawParams): which cluster set the task shader walks,
// and the debug view's tint (DEBUG_TINT_*).
struct GeometryPushConstants {
  uint32_t clusterCount;
  uint32_t debugTint;
  uint32_t quadBase;  // vertex path: first quad of this draw
  uint32_t quadCount; // vertex path: quads in this draw (0: up to QuadCount[0])
  uint32_t pass;      // GEOMETRY_PASS_*
  uint32_t clusterBase; // mesh path: first cluster of this draw (CPU frustum culling draws runs of visible chunk slots)
};
constexpr uint32_t GEOMETRY_PASS_OPAQUE = 0u;
constexpr uint32_t GEOMETRY_PASS_TRANSLUCENT = 1u; // blended water: no back-face culling
constexpr uint32_t GEOMETRY_PASS_CUTOUT = 2u;      // non-opaque blocks: alpha-tested (fragCutout)
constexpr uint32_t DEBUG_TINT_OFF = 0u;
constexpr uint32_t DEBUG_TINT_LOD = 1u;
constexpr uint32_t DEBUG_TINT_CPU = 2u;
constexpr vk::ShaderStageFlags GEOMETRY_PUSH_STAGES =
    vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eVertex |
    vk::ShaderStageFlagBits::eFragment;

// What the scene pass draws; X cycles through them.
enum class TerrainView { Both, LodOnly, RealOnly, Debug };

struct TaaParams {
  glm::vec2 inverseRenderSize;
  uint32_t historyValid;
  float depthRejectionThreshold;
};
static_assert(sizeof(TaaParams) == 16);

constexpr uint32_t TEXTURE_MIP_LEVELS = 5;
constexpr vk::Format TEXTURE_FORMAT = vk::Format::eR8G8B8A8Srgb;
constexpr vk::Format MOTION_FORMAT = vk::Format::eR16G16B16A16Sfloat;

class VulkanEngine {
public:
  /** Create the window and Vulkan state, run the main loop, then tear everything down. */
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
#pragma region Renderer State
#pragma region Window State
  GLFWwindow *window = nullptr;
#pragma endregion

#pragma region Vulkan State
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  uint32_t queueIndex = ~0u;
  vk::raii::Queue queue = nullptr;
#pragma endregion

#pragma region Swapchain State
  vk::raii::SwapchainKHR swapChain = nullptr;
  std::vector<vk::Image> swapChainImages;
  vk::SurfaceFormatKHR swapChainSurfaceFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vk::raii::ImageView> swapChainImageViews;
#pragma endregion

#pragma region Pipeline State
  // Every pipeline is created through this cache, saved to PIPELINE_CACHE_FILE on exit and loaded at startup, so later
  // launches skip the driver's shader compilation.
  static constexpr const char *PIPELINE_CACHE_FILE = "pipeline_cache.bin";
  vk::raii::PipelineCache pipelineCache = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;
  // Non-opaque blocks (leaves, glass): alpha-tested fragCutout, drawn after the opaque pipelines (GEOMETRY_PASS_CUTOUT).
  vk::raii::Pipeline cutoutPipeline = nullptr;
  vk::raii::Pipeline cutoutVertexPullingPipeline = nullptr;
  // Same packed quads, unpacked in a vertex shader via SV_VertexID instead of the mesh shader. Toggle with V.
  vk::raii::Pipeline vertexPullingPipeline = nullptr;
  bool useVertexPulling = false;
  // Line-mode copies of the two pipelines above. Toggle with T.
  vk::raii::Pipeline wireframePipeline = nullptr;
  vk::raii::Pipeline wireframeVertexPullingPipeline = nullptr;
  // Translucent pass (ocean): same shaders, alpha blended, no depth writes (kept out of the Hi-Z and TAA depth), no
  // motion writes, no culling (the surface is seen from below too).
  vk::raii::Pipeline waterPipeline = nullptr;
  vk::raii::Pipeline waterVertexPullingPipeline = nullptr;
  bool wireframeEnabled = false;
  bool contourEnabled = false; // elevation view (H)
  TerrainView terrainView = TerrainView::Both;
  vk::raii::DescriptorSetLayout taaDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout taaPipelineLayout = nullptr;
  vk::raii::Pipeline taaPipeline = nullptr;
  vk::raii::DescriptorSetLayout chunkComputeDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout chunkComputePipelineLayout = nullptr;
  vk::raii::Pipeline generateHeightsPipeline = nullptr;
  vk::raii::Pipeline retireWorkPipeline = nullptr;
  vk::raii::Pipeline allocateWorkPipeline = nullptr;
  vk::raii::Pipeline finalizeWorkPipeline = nullptr;
  vk::raii::Pipeline initWorkPoolPipeline = nullptr;
  vk::raii::Pipeline collectRelocationsPipeline = nullptr;
  vk::raii::Pipeline relocateSlotsPipeline = nullptr;
  vk::raii::Pipeline rebuildFreeStackPipeline = nullptr;
  vk::raii::Pipeline meshColumnsPipeline = nullptr;
  vk::raii::Pipeline finalizeVertexDrawPipeline = nullptr;
  vk::raii::Pipeline compactQuadsPipeline = nullptr;
  vk::raii::Pipeline copyCompactedQuadsPipeline = nullptr;
  vk::raii::PipelineLayout overlayPipelineLayout = nullptr;
  vk::raii::Pipeline overlayPipeline = nullptr;
  vk::raii::Pipeline hudPipeline = nullptr;     // overlayPipelineLayout, alpha blend
  vk::raii::Pipeline crosshairPipeline = nullptr; // overlayPipelineLayout, inverting blend
  vk::raii::Pipeline healthPipeline = nullptr;    // overlayPipelineLayout, alpha blend
  vk::raii::Pipeline gameModeSwitcherPipeline = nullptr; // overlayPipelineLayout, alpha blend
  vk::raii::PipelineLayout outlinePipelineLayout = nullptr;
  vk::raii::Pipeline outlinePipeline = nullptr;
  vk::raii::Pipeline crackPipeline = nullptr; // outlinePipelineLayout; block-breaking cracks, Minecraft's 2*src*dst blend
  // Hi-Z occlusion (mesh path): farthest-depth pyramid of last frame's depth, read by taskMain (hzb.slang).
  vk::raii::DescriptorSetLayout hzbSetLayout = nullptr;
  vk::raii::PipelineLayout hzbPipelineLayout = nullptr;
  vk::raii::Pipeline hzbBuildLowPipeline = nullptr;     // resolved depth -> mips 0..4
  vk::raii::Pipeline hzbBuildHighPipeline = nullptr;    // mip 4 -> mips 5..last
  vk::raii::DescriptorPool hzbDescriptorPool = nullptr; // own pool: rebuilt with the swapchain-sized resources
  vk::raii::DescriptorSet hzbSet{nullptr};              // the resolved depth + every mip
  vk::raii::Image hzbImage = nullptr;
  vk::raii::DeviceMemory hzbImageMemory = nullptr;
  vk::raii::ImageView hzbView = nullptr; // all mips, for taskMain
  std::vector<vk::raii::ImageView> hzbMipViews;
  vk::Extent2D hzbSize{};
  uint32_t hzbMipCount = 0;
  bool hzbImageInitialized = false;
  bool hzbValid = false; // holds the previous frame's depth, matching ubo.previousMvp
  // Toggle with R. Cost on an RTX 3070 at 1000x750: ~20 us for the depth resolve plus ~18 us for the two-dispatch
  // pyramid build (hzb.slang). Reading the MSAA depth directly cost ~53 us. Ground-level views saved 20-110 us
  // net; the top-down spawn view, with nothing occluded, lost ~40 us.
  bool hzbOcclusionEnabled = true;
  static constexpr uint32_t MAX_HZB_MIPS = 16u;        // MAX_HZB_MIPS in hzb.slang
  static constexpr uint32_t HZB_LOW_MIPS = 5u;         // mips written by hzbBuildLow (LOW_MIPS)
  static constexpr uint32_t HZB_MAX_MIP0_SIDE = 2048u; // hzbBuildHigh holds mip 5 (<= 64x64) in shared memory
  // Depth closer than this (view depth, blocks) is left out of the Hi-Z: near geometry disoccludes within a frame or
  // two of camera motion, and last frame's depth of it would cull what is newly visible behind it.
  static constexpr float HZB_IGNORE_NEAR_DISTANCE = 20.0f;
  struct HzbPushParams {
    std::array<uint32_t, 2> mip0Size;
    std::array<uint32_t, 2> depthSize;
    uint32_t mipCount;
    float ignoreNearerDepth; // reverse-Z depth at HZB_IGNORE_NEAR_DISTANCE: samples above it count as far
  };
  static constexpr float NEAR_PLANE = 0.1f;
  static constexpr float FAR_PLANE = 1000000.0f;
  /**
   * Reverse-Z depth (1 at NEAR_PLANE, 0 at FAR_PLANE) of a view depth; linear in 1 / d.
   *
   * @param d View depth, in blocks.
   * @return reverse-Z depth of `d`.
   */
  static constexpr float reverseZDepth(float d) {
    return NEAR_PLANE * (FAR_PLANE - d) / (d * (FAR_PLANE - NEAR_PLANE));
  }
#pragma endregion

#pragma region Image State
  vk::raii::Image multisampleColorImage{nullptr};
  vk::raii::DeviceMemory multisampleColorImageMemory{nullptr};
  vk::raii::ImageView multisampleColorImageView{nullptr};
  vk::raii::Image sceneColorImage{nullptr};
  vk::raii::DeviceMemory sceneColorImageMemory{nullptr};
  vk::raii::ImageView sceneColorImageView{nullptr};
  vk::raii::Image multisampleMotionImage{nullptr};
  vk::raii::DeviceMemory multisampleMotionImageMemory{nullptr};
  vk::raii::ImageView multisampleMotionImageView{nullptr};
  vk::raii::Image motionImage{nullptr};
  vk::raii::DeviceMemory motionImageMemory{nullptr};
  vk::raii::ImageView motionImageView{nullptr};
  std::array<vk::raii::Image, 2> taaHistoryImages{nullptr, nullptr};
  std::array<vk::raii::DeviceMemory, 2> taaHistoryImageMemories{nullptr, nullptr};
  std::array<vk::raii::ImageView, 2> taaHistoryImageViews{nullptr, nullptr};
  vk::raii::Image depthImage{nullptr};
  vk::raii::DeviceMemory depthImageMemory{nullptr};
  vk::raii::ImageView depthImageView{nullptr};
  // Single-sample MIN resolve of depthImage (farthest sample per pixel under reverse-Z), written by the opaque pass;
  // the water fog reads it, and recordHzb builds the pyramid from it. Reading the MSAA depth directly forced a ~40 us
  // decompression.
  vk::raii::Image resolvedDepthImage{nullptr};
  vk::raii::DeviceMemory resolvedDepthImageMemory{nullptr};
  vk::raii::ImageView resolvedDepthImageView{nullptr};

  static constexpr uint32_t MAX_TEXTURE_COUNT = 1024;
  uint32_t textureDescriptorCapacity = 0;
  std::vector<vk::raii::Image> textureImages; // one per animation frame
  // Per registry texture index: first bindless slot and frame count (createTextureImages).
  std::vector<uint32_t> textureSlots;
  std::vector<uint32_t> textureFrames;
  uint32_t hudTextureSlot = 0; // bindless slot of textures/gui/hud.png (after the block textures)
  // Bindless slot of textures/overlay/destroy_stage_0.png; stages 1-9 follow (after the HUD atlas).
  static constexpr uint32_t DESTROY_STAGES = 10;
  uint32_t destroyStageSlot = 0;
  std::vector<vk::raii::DeviceMemory> textureImageMemories;
  std::vector<vk::raii::ImageView> textureImageViews;
  blocks::Registry blockRegistry;
  std::array<uint32_t, world::kTerrainLayerBlocks.size()> terrainLayerBlocks{};
  std::array<uint32_t, world::kTerrainLayerBlocks.size()> terrainLayerFillerBlocks{};
  std::array<uint32_t, world::kTerrainLayerBlocks.size()> terrainLayerDeepBlocks{};
  uint32_t treeLogBlock = 0;
  uint32_t treeLeavesBlock = 0;
  vk::raii::Buffer blockFaceBuffer = nullptr;
  vk::raii::DeviceMemory blockFaceBufferMemory = nullptr;
  vk::raii::Sampler textureSampler = nullptr;
  vk::raii::Sampler taaSampler = nullptr;
#pragma endregion

#pragma region Buffer State
  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  static constexpr uint32_t MAX_QUAD_COUNT = 1u << 24u;
  // Quads covered by one instance of the vertex-pulling draw. 4 vertices per quad must fit a uint16 index.
  // Must match VERTEX_QUADS_PER_INSTANCE in slang.slang and chunk_compute.slang.
  static constexpr uint32_t VERTEX_QUADS_PER_INSTANCE = 16384u;
  static_assert(VERTEX_QUADS_PER_INSTANCE * 4u - 1u <= UINT16_MAX);
  // Debug text overlay; must match overlay.slang.
  static constexpr uint32_t OVERLAY_LINES = 4u;
  static constexpr uint32_t OVERLAY_COLUMNS = 28u;
  struct OverlayParams {
    glm::vec2 screenSize;
    float scale;
    uint32_t columns;
    std::array<uint32_t, OVERLAY_LINES * OVERLAY_COLUMNS / 4u> text; // glyph indices, one byte each
  };
  static_assert(sizeof(OverlayParams) == 128); // Vulkan's guaranteed maxPushConstantsSize
  // Targeted block outline; must match outline.slang.
  struct OutlineParams {
    glm::vec4 boxMin; // world space; w = line width in pixels (cracks: bindless texture slot)
    glm::vec4 boxMax;
  };
  static_assert(sizeof(OutlineParams) == 32);
  // QuadCluster in chunk_compute.slang / slang.slang: one per meshColumns group, 16 groups per chunk.
  static constexpr vk::DeviceSize QUAD_CLUSTER_SIZE = 48u;
  static constexpr uint32_t QUAD_CLUSTERS_PER_CHUNK = 16u;
  static constexpr uint32_t MAX_HORIZONTAL_CHUNK_COUNT = kClipmapMaxChunkCount; // clipmap_culler.hpp
  // Work slots (one per meshed vertical chunk, 16 QuadClusters each), handed out on the GPU by allocateWork. The pool
  // doubles (growWorkPool) whenever allocateWork runs dry, up to the most the clipmap could ever need.
  static constexpr uint32_t WORK_POOL_INITIAL_CAPACITY = 1u << 15u;
  static constexpr uint32_t WORK_POOL_MAX_CAPACITY =
      MAX_HORIZONTAL_CHUNK_COUNT * static_cast<uint32_t>(world::kMaxColumnChunks);
  uint32_t workPoolCapacity = WORK_POOL_INITIAL_CAPACITY;
  // WorkPool (compute binding 12) layout in ints; must match WORK_POOL_* in chunk_compute.slang.
  static constexpr uint32_t WORK_POOL_INTS = 16u;
  static constexpr vk::DeviceSize WORK_POOL_MESH_ARGS_OFFSET = 4u * sizeof(int32_t);
  static constexpr vk::DeviceSize WORK_POOL_COMPACT_ARGS_OFFSET = 8u * sizeof(int32_t);
  static constexpr vk::DeviceSize WORK_POOL_TASK_ARGS_OFFSET = 12u * sizeof(int32_t);
  static constexpr uint32_t HEIGHT_GROUPS_PER_COLUMN = 19u; // generateHeights groups, each with a height range
  std::vector<world::ChunkLod> visibleChunkSet;
  std::vector<uint32_t> visibleChunkNeighborLods; // parallel to visibleChunkSet
  ClipmapCuller clipmapCuller;
  std::optional<ClipmapView> lastSubmittedClipmapView;
  std::optional<ClipmapView> lastAppliedClipmapView;
  std::uint64_t latestSubmittedClipmapSequence = 0;
  std::uint64_t lastAppliedClipmapSequence = 0;
  // KNOWN ISSUE (-O0 only): at full freecam speed the drawn CPU square lags the player and GPU LOD terrain shows
  // under them. CPU generation keeps up; the lag is in the clipmap hand-over (applied clipmap vs advertised square).

  // CPU square centers of clipmap jobs submitted after the applied one (sequence, center). Their squares stay pinned
  // in cpuWorld so whichever job finishes can be applied without a hole; dropped once a newer result is applied.
  std::deque<std::pair<std::uint64_t, cpuworld::ChunkXZ>> inFlightCpuCenters;
  bool chunkBuffersDirty = true;
  // Incremental remeshing: persistent GPU slots for the visible columns and their vertical chunks. A clipmap change
  // only meshes what is new and retires what left; dead quads pile up in QuadBuffer until compactQuads packs them out.
  ChunkSlotTable chunkSlots;
  ChunkRemeshPlan remeshPlan; // built in updateUniformBuffer, recorded by recordChunkCompute in the same frame
  bool forceFullRebuild = true;
  // Pack QuadBuffer (compactQuads) at the start of the next recordChunkCompute. Bumping quadCountEpoch invalidates
  // readbacks recorded before the compaction, so a stale count can't trigger a second one.
  bool compactQuadsPending = false;
  uint32_t quadCountEpoch = 0;
  std::array<uint32_t, 8> quadCountReadbackEpoch{};

  // Compact once dead quads pass a share of the live ones (vertex path 1/4: it draws dead quads too; mesh path 1/1), or
  // once the buffer is 3/4 used and compacting would free space. A full rebuild is only for an overflowed QuadBuffer.
  static constexpr uint32_t REBUILD_MIN_DEAD_QUADS = 1u << 18u;
  static constexpr uint32_t COPY_COMPACTED_GROUPS = 1024u; // must match copyCompactedQuads in chunk_compute.slang
  // Copied back after each remesh, per frame in flight: QuadCount ({appended, live}), then WorkPool[0..3] (free-stack
  // top, high water, new work, overflow).
  static constexpr uint32_t READBACK_UINTS = 6u;
  std::vector<vk::raii::Buffer> quadCountReadbackBuffers;
  std::vector<vk::raii::DeviceMemory> quadCountReadbackMemories;
  std::vector<const uint32_t *> quadCountReadbackMapped;
  std::array<bool, 8> quadCountReadbackPending{};
  // Per frame: this remesh's new column slots (binding 9) and retired column slots (binding 10).
  std::vector<vk::raii::Buffer> columnListBuffers;
  std::vector<vk::raii::DeviceMemory> columnListBufferMemories;
  std::vector<void *> columnListBuffersMapped;
  std::vector<vk::raii::Buffer> retireListBuffers;
  std::vector<vk::raii::DeviceMemory> retireListBufferMemories;
  std::vector<void *> retireListBuffersMapped;
  std::array<float, MAX_LOD> lodDistanceSquared =
      initializeLodDistanceSquared(TARGET_SCREEN_PERCENTAGE, BASE_FOV_RADIANS, {WIDTH, HEIGHT});
  std::array<float, MAX_LOD> baseLodDistanceSquared = lodDistanceSquared; // base FOV: tiles outside the frustum
  bool quadOverflowWarned = false;
  float cachedLodFov = BASE_FOV_RADIANS;
  vk::Extent2D cachedLodExtent{WIDTH, HEIGHT};

  // CPU-meshed near chunks (cpu_world.hpp): same packed formats as the GPU-meshed buffers, drawn with a second
  // descriptor set. A column is drawn only inside the square the applied clipmap leaves out, so CPU and GPU terrain
  // never overlap or leave a gap during a hand-over. Hidden chunks keep their quads but have zeroed clusters.
  static constexpr uint32_t CPU_MAX_QUADS = 1u << 22u;
  static constexpr uint32_t CPU_MAX_CHUNK_SLOTS = 8192u;
  static constexpr vk::DeviceSize CPU_STAGING_BYTES = 16ull << 20u; // per frame in flight
  std::unique_ptr<cpuworld::CpuWorld> cpuWorld;
  Player player;
  vk::raii::Buffer cpuQuadBuffer = nullptr;
  vk::raii::DeviceMemory cpuQuadBufferMemory = nullptr;
  vk::raii::Buffer cpuChunkOffsetBuffer = nullptr;
  vk::raii::DeviceMemory cpuChunkOffsetBufferMemory = nullptr;
  vk::raii::Buffer cpuClusterBuffer = nullptr;
  vk::raii::DeviceMemory cpuClusterBufferMemory = nullptr;
  // Same layout for the translucent (ocean) clusters; their quads follow the opaque ones in each allocation.
  vk::raii::Buffer cpuTranslucentClusterBuffer = nullptr;
  vk::raii::DeviceMemory cpuTranslucentClusterBufferMemory = nullptr;
  vk::raii::Buffer cpuQuadCountBuffer = nullptr; // binding 4 of the CPU set: quad high water, for vertMain
  vk::raii::DeviceMemory cpuQuadCountBufferMemory = nullptr;
  std::vector<vk::raii::Buffer> cpuStagingBuffers;
  std::vector<vk::raii::DeviceMemory> cpuStagingMemories;
  std::vector<std::byte *> cpuStagingMapped;
  struct CpuChunkAllocation {
    int32_t cy; // chunk y, in chunks
    uint32_t slot;
    uint32_t quadOffset;
    uint32_t quadCount;
    std::array<cpuworld::QuadCluster, cpuworld::kClustersPerChunk> clusters;            // rebased onto quadOffset
    std::array<cpuworld::QuadCluster, cpuworld::kClustersPerChunk> translucentClusters; // rebased too
    uint32_t opaqueQuadCount; // quads [0, opaqueQuadCount) are opaque, the rest translucent (vertex path)
  };
  struct CpuColumnGpu {
    int32_t cx; // chunk column x, in chunks
    int32_t cz; // chunk column z, in chunks
    bool visible;
    std::vector<CpuChunkAllocation> chunks;
  };
  std::unordered_map<uint64_t, CpuColumnGpu> cpuColumns;
  SlotAllocator cpuChunkSlots;
  // CPU frustum cull of the CPU-meshed chunks (mesh path), rebuilt once per recorded frame from cullMvpCpu.
  glm::mat4 cullMvpCpu{1.0f}; // = ubo.cullMvp
  std::vector<uint8_t> cpuSlotVisible;
  std::vector<std::pair<uint32_t, uint32_t>> cpuVisibleSlotRuns; // [first slot, slot count)
  RangeAllocator cpuQuadRanges{CPU_MAX_QUADS};
  std::deque<cpuworld::ColumnMeshEvent> cpuPendingEvents; // waiting for staging space
  // Center offered to the clipmap: the CPU's ready center, once all of its geometry is uploaded.
  std::optional<cpuworld::ChunkXZ> cpuAdvertisedCenter;
  std::vector<cpuworld::ChunkXZ> pinnedCpuCenters; // reused every frame by the CPU world update
  // The camera is too far above the CPU square (world::kCpuHideHeightChunks): it stays loaded (collision), but is
  // left out of the clipmap, which hides its chunks and lets the GPU draw those columns.
  bool cpuHiddenByHeight = false;
  uint32_t maxComputeGroupsZ = 0;                  // device limit (capped at 65535), queried on first remesh
  // Debug (G): while set, the clipmap and the CPU world are driven by this frozen view / eye instead of the camera,
  // so LOD tiles and CPU chunks keep their quality and positions while the camera flies around them.
  bool lodFreezeRequested = false;
  std::optional<ClipmapView> frozenLodView;
  glm::vec3 frozenLodEye{0.0f};
  glm::mat4 frozenCullMvp{1.0f}; // set on the first frozen frame, from that frame's mvp
  glm::mat4 frozenHzbMvp{1.0f};  // the previousMvp of that frame: the view the (no longer rebuilt) Hi-Z holds
  bool frozenCullMvpValid = false;
  // This frame's CPU geometry transfers, recorded by recordCpuUploads.
  std::vector<vk::BufferCopy> cpuQuadCopies;
  std::vector<vk::BufferCopy> cpuOffsetCopies;
  std::vector<vk::BufferCopy> cpuClusterCopies;
  std::vector<vk::BufferCopy> cpuTranslucentClusterCopies;
  std::vector<uint32_t> cpuClearedSlots;
  vk::DeviceSize cpuStagingUsed = 0;
  bool cpuQuadCountDirty = false;
  bool cpuCapacityWarned = false;

  vk::raii::Buffer quadBuffer = nullptr;
  vk::raii::DeviceMemory quadBufferMemory = nullptr;
  vk::raii::Buffer chunkOffsetBuffer = nullptr;
  vk::raii::Buffer quadCompactBuffer = nullptr; // compute binding 11, see compactQuads
  vk::raii::DeviceMemory quadCompactBufferMemory = nullptr;
  vk::raii::DeviceMemory chunkOffsetBufferMemory = nullptr;
  vk::raii::Buffer chunkHeightBuffer = nullptr;
  vk::raii::DeviceMemory chunkHeightBufferMemory = nullptr;
  vk::raii::Buffer quadCountBuffer = nullptr;
  vk::raii::DeviceMemory quadCountBufferMemory = nullptr;
  vk::raii::Buffer quadIndexBuffer = nullptr;
  vk::raii::DeviceMemory quadIndexBufferMemory = nullptr;
  vk::raii::Buffer quadClusterBuffer = nullptr;
  vk::raii::DeviceMemory quadClusterBufferMemory = nullptr;
  // Same layout: each tile's ocean surface quads (translucent pass). Binding 18 of the compute set.
  vk::raii::Buffer waterClusterBuffer = nullptr;
  vk::raii::DeviceMemory waterClusterBufferMemory = nullptr;
  vk::raii::Buffer vertexDrawArgumentsBuffer = nullptr; // vertMain's indexed indirect args (finalizeVertexDraw)
  vk::raii::DeviceMemory vertexDrawArgumentsBufferMemory = nullptr;
  std::vector<vk::raii::Buffer> chunkCoordinateBuffers;
  std::vector<vk::raii::DeviceMemory> chunkCoordinateBufferMemories;
  std::vector<void *> chunkCoordinateBuffersMapped;
  // GPU-side work slot allocation (allocateWork / retireWork): bindings 5 and 12-15.
  vk::raii::Buffer chunkWorkBuffer = nullptr;
  vk::raii::DeviceMemory chunkWorkBufferMemory = nullptr;
  vk::raii::Buffer workPoolBuffer = nullptr;
  vk::raii::DeviceMemory workPoolBufferMemory = nullptr;
  vk::raii::Buffer freeWorkSlotBuffer = nullptr;
  vk::raii::DeviceMemory freeWorkSlotBufferMemory = nullptr;
  vk::raii::Buffer columnWorkSlotBuffer = nullptr;
  vk::raii::DeviceMemory columnWorkSlotBufferMemory = nullptr;
  vk::raii::Buffer columnHeightRangeBuffer = nullptr;
  vk::raii::DeviceMemory columnHeightRangeBufferMemory = nullptr;
  vk::raii::Buffer slotOwnerBuffer = nullptr; // binding 16
  vk::raii::DeviceMemory slotOwnerBufferMemory = nullptr;
  vk::raii::Buffer relocationScratchBuffer = nullptr; // binding 17
  vk::raii::DeviceMemory relocationScratchBufferMemory = nullptr;
  // Per frame: ClipmapResult::neighborLods for visibleChunkSet (compute binding 8).
  std::vector<vk::raii::Buffer> chunkNeighborLodBuffers;
  std::vector<vk::raii::DeviceMemory> chunkNeighborLodBufferMemories;
  std::vector<void *> chunkNeighborLodBuffersMapped;
#pragma endregion

#pragma region Descriptor State
  // Members are destroyed in reverse declaration order: every set from descriptorPool must be declared after it.
  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  std::vector<vk::raii::DescriptorSet> cpuDescriptorSets;
  // As cpuDescriptorSets / descriptorSets, with the translucent (ocean) clusters at binding 6.
  std::vector<vk::raii::DescriptorSet> cpuTranslucentDescriptorSets;
  std::vector<vk::raii::DescriptorSet> gpuTranslucentDescriptorSets;
  std::vector<vk::raii::DescriptorSet> taaDescriptorSets;
  std::vector<vk::raii::DescriptorSet> chunkComputeDescriptorSets;
#pragma endregion

#pragma region Command State
  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;
#pragma endregion

#pragma region Sync State
  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
  std::vector<vk::raii::Fence> inFlightFences;
  uint32_t frameIndex = 0;
  uint32_t taaHistoryWriteIndex = 0;
  uint32_t taaJitterIndex = 0;
  bool taaHistoryValid = false;
  bool taaHistoryInitialized = false;
  bool previousMvpValid = false;
  glm::mat4 previousMvp{1.0f};
#pragma endregion

#pragma region Input State
  bool framebufferResized = false;
  bool fpsLimitEnabled = true;
  // Debug overlay (top right): frames counted over a short window, shown as FPS.
  std::chrono::steady_clock::time_point fpsWindowStart = std::chrono::steady_clock::now();
  uint32_t fpsWindowFrames = 0;
  float displayedFps = 0.0f;

  // GPU pass timings (P): a timestamp after each stage of the frame, averaged and printed once a second.
  enum GpuTimestamp : uint32_t {
    TIMESTAMP_FRAME_START,
    TIMESTAMP_UPLOADS,      // CPU chunk uploads
    TIMESTAMP_COMPUTE,      // GPU chunk generation and meshing (only when something remeshes)
    TIMESTAMP_OPAQUE,       // frame-start barriers, clears, opaque terrain
    TIMESTAMP_CUTOUT,       // non-opaque terrain, then the MSAA resolves at the end of the pass
    TIMESTAMP_WATER,        // water, block outline, resolves
    TIMESTAMP_HZB,          // Hi-Z pyramid build
    TIMESTAMP_TAA,          // TAA resolve to the swapchain image
    TIMESTAMP_OVERLAY,      // debug text and crosshair
    GPU_TIMESTAMP_COUNT
  };
  static constexpr std::array<const char *, GPU_TIMESTAMP_COUNT> GPU_TIMESTAMP_NAMES{
      "", "uploads", "compute", "opaque", "cutout", "water", "hi-z", "taa", "overlay"};
  bool gpuTimingEnabled = false;
  vk::raii::QueryPool timestampQueryPool = nullptr;
  uint64_t timestampMask = 0;    // valid bits of a timestamp (0: the queue has no timestamps)
  double timestampPeriodNs = 0.0; // nanoseconds per timestamp tick
  std::array<bool, MAX_FRAMES_IN_FLIGHT> timestampsPending{};
  std::array<double, GPU_TIMESTAMP_COUNT> gpuTimeSumsMs{}; // [0]: whole frame
  uint32_t gpuTimeSamples = 0;
  std::chrono::steady_clock::time_point gpuTimeWindowStart{};
  Camera camera;
  // Block breaking (left mouse) and placing (right mouse) by Minecraft's rules, in 20 Hz interaction ticks:
  // creative breaks on click, then every BLOCK_DESTROY_DELAY_TICKS while held; survival holds the button on one block
  // until its destroy progress reaches 1 (see destroyProgressPerTick), then waits the same delay. Placing repeats every
  // BLOCK_USE_DELAY_TICKS. Adventure and spectator do neither.
  static constexpr float BLOCK_REACH = 5.0f;
  static constexpr double INTERACTION_TICK_SECONDS = 1.0 / 20.0;
  static constexpr int BLOCK_DESTROY_DELAY_TICKS = 5;
  static constexpr int BLOCK_USE_DELAY_TICKS = 4;
  bool breakClickPending = false;
  bool placeClickPending = false;
  double interactionTickAccumulator = 0.0;
  int destroyDelayTicks = 0;
  int useDelayTicks = 0;
  std::optional<cpuworld::BlockXYZ> destroyTarget; // survival: block being broken
  float destroyProgress = 0.0f;                    // 0..1 on destroyTarget
  // Hotbar slot 0-8 (keys 1-9 or the scroll wheel); slot i holds blockRegistry.blocks[i], if there is one.
  static constexpr size_t HOTBAR_SLOTS = 9;
  size_t selectedBlockIndex = 0;
  // F3+F4 game mode switcher: opens on F4 while F3 is held, F4 again cycles, releasing F3 applies the selection.
  bool f3Held = false;
  bool gameModeSwitcherOpen = false;
  Player::GameMode gameModeSelection = Player::GameMode::Creative;
  uint16_t oceanBlockId = cpuworld::kAir;
  bool cameraUnderwater = false;                   // set by updateUniformBuffer
  std::optional<cpuworld::BlockXYZ> outlineTarget; // block under the crosshair, outlined by recordOutline
#pragma endregion

#pragma region Device Extensions
  std::vector<const char *> requiredDeviceExtension = {vk::KHRSwapchainExtensionName, vk::KHRSpirv14ExtensionName,
                                                       vk::KHRSynchronization2ExtensionName,
                                                       vk::EXTMeshShaderExtensionName};
#pragma endregion
#pragma endregion

#pragma region Application Lifecycle
#pragma region Window Setup
  /** Create the GLFW window (no client API) and install the input callbacks. */
  void initWindow() {
    if (glfwInit() != GLFW_TRUE) {
      throw std::runtime_error("failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    if (window == nullptr) {
      glfwTerminate();
      throw std::runtime_error("failed to create GLFW window");
    }
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    camera.initialize(window);
  }

  /** GLFW resize callback: flags the swapchain for recreation. */
  static void framebufferResizeCallback(GLFWwindow *window, int /*width*/, int /*height*/) {
    auto app = reinterpret_cast<VulkanEngine *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  /** Display name of a terrain view mode, for the log. */
  static const char *terrainViewName(TerrainView view) {
    switch (view) {
    case TerrainView::LodOnly:
      return "LOD ONLY";
    case TerrainView::RealOnly:
      return "REAL ONLY";
    case TerrainView::Debug:
      return "DEBUG";
    default:
      return "LOD + REAL";
    }
  }

  /** GLFW mouse callback: queues a block break (left) or place (right) for the next frame. */
  static void mouseButtonCallback(GLFWwindow *window, int button, int action, int /*mods*/) {
    auto app = reinterpret_cast<VulkanEngine *>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
      app->breakClickPending = true; // a click shorter than a frame still breaks
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
      app->placeClickPending = true;
    }
  }

  /** Display name of a game mode, as the F3+F4 switcher titles it. */
  static const char *gameModeName(Player::GameMode mode) {
    switch (mode) {
    case Player::GameMode::Creative:
      return "Creative Mode";
    case Player::GameMode::Adventure:
      return "Adventure Mode";
    case Player::GameMode::Spectator:
      return "Spectator Mode";
    default:
      return "Survival Mode";
    }
  }

  /**
   * GLFW scroll callback: cycles the hotbar like Minecraft (down = next slot, wrapping); in spectator it changes the
   * fly speed instead.
   */
  static void scrollCallback(GLFWwindow *window, double /*xOffset*/, double yOffset) {
    auto app = reinterpret_cast<VulkanEngine *>(glfwGetWindowUserPointer(window));
    const int steps = yOffset > 0.0 ? 1 : (yOffset < 0.0 ? -1 : 0);
    if (steps == 0) {
      return;
    }
    if (app->player.gameMode() == Player::GameMode::Spectator) {
      app->player.scrollSpectatorSpeed(steps);
      return;
    }
    const auto slots = static_cast<int>(HOTBAR_SLOTS);
    app->selectedBlockIndex =
        static_cast<size_t>(((static_cast<int>(app->selectedBlockIndex) - steps) % slots + slots) % slots);
  }

  /**
   * F3 combos: F3+F4 opens the game mode switcher (or moves to the next mode while open), and releasing F3 applies
   * the switcher's selection.
   *
   * @return true if the key was consumed by a combo.
   */
  bool handleDebugCombo(int key, int action) {
    if (key == GLFW_KEY_F3) {
      if (action == GLFW_PRESS) {
        f3Held = true;
      } else if (action == GLFW_RELEASE) {
        f3Held = false;
        if (gameModeSwitcherOpen) {
          gameModeSwitcherOpen = false;
          player.setGameMode(gameModeSelection);
          std::clog << "Game mode: " << gameModeName(gameModeSelection) << '\n';
        }
      }
      return true;
    }
    if (!f3Held || action != GLFW_PRESS) {
      return false;
    }
    if (key == GLFW_KEY_F4) {
      if (!gameModeSwitcherOpen) {
        // Minecraft starts on the previous game mode (creative <-> survival on a fresh world).
        gameModeSwitcherOpen = true;
        gameModeSelection = player.previousGameMode() != player.gameMode() ? player.previousGameMode()
                            : player.gameMode() == Player::GameMode::Creative ? Player::GameMode::Survival
                                                                              : Player::GameMode::Creative;
      } else {
        gameModeSelection =
            static_cast<Player::GameMode>((static_cast<int>(gameModeSelection) + 1) % Player::kGameModeCount);
      }
      return true;
    }
    return false;
  }

  /** GLFW key callback: F3 combos, the camera and player, then hotbar slots (1-9) and debug toggles. */
  static void keyCallback(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/) {
    auto app = reinterpret_cast<VulkanEngine *>(glfwGetWindowUserPointer(window));
    if (app->handleDebugCombo(key, action)) {
      return;
    }
    app->camera.handleKey(window, key, action);
    app->player.handleKey(key, action);
    if (action != GLFW_PRESS) {
      return;
    }

    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
      app->selectedBlockIndex = static_cast<size_t>(key - GLFW_KEY_1);
    }
    if (key == GLFW_KEY_F) {
      app->fpsLimitEnabled = !app->fpsLimitEnabled;
      std::clog << "FPS limit " << (app->fpsLimitEnabled ? "enabled" : "disabled") << '\n';
    }
    if (key == GLFW_KEY_R) {
      app->hzbOcclusionEnabled = !app->hzbOcclusionEnabled;
      std::clog << "Hi-Z occlusion culling " << (app->hzbOcclusionEnabled ? "enabled" : "disabled") << '\n';
    }
    if (key == GLFW_KEY_T) {
      app->wireframeEnabled = !app->wireframeEnabled;
      std::clog << "Wireframe " << (app->wireframeEnabled ? "enabled" : "disabled") << '\n';
    }
    if (key == GLFW_KEY_H) {
      app->contourEnabled = !app->contourEnabled;
      std::clog << "Elevation view " << (app->contourEnabled ? "enabled" : "disabled") << '\n';
    }
    if (key == GLFW_KEY_X) {
      app->terrainView = static_cast<TerrainView>((static_cast<int>(app->terrainView) + 1) % 4);
      std::clog << "Terrain view: " << terrainViewName(app->terrainView) << '\n';
    }
    if (key == GLFW_KEY_V) {
      app->useVertexPulling = !app->useVertexPulling;
      std::clog << "Quad path: " << (app->useVertexPulling ? "vertex shader (vertex pulling)" : "mesh shader") << '\n';
    }
    if (key == GLFW_KEY_P) {
      if (app->timestampMask == 0u) {
        std::clog << "GPU timings unavailable: this queue has no timestamps\n";
      } else {
        app->gpuTimingEnabled = !app->gpuTimingEnabled;
        app->gpuTimeSumsMs = {};
        app->gpuTimeSamples = 0;
        app->gpuTimeWindowStart = std::chrono::steady_clock::now();
        std::clog << "GPU timings " << (app->gpuTimingEnabled ? "enabled (printed every second)" : "disabled")
                  << '\n';
      }
    }
    if (key == GLFW_KEY_G) {
      app->lodFreezeRequested = !app->lodFreezeRequested;
      std::clog << "LOD / chunk updates " << (app->lodFreezeRequested ? "frozen" : "unfrozen") << '\n';
    }
  }
#pragma endregion

#pragma region Renderer Initialization
  /** Create every Vulkan object the renderer needs, in dependency order. */
  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createPipelineCache();
    createSwapChain();
    createImageViews();
    createDescriptorSetLayout();
    createTaaDescriptorSetLayout();
    createGraphicsPipeline();
    createTaaPipeline();
    createOverlayPipeline();
    createChunkComputePipeline();
    createCommandPool();
    createDepthResources();
    createTextureImages();
    createTextureImageViews();
    createTextureSampler();
    createTaaSampler();
    createBlockFaceBuffer();
    createUniformBuffers();
    createChunkBuffers();
    createCpuChunkBuffers();
    createCpuWorld();
    createDescriptorPool();
    createDescriptorSets();
    createTaaDescriptorSets();
    createHzbResources();
    createCommandBuffers();
    createSyncObjects();
    createTimestampQueryPool();
    applyDebugEnvironment();
    savePipelineCache(); // every pipeline exists now; saved again on exit
  }
#pragma endregion

#pragma region Main Loop
  /** Poll events and draw frames until the window closes, capping at 144 FPS while the limit is on. */
  void mainLoop() {
    using Clock = std::chrono::steady_clock;
    constexpr auto frameDuration =
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / 144.0));
    auto nextFrame = Clock::now();

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
      ++fpsWindowFrames;
      const auto fpsNow = std::chrono::steady_clock::now();
      const float fpsWindowSeconds = std::chrono::duration<float>(fpsNow - fpsWindowStart).count();
      if (fpsWindowSeconds >= 0.5f) {
        displayedFps = static_cast<float>(fpsWindowFrames) / fpsWindowSeconds;
        fpsWindowFrames = 0;
        fpsWindowStart = fpsNow;
      }
      if (fpsLimitEnabled) {
        nextFrame += frameDuration;
        std::this_thread::sleep_until(nextFrame);

        const auto now = Clock::now();
        if (now > nextFrame + frameDuration) {
          nextFrame = now;
        }
      } else {
        nextFrame = Clock::now();
      }
    }

    device.waitIdle();
  }
#pragma endregion

#pragma region Cleanup
  /** Destroy the pipelines that depend on the swapchain (its surface format only: viewport and scissor are dynamic). */
  void destroySwapChainPipelines() {
    graphicsPipeline = nullptr;
    vertexPullingPipeline = nullptr;
    cutoutPipeline = nullptr;
    cutoutVertexPullingPipeline = nullptr;
    wireframePipeline = nullptr;
    wireframeVertexPullingPipeline = nullptr;
    waterPipeline = nullptr;
    waterVertexPullingPipeline = nullptr;
    taaPipeline = nullptr;
    overlayPipeline = nullptr;
    crosshairPipeline = nullptr;
    healthPipeline = nullptr;
    gameModeSwitcherPipeline = nullptr;
    outlinePipeline = nullptr;
    crackPipeline = nullptr;
  }

  /** Destroy the swapchain and every swapchain-sized resource, and invalidate the history that depended on them. */
  void cleanupSwapChain() {
    taaDescriptorSets.clear();
    for (auto &view : taaHistoryImageViews) {
      view = nullptr;
    }
    for (auto &image : taaHistoryImages) {
      image = nullptr;
    }
    for (auto &memory : taaHistoryImageMemories) {
      memory = nullptr;
    }
    sceneColorImageView = nullptr;
    sceneColorImage = nullptr;
    sceneColorImageMemory = nullptr;
    multisampleMotionImageView = nullptr;
    multisampleMotionImage = nullptr;
    multisampleMotionImageMemory = nullptr;
    motionImageView = nullptr;
    motionImage = nullptr;
    motionImageMemory = nullptr;
    multisampleColorImageView = nullptr;
    multisampleColorImage = nullptr;
    multisampleColorImageMemory = nullptr;
    hzbSet = nullptr;
    hzbDescriptorPool = nullptr;
    hzbMipViews.clear();
    hzbView = nullptr;
    hzbImage = nullptr;
    hzbImageMemory = nullptr;
    hzbImageInitialized = false;
    hzbValid = false;
    depthImageView = nullptr;
    resolvedDepthImageView = nullptr;
    resolvedDepthImage = nullptr;
    resolvedDepthImageMemory = nullptr;
    depthImage = nullptr;
    depthImageMemory = nullptr;
    swapChainImageViews.clear();
    swapChain = nullptr;
    taaHistoryValid = false;
    taaHistoryInitialized = false;
    taaHistoryWriteIndex = 0;
    previousMvpValid = false;
  }

  /** Save the pipeline cache and destroy what RAII does not (swapchain resources, surface, window). */
  void cleanup() {
    savePipelineCache();
    destroySwapChainPipelines();
    cleanupSwapChain();
    surface = nullptr;
    glfwDestroyWindow(window);
    glfwTerminate();
  }
#pragma endregion

#pragma region Swapchain Recreation
  /** Rebuild the swapchain and its sized resources after a resize; waits while the window is minimized. */
  void recreateSwapChain() {
    int width = 0, height = 0;
    while ((width == 0 || height == 0) && !glfwWindowShouldClose(window)) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }
    if (glfwWindowShouldClose(window)) {
      return;
    }

    device.waitIdle();

    const vk::Format previousFormat = swapChainSurfaceFormat.format;
    renderFinishedSemaphores.clear();
    cleanupSwapChain();
    createSwapChain();
    createRenderFinishedSemaphores();
    createImageViews();
    // A resize keeps the surface format, and nothing else about the swapchain is baked into the pipelines.
    if (swapChainSurfaceFormat.format != previousFormat) {
      destroySwapChainPipelines();
      createGraphicsPipeline();
      createTaaPipeline();
      createOverlayPipeline();
    }
    createDepthResources();
    createTaaDescriptorSets();
    createHzbResources();
  }
#pragma endregion
#pragma endregion

#pragma region Vulkan Setup
#pragma region Surface Setup
  /** Create the Vulkan instance, checking that the required layers and extensions are available. */
  void createInstance() {
    constexpr vk::ApplicationInfo appInfo{.pApplicationName = "engine",
                                          .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                          .pEngineName = "engine",
                                          .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                          .apiVersion = vk::ApiVersion14};

    std::vector<char const *> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layerProperties](auto const &requiredLayer) {
      return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty) {
        return strcmp(layerProperty.layerName, requiredLayer) == 0;
      });
    });
    if (unsupportedLayerIt != requiredLayers.end()) {
      throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
    }

    auto requiredExtensions = getRequiredInstanceExtensions();

    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    auto unsupportedPropertyIt =
        std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const &requiredExtension) {
          return std::ranges::none_of(extensionProperties, [requiredExtension](auto const &extensionProperty) {
            return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
          });
        });
    if (unsupportedPropertyIt != requiredExtensions.end()) {
      throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
    }

    vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo,
                                      .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
                                      .ppEnabledLayerNames = requiredLayers.data(),
                                      .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
                                      .ppEnabledExtensionNames = requiredExtensions.data()};
    instance = vk::raii::Instance(context, createInfo);
  }

  /** Install the validation message callback (validation builds only). */
  void setupDebugMessenger() {
    if (!enableValidationLayers)
      return;

    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
        .messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback};
    debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
  }

  /** Create the window surface through GLFW. */
  void createSurface() {
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, rawSurface);
  }
#pragma endregion

#pragma region Device Selection
  /**
   * Whether a GPU has everything the renderer needs.
   *
   * @return true if `candidateDevice` supports Vulkan 1.3, a graphics + compute + present queue, the required
   * extensions, features, MSAA sample count and render target formats.
   */
  bool isDeviceSuitable(vk::raii::PhysicalDevice const &candidateDevice) {
    bool supportsVulkan1_3 = candidateDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

    auto queueFamilies = candidateDevice.getQueueFamilyProperties();
    uint32_t familyIndex = 0;
    bool supportsGraphicsAndPresent = std::ranges::any_of(
        queueFamilies, [&candidateDevice, &surface = this->surface, &familyIndex](auto const &family) {
          bool const suitable = (family.queueFlags & vk::QueueFlagBits::eGraphics) &&
                                (family.queueFlags & vk::QueueFlagBits::eCompute) &&
                                candidateDevice.getSurfaceSupportKHR(familyIndex, *surface);
          familyIndex++;
          return suitable;
        });

    auto availableDeviceExtensions = candidateDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions =
        std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
          return std::ranges::any_of(
              availableDeviceExtensions, [requiredDeviceExtension](auto const &availableDeviceExtension) {
                return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
              });
        });

    auto features =
        candidateDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                                             vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                                             vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                                             vk::PhysicalDeviceMeshShaderFeaturesEXT>();
    bool supportsRequiredFeatures =
        features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
        features.template get<vk::PhysicalDeviceFeatures2>().features.shaderInt64 &&
        features.template get<vk::PhysicalDeviceFeatures2>().features.shaderStorageImageArrayDynamicIndexing &&
        features.template get<vk::PhysicalDeviceFeatures2>().features.fillModeNonSolid &&
        features.template get<vk::PhysicalDeviceFeatures2>().features.independentBlend &&
        features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
        features.template get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray &&
        features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingVariableDescriptorCount &&
        features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingSampledImageUpdateAfterBind &&
        features.template get<vk::PhysicalDeviceVulkan12Features>().shaderSampledImageArrayNonUniformIndexing &&
        features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
        features.template get<vk::PhysicalDeviceMeshShaderFeaturesEXT>().meshShader;
    const auto sampleCount = static_cast<vk::SampleCountFlags>(RENDER_SAMPLE_COUNT);
    const auto &limits = candidateDevice.getProperties().limits;
    const bool supportsRenderSampleCount =
        (limits.framebufferColorSampleCounts & sampleCount) && (limits.framebufferDepthSampleCounts & sampleCount);
    const auto motionFeatures = candidateDevice.getFormatProperties(MOTION_FORMAT).optimalTilingFeatures;
    const auto requiredMotionFeatures =
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage;
    const bool supportsMotionFormat = (motionFeatures & requiredMotionFeatures) == requiredMotionFeatures;
    const auto historyFeatures = candidateDevice.getFormatProperties(TAA_HISTORY_FORMAT).optimalTilingFeatures;
    const auto requiredHistoryFeatures =
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage;
    const bool supportsHistoryFormat = (historyFeatures & requiredHistoryFeatures) == requiredHistoryFeatures;

    return supportsVulkan1_3 && supportsGraphicsAndPresent && supportsAllRequiredExtensions &&
           supportsRequiredFeatures && supportsRenderSampleCount && supportsMotionFormat && supportsHistoryFormat;
  }

  /** Select the first suitable GPU (isDeviceSuitable). */
  void pickPhysicalDevice() {
    std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    auto const suitableDevice = std::ranges::find_if(
        physicalDevices, [&](auto const &candidateDevice) { return isDeviceSuitable(candidateDevice); });
    if (suitableDevice == physicalDevices.end()) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
    physicalDevice = *suitableDevice;
  }
#pragma endregion

#pragma region Logical Device
  /** Create the device with the required features, on the first graphics + compute + present queue family. */
  void createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    for (uint32_t familyIndex = 0; familyIndex < queueFamilyProperties.size(); familyIndex++) {
      if ((queueFamilyProperties[familyIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
          (queueFamilyProperties[familyIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
          physicalDevice.getSurfaceSupportKHR(familyIndex, *surface)) {
        queueIndex = familyIndex;
        break;
      }
    }
    if (queueIndex == ~0u) {
      throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
    }

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT, vk::PhysicalDeviceMeshShaderFeaturesEXT>
        featureChain = {
            {.features = {.independentBlend = true,
                          .fillModeNonSolid = true, // wireframe (T)
                          .samplerAnisotropy = true,
                          .shaderStorageImageArrayDynamicIndexing = true, // Hi-Z mip array (hzb.slang)
                          .shaderInt64 = true}},
            {.shaderDrawParameters = true}, // vk::PhysicalDeviceVulkan11Features
            {.shaderSampledImageArrayNonUniformIndexing = true,
             .descriptorBindingSampledImageUpdateAfterBind = true,
             .descriptorBindingVariableDescriptorCount = true,
             .runtimeDescriptorArray = true},                     // vk::PhysicalDeviceVulkan12Features
            {.synchronization2 = true, .dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
            {.extendedDynamicState = true},                       // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
            {.taskShader = true, .meshShader = true}              // vk::PhysicalDeviceMeshShaderFeaturesEXT
    };

    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
        .queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
    vk::DeviceCreateInfo deviceCreateInfo{.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                                          .queueCreateInfoCount = 1,
                                          .pQueueCreateInfos = &deviceQueueCreateInfo,
                                          .enabledExtensionCount =
                                              static_cast<uint32_t>(requiredDeviceExtension.size()),
                                          .ppEnabledExtensionNames = requiredDeviceExtension.data()};

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    queue = vk::raii::Queue(device, queueIndex, 0);
  }
#pragma endregion
#pragma endregion

#pragma region Render Setup
#pragma region Swapchain Setup
  /** Create the swapchain for the current surface size, format and present mode. */
  void createSwapChain() {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

    std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(availablePresentModes);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface = *surface,
                                                   .minImageCount = minImageCount,
                                                   .imageFormat = swapChainSurfaceFormat.format,
                                                   .imageColorSpace = swapChainSurfaceFormat.colorSpace,
                                                   .imageExtent = swapChainExtent,
                                                   .imageArrayLayers = 1,
                                                   .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                                                   .imageSharingMode = vk::SharingMode::eExclusive,
                                                   .preTransform = surfaceCapabilities.currentTransform,
                                                   .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                   .presentMode = presentMode,
                                                   .clipped = true};

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
  }

  /** Create a color view per swapchain image. */
  void createImageViews() {
    assert(swapChainImageViews.empty());

    swapChainImageViews.reserve(swapChainImages.size());
    for (auto &image : swapChainImages) {
      swapChainImageViews.emplace_back(
          createImageView(image, swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));
    }
  }
#pragma endregion

#pragma region Descriptor Layout
  /** Create the scene pipelines' set layout; the bindless texture array is sized to the device limit. */
  void createDescriptorSetLayout() {
    auto indexingProperties =
        physicalDevice
            .template getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>();
    const auto &indexingLimits = indexingProperties.template get<vk::PhysicalDeviceDescriptorIndexingProperties>();
    textureDescriptorCapacity =
        std::min({MAX_TEXTURE_COUNT, indexingLimits.maxPerStageDescriptorUpdateAfterBindSampledImages,
                  indexingLimits.maxDescriptorSetUpdateAfterBindSampledImages});
    if (textureDescriptorCapacity == 0u) {
      throw std::runtime_error("device does not support sampled image descriptors");
    }

    const vk::ShaderStageFlags geometryStages =
        vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eVertex;
    std::array<vk::DescriptorSetLayoutBinding, 10> bindings{
        {{.binding = 0,
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = geometryStages},
         {.binding = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .descriptorCount = 1,
          .stageFlags = geometryStages | vk::ShaderStageFlagBits::eFragment},
         {.binding = 2,
          .descriptorType = vk::DescriptorType::eSampler,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eFragment},
         {.binding = 3,
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = geometryStages},
         {.binding = 4, // QuadCount, vertex path
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eVertex},
         {.binding = 11,
          .descriptorType = vk::DescriptorType::eSampledImage,
          .descriptorCount = textureDescriptorCapacity,
          .stageFlags = vk::ShaderStageFlagBits::eFragment},
         {.binding = 5, // BlockFaces: fragMain, and vertMain (liquid quads per pass)
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex},
         {.binding = 6, // QuadClusters, task shader
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eTaskEXT},
         {.binding = 7, // Hi-Z, task shader
          .descriptorType = vk::DescriptorType::eSampledImage,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eTaskEXT},
         // Resolved opaque depth for the water fog: fragWater only, written in the translucent sets only.
         {.binding = 8,
          .descriptorType = vk::DescriptorType::eSampledImage,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eFragment}}};

    std::array<vk::DescriptorBindingFlags, 10> bindingFlags{};
    bindingFlags[5] = // binding 11, the bindless texture array
        vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .bindingCount = static_cast<uint32_t>(bindingFlags.size()), .pBindingFlags = bindingFlags.data()};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()),
                                                 .pBindings = bindings.data()};
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
  }

  /** Create the TAA set layout: scene color, history, motion and a sampler. */
  void createTaaDescriptorSetLayout() {
    const std::array<vk::DescriptorSetLayoutBinding, 4> bindings{{{.binding = 0,
                                                                   .descriptorType = vk::DescriptorType::eSampledImage,
                                                                   .descriptorCount = 1,
                                                                   .stageFlags = vk::ShaderStageFlagBits::eFragment},
                                                                  {.binding = 1,
                                                                   .descriptorType = vk::DescriptorType::eSampledImage,
                                                                   .descriptorCount = 1,
                                                                   .stageFlags = vk::ShaderStageFlagBits::eFragment},
                                                                  {.binding = 2,
                                                                   .descriptorType = vk::DescriptorType::eSampledImage,
                                                                   .descriptorCount = 1,
                                                                   .stageFlags = vk::ShaderStageFlagBits::eFragment},
                                                                  {.binding = 3,
                                                                   .descriptorType = vk::DescriptorType::eSampler,
                                                                   .descriptorCount = 1,
                                                                   .stageFlags = vk::ShaderStageFlagBits::eFragment}}};
    taaDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo{.bindingCount = static_cast<uint32_t>(bindings.size()),
                                                  .pBindings = bindings.data()});
  }

#pragma endregion

#pragma region Pipeline Setup
  /**
   * Create the scene pipeline layout and every terrain pipeline: mesh-shader and vertex-pulling variants of the
   * opaque, cutout, wireframe and water passes.
   */
  void createGraphicsPipeline() {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

    // Separate module on purpose (driver workaround): with task and mesh in one SPIR-V module (two
    // TaskPayloadWorkgroupEXT variables), the NVIDIA driver handed meshMain an empty payload.
    vk::raii::ShaderModule taskShaderModule = createShaderModule(readFile("shaders/task.spv"));
    vk::PipelineShaderStageCreateInfo taskShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eTaskEXT, .module = taskShaderModule, .pName = "taskMain"};
    vk::PipelineShaderStageCreateInfo meshShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eMeshEXT, .module = shaderModule, .pName = "meshMain"};
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
    vk::PipelineShaderStageCreateInfo shaderStages[] = {taskShaderStageInfo, meshShaderStageInfo, fragShaderStageInfo};
    vk::PipelineShaderStageCreateInfo vertexShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
    vk::PipelineShaderStageCreateInfo vertexPullingStages[] = {vertexShaderStageInfo, fragShaderStageInfo};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

    vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable = vk::False,
                                                        .rasterizerDiscardEnable = vk::False,
                                                        .polygonMode = vk::PolygonMode::eFill,
                                                        .cullMode = vk::CullModeFlagBits::eBack,
                                                        .frontFace = vk::FrontFace::eCounterClockwise,
                                                        .depthBiasEnable = vk::False,
                                                        .lineWidth = 1.0f};

    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = RENDER_SAMPLE_COUNT,
                                                         .sampleShadingEnable = vk::False};

    vk::PipelineDepthStencilStateCreateInfo depthStencil{.depthTestEnable = vk::True,
                                                         .depthWriteEnable = vk::True,
                                                         .depthCompareOp = vk::CompareOp::eGreater, // reverse-Z
                                                         .depthBoundsTestEnable = vk::False,
                                                         .stencilTestEnable = vk::False};
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const std::array<vk::PipelineColorBlendAttachmentState, 2> colorBlendAttachments{colorBlendAttachment,
                                                                                     colorBlendAttachment};
    vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False,
                                                        .logicOp = vk::LogicOp::eCopy,
                                                        .attachmentCount =
                                                            static_cast<uint32_t>(colorBlendAttachments.size()),
                                                        .pAttachments = colorBlendAttachments.data()};

    // Cull mode is dynamic so the LOD freeze (G) can turn off hardware backface culling and leave it to the frozen
    // view's shader-side culling.
    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor,
                                                   vk::DynamicState::eCullMode};
    vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                                                    .pDynamicStates = dynamicStates.data()};

    const vk::PushConstantRange geometryPushRange{
        .stageFlags = GEOMETRY_PUSH_STAGES, .offset = 0, .size = sizeof(GeometryPushConstants)};
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 1,
                                                    .pSetLayouts = &*descriptorSetLayout,
                                                    .pushConstantRangeCount = 1,
                                                    .pPushConstantRanges = &geometryPushRange};
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::Format depthFormat = findDepthFormat();

    const std::array<vk::Format, 2> colorFormats{swapChainSurfaceFormat.format, MOTION_FORMAT};
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = 3,
         .pStages = shaderStages,
         .pVertexInputState = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState = &multisampling,
         .pDepthStencilState = &depthStencil,
         .pColorBlendState = &colorBlending,
         .pDynamicState = &dynamicState,
         .layout = pipelineLayout,
         .renderPass = nullptr},
        {.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
         .pColorAttachmentFormats = colorFormats.data(),
         .depthAttachmentFormat = depthFormat}};

    graphicsPipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    // Vertex input stays empty: the vertex shader pulls from QuadBuffer itself.
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 2;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = vertexPullingStages;
    vertexPullingPipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    // Cutout: separate from the opaque pipelines so theirs has no discard and keeps early depth testing.
    shaderStages[2].pName = "fragCutout";
    vertexPullingStages[1].pName = "fragCutout";
    cutoutVertexPullingPipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 3;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = shaderStages;
    cutoutPipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    shaderStages[2].pName = "fragMain";
    vertexPullingStages[1].pName = "fragMain";
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 2;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = vertexPullingStages;

    rasterizer.polygonMode = vk::PolygonMode::eLine;
    wireframeVertexPullingPipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 3;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = shaderStages;
    wireframePipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    // Water: blended over the terrain, keeping its depth and motion.
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    depthStencil.depthWriteEnable = vk::False;
    const vk::PipelineColorBlendAttachmentState waterColorBlend{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const vk::PipelineColorBlendAttachmentState waterMotionBlend{.blendEnable = vk::False, .colorWriteMask = {}};
    const std::array<vk::PipelineColorBlendAttachmentState, 2> waterBlendAttachments{waterColorBlend, waterMotionBlend};
    colorBlending.pAttachments = waterBlendAttachments.data();
    shaderStages[2].pName = "fragWater";
    vertexPullingStages[1].pName = "fragWater";
    waterPipeline = vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 2;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = vertexPullingStages;
    waterVertexPullingPipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  /** Create the TAA resolve pipeline, writing the swapchain image and the next history image. */
  void createTaaPipeline() {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/taa.spv"));
    const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{
        {{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "taaVertexMain"},
         {.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "taaFragmentMain"}}};
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{.polygonMode = vk::PolygonMode::eFill,
                                                        .cullMode = vk::CullModeFlagBits::eNone,
                                                        .frontFace = vk::FrontFace::eCounterClockwise,
                                                        .lineWidth = 1.0f};
    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const std::array<vk::PipelineColorBlendAttachmentState, 2> blendAttachments{colorBlendAttachment,
                                                                                colorBlendAttachment};
    vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOp = vk::LogicOp::eCopy,
                                                        .attachmentCount =
                                                            static_cast<uint32_t>(blendAttachments.size()),
                                                        .pAttachments = blendAttachments.data()};
    const std::array<vk::DynamicState, 2> dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                                                    .pDynamicStates = dynamicStates.data()};
    const vk::DescriptorSetLayout taaSetLayout = taaDescriptorSetLayout;
    const vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = sizeof(TaaParams)};
    taaPipelineLayout =
        vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1,
                                                                      .pSetLayouts = &taaSetLayout,
                                                                      .pushConstantRangeCount = 1,
                                                                      .pPushConstantRanges = &pushConstantRange});

    const std::array<vk::Format, 2> colorFormats{swapChainSurfaceFormat.format, TAA_HISTORY_FORMAT};
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = static_cast<uint32_t>(shaderStages.size()),
         .pStages = shaderStages.data(),
         .pVertexInputState = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState = &multisampling,
         .pColorBlendState = &colorBlending,
         .pDynamicState = &dynamicState,
         .layout = taaPipelineLayout,
         .renderPass = nullptr},
        {.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
         .pColorAttachmentFormats = colorFormats.data()}};
    taaPipeline = vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  /**
   * Create the screen-space overlay pipelines (text, HUD, health: alpha blend; crosshair: inverting blend,
   * 1 - destination), all on OverlayParams push constants, and the block outline pipeline.
   */
  void createOverlayPipeline() {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/overlay.spv"));
    const vk::PushConstantRange pushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex |
                                                                vk::ShaderStageFlagBits::eFragment,
                                                  .offset = 0,
                                                  .size = sizeof(OverlayParams)};
    const vk::DescriptorSetLayout setLayout = descriptorSetLayout;
    overlayPipelineLayout =
        vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1,
                                                                      .pSetLayouts = &setLayout,
                                                                      .pushConstantRangeCount = 1,
                                                                      .pPushConstantRanges = &pushConstantRange});
    overlayPipeline = createOverlayStylePipeline(shaderModule, "overlayVertexMain", "overlayFragmentMain",
                                                 vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha);
    hudPipeline = createOverlayStylePipeline(shaderModule, "hudVertexMain", "hudFragmentMain",
                                             vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha);
    crosshairPipeline = createOverlayStylePipeline(shaderModule, "crosshairVertexMain", "crosshairFragmentMain",
                                                   vk::BlendFactor::eOneMinusDstColor, vk::BlendFactor::eZero);
    healthPipeline = createOverlayStylePipeline(shaderModule, "healthVertexMain", "healthFragmentMain",
                                                vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha);
    gameModeSwitcherPipeline =
        createOverlayStylePipeline(shaderModule, "switcherVertexMain", "switcherFragmentMain",
                                   vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha);
    createOutlinePipeline();
  }

  /**
   * Create the block outline pipeline: a scene-pass pipeline (MSAA, color + motion, depth test without writes) on the
   * scene uniform buffer, so the outline is jittered and reprojected like the terrain.
   */
  void createOutlinePipeline() {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/outline.spv"));
    const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{
        {{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "outlineVertexMain"},
         {.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "outlineFragmentMain"}}};
    const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .lineWidth = 1.0f};
    const vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = RENDER_SAMPLE_COUNT};
    const vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True, .depthWriteEnable = vk::False, .depthCompareOp = vk::CompareOp::eGreaterOrEqual};
    const vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    // Motion is overwritten: the outline lies on the surface it covers, so their motion matches anyway.
    const vk::PipelineColorBlendAttachmentState motionAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const std::array<vk::PipelineColorBlendAttachmentState, 2> blendAttachments{blendAttachment, motionAttachment};
    const vk::PipelineColorBlendStateCreateInfo colorBlending{
        .attachmentCount = static_cast<uint32_t>(blendAttachments.size()), .pAttachments = blendAttachments.data()};
    const std::array<vk::DynamicState, 2> dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};
    const vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(OutlineParams)};
    outlinePipelineLayout =
        vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1,
                                                                      .pSetLayouts = &*descriptorSetLayout,
                                                                      .pushConstantRangeCount = 1,
                                                                      .pPushConstantRanges = &pushConstantRange});

    const std::array<vk::Format, 2> colorFormats{swapChainSurfaceFormat.format, MOTION_FORMAT};
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = static_cast<uint32_t>(shaderStages.size()),
         .pStages = shaderStages.data(),
         .pVertexInputState = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState = &multisampling,
         .pDepthStencilState = &depthStencil,
         .pColorBlendState = &colorBlending,
         .pDynamicState = &dynamicState,
         .layout = outlinePipelineLayout,
         .renderPass = nullptr},
        {.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
         .pColorAttachmentFormats = colorFormats.data(),
         .depthAttachmentFormat = findDepthFormat()}};
    outlinePipeline =
        vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    // Cracks: the block's six faces, color = src * dst + dst * src (Minecraft's crumbling blend), alpha kept.
    const std::array<vk::PipelineShaderStageCreateInfo, 2> crackStages{
        {{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "crackVertexMain"},
         {.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "crackFragmentMain"}}};
    const vk::PipelineColorBlendAttachmentState crackBlend{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eDstColor,
        .dstColorBlendFactor = vk::BlendFactor::eSrcColor,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const std::array<vk::PipelineColorBlendAttachmentState, 2> crackBlends{crackBlend, motionAttachment};
    const vk::PipelineColorBlendStateCreateInfo crackColorBlending{
        .attachmentCount = static_cast<uint32_t>(crackBlends.size()), .pAttachments = crackBlends.data()};
    auto &crackInfo = pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>();
    crackInfo.pStages = crackStages.data();
    crackInfo.pColorBlendState = &crackColorBlending;
    crackPipeline = vk::raii::Pipeline(device, pipelineCache, crackInfo);
  }

  /** Draw the targeted block's outline; last draw of the scene pass (reuses its viewport and scissor). */
  void recordOutline(vk::raii::CommandBuffer &commandBuffer) {
    if (!outlineTarget) {
      return;
    }
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *outlinePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *outlinePipelineLayout, 0,
                                     *descriptorSets[frameIndex], nullptr);
    // A hair larger than the block, so its edges sit just outside the faces they outline.
    constexpr float grow = 0.002f; // blocks
    const glm::vec3 block{static_cast<float>((*outlineTarget)[0]), static_cast<float>((*outlineTarget)[1]),
                          static_cast<float>((*outlineTarget)[2])};
    const float lineWidth = std::max(1.0f, std::round(static_cast<float>(swapChainExtent.height) / 400.0f));
    const OutlineParams params{.boxMin = glm::vec4(block - grow, lineWidth),
                               .boxMax = glm::vec4(block + 1.0f + grow, 0.0f)};
    constexpr vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    commandBuffer.pushConstants(*outlinePipelineLayout, stages, 0, sizeof(params), &params);
    commandBuffer.draw(72, 1, 0, 0);

    // Survival cracks on the block being broken (Minecraft shows stage floor(progress * 10) from the first tick).
    if (destroyTarget && destroyTarget == outlineTarget) {
      const auto stage = std::min(static_cast<uint32_t>(destroyProgress * 10.0f), DESTROY_STAGES - 1);
      const OutlineParams crackParams{.boxMin = glm::vec4(block, static_cast<float>(destroyStageSlot + stage)),
                                      .boxMax = glm::vec4(block + 1.0f, 0.0f)};
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *crackPipeline);
      commandBuffer.pushConstants(*outlinePipelineLayout, stages, 0, sizeof(crackParams), &crackParams);
      commandBuffer.draw(36, 1, 0, 0);
    }
  }

  /**
   * Create a single-sample, depthless, fullscreen-style overlay pipeline on overlayPipelineLayout.
   *
   * @param vertexEntry Vertex entry point name in `shaderModule`.
   * @param fragmentEntry Fragment entry point name in `shaderModule`.
   * @param srcColorFactor Source color blend factor.
   * @param dstColorFactor Destination color blend factor.
   * @return the pipeline.
   */
  vk::raii::Pipeline createOverlayStylePipeline(const vk::raii::ShaderModule &shaderModule, const char *vertexEntry,
                                                const char *fragmentEntry, vk::BlendFactor srcColorFactor,
                                                vk::BlendFactor dstColorFactor) {
    const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{
        {{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = vertexEntry},
         {.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = fragmentEntry}}};
    const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .lineWidth = 1.0f};
    const vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    const vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = srcColorFactor,
        .dstColorBlendFactor = dstColorFactor,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const vk::PipelineColorBlendStateCreateInfo colorBlending{.attachmentCount = 1, .pAttachments = &blendAttachment};
    const std::array<vk::DynamicState, 2> dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};
    const vk::Format colorFormat = swapChainSurfaceFormat.format;
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = static_cast<uint32_t>(shaderStages.size()),
         .pStages = shaderStages.data(),
         .pVertexInputState = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState = &multisampling,
         .pColorBlendState = &colorBlending,
         .pDynamicState = &dynamicState,
         .layout = overlayPipelineLayout,
         .renderPass = nullptr},
        {.colorAttachmentCount = 1, .pColorAttachmentFormats = &colorFormat}};
    return vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  /**
   * Glyph index of a character in overlay.slang's FONT table (must match its order).
   *
   * @return the glyph index of `c` (case-insensitive), or 0 (space) if the font lacks it.
   */
  static uint8_t overlayGlyphIndex(char c) {
    static constexpr std::string_view glyphs = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:.+-/()";
    if (c >= 'a' && c <= 'z') {
      c = static_cast<char>(c - 'a' + 'A');
    }
    const size_t index = glyphs.find(c);
    return index == std::string_view::npos ? 0u : static_cast<uint8_t>(index);
  }

  /**
   * Draw the HUD on the presented image after TAA: top-right debug text (FPS, geometry path, terrain view, player
   * position), the selected block's texture, the crosshair and health.
   *
   * @param imageIndex Swapchain image to draw on.
   */
  void recordOverlay(vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) {
    // Formatted into fixed buffers: no heap allocation per frame (the XYZ line outgrew std::string's SSO).
    const char *mode = player.waitingForTerrain()                 ? "LOADING"
                       : player.mode() == Player::Mode::Spectator ? ""
                       : player.mode() == Player::Mode::Flying    ? " FLYING"
                       : player.swimming()                        ? " SWIMMING"
                                                                  : " WALKING";
    static constexpr std::array<const char *, Player::kGameModeCount> GAME_MODE_LABELS{"CREATIVE", "SURVIVAL",
                                                                                       "ADVENTURE", "SPECTATOR"};
    const char *gameModeLabel = GAME_MODE_LABELS[static_cast<size_t>(player.gameMode())];
    const glm::dvec3 &feet = player.feetPosition();
    std::array<std::array<char, OVERLAY_COLUMNS + 1>, OVERLAY_LINES> buffers{};
    std::array<std::string_view, OVERLAY_LINES> lines{};
    // printf into line `row`, truncated to OVERLAY_COLUMNS.
    const auto print = [&](uint32_t row, const char *format, auto... args) {
      const int written = std::snprintf(buffers[row].data(), buffers[row].size(), format, args...);
      lines[row] = {buffers[row].data(), std::min<size_t>(static_cast<size_t>(std::max(written, 0)), OVERLAY_COLUMNS)};
    };
    // SKY: skylight at the feet cell, like F3's client light.
    const std::optional<uint32_t> feetSkyLight =
        cpuWorld ? cpuWorld->skyLightAt(static_cast<int32_t>(std::floor(feet.x)), static_cast<int32_t>(std::floor(feet.y)),
                                        static_cast<int32_t>(std::floor(feet.z)))
                 : std::nullopt;
    if (feetSkyLight) {
      print(0, "FPS %d SKY %u", static_cast<int>(std::lround(displayedFps)), *feetSkyLight);
    } else {
      print(0, "FPS %d", static_cast<int>(std::lround(displayedFps)));
    }
    print(1, "%s %s%s", useVertexPulling ? "VERTEX" : "MESH", gameModeLabel, mode);
    print(2, "VIEW %s", terrainViewName(terrainView));
    print(3, "XYZ %lld %lld %lld", static_cast<long long>(std::floor(feet.x)), static_cast<long long>(std::floor(feet.y)),
          static_cast<long long>(std::floor(feet.z)));
    OverlayParams params{
        .screenSize = {static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height)},
        .scale = std::max(1.0f, std::round(static_cast<float>(swapChainExtent.height) / 540.0f)),
        .columns = 0,
        .text = {}};
    for (const auto &line : lines) {
      params.columns = std::max(params.columns, static_cast<uint32_t>(std::min<size_t>(line.size(), OVERLAY_COLUMNS)));
    }
    std::array<uint8_t, OVERLAY_LINES * OVERLAY_COLUMNS> glyphs{};
    for (uint32_t row = 0; row < OVERLAY_LINES; ++row) {
      const std::string_view line = lines[row];
      const uint32_t length = static_cast<uint32_t>(std::min<size_t>(line.size(), OVERLAY_COLUMNS));
      const uint32_t padding = params.columns - length; // right-align
      for (uint32_t i = 0; i < length; ++i) {
        glyphs[row * OVERLAY_COLUMNS + padding + i] = overlayGlyphIndex(line[i]);
      }
    }
    std::memcpy(params.text.data(), glyphs.data(), glyphs.size());

    // The TAA pass just wrote this image as a color attachment; order the overlay's blend after it.
    const vk::MemoryBarrier2 taaDone{.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                     .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                                     .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                     .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                                                      vk::AccessFlagBits2::eColorAttachmentWrite};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &taaDone});

    const vk::RenderingAttachmentInfo overlayAttachment{.imageView = swapChainImageViews[imageIndex],
                                                        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                        .loadOp = vk::AttachmentLoadOp::eLoad,
                                                        .storeOp = vk::AttachmentStoreOp::eStore};
    commandBuffer.beginRendering(vk::RenderingInfo{.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
                                                   .layerCount = 1,
                                                   .colorAttachmentCount = 1,
                                                   .pColorAttachments = &overlayAttachment});
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *overlayPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                                              static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.pushConstants(*overlayPipelineLayout,
                                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(params), &params);
    commandBuffer.draw(6, 1, 0, 0);
    // Minecraft's HUD per game mode: spectator shows none of it, creative only the hotbar, survival and adventure
    // add health, hunger and XP.
    if (player.gameMode() != Player::GameMode::Spectator) {
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *crosshairPipeline);
      commandBuffer.draw(18, 1, 0, 0);

      // Hotbar, XP bar and hunger (overlay.slang hudFragmentMain): slots 1-9 hold the first 9 registry blocks.
      OverlayParams hudParams{.screenSize = params.screenSize, .scale = params.scale, .columns = 0, .text = {}};
      const size_t slotCount = std::min<size_t>(blockRegistry.blocks.size(), 9);
      hudParams.columns = static_cast<uint32_t>(selectedBlockIndex) | (player.survivalLike() ? 0x100u : 0u); // | XP, hunger
      for (size_t slot = 0; slot < slotCount; ++slot) {
        const blocks::Block &block = blockRegistry.blocks[slot];
        const uint32_t topTexture = textureSlots[block.faces[2].texture]; // +y
        const uint32_t sideTexture = textureSlots[block.faces[4].texture]; // +z
        hudParams.text[slot] = topTexture | (sideTexture << 16u);
        const auto channel = [&](size_t i) {
          return static_cast<uint32_t>(std::lround(std::clamp(block.tint[i], 0.0f, 1.0f) * 255.0f));
        };
        hudParams.text[9 + slot] = channel(0) | (channel(1) << 8u) | (channel(2) << 16u) | 0xff000000u;
      }
      hudParams.text[18] = hudTextureSlot;
      hudParams.text[19] = static_cast<uint32_t>(slotCount);
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *hudPipeline);
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *overlayPipelineLayout, 0,
                                       *descriptorSets[frameIndex], nullptr);
      commandBuffer.pushConstants(*overlayPipelineLayout,
                                  vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                  sizeof(hudParams), &hudParams);
      commandBuffer.draw(6, 1, 0, 0);

      OverlayParams healthParams{
          .screenSize = params.screenSize,
          .scale = params.scale,
          .columns = static_cast<uint32_t>(std::ceil(std::max(player.health(), 0.0f))),
          .text = {}};
      healthParams.text[0] = player.hurtFlashing() ? 1u : 0u;
      if (player.survivalLike()) {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *healthPipeline);
        commandBuffer.pushConstants(*overlayPipelineLayout,
                                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                    sizeof(healthParams), &healthParams);
        commandBuffer.draw(6, 1, 0, 0);
      }
    }
    if (gameModeSwitcherOpen) {
      recordGameModeSwitcher(commandBuffer, params.screenSize);
    }
    commandBuffer.endRendering();
  }

  /** MC_FONT index (overlay.slang) of a switcher text character; space for anything missing. */
  static uint8_t minecraftGlyphIndex(char c) {
    static constexpr std::string_view glyphs = " CSAMNFacdeilnoprtuvx4";
    const size_t index = glyphs.find(c);
    return index == std::string_view::npos ? 0u : static_cast<uint8_t>(index);
  }

  /**
   * Draw the F3+F4 game mode switcher (overlay.slang switcherFragmentMain): params.columns = selected mode, text line
   * 0 = its title, line 1 = the "F4 Next" hint (MC_FONT indices, length in each line's last byte), text[18] = the
   * HUD atlas slot.
   */
  void recordGameModeSwitcher(vk::raii::CommandBuffer &commandBuffer, const glm::vec2 &screenSize) {
    OverlayParams switcherParams{.screenSize = screenSize, .scale = 1.0f, .columns = 0, .text = {}};
    switcherParams.columns = static_cast<uint32_t>(gameModeSelection);
    std::array<uint8_t, OVERLAY_LINES * OVERLAY_COLUMNS> glyphs{};
    const auto writeLine = [&](uint32_t row, std::string_view line) {
      for (size_t i = 0; i < std::min<size_t>(line.size(), OVERLAY_COLUMNS); ++i) {
        glyphs[row * OVERLAY_COLUMNS + i] = minecraftGlyphIndex(line[i]);
      }
      glyphs[row * OVERLAY_COLUMNS + OVERLAY_COLUMNS - 1] = static_cast<uint8_t>(std::min<size_t>(line.size(), OVERLAY_COLUMNS - 1));
    };
    writeLine(0, gameModeName(gameModeSelection));
    writeLine(1, "F4 Next");
    std::memcpy(switcherParams.text.data(), glyphs.data(), glyphs.size());
    switcherParams.text[18] = hudTextureSlot; // after lines 0-1, where the shader's atlasTexel looks
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *gameModeSwitcherPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *overlayPipelineLayout, 0,
                                     *descriptorSets[frameIndex], nullptr);
    commandBuffer.pushConstants(*overlayPipelineLayout,
                                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                sizeof(switcherParams), &switcherParams);
    commandBuffer.draw(6, 1, 0, 0);
  }

  /** Create the Hi-Z set layout, pipeline layout and the two pyramid build pipelines (hzb.slang). */
  void createHzbPipelines() {
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
        {{.binding = 0,
          .descriptorType = vk::DescriptorType::eSampledImage,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eCompute},
         {.binding = 1, // one storage view per mip; unused slots repeat the last mip
          .descriptorType = vk::DescriptorType::eStorageImage,
          .descriptorCount = MAX_HZB_MIPS,
          .stageFlags = vk::ShaderStageFlagBits::eCompute}}};
    hzbSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo{.bindingCount = static_cast<uint32_t>(bindings.size()),
                                                  .pBindings = bindings.data()});
    const vk::PushConstantRange pushRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(HzbPushParams)};
    const vk::DescriptorSetLayout setLayout = hzbSetLayout;
    hzbPipelineLayout =
        vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1,
                                                                      .pSetLayouts = &setLayout,
                                                                      .pushConstantRangeCount = 1,
                                                                      .pPushConstantRanges = &pushRange});
    vk::raii::ShaderModule module = createShaderModule(readFile("shaders/hzb.spv"));
    auto createHzbPipeline = [&](const char *entry) {
      return vk::raii::Pipeline(
          device, pipelineCache,
          vk::ComputePipelineCreateInfo{
              .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = entry},
              .layout = hzbPipelineLayout});
    };
    hzbBuildLowPipeline = createHzbPipeline("hzbBuildLow");
    hzbBuildHighPipeline = createHzbPipeline("hzbBuildHigh");
  }

  /**
   * Create the swapchain-sized Hi-Z resources: the pyramid (mip 0 is half the next power of two of the swapchain
   * size), its views, its descriptor set, and binding 7 of every scene set.
   */
  void createHzbResources() {
    auto halfNextPowerOfTwo = [](uint32_t v) {
      uint32_t power = 1;
      while (power < v) {
        power <<= 1u;
      }
      return std::max(power / 2u, 1u);
    };
    hzbSize = vk::Extent2D{std::min(halfNextPowerOfTwo(swapChainExtent.width), HZB_MAX_MIP0_SIDE),
                           std::min(halfNextPowerOfTwo(swapChainExtent.height), HZB_MAX_MIP0_SIDE)};
    hzbMipCount = 1;
    while ((std::max(hzbSize.width, hzbSize.height) >> hzbMipCount) != 0u) {
      ++hzbMipCount;
    }
    auto [image, memory] = createImage(hzbSize.width, hzbSize.height, vk::Format::eR32Sfloat, vk::ImageTiling::eOptimal,
                                       vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                                       vk::MemoryPropertyFlagBits::eDeviceLocal, hzbMipCount);
    hzbImage = std::move(image);
    hzbImageMemory = std::move(memory);
    hzbView = createImageView(*hzbImage, vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor, hzbMipCount);
    hzbMipViews.clear();
    for (uint32_t mip = 0; mip < hzbMipCount; ++mip) {
      hzbMipViews.emplace_back(
          device, vk::ImageViewCreateInfo{.image = *hzbImage,
                                          .viewType = vk::ImageViewType::e2D,
                                          .format = vk::Format::eR32Sfloat,
                                          .subresourceRange = {vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1}});
    }
    hzbImageInitialized = false;
    hzbValid = false;

    // One set: the resolved depth (a single image shared by every frame) and every mip.
    const std::array<vk::DescriptorPoolSize, 2> poolSizes{
        {{.type = vk::DescriptorType::eSampledImage, .descriptorCount = 1},
         {.type = vk::DescriptorType::eStorageImage, .descriptorCount = MAX_HZB_MIPS}}};
    hzbDescriptorPool = vk::raii::DescriptorPool(
        device, vk::DescriptorPoolCreateInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                             .maxSets = 1,
                                             .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                             .pPoolSizes = poolSizes.data()});
    hzbSet = std::move(device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
        .descriptorPool = *hzbDescriptorPool, .descriptorSetCount = 1, .pSetLayouts = &*hzbSetLayout})[0]);

    std::array<vk::DescriptorImageInfo, MAX_HZB_MIPS> mipInfos{};
    for (uint32_t mip = 0; mip < MAX_HZB_MIPS; ++mip) {
      mipInfos[mip] = {.imageView = *hzbMipViews[std::min(mip, hzbMipCount - 1u)],
                       .imageLayout = vk::ImageLayout::eGeneral};
    }
    const vk::DescriptorImageInfo depthInfo{.imageView = *resolvedDepthImageView,
                                            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    const std::array<vk::WriteDescriptorSet, 2> writes{{{.dstSet = *hzbSet,
                                                         .dstBinding = 0,
                                                         .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &depthInfo},
                                                        {.dstSet = *hzbSet,
                                                         .dstBinding = 1,
                                                         .descriptorCount = MAX_HZB_MIPS,
                                                         .descriptorType = vk::DescriptorType::eStorageImage,
                                                         .pImageInfo = mipInfos.data()}}};
    device.updateDescriptorSets(writes, {});

    const vk::DescriptorImageInfo taskInfo{.imageView = *hzbView, .imageLayout = vk::ImageLayout::eGeneral};
    std::vector<vk::DescriptorSet> sceneSets;
    for (const auto &set : descriptorSets) {
      sceneSets.push_back(*set);
    }
    for (const auto &set : cpuDescriptorSets) {
      sceneSets.push_back(*set);
    }
    for (const auto &set : cpuTranslucentDescriptorSets) {
      sceneSets.push_back(*set);
    }
    for (const auto &set : gpuTranslucentDescriptorSets) {
      sceneSets.push_back(*set);
    }
    for (const vk::DescriptorSet set : sceneSets) {
      device.updateDescriptorSets(vk::WriteDescriptorSet{.dstSet = set,
                                                         .dstBinding = 7,
                                                         .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &taskInfo},
                                  {});
    }
  }

  /** Build the Hi-Z pyramid from the resolved depth after the scene pass, for the next frame's taskMain. */
  void recordHzb(vk::raii::CommandBuffer &commandBuffer) {
    const vk::ImageSubresourceRange allMips{vk::ImageAspectFlagBits::eColor, 0, hzbMipCount, 0, 1};
    // No barrier on the resolved depth: it was made visible to compute after the opaque pass and only read since (one
    // here would wait for the water pass). Only this frame's taskMain reads of the pyramid must finish first.
    const std::array<vk::ImageMemoryBarrier2, 1> before{
        {{.srcStageMask =
              hzbImageInitialized ? vk::PipelineStageFlagBits2::eTaskShaderEXT : vk::PipelineStageFlagBits2::eTopOfPipe,
          .srcAccessMask = hzbImageInitialized ? vk::AccessFlagBits2::eShaderRead : vk::AccessFlagBits2::eNone,
          .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
          .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
          .oldLayout = hzbImageInitialized ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
          .newLayout = vk::ImageLayout::eGeneral,
          .image = *hzbImage,
          .subresourceRange = allMips}}};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = static_cast<uint32_t>(before.size()),
                                                      .pImageMemoryBarriers = before.data()});
    hzbImageInitialized = true;

    const auto mipBarrier = [&](vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess) {
      const vk::ImageMemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                            .dstStageMask = dstStage,
                                            .dstAccessMask = dstAccess,
                                            .oldLayout = vk::ImageLayout::eGeneral,
                                            .newLayout = vk::ImageLayout::eGeneral,
                                            .image = *hzbImage,
                                            .subresourceRange = allMips};
      commandBuffer.pipelineBarrier2(
          vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    };

    const HzbPushParams params{.mip0Size = {hzbSize.width, hzbSize.height},
                               .depthSize = {swapChainExtent.width, swapChainExtent.height},
                               .mipCount = hzbMipCount,
                               .ignoreNearerDepth = reverseZDepth(HZB_IGNORE_NEAR_DISTANCE)};
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *hzbPipelineLayout, 0, *hzbSet,
                                     nullptr);
    commandBuffer.pushConstants(*hzbPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), &params);
    // One thread per mip 0 texel; each 16x16 group reduces its block to one texel of mip 4.
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *hzbBuildLowPipeline);
    commandBuffer.dispatch((hzbSize.width + 15u) / 16u, (hzbSize.height + 15u) / 16u, 1);
    if (hzbMipCount > HZB_LOW_MIPS) {
      mipBarrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *hzbBuildHighPipeline);
      commandBuffer.dispatch(1, 1, 1);
    }
    mipBarrier(vk::PipelineStageFlagBits2::eTaskShaderEXT, vk::AccessFlagBits2::eShaderRead);
    hzbValid = true;
  }

  /** Create the chunk compute set layout (19 storage buffers), pipeline layout and every chunk_compute.slang pipeline. */
  void createChunkComputePipeline() {
    std::array<vk::DescriptorSetLayoutBinding, 19> bindings{};
    for (uint32_t binding = 0; binding < bindings.size(); ++binding) {
      bindings[binding] = {.binding = binding,
                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                           .descriptorCount = 1,
                           .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    chunkComputeDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo{.bindingCount = static_cast<uint32_t>(bindings.size()),
                                                  .pBindings = bindings.data()});

    const vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(ChunkComputeParams)};
    const vk::DescriptorSetLayout computeSetLayout = chunkComputeDescriptorSetLayout;
    chunkComputePipelineLayout =
        vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1,
                                                                      .pSetLayouts = &computeSetLayout,
                                                                      .pushConstantRangeCount = 1,
                                                                      .pPushConstantRanges = &pushConstantRange});

    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/chunk_compute.spv"));
    auto createComputePipeline = [&](const char *entryPoint) {
      vk::PipelineShaderStageCreateInfo stage{
          .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = entryPoint};
      return vk::raii::Pipeline(device, pipelineCache,
                                vk::ComputePipelineCreateInfo{.stage = stage, .layout = chunkComputePipelineLayout});
    };

    generateHeightsPipeline = createComputePipeline("generateHeights");
    retireWorkPipeline = createComputePipeline("retireWork");
    allocateWorkPipeline = createComputePipeline("allocateWork");
    finalizeWorkPipeline = createComputePipeline("finalizeWork");
    initWorkPoolPipeline = createComputePipeline("initWorkPool");
    collectRelocationsPipeline = createComputePipeline("collectRelocations");
    relocateSlotsPipeline = createComputePipeline("relocateSlots");
    rebuildFreeStackPipeline = createComputePipeline("rebuildFreeStack");
    meshColumnsPipeline = createComputePipeline("meshColumns");
    finalizeVertexDrawPipeline = createComputePipeline("finalizeVertexDraw");
    compactQuadsPipeline = createComputePipeline("compactQuads");
    copyCompactedQuadsPipeline = createComputePipeline("copyCompactedQuads");
    createHzbPipelines();
  }
#pragma endregion
#pragma endregion

#pragma region Resource Setup
#pragma region Command Pool
  /** Create the resettable command pool on the render queue family. */
  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       .queueFamilyIndex = queueIndex};
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }
#pragma endregion

#pragma region Depth Image
  /**
   * Create the swapchain-sized render targets: scene color, MSAA color and motion, their resolves, MSAA depth, the
   * resolved depth and the two TAA history images.
   */
  void createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    // One set of targets shared by every frame: frames run on one queue, and each frame's first barriers order its
    // writes after the previous frame's reads. Per-image copies would only cost VRAM.
    const auto makeTarget = [&](vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
                                vk::SampleCountFlagBits samples, vk::raii::Image &image, vk::raii::DeviceMemory &memory,
                                vk::raii::ImageView &view) {
      std::tie(image, memory) = createImage(swapChainExtent.width, swapChainExtent.height, format,
                                            vk::ImageTiling::eOptimal, usage, vk::MemoryPropertyFlagBits::eDeviceLocal,
                                            1, samples);
      view = createImageView(*image, format, aspect);
    };
    const auto color = vk::ImageAspectFlagBits::eColor;
    const auto single = vk::SampleCountFlagBits::e1; // no MSAA
    makeTarget(swapChainSurfaceFormat.format,
               vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, // never copied from
               color, single, sceneColorImage, sceneColorImageMemory, sceneColorImageView);
    makeTarget(swapChainSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment, color, RENDER_SAMPLE_COUNT,
               multisampleColorImage, multisampleColorImageMemory, multisampleColorImageView);
    makeTarget(MOTION_FORMAT, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, color,
               single, motionImage, motionImageMemory, motionImageView);
    makeTarget(MOTION_FORMAT, vk::ImageUsageFlagBits::eColorAttachment, color, RENDER_SAMPLE_COUNT,
               multisampleMotionImage, multisampleMotionImageMemory, multisampleMotionImageView);
    // Never sampled (Hi-Z and the water fog read the resolve below), so the driver can keep it compressed.
    makeTarget(depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth,
               RENDER_SAMPLE_COUNT, depthImage, depthImageMemory, depthImageView);
    makeTarget(depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
               vk::ImageAspectFlagBits::eDepth, single, resolvedDepthImage, resolvedDepthImageMemory,
               resolvedDepthImageView);

    for (size_t i = 0; i < taaHistoryImages.size(); ++i) {
      auto [historyImage, historyImageMemory] =
          createImage(swapChainExtent.width, swapChainExtent.height, TAA_HISTORY_FORMAT, vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);
      taaHistoryImages[i] = std::move(historyImage);
      taaHistoryImageMemories[i] = std::move(historyImageMemory);
      taaHistoryImageViews[i] =
          createImageView(taaHistoryImages[i], TAA_HISTORY_FORMAT, vk::ImageAspectFlagBits::eColor);
    }
  }

  /**
   * First format of `candidates` that supports `features` with `tiling`.
   *
   * @return the format; throws if none qualifies.
   */
  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features) {
    for (const auto format : candidates) {
      vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(format);
      if (((tiling == vk::ImageTiling::eLinear) && ((formatProperties.linearTilingFeatures & features) == features)) ||
          ((tiling == vk::ImageTiling::eOptimal) &&
           ((formatProperties.optimalTilingFeatures & features) == features))) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  /** Depth format usable as an attachment and sampled image (D32 preferred). */
  vk::Format findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage);
  }
#pragma endregion

#pragma region Texture Setup
  /**
   * Upload one 16x16 RGBA texture frame into a new image and blit its mip chain.
   *
   * @param pixels 16 rows of 64 bytes (RGBA8).
   * @param image Receives the created image.
   * @param imageMemory Receives its memory.
   */
  void createTextureImage(const stbi_uc *pixels, int texWidth, int texHeight, vk::raii::Image &image,
                          vk::raii::DeviceMemory &imageMemory) {
    const auto formatFeatures = physicalDevice.getFormatProperties(TEXTURE_FORMAT).optimalTilingFeatures;
    constexpr auto requiredBlitFeatures = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst |
                                          vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
    if ((formatFeatures & requiredBlitFeatures) != requiredBlitFeatures) {
      throw std::runtime_error("texture format does not support linear mipmap blits");
    }
    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();

    std::tie(image, imageMemory) = createImage(
        texWidth, texHeight, TEXTURE_FORMAT, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, TEXTURE_MIP_LEVELS);

    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
    transitionImageLayout(commandBuffer, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 0,
                          TEXTURE_MIP_LEVELS);
    copyBufferToImage(commandBuffer, stagingBuffer, image, static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;
    for (uint32_t mipLevel = 1; mipLevel < TEXTURE_MIP_LEVELS; ++mipLevel) {
      transitionImageLayout(commandBuffer, image, vk::ImageLayout::eTransferDstOptimal,
                            vk::ImageLayout::eTransferSrcOptimal, mipLevel - 1u, 1);

      const int32_t nextMipWidth = std::max(mipWidth / 2, 1);
      const int32_t nextMipHeight = std::max(mipHeight / 2, 1);
      vk::ImageBlit blit{
          .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = mipLevel - 1u,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
          .srcOffsets = std::array<vk::Offset3D, 2>{vk::Offset3D{0, 0, 0}, vk::Offset3D{mipWidth, mipHeight, 1}},
          .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = mipLevel,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
          .dstOffsets =
              std::array<vk::Offset3D, 2>{vk::Offset3D{0, 0, 0}, vk::Offset3D{nextMipWidth, nextMipHeight, 1}}};
      commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal,
                              blit, vk::Filter::eLinear);

      transitionImageLayout(commandBuffer, image, vk::ImageLayout::eTransferSrcOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal, mipLevel - 1u, 1);
      mipWidth = nextMipWidth;
      mipHeight = nextMipHeight;
    }
    transitionImageLayout(commandBuffer, image, vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal, TEXTURE_MIP_LEVELS - 1u, 1);
    endSingleTimeCommands(std::move(commandBuffer));
  }

  /** Load the block registry, resolve the terrain block IDs, and upload every texture frame to the bindless array. */
  void createTextureImages() {
    blockRegistry = blocks::loadRegistry("blocks.txt", "textures/");
    for (size_t layer = 0; layer < world::kTerrainLayerBlocks.size(); ++layer) {
      terrainLayerBlocks[layer] = blockRegistry.idOf(world::kTerrainLayerBlocks[layer]);
      terrainLayerFillerBlocks[layer] = blockRegistry.idOf(world::kTerrainLayerFillerBlocks[layer]);
      terrainLayerDeepBlocks[layer] = blockRegistry.idOf(world::kTerrainLayerDeepBlocks[layer]);
    }
    treeLogBlock = blockRegistry.idOf(world::kTreeLogBlock);
    treeLeavesBlock = blockRegistry.idOf(world::kTreeLeavesBlock);

    const auto &texturePaths = blockRegistry.texturePaths;
    if (texturePaths.size() > textureDescriptorCapacity) {
      throw std::runtime_error("texture descriptor capacity is too small for the loaded textures");
    }

    // A texture file is 16 px wide and 16 * N tall: N animation frames, top first, each its own bindless texture at
    // consecutive slots.
    textureSlots.clear();
    textureFrames.clear();
    for (const std::string &path : texturePaths) {
      int texWidth, texHeight, texChannels;
      stbi_uc *pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
      if (!pixels) {
        throw std::runtime_error("failed to load texture image: " + path);
      }
      const int frames = texHeight / 16;
      if (texWidth != 16 || texHeight % 16 != 0 || frames < 1 || frames > 255) {
        stbi_image_free(pixels);
        throw std::runtime_error("block texture must be 16 px wide and 16 * N (N = 1..255 frames) tall: " + path);
      }
      textureSlots.push_back(static_cast<uint32_t>(textureImages.size()));
      textureFrames.push_back(static_cast<uint32_t>(frames));
      for (int frame = 0; frame < frames; ++frame) {
        if (textureImages.size() >= textureDescriptorCapacity) {
          stbi_image_free(pixels);
          throw std::runtime_error("texture descriptor capacity is too small for the loaded texture frames");
        }
        textureImages.emplace_back(nullptr);
        textureImageMemories.emplace_back(nullptr);
        createTextureImage(pixels + static_cast<size_t>(frame) * 16u * 16u * 4u, 16, 16, textureImages.back(),
                           textureImageMemories.back());
      }
      stbi_image_free(pixels);
    }

    // HUD sprite atlas (overlay.slang hudFragmentMain), right after the block textures.
    int hudWidth, hudHeight, hudChannels;
    stbi_uc *hudPixels = stbi_load("textures/gui/hud.png", &hudWidth, &hudHeight, &hudChannels, STBI_rgb_alpha);
    if (!hudPixels) {
      throw std::runtime_error("failed to load texture image: textures/gui/hud.png");
    }
    if (textureImages.size() >= textureDescriptorCapacity) {
      stbi_image_free(hudPixels);
      throw std::runtime_error("texture descriptor capacity is too small for the HUD atlas");
    }
    hudTextureSlot = static_cast<uint32_t>(textureImages.size());
    textureImages.emplace_back(nullptr);
    textureImageMemories.emplace_back(nullptr);
    createTextureImage(hudPixels, hudWidth, hudHeight, textureImages.back(), textureImageMemories.back());
    stbi_image_free(hudPixels);

    // Block-breaking crack stages (outline.slang crackFragmentMain), after the atlas.
    destroyStageSlot = static_cast<uint32_t>(textureImages.size());
    for (uint32_t stage = 0; stage < DESTROY_STAGES; ++stage) {
      const std::string path = "textures/overlay/destroy_stage_" + std::to_string(stage) + ".png";
      int width, height, channels;
      stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
      if (!pixels) {
        throw std::runtime_error("failed to load texture image: " + path);
      }
      if (textureImages.size() >= textureDescriptorCapacity) {
        stbi_image_free(pixels);
        throw std::runtime_error("texture descriptor capacity is too small for the destroy stages");
      }
      textureImages.emplace_back(nullptr);
      textureImageMemories.emplace_back(nullptr);
      createTextureImage(pixels, width, height, textureImages.back(), textureImageMemories.back());
      stbi_image_free(pixels);
    }
  }

  /** Upload the per-face block table (GpuBlockFace), read by the shaders by block ID and normal. */
  void createBlockFaceBuffer() {
    // Indexed by block ID, which has a gap between the opaque (0..) and non-opaque (512..) ranges; gaps stay zero.
    std::vector<GpuBlockFace> faces((blockRegistry.maxId() + 1u) * blocks::kFaceCount, GpuBlockFace{});
    for (const blocks::Block &block : blockRegistry.blocks) {
      for (size_t faceIndex = 0; faceIndex < blocks::kFaceCount; ++faceIndex) {
        const blocks::Face &face = block.faces[faceIndex];
        const uint32_t lodTexture = face.hasLodTexture ? face.lodTexture : face.texture;
        const uint32_t frameMs =
            static_cast<uint32_t>(std::lround(std::clamp(block.frameTime, 0.001f, 60.0f) * 1000.0f));
        faces[block.id * blocks::kFaceCount + faceIndex] = {
            .texture = textureSlots[face.texture],
            .lodTexture = textureSlots[lodTexture],
            .flags = (face.randomRotation ? BLOCK_FACE_RANDOM_ROTATION : 0u) |
                     (face.hasLodTexture ? BLOCK_FACE_HAS_LOD_TEXTURE : 0u) |
                     (block.fluid != blocks::Fluid::None ? BLOCK_FACE_FLUID : 0u),
            .animation = textureFrames[face.texture] | (textureFrames[lodTexture] << 8u) | (frameMs << 16u),
            .lodTint = {block.lodTint[0], block.lodTint[1], block.lodTint[2], 1.0f},
            .tint = {block.tint[0], block.tint[1], block.tint[2], 1.0f},
            .farTint = {block.farTint[0], block.farTint[1], block.farTint[2], 1.0f}};
      }
    }

    const vk::DeviceSize size = faces.size() * sizeof(GpuBlockFace);
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data = stagingBufferMemory.mapMemory(0, size);
    std::memcpy(data, faces.data(), static_cast<size_t>(size));
    stagingBufferMemory.unmapMemory();

    std::tie(blockFaceBuffer, blockFaceBufferMemory) =
        createBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyBuffer(stagingBuffer, blockFaceBuffer, size);
  }

  /** Create a mipmapped view per texture frame. */
  void createTextureImageViews() {
    textureImageViews.reserve(textureImages.size());
    for (const auto &image : textureImages) {
      textureImageViews.emplace_back(
          createImageView(*image, TEXTURE_FORMAT, vk::ImageAspectFlagBits::eColor, TEXTURE_MIP_LEVELS));
    }
  }

  /** Create the block texture sampler: nearest texels, linear mips, max anisotropy. */
  void createTextureSampler() {
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{.magFilter = vk::Filter::eNearest,
                                      .minFilter = vk::Filter::eNearest,
                                      .mipmapMode = vk::SamplerMipmapMode::eLinear,
                                      .addressModeU = vk::SamplerAddressMode::eRepeat,
                                      .addressModeV = vk::SamplerAddressMode::eRepeat,
                                      .addressModeW = vk::SamplerAddressMode::eRepeat,
                                      .mipLodBias = 0.0f,
                                      .anisotropyEnable = vk::True,
                                      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
                                      .compareEnable = vk::False,
                                      .compareOp = vk::CompareOp::eAlways,
                                      .minLod = 0.0f,
                                      .maxLod = static_cast<float>(TEXTURE_MIP_LEVELS - 1u)};
    textureSampler = vk::raii::Sampler(device, samplerInfo);
  }

  /** Create the TAA sampler: bilinear, clamped, single mip. */
  void createTaaSampler() {
    const vk::SamplerCreateInfo samplerInfo{.magFilter = vk::Filter::eLinear,
                                            .minFilter = vk::Filter::eLinear,
                                            .mipmapMode = vk::SamplerMipmapMode::eNearest,
                                            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                                            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                                            .maxAnisotropy = 1.0f,
                                            .compareOp = vk::CompareOp::eAlways,
                                            .minLod = 0.0f,
                                            .maxLod = 0.0f};
    taaSampler = vk::raii::Sampler(device, samplerInfo);
  }
#pragma endregion

#pragma region Image Helpers
  /** Create a 2D view of the first `mipLevels` mips of `image`. */
  vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format, vk::ImageAspectFlags aspectFlags,
                                      uint32_t mipLevels = 1) {
    vk::ImageViewCreateInfo viewInfo{.image = image,
                                     .viewType = vk::ImageViewType::e2D,
                                     .format = format,
                                     .subresourceRange = {.aspectMask = aspectFlags,
                                                          .baseMipLevel = 0,
                                                          .levelCount = mipLevels,
                                                          .baseArrayLayer = 0,
                                                          .layerCount = 1}};
    return vk::raii::ImageView(device, viewInfo);
  }

  /**
   * Create a 2D image and bind it to newly allocated memory.
   *
   * @param properties Required memory properties.
   * @return the image and its memory.
   */
  std::pair<vk::raii::Image, vk::raii::DeviceMemory>
  createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
              vk::MemoryPropertyFlags properties, uint32_t mipLevels = 1,
              vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1) {
    vk::ImageCreateInfo imageInfo{.imageType = vk::ImageType::e2D,
                                  .format = format,
                                  .extent = {width, height, 1},
                                  .mipLevels = mipLevels,
                                  .arrayLayers = 1,
                                  .samples = samples,
                                  .tiling = tiling,
                                  .usage = usage,
                                  .sharingMode = vk::SharingMode::eExclusive};

    vk::raii::Image image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memoryRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memoryRequirements.size,
                                     .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties)};
    vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(imageMemory, 0);

    return {std::move(image), std::move(imageMemory)};
  }
#pragma endregion

#pragma region Upload Helpers
  /**
   * Record a layout transition of a color image's mips for the texture upload path.
   *
   * Supports undefined -> transfer dst, transfer dst -> transfer src, and transfer dst/src -> shader read only;
   * throws on any other pair.
   */
  void transitionImageLayout(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Image &image,
                             vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t baseMipLevel = 0,
                             uint32_t levelCount = 1) {
    vk::ImageMemoryBarrier barrier{.oldLayout = oldLayout,
                                   .newLayout = newLayout,
                                   .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                                   .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                                   .image = image,
                                   .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                        .baseMipLevel = baseMipLevel,
                                                        .levelCount = levelCount,
                                                        .layerCount = 1}};

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if ((oldLayout == vk::ImageLayout::eTransferDstOptimal ||
                oldLayout == vk::ImageLayout::eTransferSrcOptimal) &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = oldLayout == vk::ImageLayout::eTransferDstOptimal ? vk::AccessFlagBits::eTransferWrite
                                                                                : vk::AccessFlagBits::eTransferRead;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
  }

  /** Record a copy of tightly packed pixels from `buffer` into mip 0 of `image` (in transfer dst layout). */
  void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Buffer &buffer, vk::raii::Image &image,
                         uint32_t width, uint32_t height) {
    vk::BufferImageCopy region{.bufferOffset = 0,
                               .bufferRowLength = 0,
                               .bufferImageHeight = 0,
                               .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                    .mipLevel = 0,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = 1},
                               .imageOffset = {0, 0, 0},
                               .imageExtent = {width, height, 1}};
    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
  }
#pragma endregion

#pragma region Buffer Setup
  /**
   * Create a buffer and bind it to newly allocated memory.
   *
   * @param size Size in bytes.
   * @param properties Required memory properties.
   * @return the buffer and its memory.
   */
  std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                                   vk::MemoryPropertyFlags properties) {
    vk::BufferCreateInfo bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
    vk::raii::Buffer buffer = vk::raii::Buffer(device, bufferInfo);
    vk::MemoryRequirements memoryRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memoryRequirements.size,
                                     .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties)};
    vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
    buffer.bindMemory(*bufferMemory, 0);
    return {std::move(buffer), std::move(bufferMemory)};
  }

  /** Create a persistently mapped uniform buffer per frame in flight. */
  void createUniformBuffers() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
      auto [buffer, bufferMemory] =
          createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      uniformBuffers.emplace_back(std::move(buffer));
      uniformBuffersMemory.emplace_back(std::move(bufferMemory));
      uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
    }
  }

  /**
   * Create the GPU clipmap's buffers: quads, offsets, heights, counts, the static quad index buffer, the work pool
   * and the per-frame host-visible lists and readbacks.
   */
  void createChunkBuffers() {
    const vk::DeviceSize bufferSize = sizeof(uint64_t) * MAX_QUAD_COUNT;
    auto [quadBufferValue, quadBufferMemoryValue] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    quadBuffer = std::move(quadBufferValue);
    quadBufferMemory = std::move(quadBufferMemoryValue);

    auto [offsetBufferValue, offsetBufferMemoryValue] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    chunkOffsetBuffer = std::move(offsetBufferValue);
    chunkOffsetBufferMemory = std::move(offsetBufferMemoryValue);

    // compactQuads output: packed QuadBuffer, then packed ChunkOffsets.
    auto [compactBufferValue, compactBufferMemoryValue] = createBuffer(
        2u * bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    quadCompactBuffer = std::move(compactBufferValue);
    quadCompactBufferMemory = std::move(compactBufferMemoryValue);

    constexpr uint32_t HEIGHT_STRIDE = CHUNK_SIZE + 2u; // one block of border on each side
    const vk::DeviceSize heightBufferSize =
        sizeof(int32_t) * MAX_HORIZONTAL_CHUNK_COUNT * HEIGHT_STRIDE * HEIGHT_STRIDE;
    auto [heightBufferValue, heightBufferMemoryValue] = createBuffer(
        heightBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    chunkHeightBuffer = std::move(heightBufferValue);
    chunkHeightBufferMemory = std::move(heightBufferMemoryValue);

    auto [quadCountBufferValue, quadCountBufferMemoryValue] =
        createBuffer(2u * sizeof(uint32_t),
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    quadCountBuffer = std::move(quadCountBufferValue);
    quadCountBufferMemory = std::move(quadCountBufferMemoryValue);

    // Static {0,1,2, 0,2,3} + 4q pattern for VERTEX_QUADS_PER_INSTANCE quads, staged into device-local memory since
    // every vertex-pulling draw reads it.
    {
      const vk::DeviceSize indexBufferSize = sizeof(uint16_t) * 6u * VERTEX_QUADS_PER_INSTANCE;
      auto [stagingBuffer, stagingBufferMemory] =
          createBuffer(indexBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      auto *indices = static_cast<uint16_t *>(stagingBufferMemory.mapMemory(0, indexBufferSize));
      for (uint32_t quad = 0; quad < VERTEX_QUADS_PER_INSTANCE; ++quad) {
        const auto base = static_cast<uint16_t>(quad * 4u);
        const std::array<uint16_t, 6> quadIndices{
            base, static_cast<uint16_t>(base + 1u), static_cast<uint16_t>(base + 2u),
            base, static_cast<uint16_t>(base + 2u), static_cast<uint16_t>(base + 3u)};
        std::memcpy(indices + quad * 6u, quadIndices.data(), sizeof(quadIndices));
      }
      stagingBufferMemory.unmapMemory();
      std::tie(quadIndexBuffer, quadIndexBufferMemory) =
          createBuffer(indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eDeviceLocal);
      copyBuffer(stagingBuffer, quadIndexBuffer, indexBufferSize);
    }

    createWorkPoolBuffers();

    auto [vertexDrawArgumentsValue, vertexDrawArgumentsMemoryValue] =
        createBuffer(sizeof(vk::DrawIndexedIndirectCommand),
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    vertexDrawArgumentsBuffer = std::move(vertexDrawArgumentsValue);
    vertexDrawArgumentsBufferMemory = std::move(vertexDrawArgumentsMemoryValue);

    const vk::DeviceSize coordinateBufferSize = sizeof(world::ChunkLod) * MAX_HORIZONTAL_CHUNK_COUNT;
    const vk::BufferUsageFlags gpuOnly = vk::BufferUsageFlagBits::eStorageBuffer;
    // Set up by initWorkPool: the first remesh is a full rebuild, recorded before the first draw that reads it.
    std::tie(workPoolBuffer, workPoolBufferMemory) =
        createBuffer(sizeof(int32_t) * WORK_POOL_INTS,
                     gpuOnly | vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(columnWorkSlotBuffer, columnWorkSlotBufferMemory) =
        createBuffer(sizeof(uint32_t) * MAX_HORIZONTAL_CHUNK_COUNT * world::kMaxColumnChunks, gpuOnly,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(columnHeightRangeBuffer, columnHeightRangeBufferMemory) =
        createBuffer(2u * sizeof(int32_t) * MAX_HORIZONTAL_CHUNK_COUNT * HEIGHT_GROUPS_PER_COLUMN, gpuOnly,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    for (size_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
      auto [coordinateBuffer, coordinateBufferMemory] =
          createBuffer(coordinateBufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      chunkCoordinateBuffers.emplace_back(std::move(coordinateBuffer));
      chunkCoordinateBufferMemories.emplace_back(std::move(coordinateBufferMemory));
      chunkCoordinateBuffersMapped.emplace_back(
          chunkCoordinateBufferMemories.back().mapMemory(0, coordinateBufferSize));
      std::memset(chunkCoordinateBuffersMapped.back(), 0, static_cast<size_t>(coordinateBufferSize));

      const vk::DeviceSize neighborLodBufferSize = sizeof(uint32_t) * MAX_HORIZONTAL_CHUNK_COUNT;
      auto [neighborLodBuffer, neighborLodBufferMemory] =
          createBuffer(neighborLodBufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      chunkNeighborLodBuffers.emplace_back(std::move(neighborLodBuffer));
      chunkNeighborLodBufferMemories.emplace_back(std::move(neighborLodBufferMemory));
      chunkNeighborLodBuffersMapped.emplace_back(
          chunkNeighborLodBufferMemories.back().mapMemory(0, neighborLodBufferSize));
      std::memset(chunkNeighborLodBuffersMapped.back(), 0, static_cast<size_t>(neighborLodBufferSize));

      const vk::DeviceSize columnListSize = sizeof(uint32_t) * MAX_HORIZONTAL_CHUNK_COUNT;
      auto [columnListBuffer, columnListMemory] =
          createBuffer(columnListSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      columnListBuffers.emplace_back(std::move(columnListBuffer));
      columnListBufferMemories.emplace_back(std::move(columnListMemory));
      columnListBuffersMapped.emplace_back(columnListBufferMemories.back().mapMemory(0, columnListSize));

      const vk::DeviceSize retireListSize = sizeof(uint32_t) * MAX_HORIZONTAL_CHUNK_COUNT;
      auto [retireListBuffer, retireListMemory] =
          createBuffer(retireListSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      retireListBuffers.emplace_back(std::move(retireListBuffer));
      retireListBufferMemories.emplace_back(std::move(retireListMemory));
      retireListBuffersMapped.emplace_back(retireListBufferMemories.back().mapMemory(0, retireListSize));

      auto [readbackBuffer, readbackMemory] =
          createBuffer(READBACK_UINTS * sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      quadCountReadbackBuffers.emplace_back(std::move(readbackBuffer));
      quadCountReadbackMemories.emplace_back(std::move(readbackMemory));
      quadCountReadbackMapped.emplace_back(static_cast<const uint32_t *>(
          quadCountReadbackMemories.back().mapMemory(0, READBACK_UINTS * sizeof(uint32_t))));
    }
  }

  /**
   * Create every buffer sized by workPoolCapacity (compute bindings 5, 6, 13, 16, 17 and 18). Contents start
   * undefined: a full rebuild (clears + initWorkPool) always follows.
   */
  void createWorkPoolBuffers() {
    const vk::BufferUsageFlags gpuOnly = vk::BufferUsageFlagBits::eStorageBuffer;
    std::tie(quadClusterBuffer, quadClusterBufferMemory) =
        createBuffer(QUAD_CLUSTER_SIZE * QUAD_CLUSTERS_PER_CHUNK * workPoolCapacity,
                     gpuOnly | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(waterClusterBuffer, waterClusterBufferMemory) =
        createBuffer(QUAD_CLUSTER_SIZE * QUAD_CLUSTERS_PER_CHUNK * workPoolCapacity,
                     gpuOnly | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(chunkWorkBuffer, chunkWorkBufferMemory) = createBuffer(sizeof(std::array<int32_t, 4>) * workPoolCapacity,
                                                                    gpuOnly, vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(freeWorkSlotBuffer, freeWorkSlotBufferMemory) =
        createBuffer(sizeof(uint32_t) * workPoolCapacity, gpuOnly, vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(slotOwnerBuffer, slotOwnerBufferMemory) =
        createBuffer(sizeof(uint32_t) * workPoolCapacity, gpuOnly, vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(relocationScratchBuffer, relocationScratchBufferMemory) =
        createBuffer(2u * sizeof(uint32_t) * workPoolCapacity, gpuOnly, vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  /**
   * Grow the work slot pool after allocateWork ran dry (at least doubling, with headroom), rewrite the descriptors
   * and schedule a full remesh. Rare, so it simply waits for the GPU to go idle.
   *
   * @param demand Work slots the last remesh needed.
   */
  void growWorkPool(uint64_t demand) {
    const uint32_t oldCapacity = workPoolCapacity;
    const uint64_t wanted = std::max<uint64_t>(2ull * workPoolCapacity, demand + demand / 4u);
    workPoolCapacity = static_cast<uint32_t>(std::min<uint64_t>(std::bit_ceil(wanted), WORK_POOL_MAX_CAPACITY));
    std::cerr << "GPU work slot pool: " << oldCapacity << " slots ran out, growing to " << workPoolCapacity << "\n";
    device.waitIdle();
    createWorkPoolBuffers();
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      const std::array<std::pair<uint32_t, vk::Buffer>, 6> computeBindings{{{5u, *chunkWorkBuffer},
                                                                            {6u, *quadClusterBuffer},
                                                                            {13u, *freeWorkSlotBuffer},
                                                                            {16u, *slotOwnerBuffer},
                                                                            {17u, *relocationScratchBuffer},
                                                                            {18u, *waterClusterBuffer}}};
      std::array<vk::DescriptorBufferInfo, computeBindings.size() + 2> infos{};
      std::array<vk::WriteDescriptorSet, computeBindings.size() + 2> writes{};
      for (size_t binding = 0; binding < computeBindings.size(); ++binding) {
        infos[binding] = vk::DescriptorBufferInfo{
            .buffer = computeBindings[binding].second, .offset = 0, .range = vk::WholeSize};
        writes[binding] = {.dstSet = chunkComputeDescriptorSets[i],
                           .dstBinding = computeBindings[binding].first,
                           .descriptorCount = 1,
                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                           .pBufferInfo = &infos[binding]};
      }
      // The GPU clipmap's graphics sets read the clusters too (task shader, binding 6).
      const size_t opaque = computeBindings.size();
      infos[opaque] = vk::DescriptorBufferInfo{.buffer = *quadClusterBuffer, .offset = 0, .range = vk::WholeSize};
      writes[opaque] = {.dstSet = descriptorSets[i],
                        .dstBinding = 6,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &infos[opaque]};
      infos[opaque + 1] = vk::DescriptorBufferInfo{.buffer = *waterClusterBuffer, .offset = 0, .range = vk::WholeSize};
      writes[opaque + 1] = {.dstSet = gpuTranslucentDescriptorSets[i],
                            .dstBinding = 6,
                            .descriptorCount = 1,
                            .descriptorType = vk::DescriptorType::eStorageBuffer,
                            .pBufferInfo = &infos[opaque + 1]};
      device.updateDescriptorSets(writes, {});
    }
    forceFullRebuild = true;
    chunkBuffersDirty = true;
  }

  /** Create the CPU-meshed chunks' device buffers (cleared to no clusters) and per-frame staging buffers. */
  void createCpuChunkBuffers() {
    const vk::BufferUsageFlags deviceUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    const vk::DeviceSize quadBytes = sizeof(uint64_t) * CPU_MAX_QUADS;
    std::tie(cpuQuadBuffer, cpuQuadBufferMemory) =
        createBuffer(quadBytes, deviceUsage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(cpuChunkOffsetBuffer, cpuChunkOffsetBufferMemory) =
        createBuffer(quadBytes, deviceUsage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(cpuClusterBuffer, cpuClusterBufferMemory) =
        createBuffer(QUAD_CLUSTER_SIZE * QUAD_CLUSTERS_PER_CHUNK * CPU_MAX_CHUNK_SLOTS, deviceUsage,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(cpuTranslucentClusterBuffer, cpuTranslucentClusterBufferMemory) =
        createBuffer(QUAD_CLUSTER_SIZE * QUAD_CLUSTERS_PER_CHUNK * CPU_MAX_CHUNK_SLOTS, deviceUsage,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::tie(cpuQuadCountBuffer, cpuQuadCountBufferMemory) =
        createBuffer(2u * sizeof(uint32_t), deviceUsage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    for (size_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
      auto [staging, stagingMemory] =
          createBuffer(CPU_STAGING_BYTES, vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      cpuStagingBuffers.emplace_back(std::move(staging));
      cpuStagingMemories.emplace_back(std::move(stagingMemory));
      cpuStagingMapped.push_back(static_cast<std::byte *>(cpuStagingMemories.back().mapMemory(0, CPU_STAGING_BYTES)));
    }
    auto commandBuffer = beginSingleTimeCommands();
    commandBuffer.fillBuffer(*cpuClusterBuffer, 0, vk::WholeSize, 0u);
    commandBuffer.fillBuffer(*cpuTranslucentClusterBuffer, 0, vk::WholeSize, 0u);
    commandBuffer.fillBuffer(*cpuQuadCountBuffer, 0, vk::WholeSize, 0u);
    endSingleTimeCommands(std::move(commandBuffer));
  }

  /**
   * Tick fluids, break (left) / place (right) blocks on click or hold, then pick the block to outline (after editing,
   * so the outline moves on at once).
   *
   * @param eye Camera position, in world blocks.
   * @param frameDeltaSeconds Frame time, in seconds (drives the 20 Hz interaction ticks).
   */
  void updateBlockInteraction(const glm::vec3 &eye, float frameDeltaSeconds) {
    const bool breakClicked = std::exchange(breakClickPending, false) && camera.cursorCaptured();
    const bool placeClicked = std::exchange(placeClickPending, false) && camera.cursorCaptured();
    outlineTarget.reset();
    cpuWorld->tickFluids(glfwGetTime()); // ocean flowing into dug cells, one block per flow step
    const Player::GameMode gameMode = player.gameMode();
    if (gameMode == Player::GameMode::Spectator) {
      destroyTarget.reset();
      return;
    }
    const bool canEdit = gameMode != Player::GameMode::Adventure;
    const glm::vec3 &front = camera.front();
    const auto raycastTarget = [&] {
      return cpuWorld->raycast({eye.x, eye.y, eye.z}, {front.x, front.y, front.z}, BLOCK_REACH);
    };
    std::optional<cpuworld::BlockHit> hit = raycastTarget();
    const auto breakTarget = [&] {
      cpuWorld->breakBlock(hit->block[0], hit->block[1], hit->block[2]);
      destroyTarget.reset();
      destroyProgress = 0.0f;
      hit = raycastTarget();
    };
    const auto placeOnTarget = [&] {
      if (!hit || hit->normal == cpuworld::BlockXYZ{} || selectedBlockIndex >= blockRegistry.blocks.size()) {
        return;
      }
      const cpuworld::BlockXYZ cell{hit->block[0] + hit->normal[0], hit->block[1] + hit->normal[1],
                                    hit->block[2] + hit->normal[2]};
      if (!player.overlapsBlock(cell[0], cell[1], cell[2])) {
        const auto blockId = static_cast<uint16_t>(blockRegistry.blocks[selectedBlockIndex].id);
        if (cpuWorld->placeBlock(cell[0], cell[1], cell[2], blockId)) {
          hit = raycastTarget();
        }
      }
    };

    // Clicks act at once (Minecraft's startDestroyBlock / useItemOn), holding acts in ticks.
    if (breakClicked && hit && canEdit) {
      if (gameMode == Player::GameMode::Creative || destroyProgressPerTick(hit->block) >= 1.0f) {
        breakTarget();
        destroyDelayTicks = gameMode == Player::GameMode::Creative ? BLOCK_DESTROY_DELAY_TICKS : 0;
      } else {
        destroyTarget = hit->block;
        destroyProgress = 0.0f;
      }
    }
    if (placeClicked && canEdit) {
      placeOnTarget();
      useDelayTicks = BLOCK_USE_DELAY_TICKS;
    }

    const bool breakHeld = camera.cursorCaptured() && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool placeHeld = camera.cursorCaptured() && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (!breakHeld) {
      destroyTarget.reset();
      destroyProgress = 0.0f;
    }
    interactionTickAccumulator = std::min(interactionTickAccumulator + static_cast<double>(frameDeltaSeconds),
                                          10.0 * INTERACTION_TICK_SECONDS);
    while (interactionTickAccumulator >= INTERACTION_TICK_SECONDS) {
      interactionTickAccumulator -= INTERACTION_TICK_SECONDS;
      if (useDelayTicks > 0) {
        --useDelayTicks;
      } else if (placeHeld && canEdit) {
        placeOnTarget();
        useDelayTicks = BLOCK_USE_DELAY_TICKS;
      }
      if (destroyDelayTicks > 0) {
        --destroyDelayTicks;
        continue;
      }
      if (!breakHeld || !hit || !canEdit) {
        continue;
      }
      if (gameMode == Player::GameMode::Creative) {
        breakTarget();
        destroyDelayTicks = BLOCK_DESTROY_DELAY_TICKS;
        continue;
      }
      if (destroyTarget != hit->block) { // looked at another block: start over
        destroyTarget = hit->block;
        destroyProgress = 0.0f;
      }
      destroyProgress += destroyProgressPerTick(hit->block);
      if (destroyProgress >= 1.0f) {
        breakTarget();
        destroyDelayTicks = BLOCK_DESTROY_DELAY_TICKS;
      }
    }
    if (hit) {
      outlineTarget = hit->block;
    }
  }

  /**
   * Survival destroy progress per tick on a block, by hand (Minecraft's getDestroyProgress): 1 / (hardness * 30),
   * or 1 / (hardness * 100) for blocks that need a tool; 5x slower with the eyes underwater or off the ground.
   *
   * @return progress per tick; >= 1 breaks instantly.
   */
  float destroyProgressPerTick(const cpuworld::BlockXYZ &cell) const {
    const uint16_t id = cpuWorld->blockId(cell[0], cell[1], cell[2]);
    const auto block = std::ranges::find_if(blockRegistry.blocks, [&](const blocks::Block &b) { return b.id == id; });
    if (block == blockRegistry.blocks.end() || block->hardness <= 0.0f) {
      return 1.0f;
    }
    float speed = 1.0f;
    if (player.eyesUnderwater()) {
      speed /= 5.0f;
    }
    if (!player.grounded()) {
      speed /= 5.0f;
    }
    return speed / block->hardness / (block->requiresTool ? 100.0f : 30.0f);
  }

  /** Create the CPU world with the registry's terrain block IDs and spawn the player (ENGINE_SPAWN overrides the spot). */
  void createCpuWorld() {
    cpuworld::TerrainBlockIds blockIds;
    for (size_t layer = 0; layer < blockIds.top.size(); ++layer) {
      blockIds.top[layer] = static_cast<uint16_t>(terrainLayerBlocks[layer]);
      blockIds.filler[layer] = static_cast<uint16_t>(terrainLayerFillerBlocks[layer]);
      blockIds.deep[layer] = static_cast<uint16_t>(terrainLayerDeepBlocks[layer]);
    }
    blockIds.ocean = static_cast<uint16_t>(blockRegistry.idOf("ocean"));
    blockIds.log = static_cast<uint16_t>(treeLogBlock);
    blockIds.leaves = static_cast<uint16_t>(treeLeavesBlock);
    oceanBlockId = blockIds.ocean;
    cpuWorld = std::make_unique<cpuworld::CpuWorld>(blockIds);
    // ENGINE_SPAWN="x,z[,above]" spawns over world block (x, z), `above` blocks over the ground (testing).
    int32_t spawnX = 0;
    int32_t spawnZ = 0;
    int32_t spawnAbove = 0;
    if (const char *env = std::getenv("ENGINE_SPAWN"); env != nullptr) {
      std::sscanf(env, "%d,%d,%d", &spawnX, &spawnZ, &spawnAbove);
    }
    const int32_t spawnHeight =
        std::max(static_cast<int32_t>(std::floor(terrain::sampleHeight(spawnX, spawnZ))), world::kMinTerrainHeight) +
        spawnAbove;
    player.spawn({spawnX + 0.5, static_cast<double>(spawnHeight), spawnZ + 0.5});
    if (spawnAbove > 0) {
      player.setSpectator(); // stay up there (no gravity)
    }
  }

  /**
   * Map key of a chunk column: cx in the high 32 bits, cz in the low.
   *
   * @param cx Chunk column x, in chunks.
   * @param cz Chunk column z, in chunks.
   * @return the packed key.
   */
  static uint64_t cpuColumnKey(int32_t cx, int32_t cz) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32u) | static_cast<uint32_t>(cz);
  }

  /**
   * Copy a chunk's opaque and translucent clusters into this frame's staging buffer and queue their copies to its slot.
   *
   * @param frame Frame in flight whose staging buffer is used.
   */
  void stageCpuClusters(const CpuChunkAllocation &chunk, uint32_t frame) {
    const vk::DeviceSize slotOffset = QUAD_CLUSTER_SIZE * QUAD_CLUSTERS_PER_CHUNK * chunk.slot;
    std::memcpy(cpuStagingMapped[frame] + cpuStagingUsed, chunk.clusters.data(), sizeof(chunk.clusters));
    cpuClusterCopies.push_back({cpuStagingUsed, slotOffset, sizeof(chunk.clusters)});
    cpuStagingUsed += sizeof(chunk.clusters);
    std::memcpy(cpuStagingMapped[frame] + cpuStagingUsed, chunk.translucentClusters.data(),
                sizeof(chunk.translucentClusters));
    cpuTranslucentClusterCopies.push_back({cpuStagingUsed, slotOffset, sizeof(chunk.translucentClusters)});
    cpuStagingUsed += sizeof(chunk.translucentClusters);
  }

  /**
   * Per frame, after updateVisibleClipmap: move the CPU square with the player, show exactly the columns of the
   * applied clipmap's square, turn finished / unloaded columns into this frame's transfers, and offer the ready square
   * to the clipmap once it is uploaded.
   *
   * @param eye Camera (or frozen) position, in world blocks.
   * @param frame Frame in flight whose staging buffer is used.
   */
  void updateCpuWorld(const glm::vec3 &eye, uint32_t frame) {
    const cpuworld::ChunkXZ desiredCenter{cpuworld::floorDiv(static_cast<int32_t>(std::floor(eye.x)), CHUNK_SIZE),
                                    cpuworld::floorDiv(static_cast<int32_t>(std::floor(eye.z)), CHUNK_SIZE)};
    const std::optional<cpuworld::ChunkXZ> appliedCenter =
        lastAppliedClipmapView ? lastAppliedClipmapView->cpuRegionCenter : std::nullopt;
    const auto shouldShow = [&](int32_t cx, int32_t cz) {
      return appliedCenter && cpuworld::CpuWorld::inSquare(*appliedCenter, cx, cz);
    };
    std::vector<cpuworld::ChunkXZ> &pinned = pinnedCpuCenters;
    pinned.clear();
    for (const auto &center : {appliedCenter, cpuAdvertisedCenter}) {
      if (center) {
        pinned.push_back(*center);
      }
    }
    for (const auto &[sequence, center] : inFlightCpuCenters) {
      pinned.push_back(center);
    }
    cpuWorld->update(desiredCenter, pinned);
    for (auto &event : cpuWorld->takeMeshEvents()) {
      cpuPendingEvents.push_back(std::move(event));
    }

    cpuQuadCopies.clear();
    cpuOffsetCopies.clear();
    cpuClusterCopies.clear();
    cpuTranslucentClusterCopies.clear();
    cpuClearedSlots.clear();
    cpuStagingUsed = 0;
    std::byte *staging = cpuStagingMapped[frame];

    // Visibility first, unconditionally: it must change in the same frame as the applied clipmap. Only clusters are
    // written (768 bytes per chunk), so it always fits in staging.
    for (auto &[columnKey, column] : cpuColumns) {
      const bool show = shouldShow(column.cx, column.cz);
      if (show == column.visible) {
        continue;
      }
      column.visible = show;
      for (const CpuChunkAllocation &chunk : column.chunks) {
        if (show) {
          stageCpuClusters(chunk, frame);
        } else {
          cpuClearedSlots.push_back(chunk.slot);
        }
      }
    }

    while (!cpuPendingEvents.empty()) {
      cpuworld::ColumnMeshEvent &event = cpuPendingEvents.front();
      vk::DeviceSize bytes = 0;
      for (const cpuworld::ChunkMesh &mesh : event.chunks) {
        bytes += 2u * sizeof(uint64_t) * mesh.quads.size() + sizeof(mesh.clusters) + sizeof(mesh.translucentClusters);
      }
      if (cpuStagingUsed != 0u && cpuStagingUsed + bytes > CPU_STAGING_BYTES) {
        break; // next frame
      }
      if (cpuStagingUsed + bytes > CPU_STAGING_BYTES) {
        throw std::runtime_error("CPU chunk column exceeds the staging buffer");
      }

      const uint64_t columnKey = cpuColumnKey(event.cx, event.cz);
      const auto releaseChunk = [&](const CpuChunkAllocation &chunk) {
        cpuClearedSlots.push_back(chunk.slot);
        cpuQuadRanges.release(chunk.quadOffset, chunk.quadCount);
        cpuChunkSlots.release(chunk.slot);
        cpuQuadCountDirty = true;
      };
      // Allocate a quad range and slot for a chunk mesh and stage its quads, offsets and (if shown) clusters.
      const auto uploadChunk = [&](CpuColumnGpu &column, cpuworld::ChunkMesh &mesh) {
        const uint32_t quadCount = static_cast<uint32_t>(mesh.quads.size());
        if (quadCount == 0u) {
          return;
        }
        const std::optional<uint32_t> quadOffset = cpuQuadRanges.allocate(quadCount);
        if (!quadOffset || cpuChunkSlots.highWater() >= CPU_MAX_CHUNK_SLOTS) {
          if (quadOffset) {
            cpuQuadRanges.release(*quadOffset, quadCount);
          }
          if (!cpuCapacityWarned) {
            std::clog << "CPU chunk buffers are full; some near chunks are not drawn\n";
            cpuCapacityWarned = true;
          }
          return;
        }
        // counts01/23/45: two 16-bit per-face quad counts each.
        uint32_t opaqueQuadCount = 0;
        for (const cpuworld::QuadCluster &cluster : mesh.clusters) {
          opaqueQuadCount += (cluster.counts01 & 0xffffu) + (cluster.counts01 >> 16u) + (cluster.counts23 & 0xffffu) +
                             (cluster.counts23 >> 16u) + (cluster.counts45 & 0xffffu) + (cluster.counts45 >> 16u);
        }
        CpuChunkAllocation chunk{mesh.cy,       cpuChunkSlots.allocate(), *quadOffset,    quadCount,
                                 mesh.clusters, mesh.translucentClusters, opaqueQuadCount};
        for (cpuworld::QuadCluster &cluster : chunk.clusters) {
          cluster.base += *quadOffset;
        }
        for (cpuworld::QuadCluster &cluster : chunk.translucentClusters) {
          cluster.base += *quadOffset;
        }

        const vk::DeviceSize quadBytes = sizeof(uint64_t) * quadCount;
        std::memcpy(staging + cpuStagingUsed, mesh.quads.data(), quadBytes);
        cpuQuadCopies.push_back({cpuStagingUsed, sizeof(uint64_t) * *quadOffset, quadBytes});
        cpuStagingUsed += quadBytes;
        std::fill_n(reinterpret_cast<uint64_t *>(staging + cpuStagingUsed), quadCount, mesh.chunkOffset);
        cpuOffsetCopies.push_back({cpuStagingUsed, sizeof(uint64_t) * *quadOffset, quadBytes});
        cpuStagingUsed += quadBytes;
        // Hidden chunks leave their slot zeroed (freed slots are cleared, new ones start at zero).
        if (column.visible) {
          stageCpuClusters(chunk, frame);
        }
        column.chunks.push_back(chunk);
        cpuQuadCountDirty = true;
      };

      const auto existing = cpuColumns.find(columnKey);
      if (event.partial) {
        // Block edit: swap only the remeshed chunks. A column already unloaded again has nothing to update.
        if (existing != cpuColumns.end()) {
          CpuColumnGpu &column = existing->second;
          for (cpuworld::ChunkMesh &mesh : event.chunks) {
            std::erase_if(column.chunks, [&](const CpuChunkAllocation &chunk) {
              if (chunk.cy != mesh.cy) {
                return false;
              }
              releaseChunk(chunk);
              return true;
            });
            uploadChunk(column, mesh);
          }
        }
      } else {
        if (existing != cpuColumns.end()) {
          for (const CpuChunkAllocation &chunk : existing->second.chunks) {
            releaseChunk(chunk);
          }
          cpuColumns.erase(existing);
        }
        if (!event.removed) {
          CpuColumnGpu column{event.cx, event.cz, shouldShow(event.cx, event.cz), {}};
          for (cpuworld::ChunkMesh &mesh : event.chunks) {
            uploadChunk(column, mesh);
          }
          cpuColumns.emplace(columnKey, std::move(column));
        }
      }
      cpuPendingEvents.pop_front();
    }

    // Offer the ready square only when everything handed over so far is on its way to the GPU.
    if (cpuPendingEvents.empty()) {
      cpuAdvertisedCenter = cpuWorld->readyCenter();
    }
  }

  /** Record this frame's CPU geometry transfers (clears of hidden / freed clusters, then staging copies) before its draws. */
  void recordCpuUploads(vk::raii::CommandBuffer &commandBuffer) {
    if (cpuQuadCopies.empty() && cpuClusterCopies.empty() && cpuClearedSlots.empty() && !cpuQuadCountDirty) {
      return;
    }
    const vk::MemoryBarrier2 afterDraws{.srcStageMask = vk::PipelineStageFlagBits2::eTaskShaderEXT |
                                                        vk::PipelineStageFlagBits2::eMeshShaderEXT |
                                                        vk::PipelineStageFlagBits2::eVertexShader,
                                        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                                        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &afterDraws});

    for (const uint32_t slot : cpuClearedSlots) {
      for (const vk::Buffer buffer : {*cpuClusterBuffer, *cpuTranslucentClusterBuffer}) {
        commandBuffer.fillBuffer(buffer, QUAD_CLUSTER_SIZE * QUAD_CLUSTERS_PER_CHUNK * slot,
                                 QUAD_CLUSTER_SIZE * QUAD_CLUSTERS_PER_CHUNK, 0u);
      }
    }
    if (!cpuClearedSlots.empty()) {
      // A cleared slot may be reused by this frame's copies.
      const vk::MemoryBarrier2 fillsDone{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                         .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                         .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                         .dstAccessMask = vk::AccessFlagBits2::eTransferWrite};
      commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &fillsDone});
    }
    const vk::Buffer staging = *cpuStagingBuffers[frameIndex];
    if (!cpuQuadCopies.empty()) {
      commandBuffer.copyBuffer(staging, *cpuQuadBuffer, cpuQuadCopies);
      commandBuffer.copyBuffer(staging, *cpuChunkOffsetBuffer, cpuOffsetCopies);
    }
    if (!cpuClusterCopies.empty()) {
      commandBuffer.copyBuffer(staging, *cpuClusterBuffer, cpuClusterCopies);
      commandBuffer.copyBuffer(staging, *cpuTranslucentClusterBuffer, cpuTranslucentClusterCopies);
    }
    if (cpuQuadCountDirty) {
      const std::array<uint32_t, 2> quadCount{cpuQuadRanges.highWater(), 0u};
      commandBuffer.updateBuffer<uint32_t>(*cpuQuadCountBuffer, 0, quadCount);
      cpuQuadCountDirty = false;
    }

    const vk::MemoryBarrier2 uploadsDone{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                         .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                         .dstStageMask = vk::PipelineStageFlagBits2::eTaskShaderEXT |
                                                         vk::PipelineStageFlagBits2::eMeshShaderEXT |
                                                         vk::PipelineStageFlagBits2::eVertexShader,
                                         .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &uploadsDone});
    cpuQuadCopies.clear();
    cpuOffsetCopies.clear();
    cpuClusterCopies.clear();
    cpuTranslucentClusterCopies.clear();
    cpuClearedSlots.clear();
  }

  /**
   * Submit an async clipmap cull when the view leaves the last submitted one's cover, apply a finished result when
   * usable, and fall back to a synchronous build when the applied set is too stale.
   *
   * @param view This frame's (or the frozen) clipmap view.
   */
  void updateVisibleClipmap(const ClipmapView &view) {
    if (!lastSubmittedClipmapView || !clipmapViewCanCover(*lastSubmittedClipmapView, view)) {
      latestSubmittedClipmapSequence = clipmapCuller.submit(view);
      if (view.cpuRegionCenter) {
        inFlightCpuCenters.emplace_back(latestSubmittedClipmapSequence, *view.cpuRegionCenter);
        constexpr size_t maxPinned = 8; // the culler only runs the newest pending job, so older ones mostly vanish
        while (inFlightCpuCenters.size() > maxPinned) {
          inFlightCpuCenters.pop_front();
        }
      }
      lastSubmittedClipmapView = view;
    }
    // A result built around a CPU square that isn't fully loaded and uploaded any more would leave a hole. Once
    // applied, its square is pinned in cpuWorld (updateCpuWorld runs after this).
    const auto cpuCenterUsable = [&](const std::optional<cpuworld::ChunkXZ> &center) {
      return !center || center == cpuAdvertisedCenter ||
             (lastAppliedClipmapView && center == lastAppliedClipmapView->cpuRegionCenter) ||
             (cpuPendingEvents.empty() && cpuWorld->squareLoaded(*center));
    };
    if (auto completion = clipmapCuller.takeCompletion();
        completion && completion->sequence > lastAppliedClipmapSequence &&
        clipmapViewCanCover(completion->view, view, kClipmapStaleCover) &&
        cpuCenterUsable(completion->view.cpuRegionCenter)) {
      lastAppliedClipmapSequence = completion->sequence;
      while (!inFlightCpuCenters.empty() && inFlightCpuCenters.front().first <= completion->sequence) {
        inFlightCpuCenters.pop_front();
      }
      lastAppliedClipmapView = completion->view;
      if (completion->result.chunks != visibleChunkSet || completion->result.neighborLods != visibleChunkNeighborLods) {
        visibleChunkSet = std::move(completion->result.chunks);
        visibleChunkNeighborLods = std::move(completion->result.neighborLods);
        chunkBuffersDirty = true;
      }
    }
    // Synchronous fallback only beyond the stale tolerance (first frame, FOV/LOD change, teleport, fast turn). Normal
    // movement keeps the previous set until the async result lands: rebuilding here stalled frames for several ms.
    if (!lastAppliedClipmapView || !clipmapViewCanCover(*lastAppliedClipmapView, view, kClipmapStaleCover)) {
      ClipmapResult result = buildVisibleClipmap(view);
      lastAppliedClipmapSequence = latestSubmittedClipmapSequence;
      inFlightCpuCenters.clear();
      lastAppliedClipmapView = view;
      if (result.chunks != visibleChunkSet || result.neighborLods != visibleChunkNeighborLods) {
        visibleChunkSet = std::move(result.chunks);
        visibleChunkNeighborLods = std::move(result.neighborLods);
        chunkBuffersDirty = true;
      }
    }
  }

  struct ClipmapCoverTolerance {
    float maxPositionDelta;
    float minForwardDot;
    bool sameCpuRegion;
  };
  // Submit a new async cull past 2 chunks / 5 degrees (or on a new CPU square); keep the previous set while it runs,
  // up to 12 chunks / 7.5 degrees, then build synchronously. The position limit is loose (a lagging set is only
  // slightly coarser near the camera); the direction limit is tight (the frustum-culled set has nothing off-view).
  // minForwardDot values are cos(5 deg) and cos(7.5 deg).
  static constexpr ClipmapCoverTolerance kClipmapSubmitCover{static_cast<float>(CHUNK_SIZE) * 2.0f, 0.9961947f, true};
  static constexpr ClipmapCoverTolerance kClipmapStaleCover{static_cast<float>(CHUNK_SIZE) * 12.0f, 0.9914449f, false};

  /**
   * Whether a clipmap built for `builtView` can still stand in for `currentView`.
   *
   * @param tolerance Allowed camera movement and turn, and whether the CPU square must match.
   * @return true if the views match in everything but position / direction, and those are within `tolerance`.
   */
  static bool clipmapViewCanCover(const ClipmapView &builtView, const ClipmapView &currentView,
                                  const ClipmapCoverTolerance &tolerance = kClipmapSubmitCover) {
    const float maxPositionDelta = tolerance.maxPositionDelta;
    const float minForwardDot = tolerance.minForwardDot;
    const float dx = builtView.position[0] - currentView.position[0];
    const float dy = builtView.position[1] - currentView.position[1];
    const float dz = builtView.position[2] - currentView.position[2];
    const float positionDeltaSquared = dx * dx + dy * dy + dz * dz;
    const float forwardDot = builtView.forward[0] * currentView.forward[0] +
                             builtView.forward[1] * currentView.forward[1] +
                             builtView.forward[2] * currentView.forward[2];
    // Without frustum culling and unzoomed, the result doesn't depend on the view direction.
    const bool directionDependent =
        builtView.frustumCull || builtView.lodDistanceSquared != builtView.outsideLodDistanceSquared;
    const bool directionCovered = !directionDependent || forwardDot >= minForwardDot;
    return positionDeltaSquared <= maxPositionDelta * maxPositionDelta && directionCovered &&
           builtView.frustumCull == currentView.frustumCull &&
           builtView.lodDistanceSquared == currentView.lodDistanceSquared &&
           builtView.outsideLodDistanceSquared == currentView.outsideLodDistanceSquared &&
           builtView.verticalFovRadians == currentView.verticalFovRadians &&
           builtView.aspectRatio == currentView.aspectRatio &&
           (!tolerance.sameCpuRegion || builtView.cpuRegionCenter == currentView.cpuRegionCenter);
  }
#pragma endregion

#pragma region Descriptor Setup
  // Scene descriptor sets per frame: GPU clipmap, CPU chunks, and the translucent clusters of each.
  static constexpr uint32_t SCENE_SET_KINDS = 4u;

  /** Create the shared descriptor pool for the scene, TAA and chunk compute sets (update-after-bind for the textures). */
  void createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 4> poolSize{
        {{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 48u * MAX_FRAMES_IN_FLIGHT},
         {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 2u * MAX_FRAMES_IN_FLIGHT},
         {.type = vk::DescriptorType::eSampledImage,
          // Per scene set: the textures, Hi-Z and the opaque depth; then the TAA sets' three images.
          .descriptorCount =
              SCENE_SET_KINDS * MAX_FRAMES_IN_FLIGHT * (static_cast<uint32_t>(textureImageViews.size()) + 2u) +
              3u * MAX_FRAMES_IN_FLIGHT},
         {.type = vk::DescriptorType::eSampler, .descriptorCount = SCENE_SET_KINDS * MAX_FRAMES_IN_FLIGHT}}};
    vk::DescriptorPoolCreateInfo poolInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                                                   vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
                                          .maxSets = (SCENE_SET_KINDS + 2u) * MAX_FRAMES_IN_FLIGHT,
                                          .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
                                          .pPoolSizes = poolSize.data()};
    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
  }

  /**
   * Allocate and write the per-frame scene sets (all four kinds) and chunk compute sets. Binding 7 (Hi-Z) is
   * written separately by createHzbResources.
   */
  void createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    std::vector<uint32_t> textureDescriptorCounts(MAX_FRAMES_IN_FLIGHT,
                                                  static_cast<uint32_t>(textureImageViews.size()));
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{
        .descriptorSetCount = static_cast<uint32_t>(textureDescriptorCounts.size()),
        .pDescriptorCounts = textureDescriptorCounts.data()};
    vk::DescriptorSetAllocateInfo allocInfo{.descriptorPool = descriptorPool,
                                            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                                            .pSetLayouts = layouts.data()};
    allocInfo.pNext = &variableCountInfo;

    descriptorSets.clear();
    descriptorSets = device.allocateDescriptorSets(allocInfo);
    cpuDescriptorSets.clear();
    cpuDescriptorSets = device.allocateDescriptorSets(allocInfo);
    cpuTranslucentDescriptorSets.clear();
    cpuTranslucentDescriptorSets = device.allocateDescriptorSets(allocInfo);
    gpuTranslucentDescriptorSets.clear();
    gpuTranslucentDescriptorSets = device.allocateDescriptorSets(allocInfo);

    std::vector<vk::DescriptorImageInfo> textureImageInfos;
    textureImageInfos.reserve(textureImageViews.size());
    for (const auto &imageView : textureImageViews) {
      textureImageInfos.push_back(
          {.sampler = nullptr, .imageView = imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal});
    }

    for (size_t setIndex = 0; setIndex < SCENE_SET_KINDS * MAX_FRAMES_IN_FLIGHT; setIndex++) {
      const size_t i = setIndex % MAX_FRAMES_IN_FLIGHT;
      // Kinds: 0 GPU clipmap, 1 CPU chunks, 2 CPU translucent, 3 GPU translucent (translucent: binding 6 differs).
      const size_t kind = setIndex / MAX_FRAMES_IN_FLIGHT;
      const bool cpuSet = kind == 1u || kind == 2u;
      const bool translucentSet = kind >= 2u;
      const vk::DescriptorSet set = kind == 0u   ? *descriptorSets[i]
                                    : kind == 1u ? *cpuDescriptorSets[i]
                                    : kind == 2u ? *cpuTranslucentDescriptorSets[i]
                                                 : *gpuTranslucentDescriptorSets[i];
      vk::DescriptorBufferInfo uniformBufferInfo{
          .buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};
      vk::DescriptorBufferInfo quadBufferInfo{.buffer = cpuSet ? *cpuQuadBuffer : *quadBuffer,
                                              .offset = 0,
                                              .range = sizeof(uint64_t) * (cpuSet ? CPU_MAX_QUADS : MAX_QUAD_COUNT)};
      vk::DescriptorBufferInfo chunkOffsetBufferInfo{.buffer = cpuSet ? *cpuChunkOffsetBuffer : *chunkOffsetBuffer,
                                                     .offset = 0,
                                                     .range =
                                                         sizeof(uint64_t) * (cpuSet ? CPU_MAX_QUADS : MAX_QUAD_COUNT)};
      vk::DescriptorImageInfo samplerInfo{
          .sampler = textureSampler, .imageView = nullptr, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

      vk::DescriptorBufferInfo blockFaceBufferInfo{.buffer = blockFaceBuffer, .offset = 0, .range = vk::WholeSize};
      vk::DescriptorBufferInfo quadCountBufferInfo{
          .buffer = cpuSet ? *cpuQuadCountBuffer : *quadCountBuffer, .offset = 0, .range = sizeof(uint32_t)};
      vk::DescriptorBufferInfo clusterBufferInfo{
          .buffer = cpuSet ? (translucentSet ? *cpuTranslucentClusterBuffer : *cpuClusterBuffer)
                           : (translucentSet ? *waterClusterBuffer : *quadClusterBuffer),
          .offset = 0,
          .range = vk::WholeSize};
      std::array<vk::WriteDescriptorSet, 8> descriptorWrites{{
          {.dstSet = set,
           .dstBinding = 0,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &quadBufferInfo},
          {.dstSet = set,
           .dstBinding = 1,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eUniformBuffer,
           .pBufferInfo = &uniformBufferInfo},
          {.dstSet = set,
           .dstBinding = 2,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eSampler,
           .pImageInfo = &samplerInfo},
          {.dstSet = set,
           .dstBinding = 3,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &chunkOffsetBufferInfo},
          {.dstSet = set,
           .dstBinding = 11,
           .dstArrayElement = 0,
           .descriptorCount = static_cast<uint32_t>(textureImageInfos.size()),
           .descriptorType = vk::DescriptorType::eSampledImage,
           .pImageInfo = textureImageInfos.data()},
          {.dstSet = set,
           .dstBinding = 5,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &blockFaceBufferInfo},
          {.dstSet = set,
           .dstBinding = 4,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &quadCountBufferInfo},
          {.dstSet = set,
           .dstBinding = 6,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &clusterBufferInfo},
      }};
      // Binding 7 (Hi-Z) is written by createHzbResources, on every swapchain rebuild.
      device.updateDescriptorSets(descriptorWrites, {});
    }

    std::vector<vk::DescriptorSetLayout> computeLayouts(MAX_FRAMES_IN_FLIGHT, chunkComputeDescriptorSetLayout);
    chunkComputeDescriptorSets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{.descriptorPool = descriptorPool,
                                      .descriptorSetCount = static_cast<uint32_t>(computeLayouts.size()),
                                      .pSetLayouts = computeLayouts.data()});

    const vk::DeviceSize coordinateBufferSize = sizeof(world::ChunkLod) * MAX_HORIZONTAL_CHUNK_COUNT;
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      std::array<vk::DescriptorBufferInfo, 19> bufferInfos{{
          // 34 = HEIGHT_STRIDE (CHUNK_SIZE + 2) in createChunkBuffers.
          {.buffer = chunkHeightBuffer, .offset = 0, .range = sizeof(int32_t) * MAX_HORIZONTAL_CHUNK_COUNT * 34u * 34u},
          {.buffer = chunkCoordinateBuffers[i], .offset = 0, .range = coordinateBufferSize},
          {.buffer = quadBuffer, .offset = 0, .range = sizeof(uint64_t) * MAX_QUAD_COUNT},
          {.buffer = chunkOffsetBuffer, .offset = 0, .range = sizeof(uint64_t) * MAX_QUAD_COUNT},
          {.buffer = quadCountBuffer, .offset = 0, .range = 2u * sizeof(uint32_t)},
          {.buffer = chunkWorkBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = quadClusterBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = vertexDrawArgumentsBuffer, .offset = 0, .range = sizeof(vk::DrawIndexedIndirectCommand)},
          {.buffer = chunkNeighborLodBuffers[i], .offset = 0, .range = sizeof(uint32_t) * MAX_HORIZONTAL_CHUNK_COUNT},
          {.buffer = columnListBuffers[i], .offset = 0, .range = vk::WholeSize},
          {.buffer = retireListBuffers[i], .offset = 0, .range = vk::WholeSize},
          {.buffer = quadCompactBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = workPoolBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = freeWorkSlotBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = columnWorkSlotBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = columnHeightRangeBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = slotOwnerBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = relocationScratchBuffer, .offset = 0, .range = vk::WholeSize},
          {.buffer = waterClusterBuffer, .offset = 0, .range = vk::WholeSize},
      }};
      std::array<vk::WriteDescriptorSet, 19> writes{};
      for (uint32_t binding = 0; binding < writes.size(); ++binding) {
        writes[binding] = {.dstSet = chunkComputeDescriptorSets[i],
                           .dstBinding = binding,
                           .dstArrayElement = 0,
                           .descriptorCount = 1,
                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                           .pBufferInfo = &bufferInfos[binding]};
      }
      device.updateDescriptorSets(writes, {});
    }
  }

  /**
   * Write the TAA sets (one per history image read) and the translucent scene sets' resolved-depth binding. Done once
   * per swapchain (re)creation: the render targets are shared by all frames.
   */
  void createTaaDescriptorSets() {
    static_assert(MAX_FRAMES_IN_FLIGHT == 2, "the descriptor pool holds MAX_FRAMES_IN_FLIGHT TAA sets, one per history");
    std::vector<vk::DescriptorSetLayout> layouts(taaHistoryImages.size(), taaDescriptorSetLayout);
    taaDescriptorSets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{.descriptorPool = descriptorPool,
                                      .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                                      .pSetLayouts = layouts.data()});
    for (uint32_t historyReadIndex = 0; historyReadIndex < taaDescriptorSets.size(); ++historyReadIndex) {
      const std::array<vk::DescriptorImageInfo, 4> imageInfos{
          {{.imageView = sceneColorImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
           {.imageView = taaHistoryImageViews[historyReadIndex],
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
           {.imageView = motionImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
           {.sampler = taaSampler}}};
      std::array<vk::WriteDescriptorSet, 4> writes{};
      for (uint32_t binding = 0; binding < writes.size(); ++binding) {
        writes[binding] = {.dstSet = taaDescriptorSets[historyReadIndex],
                           .dstBinding = binding,
                           .dstArrayElement = 0,
                           .descriptorCount = 1,
                           .descriptorType =
                               binding == 3u ? vk::DescriptorType::eSampler : vk::DescriptorType::eSampledImage,
                           .pImageInfo = &imageInfos[binding]};
      }
      device.updateDescriptorSets(writes, {});
    }

    // Translucent scene sets: the resolved opaque depth, for the water fog.
    const vk::DescriptorImageInfo depthInfo{.imageView = resolvedDepthImageView,
                                            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    std::vector<vk::WriteDescriptorSet> depthWrites;
    for (const auto *sets : {&cpuTranslucentDescriptorSets, &gpuTranslucentDescriptorSets}) {
      for (const auto &set : *sets) {
        depthWrites.push_back({.dstSet = *set,
                               .dstBinding = 8,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eSampledImage,
                               .pImageInfo = &depthInfo});
      }
    }
    device.updateDescriptorSets(depthWrites, {});
  }

#pragma endregion
#pragma endregion

#pragma region Command Setup
#pragma region Single-Time Submit
  /** Allocate and begin a one-time-submit command buffer. */
  vk::raii::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
    vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo);

    return commandBuffer;
  }

  /** End, submit and wait for a command buffer from beginSingleTimeCommands. */
  void endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
  }
#pragma endregion

#pragma region Buffer Helpers
  /** Copy `size` bytes between buffers and wait for it. */
  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size) {
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
    commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size = size});
    endSingleTimeCommands(std::move(commandBuffer));
  }

  /**
   * Find a memory type allowed by `typeFilter` with all of `properties`.
   *
   * @param typeFilter Bit mask of acceptable memory type indices (memoryTypeBits).
   * @return the memory type index; throws if none qualifies.
   */
  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }
#pragma endregion

#pragma region Command Allocation
  /** Allocate a command buffer per frame in flight. */
  void createCommandBuffers() {
    commandBuffers.clear();
    vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool,
                                            .level = vk::CommandBufferLevel::ePrimary,
                                            .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }
#pragma endregion

#pragma region Command Recording
  /**
   * Frustum-cull the CPU-meshed chunks (each whole 32^3 cube against 5 planes, no far plane) into runs of visible
   * slots, so the mesh path only launches task groups over those. Gaps of up to 2 slots are merged to keep the draw
   * count down; the task shader's per-cluster test drops what slips through.
   */
  void buildCpuVisibleSlotRuns() {
    cpuVisibleSlotRuns.clear();
    const uint32_t slotCount = cpuChunkSlots.highWater();
    if (slotCount == 0u) {
      return;
    }
    cpuSlotVisible.assign(slotCount, 0u);
    // Planes from the rows of the clip matrix (Gribb-Hartmann), inside when dot(n, p) + d >= 0. Vulkan reverse-Z: the
    // near plane is z <= w. glm is column-major: row i is m[0][i], m[1][i], m[2][i], m[3][i].
    const glm::mat4 &m = cullMvpCpu;
    const auto row = [&](int index) { return glm::vec4(m[0][index], m[1][index], m[2][index], m[3][index]); };
    const std::array<glm::vec4, 5> planes{row(3) + row(0), row(3) - row(0), row(3) + row(1), row(3) - row(1),
                                          row(3) - row(2)};
    constexpr float half = static_cast<float>(CHUNK_SIZE) * 0.5f;
    std::array<float, 5> radius{}; // projected half-extent of the cube on each plane normal
    for (size_t plane = 0; plane < planes.size(); ++plane) {
      radius[plane] = (std::abs(planes[plane].x) + std::abs(planes[plane].y) + std::abs(planes[plane].z)) * half;
    }
    for (const auto &[columnKey, column] : cpuColumns) {
      if (!column.visible) {
        continue; // its clusters are cleared on the GPU anyway
      }
      const float centerX = static_cast<float>(column.cx * CHUNK_SIZE) + half;
      const float centerZ = static_cast<float>(column.cz * CHUNK_SIZE) + half;
      for (const CpuChunkAllocation &chunk : column.chunks) {
        const float centerY = static_cast<float>(chunk.cy * CHUNK_SIZE) + half;
        bool inside = true;
        for (size_t plane = 0; plane < planes.size() && inside; ++plane) {
          inside = planes[plane].x * centerX + planes[plane].y * centerY + planes[plane].z * centerZ +
                       planes[plane].w >=
                   -radius[plane];
        }
        cpuSlotVisible[chunk.slot] = inside ? 1u : 0u;
      }
    }
    constexpr uint32_t maxMergedGap = 2u; // one task group covers 2 slots (32 clusters)
    for (uint32_t slot = 0; slot < slotCount;) {
      if (cpuSlotVisible[slot] == 0u) {
        ++slot;
        continue;
      }
      if (!cpuVisibleSlotRuns.empty()) {
        auto &[first, count] = cpuVisibleSlotRuns.back();
        if (slot - (first + count) <= maxMergedGap) {
          count = slot + 1u - first;
          ++slot;
          continue;
        }
      }
      cpuVisibleSlotRuns.emplace_back(slot, 1u);
      ++slot;
    }
  }

  /**
   * Record the whole frame: CPU uploads, chunk compute, the opaque / cutout / water scene passes, Hi-Z build, TAA
   * resolve to the swapchain image and the overlay.
   *
   * @param imageIndex Acquired swapchain image.
   */
  void recordCommandBuffer(uint32_t imageIndex) {
    auto &commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});
    if (gpuTimingEnabled) {
      commandBuffer.resetQueryPool(*timestampQueryPool, frameIndex * GPU_TIMESTAMP_COUNT, GPU_TIMESTAMP_COUNT);
      timestampsPending[frameIndex] = true;
    }
    writeGpuTimestamp(commandBuffer, TIMESTAMP_FRAME_START);
    recordCpuUploads(commandBuffer);
    writeGpuTimestamp(commandBuffer, TIMESTAMP_UPLOADS);
    recordChunkCompute(commandBuffer);
    writeGpuTimestamp(commandBuffer, TIMESTAMP_COMPUTE);

    const auto colorOutputStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    const auto colorWrite = vk::AccessFlagBits2::eColorAttachmentWrite;
    // Render targets are shared by every frame: each source stage below is the previous frame's last use of that
    // image, so this frame's writes wait for it.
    const auto fragmentShaderStage = vk::PipelineStageFlagBits2::eFragmentShader;
    // MSAA color and motion: last written (and resolved) by the previous water pass.
    transition_image_layout(*multisampleColorImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, colorWrite, colorWrite, colorOutputStage,
                            colorOutputStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(*multisampleMotionImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, colorWrite, colorWrite, colorOutputStage,
                            colorOutputStage, vk::ImageAspectFlagBits::eColor);
    // Scene color and motion resolves: last read by the previous TAA pass.
    transition_image_layout(*sceneColorImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite, fragmentShaderStage,
                            colorOutputStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(*motionImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite, fragmentShaderStage,
                            colorOutputStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(
        *depthImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);
    // Depth resolves run in the color attachment output stage with color attachment access. Last read by the
    // previous frame's water pass and Hi-Z build.
    transition_image_layout(*resolvedDepthImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eDepthAttachmentOptimal, {}, colorWrite,
                            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
                            colorOutputStage, vk::ImageAspectFlagBits::eDepth);
    // Source stage = the acquire semaphore's wait stage, so the transition chains after the acquire (TOP_OF_PIPE could
    // run before the presentation engine is done reading the image).
    transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite, colorOutputStage,
                            colorOutputStage, vk::ImageAspectFlagBits::eColor);
    const uint32_t historyReadIndex = 1u - taaHistoryWriteIndex;
    if (!taaHistoryInitialized) {
      for (auto &historyImage : taaHistoryImages) {
        transition_image_layout(*historyImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, {},
                                vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eTopOfPipe,
                                vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
      }
    }
    transition_image_layout(*taaHistoryImages[taaHistoryWriteIndex], vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::ImageLayout::eColorAttachmentOptimal, vk::AccessFlagBits2::eShaderRead, colorWrite,
                            vk::PipelineStageFlagBits2::eFragmentShader, colorOutputStage,
                            vk::ImageAspectFlagBits::eColor);

    // Sky, or the underwater fog color while the camera is in the ocean (WATER_FOG_COLOR in slang.slang).
    vk::ClearValue clearColor = cameraUnderwater ? vk::ClearColorValue(0.06f, 0.18f, 0.34f, 1.0f)
                                                 : vk::ClearColorValue(0.55f, 0.72f, 0.9f, 1.0f);
    // Reverse-Z: 1 at the near plane, 0 at the far plane (see the projection in updateUniformBuffer).
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(0.0f, 0);

    // Two passes over the same MSAA targets: the opaque pass resolves depth (farthest sample) into resolvedDepthImage;
    // the water pass loads the targets, reads that depth (water fog), and resolves color and motion for TAA.
    std::array<vk::RenderingAttachmentInfo, 2> sceneAttachments{
        {{.imageView = multisampleColorImageView,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = clearColor},
         {.imageView = multisampleMotionImageView,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)}}};

    vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = depthImageView,
                                                       .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                                                       .resolveMode =
                                                           vk::ResolveModeFlagBits::eMin, // farthest sample (reverse-Z)
                                                       .resolveImageView = *resolvedDepthImageView,
                                                       .resolveImageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eClear,
                                                       .storeOp = vk::AttachmentStoreOp::eStore,
                                                       .clearValue = clearDepth};

    vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
                                       .layerCount = 1,
                                       .colorAttachmentCount = static_cast<uint32_t>(sceneAttachments.size()),
                                       .pColorAttachments = sceneAttachments.data(),
                                       .pDepthAttachment = &depthAttachmentInfo};

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               wireframeEnabled
                                   ? (useVertexPulling ? *wireframeVertexPullingPipeline : *wireframePipeline)
                                   : (useVertexPulling ? *vertexPullingPipeline : *graphicsPipeline));
    commandBuffer.setCullMode(frozenLodView ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
    // Negative height flips Y for Vulkan.
    commandBuffer.setViewport(0, vk::Viewport(0.0f, static_cast<float>(swapChainExtent.height),
                                              static_cast<float>(swapChainExtent.width),
                                              -static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    const bool debugView = terrainView == TerrainView::Debug;
    uint32_t geometryPass = GEOMETRY_PASS_OPAQUE;
    // Push GeometryPushConstants for the next draw (in the current geometryPass).
    const auto pushGeometry = [&](uint32_t clusterCount, uint32_t debugTint, uint32_t quadBase = 0u,
                                  uint32_t quadCount = 0u, uint32_t clusterBase = 0u) {
      const GeometryPushConstants push{clusterCount, debugView ? debugTint : DEBUG_TINT_OFF, quadBase, quadCount,
                                       geometryPass, clusterBase};
      commandBuffer.pushConstants(*pipelineLayout, GEOMETRY_PUSH_STAGES, 0, sizeof(push), &push);
    };
    // 32 clusters per task group, spread over x and y to stay under the 65535 group limit per dimension.
    const auto drawClusters = [&](uint32_t clusterCount, uint32_t debugTint, uint32_t clusterBase = 0u) {
      if (clusterCount == 0u) {
        return;
      }
      pushGeometry(clusterCount, debugTint, 0u, 0u, clusterBase);
      const uint32_t taskGroups = (clusterCount + 31u) / 32u;
      commandBuffer.drawMeshTasksEXT(std::min(taskGroups, 65535u), (taskGroups + 65534u) / 65535u, 1);
    };
    // CPU-meshed chunks on the mesh path: only the slot runs that passed the CPU frustum cull.
    if (!useVertexPulling) {
      buildCpuVisibleSlotRuns();
    }
    const auto drawCpuClusters = [&] {
      for (const auto &[first, count] : cpuVisibleSlotRuns) {
        drawClusters(count * QUAD_CLUSTERS_PER_CHUNK, DEBUG_TINT_CPU, first * QUAD_CLUSTERS_PER_CHUNK);
      }
    };
    if (useVertexPulling) {
      commandBuffer.bindIndexBuffer(*quadIndexBuffer, 0, vk::IndexType::eUint16);
    }
    // Terrain draws, recorded once per pass: GPU-meshed clipmap, then CPU-meshed near chunks.
    const auto drawTerrain = [&] {
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                       *descriptorSets[frameIndex], nullptr);
      // GPU-meshed clipmap (LOD terrain, with the hole the CPU square fills).
      if (terrainView != TerrainView::RealOnly) {
        if (useVertexPulling) {
          pushGeometry(0u, DEBUG_TINT_LOD);
          commandBuffer.drawIndexedIndirect(*vertexDrawArgumentsBuffer, 0, 1, sizeof(vk::DrawIndexedIndirectCommand));
        } else {
          // finalizeWork wrote the group count from the work slot high water. Clusters past it (the folded grid
          // overshoots) are zero, so the bound is just the buffer.
          pushGeometry(workPoolCapacity * QUAD_CLUSTERS_PER_CHUNK, DEBUG_TINT_LOD);
          commandBuffer.drawMeshTasksIndirectEXT(*workPoolBuffer, WORK_POOL_TASK_ARGS_OFFSET, 1,
                                                 sizeof(vk::DrawMeshTasksIndirectCommandEXT));
        }
      }
      // CPU-meshed near chunks: same pipelines, their own buffers.
      if (terrainView != TerrainView::LodOnly && cpuChunkSlots.highWater() != 0u) {
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                         *cpuDescriptorSets[frameIndex], nullptr);
        if (useVertexPulling) {
          // One draw per visible chunk (hidden chunks keep their quads in the buffer).
          for (const auto &[columnKey, column] : cpuColumns) {
            if (!column.visible) {
              continue;
            }
            for (const CpuChunkAllocation &chunk : column.chunks) {
              for (uint32_t first = 0; first < chunk.opaqueQuadCount; first += VERTEX_QUADS_PER_INSTANCE) {
                const uint32_t count = std::min(chunk.opaqueQuadCount - first, VERTEX_QUADS_PER_INSTANCE);
                pushGeometry(0u, DEBUG_TINT_CPU, chunk.quadOffset + first, count);
                commandBuffer.drawIndexed(count * 6u, 1, 0, 0, 0);
              }
            }
          }
        } else {
          drawCpuClusters();
        }
      }
    };
    // Opaque blocks first (no discard: early depth testing), then the alpha-tested non-opaque ones against that depth.
    drawTerrain();
    writeGpuTimestamp(commandBuffer, TIMESTAMP_OPAQUE);
    if (!wireframeEnabled) {
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 useVertexPulling ? *cutoutVertexPullingPipeline : *cutoutPipeline);
    }
    geometryPass = GEOMETRY_PASS_CUTOUT;
    drawTerrain();
    commandBuffer.endRendering();
    writeGpuTimestamp(commandBuffer, TIMESTAMP_CUTOUT);

    // Opaque -> water pass: the depth resolve becomes a shader input (water fog, Hi-Z), and the MSAA targets' stores
    // complete before the water pass loads them.
    const vk::ImageMemoryBarrier2 opaqueDepthReady{.srcStageMask = colorOutputStage,
                                                   .srcAccessMask = colorWrite,
                                                   .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader |
                                                                   vk::PipelineStageFlagBits2::eComputeShader,
                                                   .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                                                   .oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                                                   .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                                   .image = *resolvedDepthImage,
                                                   .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}};
    const vk::MemoryBarrier2 attachmentsStored{
        .srcStageMask = colorOutputStage | vk::PipelineStageFlagBits2::eLateFragmentTests,
        .srcAccessMask = colorWrite | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .dstStageMask = colorOutputStage | vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                        vk::PipelineStageFlagBits2::eLateFragmentTests,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead | colorWrite |
                         vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                      .pMemoryBarriers = &attachmentsStored,
                                                      .imageMemoryBarrierCount = 1,
                                                      .pImageMemoryBarriers = &opaqueDepthReady});

    const std::array<vk::RenderingAttachmentInfo, 2> waterAttachments{
        {{.imageView = multisampleColorImageView,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .resolveMode = vk::ResolveModeFlagBits::eAverage,
          .resolveImageView = sceneColorImageView,
          .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eLoad,
          .storeOp = vk::AttachmentStoreOp::eDontCare},
         {.imageView = multisampleMotionImageView,
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .resolveMode = vk::ResolveModeFlagBits::eAverage,
          .resolveImageView = motionImageView,
          .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eLoad,
          .storeOp = vk::AttachmentStoreOp::eDontCare}}};
    // Depth-tested, never written: nothing after this pass reads the MSAA depth.
    const vk::RenderingAttachmentInfo waterDepthAttachment{.imageView = depthImageView,
                                                           .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                                                           .loadOp = vk::AttachmentLoadOp::eLoad,
                                                           .storeOp = vk::AttachmentStoreOp::eDontCare};
    commandBuffer.beginRendering(
        vk::RenderingInfo{.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
                          .layerCount = 1,
                          .colorAttachmentCount = static_cast<uint32_t>(waterAttachments.size()),
                          .pColorAttachments = waterAttachments.data(),
                          .pDepthAttachment = &waterDepthAttachment});
    commandBuffer.setViewport(0, vk::Viewport(0.0f, static_cast<float>(swapChainExtent.height),
                                              static_cast<float>(swapChainExtent.width),
                                              -static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    // Translucent pass (ocean): blended over and depth-tested against every opaque draw.
    if (!wireframeEnabled) {
      geometryPass = GEOMETRY_PASS_TRANSLUCENT;
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 useVertexPulling ? *waterVertexPullingPipeline : *waterPipeline);
      commandBuffer.setCullMode(vk::CullModeFlagBits::eNone);
    }
    if (!wireframeEnabled && terrainView != TerrainView::RealOnly) {
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                       *gpuTranslucentDescriptorSets[frameIndex], nullptr);
      if (useVertexPulling) {
        // Every GPU quad again; vertMain keeps only liquids in this pass.
        pushGeometry(0u, DEBUG_TINT_LOD);
        commandBuffer.drawIndexedIndirect(*vertexDrawArgumentsBuffer, 0, 1, sizeof(vk::DrawIndexedIndirectCommand));
      } else {
        pushGeometry(workPoolCapacity * QUAD_CLUSTERS_PER_CHUNK, DEBUG_TINT_LOD);
        commandBuffer.drawMeshTasksIndirectEXT(*workPoolBuffer, WORK_POOL_TASK_ARGS_OFFSET, 1,
                                               sizeof(vk::DrawMeshTasksIndirectCommandEXT));
      }
    }
    if (terrainView != TerrainView::LodOnly && cpuChunkSlots.highWater() != 0u && !wireframeEnabled) {
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                       *cpuTranslucentDescriptorSets[frameIndex], nullptr);
      if (useVertexPulling) {
        for (const auto &[columnKey, column] : cpuColumns) {
          if (!column.visible) {
            continue;
          }
          for (const CpuChunkAllocation &chunk : column.chunks) {
            for (uint32_t first = chunk.opaqueQuadCount; first < chunk.quadCount; first += VERTEX_QUADS_PER_INSTANCE) {
              const uint32_t count = std::min(chunk.quadCount - first, VERTEX_QUADS_PER_INSTANCE);
              pushGeometry(0u, DEBUG_TINT_CPU, chunk.quadOffset + first, count);
              commandBuffer.drawIndexed(count * 6u, 1, 0, 0, 0);
            }
          }
        }
      } else {
        drawCpuClusters();
      }
      geometryPass = GEOMETRY_PASS_OPAQUE;
    }
    recordOutline(commandBuffer);
    commandBuffer.endRendering();
    writeGpuTimestamp(commandBuffer, TIMESTAMP_WATER);
    if (!useVertexPulling && hzbOcclusionEnabled && !frozenLodView) {
      recordHzb(commandBuffer);
    }
    writeGpuTimestamp(commandBuffer, TIMESTAMP_HZB);

    const auto fragmentStage = vk::PipelineStageFlagBits2::eFragmentShader;
    transition_image_layout(*sceneColorImage, vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal, colorWrite, vk::AccessFlagBits2::eShaderRead,
                            colorOutputStage, fragmentStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(*motionImage, vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal, colorWrite, vk::AccessFlagBits2::eShaderRead,
                            colorOutputStage, fragmentStage, vk::ImageAspectFlagBits::eColor);

    const vk::RenderingAttachmentInfo taaOutputAttachment{.imageView = swapChainImageViews[imageIndex],
                                                          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                          .loadOp = vk::AttachmentLoadOp::eDontCare,
                                                          .storeOp = vk::AttachmentStoreOp::eStore};
    const vk::RenderingAttachmentInfo historyOutputAttachment{.imageView = taaHistoryImageViews[taaHistoryWriteIndex],
                                                              .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                              .loadOp = vk::AttachmentLoadOp::eDontCare,
                                                              .storeOp = vk::AttachmentStoreOp::eStore};
    const std::array<vk::RenderingAttachmentInfo, 2> taaAttachments{taaOutputAttachment, historyOutputAttachment};
    const vk::RenderingInfo taaRenderingInfo{.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
                                             .layerCount = 1,
                                             .colorAttachmentCount = static_cast<uint32_t>(taaAttachments.size()),
                                             .pColorAttachments = taaAttachments.data()};
    commandBuffer.beginRendering(taaRenderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *taaPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, static_cast<float>(swapChainExtent.height),
                                              static_cast<float>(swapChainExtent.width),
                                              -static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *taaPipelineLayout, 0,
                                     *taaDescriptorSets[historyReadIndex], nullptr);
    const TaaParams taaParams{
        {1.0f / static_cast<float>(swapChainExtent.width), 1.0f / static_cast<float>(swapChainExtent.height)},
        (TAA_ENABLED ? 2u : 0u) | (TAA_ENABLED && taaHistoryValid ? 1u : 0u),
        TAA_DEPTH_REJECTION_THRESHOLD};
    commandBuffer.pushConstants(*taaPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(taaParams), &taaParams);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();
    writeGpuTimestamp(commandBuffer, TIMESTAMP_TAA);

    transition_image_layout(*taaHistoryImages[taaHistoryWriteIndex], vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal, colorWrite, vk::AccessFlagBits2::eShaderRead,
                            colorOutputStage, fragmentStage, vk::ImageAspectFlagBits::eColor);
    recordOverlay(commandBuffer, imageIndex);
    writeGpuTimestamp(commandBuffer, TIMESTAMP_OVERLAY);
    transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::ePresentSrcKHR, colorWrite, {}, colorOutputStage,
                            vk::PipelineStageFlagBits2::eBottomOfPipe, vk::ImageAspectFlagBits::eColor);
    taaHistoryInitialized = true;
    taaHistoryValid = true;
    taaHistoryWriteIndex = historyReadIndex;
    commandBuffer.end();
  }

  /**
   * Record the GPU clipmap's chunk compute for this frame, if anything changed: full-rebuild clears, retiring and
   * allocating work slots, height generation, meshing, quad compaction and the readback of the quad counts.
   */
  void recordChunkCompute(vk::raii::CommandBuffer &commandBuffer) {
    const bool remesh = chunkBuffersDirty;
    // A full rebuild starts from an empty buffer, so it leaves nothing to compact.
    const bool compact = compactQuadsPending && !(remesh && remeshPlan.fullRebuild);
    compactQuadsPending = false;
    if (!remesh && !compact) {
      return;
    }

    vk::MemoryBarrier2 beforeWrite{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTaskShaderEXT |
                        vk::PipelineStageFlagBits2::eMeshShaderEXT | vk::PipelineStageFlagBits2::eVertexShader |
                        vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eDrawIndirect,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                         vk::AccessFlagBits2::eIndirectCommandRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &beforeWrite});
    const auto computeAfterCompute = [&] {
      vk::MemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
      commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    };

    if (remesh && remeshPlan.fullRebuild) {
      // Every slot is new: start from an empty quad buffer and clusters with no quads.
      commandBuffer.fillBuffer(*quadCountBuffer, 0, 2u * sizeof(uint32_t), 0);
      commandBuffer.fillBuffer(*quadClusterBuffer, 0, vk::WholeSize, 0);
      commandBuffer.fillBuffer(*waterClusterBuffer, 0, vk::WholeSize, 0);
      vk::MemoryBarrier2 clearDone{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                   .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                   .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                   .dstAccessMask =
                                       vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
      commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearDone});
    }

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *chunkComputePipelineLayout, 0,
                                     *chunkComputeDescriptorSets[frameIndex], nullptr);
    // Without a remesh this frame, remeshPlan is the last one (already done): run none of its passes.
    const ChunkComputeParams computeParams{
        .horizontalChunkCount = remesh ? static_cast<uint32_t>(remeshPlan.newColumns.size()) : 0u,
        .retireCount = remesh ? static_cast<uint32_t>(remeshPlan.retiredColumns.size()) : 0u,
        .workPoolCapacity = workPoolCapacity,
        .oceanBlock = oceanBlockId,
        .layerBlocks = terrainLayerBlocks,
        .layerFillerBlocks = terrainLayerFillerBlocks,
        .layerDeepBlocks = terrainLayerDeepBlocks,
        .zBase = 0u,
        .logBlock = treeLogBlock,
        .leavesBlock = treeLeavesBlock};
    commandBuffer.pushConstants(*chunkComputePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(computeParams), &computeParams);
    // One item per Z group, split into batches under the device's Z group limit, each pushing its first item as zBase.
    // Ends with zBase back at 0.
    if (maxComputeGroupsZ == 0u) { // queried once: getProperties() copies the whole ~800-byte properties struct
      maxComputeGroupsZ = std::min(physicalDevice.getProperties().limits.maxComputeWorkGroupCount[2], 65535u);
    }
    const uint32_t maxGroupsZ = maxComputeGroupsZ;
    const auto dispatchBatchedZ = [&](uint32_t groupsX, uint32_t groupsY, uint32_t items) {
      for (uint32_t base = 0; base < items; base += maxGroupsZ) {
        if (base != 0u) {
          commandBuffer.pushConstants(*chunkComputePipelineLayout, vk::ShaderStageFlagBits::eCompute, offsetof(ChunkComputeParams, zBase), sizeof(uint32_t), &base);
        }
        commandBuffer.dispatch(groupsX, groupsY, std::min(items - base, maxGroupsZ));
      }
      if (items > maxGroupsZ) {
        const uint32_t zero = 0u;
        commandBuffer.pushConstants(*chunkComputePipelineLayout, vk::ShaderStageFlagBits::eCompute, offsetof(ChunkComputeParams, zBase), sizeof(uint32_t), &zero);
      }
    };
    // finalizeWork's arguments, read by the next indirect dispatch.
    const auto indirectAfterCompute = [&] {
      vk::MemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                 .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect |
                                                 vk::PipelineStageFlagBits2::eComputeShader,
                                 .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
                                                  vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
      commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    };

    if (remesh && remeshPlan.fullRebuild) {
      // Every work slot back on the free stack (no column holds one after the clears above).
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *initWorkPoolPipeline);
      commandBuffer.dispatch((workPoolCapacity + 255u) / 256u, 1, 1);
      indirectAfterCompute(); // also resets the high water that compactQuads / finalizeWork read
    }
    // Retire before allocating: new columns may reuse the freed work slots. Retired clusters drop to zero quads, so
    // compactQuads (which walks every slot below the high water) skips them.
    if (computeParams.retireCount != 0u) {
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *retireWorkPipeline);
      dispatchBatchedZ(QUAD_CLUSTERS_PER_CHUNK, static_cast<uint32_t>(world::kMaxColumnChunks),
                       computeParams.retireCount);
      computeAfterCompute();
    }
    if (compact) {
      // QuadCount[0] restarts at 0 and compactQuads bumps it to the packed size; QuadCount[1] (live) is unchanged.
      vk::MemoryBarrier2 countWritable{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                       .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                       .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                       .dstAccessMask = vk::AccessFlagBits2::eTransferWrite};
      commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &countWritable});
      commandBuffer.fillBuffer(*quadCountBuffer, 0, sizeof(uint32_t), 0);
      vk::MemoryBarrier2 countCleared{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .dstAccessMask =
                                          vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
      commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &countCleared});
      // Pack the live work slots below the live count first, so the high water drops back to it.
      const uint32_t poolGroups = (workPoolCapacity + 255u) / 256u;
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *collectRelocationsPipeline);
      commandBuffer.dispatch(poolGroups, 1, 1);
      computeAfterCompute();
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *relocateSlotsPipeline);
      commandBuffer.dispatch(poolGroups, 1, 1);
      computeAfterCompute();
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *rebuildFreeStackPipeline);
      commandBuffer.dispatch(poolGroups, 1, 1);
      computeAfterCompute();
      // Indirect args from the last finalizeWork (old high water); slots past the new one are empty now.
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *compactQuadsPipeline);
      commandBuffer.dispatchIndirect(*workPoolBuffer, WORK_POOL_COMPACT_ARGS_OFFSET);
      computeAfterCompute();
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *copyCompactedQuadsPipeline);
      commandBuffer.dispatch(COPY_COMPACTED_GROUPS, 1, 1);
      computeAfterCompute();
    }
    if (computeParams.horizontalChunkCount != 0u) {
      // Noise once per column: heights plus their range per group...
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *generateHeightsPipeline);
      dispatchBatchedZ(HEIGHT_GROUPS_PER_COLUMN, 1, computeParams.horizontalChunkCount);
      computeAfterCompute();
      // ...then only the vertical chunks those heights reach get work slots and meshColumns items.
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *allocateWorkPipeline);
      commandBuffer.dispatch((computeParams.horizontalChunkCount + 63u) / 64u, 1, 1);
      computeAfterCompute();
    }
    // Also after a compaction alone: the high water may have dropped, and the relocation counters need resetting.
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *finalizeWorkPipeline);
    commandBuffer.dispatch(1, 1, 1);
    indirectAfterCompute();
    if (remesh) {
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *meshColumnsPipeline);
      // One 8x8-column tile (one QuadCluster) per group; 2 or 4 tiles per group measured the same or slower.
      commandBuffer.dispatchIndirect(*workPoolBuffer, WORK_POOL_MESH_ARGS_OFFSET);
    }

    vk::MemoryBarrier2 quadCountReady{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &quadCountReady});
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *finalizeVertexDrawPipeline);
    commandBuffer.dispatch(1, 1, 1);

    vk::MemoryBarrier2 meshReadyBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTaskShaderEXT | vk::PipelineStageFlagBits2::eMeshShaderEXT |
                        vk::PipelineStageFlagBits2::eVertexShader | vk::PipelineStageFlagBits2::eDrawIndirect,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eIndirectCommandRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &meshReadyBarrier});

    // Quad counts back to the CPU (read once this frame's fence signals) to decide on compaction.
    vk::MemoryBarrier2 countReadable{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                     .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                     .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                     .dstAccessMask = vk::AccessFlagBits2::eTransferRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &countReadable});
    commandBuffer.copyBuffer(*quadCountBuffer, *quadCountReadbackBuffers[frameIndex],
                             vk::BufferCopy{0, 0, 2u * sizeof(uint32_t)});
    commandBuffer.copyBuffer(*workPoolBuffer, *quadCountReadbackBuffers[frameIndex],
                             vk::BufferCopy{0, 2u * sizeof(uint32_t), 4u * sizeof(uint32_t)});
    if (compact || (remesh && remeshPlan.fullRebuild)) {
      ++quadCountEpoch; // counts read back from earlier frames predate this
    }
    quadCountReadbackPending[frameIndex] = true;
    quadCountReadbackEpoch[frameIndex] = quadCountEpoch;
    chunkBuffersDirty = false;
  }

  /** Record a synchronization2 layout transition of mip 0 / layer 0 of `image` into this frame's command buffer. */
  void transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask,
                               vk::ImageAspectFlags image_aspect_flags) {
    vk::ImageMemoryBarrier2 barrier = {.srcStageMask = src_stage_mask,
                                       .srcAccessMask = src_access_mask,
                                       .dstStageMask = dst_stage_mask,
                                       .dstAccessMask = dst_access_mask,
                                       .oldLayout = old_layout,
                                       .newLayout = new_layout,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .image = image,
                                       .subresourceRange = {.aspectMask = image_aspect_flags,
                                                            .baseMipLevel = 0,
                                                            .levelCount = 1,
                                                            .baseArrayLayer = 0,
                                                            .layerCount = 1}};
    vk::DependencyInfo dependency_info = {
        .dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
    commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
  }
#pragma endregion

#pragma region Frame Sync
  /** Create the per-frame acquire semaphores and fences (signaled), and the per-image render-finished semaphores. */
  void createSyncObjects() {
    assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

    createRenderFinishedSemaphores();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
      inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
  }

  /** Create a render-finished semaphore per swapchain image. */
  void createRenderFinishedSemaphores() {
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }
  }

#pragma endregion
#pragma endregion

#pragma region Frame Rendering
#pragma region Camera Update
  /**
   * Halton low-discrepancy sequence, for the TAA jitter.
   *
   * @return element `index` of the base-`base` sequence, in [0, 1).
   */
  static float halton(uint32_t index, uint32_t base) {
    float result = 0.0f;
    float fraction = 1.0f;
    while (index != 0u) {
      fraction /= static_cast<float>(base);
      result += fraction * static_cast<float>(index % base);
      index /= base;
    }
    return result;
  }
#pragma endregion

#pragma region Uniform Update
  /**
   * Per-frame CPU update: camera, player, block interaction, LOD distances, clipmap and CPU world, the remesh plan,
   * and this frame's uniform buffer (jittered reverse-Z projection, cull matrices, fog and Hi-Z parameters).
   *
   * @param currentImage Frame in flight (indexes the per-frame buffers, not the swapchain).
   */
  void updateUniformBuffer(uint32_t currentImage) {
    static auto lastTime = std::chrono::steady_clock::now();

    const auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
    lastTime = currentTime;

    UniformBufferObject ubo{};
    camera.updateLook(window);
    player.update(window, deltaTime, camera, *cpuWorld);
    const glm::vec3 eye = player.eyePosition();
    const glm::mat4 view = camera.view(eye);
    updateBlockInteraction(eye, deltaTime); // before updateCpuWorld, which uploads the remeshed chunks this frame

    const bool zoomed = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
    // LODs and the clipmap use the unwidened FOV, so the sprint FOV animation doesn't rebuild the clipmap every frame.
    const float fov = zoomed ? 2.0f * std::atan(std::tan(BASE_FOV_RADIANS * 0.5f) * 0.5f) : BASE_FOV_RADIANS;
    const float projectionFov = fov * player.fovMultiplier();
    if (fov != cachedLodFov || swapChainExtent.width != cachedLodExtent.width ||
        swapChainExtent.height != cachedLodExtent.height) {
      lodDistanceSquared =
          initializeLodDistanceSquared(TARGET_SCREEN_PERCENTAGE, fov, {swapChainExtent.width, swapChainExtent.height});
      baseLodDistanceSquared = initializeLodDistanceSquared(TARGET_SCREEN_PERCENTAGE, BASE_FOV_RADIANS,
                                                            {swapChainExtent.width, swapChainExtent.height});
      cachedLodFov = fov;
      cachedLodExtent = swapChainExtent;
    }
    const glm::vec3 worldUp{0.0f, 1.0f, 0.0f};
    const glm::vec3 cameraRight = glm::normalize(glm::cross(camera.front(), worldUp));
    const glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, camera.front()));
    const float aspectRatio = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
    // Hysteresis of half a chunk so hovering at the threshold doesn't flip the hand-over every frame.
    std::optional<cpuworld::ChunkXZ> clipmapCpuCenter = cpuAdvertisedCenter;
    if (clipmapCpuCenter) {
      const glm::vec3 &lodEye = frozenLodView ? frozenLodEye : eye;
      const float heightAbove = lodEye.y - static_cast<float>(cpuWorld->squareTopY(*clipmapCpuCenter));
      constexpr float hideAbove = static_cast<float>(world::kCpuHideHeightChunks * CHUNK_SIZE);
      cpuHiddenByHeight = heightAbove > (cpuHiddenByHeight ? hideAbove - CHUNK_SIZE * 0.5f : hideAbove);
      if (cpuHiddenByHeight) {
        clipmapCpuCenter.reset();
      }
    }
    const ClipmapView clipmapView{{eye.x, eye.y, eye.z},
                                  {camera.front().x, camera.front().y, camera.front().z},
                                  {cameraRight.x, cameraRight.y, cameraRight.z},
                                  {cameraUp.x, cameraUp.y, cameraUp.z},
                                  lodDistanceSquared,
                                  baseLodDistanceSquared,
                                  fov,
                                  aspectRatio,
                                  useVertexPulling,
                                  clipmapCpuCenter};
    if (lodFreezeRequested && !frozenLodView) {
      frozenLodView = clipmapView;
      frozenLodEye = eye;
      frozenCullMvpValid = false;
    } else if (!lodFreezeRequested) {
      frozenLodView.reset();
    }
    updateVisibleClipmap(frozenLodView ? *frozenLodView : clipmapView);
    updateCpuWorld(frozenLodView ? frozenLodEye : eye, currentImage);
    if (chunkBuffersDirty) {
      remeshPlan = chunkSlots.update(visibleChunkSet, visibleChunkNeighborLods, forceFullRebuild);
      forceFullRebuild = false;
      // Slot-indexed column data is rewritten whole: this frame's copy is the one the remesh reads.
      std::memcpy(chunkCoordinateBuffersMapped[currentImage], chunkSlots.slotCoords().data(),
                  chunkSlots.columnSlotCount() * sizeof(world::ChunkLod));
      std::memcpy(chunkNeighborLodBuffersMapped[currentImage], chunkSlots.slotNeighborLods().data(),
                  chunkSlots.columnSlotCount() * sizeof(uint32_t));
      std::memcpy(columnListBuffersMapped[currentImage], remeshPlan.newColumns.data(),
                  remeshPlan.newColumns.size() * sizeof(uint32_t));
      std::memcpy(retireListBuffersMapped[currentImage], remeshPlan.retiredColumns.data(),
                  remeshPlan.retiredColumns.size() * sizeof(uint32_t));
    }

    constexpr float nearPlane = NEAR_PLANE;
    constexpr float farPlane = FAR_PLANE;
    // Reverse-Z (near and far swapped: depth 1 at near, 0 at far) keeps float depth precision out to the far plane;
    // standard Z z-fought past a few thousand blocks. Explicitly ZO: glm's default -1..1 (GLM_FORCE_DEPTH_ZERO_TO_ONE
    // is not set) with swapped planes would put everything past the near plane at negative depth.
    glm::mat4 proj = glm::perspectiveRH_ZO(
        projectionFov, static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), farPlane,
        nearPlane);
    const uint32_t jitterSample = (taaJitterIndex++ % 8u) + 1u;
    const float jitterX = TAA_ENABLED ? (halton(jitterSample, 2u) - 0.5f) * TAA_JITTER_PIXELS : 0.0f;
    const float jitterY = TAA_ENABLED ? (halton(jitterSample, 3u) - 0.5f) * TAA_JITTER_PIXELS : 0.0f;
    proj[2][0] -= 2.0f * jitterX / static_cast<float>(swapChainExtent.width);
    proj[2][1] += 2.0f * jitterY / static_cast<float>(swapChainExtent.height);
    ubo.mvp = proj * view;
    ubo.previousMvp = previousMvpValid ? previousMvp : ubo.mvp;
    ubo.renderSize = {static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height)};
    ubo.lodAoLighteningPerLevel = LOD_AO_LIGHTENING_PER_LEVEL;
    ubo.contourMode = contourEnabled ? 1u : 0u;
    // far_tint fades in over [LOD 2, LOD 4) distance, so LOD 3 trees sit mid-fade and no LOD seam shows a step.
    ubo.farTintStart = std::sqrt(lodDistanceSquared[2]);
    ubo.nearPlane = nearPlane;
    ubo.farPlane = farPlane;
    // w: seconds for texture animation and fake waves, wrapped so a float keeps millisecond precision.
    ubo.cameraPosition = glm::vec4(eye, static_cast<float>(std::fmod(glfwGetTime(), 3600.0)));
    // z: 1 while the camera is inside an ocean cell (underwater fog in fragMain).
    cameraUnderwater =
        cpuWorld->blockId(static_cast<int32_t>(std::floor(eye.x)), static_cast<int32_t>(std::floor(eye.y)),
                          static_cast<int32_t>(std::floor(eye.z))) == oceanBlockId;
    // Border fog end: the clipmap radius less its allowed lag (stale cover) and some slack, so its edge stays fogged.
    const float borderFogEnd =
        kClipmapRadius - kClipmapStaleCover.maxPositionDelta - static_cast<float>(CHUNK_SIZE) * 4.0f;
    ubo.viewRayScale = {1.0f / proj[0][0], 1.0f / proj[1][1], cameraUnderwater ? 1.0f : 0.0f, borderFogEnd};
    // Only the mesh path builds the Hi-Z; after a vertex-path frame it no longer matches previousMvp.
    if (useVertexPulling || !hzbOcclusionEnabled) {
      hzbValid = false;
    }
    ubo.hzbParams = {static_cast<float>(hzbSize.width), static_cast<float>(hzbSize.height),
                     static_cast<float>(hzbMipCount), hzbValid ? 1.0f : 0.0f};
    if (frozenLodView) {
      if (!frozenCullMvpValid) {
        frozenCullMvp = ubo.mvp;
        frozenHzbMvp = ubo.previousMvp;
        frozenCullMvpValid = true;
      }
      ubo.cullMvp = frozenCullMvp;
      ubo.cullCameraPosition = glm::vec4(frozenLodEye, 0.0f);
      ubo.cullPreviousMvp = frozenHzbMvp; // recordHzb is skipped while frozen: the Hi-Z keeps the frozen view's depth
    } else {
      ubo.cullMvp = ubo.mvp;
      ubo.cullCameraPosition = glm::vec4(eye, 0.0f);
      ubo.cullPreviousMvp = ubo.previousMvp;
    }
    cullMvpCpu = ubo.cullMvp;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    previousMvp = ubo.mvp;
    previousMvpValid = true;
  }
#pragma endregion

#pragma region Frame Submit
  /**
   * Draw one frame: wait for the frame's fence, act on its GPU readbacks (compaction, work pool growth), update,
   * record, submit and present; recreates the swapchain when it is out of date or resized.
   */
  void drawFrame() {
    // inFlightFences, presentCompleteSemaphores and commandBuffers are indexed by frameIndex; renderFinishedSemaphores
    // by imageIndex.
    vk::Result result;
    uint32_t imageIndex;
    {
      const auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
      if (fenceResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for fence!");
      }
      collectGpuTimestamps();
      if (quadCountReadbackPending[frameIndex]) {
        quadCountReadbackPending[frameIndex] = false;
        // Layout: READBACK_UINTS.
        const uint32_t appended = quadCountReadbackMapped[frameIndex][0];
        const uint32_t live = quadCountReadbackMapped[frameIndex][1];
        const uint32_t freeSlots = quadCountReadbackMapped[frameIndex][2];
        const uint32_t slotHighWater = quadCountReadbackMapped[frameIndex][3];
        const uint32_t workPoolOverflow = quadCountReadbackMapped[frameIndex][5];
        const uint32_t liveSlots = workPoolCapacity > freeSlots ? workPoolCapacity - freeSlots : 0u;
        // Retired slots leave holes the draw and compaction still walk; pack once they pass 1/4 of the live slots.
        const bool slotsSparse = slotHighWater > liveSlots + liveSlots / 4u + 1024u;
        const uint32_t deadQuads = appended > live ? appended - live : 0u;
        const uint32_t deadLimit = useVertexPulling ? live / 4u : live;
        const bool garbageHeavy = deadQuads > deadLimit && deadQuads > REBUILD_MIN_DEAD_QUADS;
        const bool nearlyFull = appended > MAX_QUAD_COUNT / 4u * 3u && live < MAX_QUAD_COUNT / 2u;
        if (quadCountReadbackEpoch[frameIndex] != quadCountEpoch) {
          // Recorded before the last compaction or rebuild: these counts no longer describe QuadBuffer.
        } else if (workPoolOverflow != 0u) {
          // allocateWork ran dry and dropped chunks. Failed pops still took their count off the stack top, so
          // capacity - top (top gone negative) is everything asked for.
          const int64_t stackTop = static_cast<int32_t>(freeSlots);
          growWorkPool(static_cast<uint64_t>(std::max<int64_t>(int64_t{workPoolCapacity} - stackTop, 0)));
        } else if (appended > MAX_QUAD_COUNT) {
          // Quads past the end were dropped (holes). Warn once: a set that doesn't fit rebuilds every frame.
          if (!quadOverflowWarned) {
            std::cerr << "QuadBuffer overflow: " << appended << " quads > capacity " << MAX_QUAD_COUNT
                      << "; some chunks are missing\n";
            quadOverflowWarned = true;
          }
          forceFullRebuild = true;
          chunkBuffersDirty = true;
        } else if (garbageHeavy || nearlyFull || slotsSparse) {
          compactQuadsPending = true;
        }
      }

      auto [acquireResult, acquiredImageIndex] =
          swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);
      result = acquireResult;
      imageIndex = acquiredImageIndex;

      if (result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain();
        return;
      }
      if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
        throw std::runtime_error("failed to acquire swap chain image!");
      }
    }
    updateUniformBuffer(frameIndex);

    // Reset only once work is sure to be submitted.
    device.resetFences(*inFlightFences[frameIndex]);

    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{.waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
                                    .pWaitDstStageMask = &waitDestinationStageMask,
                                    .commandBufferCount = 1,
                                    .pCommandBuffers = &*commandBuffers[frameIndex],
                                    .signalSemaphoreCount = 1,
                                    .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]};
    {
      queue.submit(submitInfo, *inFlightFences[frameIndex]);

      const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                              .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                                              .swapchainCount = 1,
                                              .pSwapchains = &*swapChain,
                                              .pImageIndices = &imageIndex};
      result = queue.presentKHR(presentInfoKHR);
    }
    // VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS: out-of-date is returned, not thrown.
    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else {
      // Any error code already threw.
      assert(result == vk::Result::eSuccess);
    }
    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
  }
#pragma endregion
#pragma endregion

#pragma region Vulkan Utilities
#pragma region Shader Module
  /** Create a shader module from SPIR-V bytes. */
  [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const {
    vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(),
                                          .pCode = reinterpret_cast<const uint32_t *>(code.data())};
    vk::raii::ShaderModule shaderModule{device, createInfo};

    return shaderModule;
  }
#pragma endregion

#pragma region Swapchain Configuration
  /** Swapchain image count: at least 3, capped at the surface maximum. */
  static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
  }

  /** Surface format: BGRA8 sRGB if offered, else the first one. */
  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    assert(!availableFormats.empty());
    const auto srgbFormat = std::ranges::find_if(availableFormats, [](const auto &format) {
      return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    return srgbFormat != availableFormats.end() ? *srgbFormat : availableFormats[0];
  }

  /** Present mode: mailbox, else immediate, else FIFO (vsync). */
  static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes) {
    assert(std::ranges::any_of(availablePresentModes,
                               [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
    // Uncapped first: mailbox (no tearing), then immediate (tears). Many Windows drivers (and Proton) offer only
    // immediate + FIFO.
    for (const vk::PresentModeKHR preferred : {vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eImmediate}) {
      if (std::ranges::find(availablePresentModes, preferred) != availablePresentModes.end()) {
        return preferred;
      }
    }
    return vk::PresentModeKHR::eFifo;
  }

  /** Swapchain extent: the surface's current extent, or the framebuffer size clamped to the surface limits. */
  vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
  }
#pragma endregion

#pragma region Instance Extensions
  /** Instance extensions: GLFW's, plus debug utils with validation. */
  std::vector<const char *> getRequiredInstanceExtensions() {
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr || glfwExtensionCount == 0) {
      throw std::runtime_error("failed to get required GLFW Vulkan extensions");
    }

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidationLayers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
  }
#pragma endregion

#pragma region Validation Callback
  /**
   * Validation layer callback: prints warnings and errors.
   *
   * @return VK_FALSE (never aborts the call).
   */
  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                        vk::DebugUtilsMessageTypeFlagsEXT type,
                                                        const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                        void *) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
        severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
      std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
    }

    return vk::False;
  }
#pragma endregion

#pragma region GPU Timing and Debug Environment
  /** Create the GPU timing query pool, if the queue has timestamps; ENGINE_GPU_TIMING=1 starts with timing on. */
  void createTimestampQueryPool() {
    const uint32_t validBits = physicalDevice.getQueueFamilyProperties()[queueIndex].timestampValidBits;
    if (validBits == 0u) {
      return; // P reports timings as unavailable
    }
    timestampMask = validBits >= 64u ? ~uint64_t{0} : (uint64_t{1} << validBits) - 1u;
    timestampPeriodNs = physicalDevice.getProperties().limits.timestampPeriod;
    timestampQueryPool = vk::raii::QueryPool(
        device, vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp,
                                        .queryCount = GPU_TIMESTAMP_COUNT * MAX_FRAMES_IN_FLIGHT});
    // ENGINE_GPU_TIMING=1 starts with the timings on (same as pressing P).
    if (const char *env = std::getenv("ENGINE_GPU_TIMING"); env != nullptr && env[0] == '1') {
      gpuTimingEnabled = true;
      gpuTimeWindowStart = std::chrono::steady_clock::now();
    }
  }

  /**
   * Apply the testing environment variables: ENGINE_LOOK="yaw,pitch" (degrees), ENGINE_FORCE_LOD=n,
   * ENGINE_VIEW=n (terrain view, as X cycles: 3 = LOD debug colors), ENGINE_CONTOUR=1 (elevation view).
   */
  void applyDebugEnvironment() {
    if (const char *env = std::getenv("ENGINE_LOOK"); env != nullptr) {
      float yaw = -90.0f;
      float pitch = -20.0f;
      std::sscanf(env, "%f,%f", &yaw, &pitch);
      camera.setLook(yaw, pitch);
    }
    if (const char *env = std::getenv("ENGINE_FORCE_LOD"); env != nullptr) {
      gForcedMinLod = static_cast<uint32_t>(std::atoi(env));
    }
    if (const char *env = std::getenv("ENGINE_VIEW"); env != nullptr) {
      terrainView = static_cast<TerrainView>(std::atoi(env) % 4);
    }
    if (const char *env = std::getenv("ENGINE_CONTOUR"); env != nullptr && env[0] == '1') {
      contourEnabled = true;
    }
  }

  /** Write timestamp `stamp` of this frame slot, while GPU timing is on. */
  void writeGpuTimestamp(vk::raii::CommandBuffer &commandBuffer, GpuTimestamp stamp) {
    if (gpuTimingEnabled) {
      // ALL_COMMANDS: written once every earlier command has finished, so each interval is that stage's GPU time.
      commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *timestampQueryPool,
                                    frameIndex * GPU_TIMESTAMP_COUNT + stamp);
    }
  }

  /** After this frame slot's fence: accumulate its GPU timestamps and print the averages once a second. */
  void collectGpuTimestamps() {
    if (!timestampsPending[frameIndex]) {
      return;
    }
    timestampsPending[frameIndex] = false;
    const auto [result, ticks] = timestampQueryPool.getResults<uint64_t>(
        frameIndex * GPU_TIMESTAMP_COUNT, GPU_TIMESTAMP_COUNT, GPU_TIMESTAMP_COUNT * sizeof(uint64_t),
        sizeof(uint64_t), vk::QueryResultFlagBits::e64);
    if (result != vk::Result::eSuccess) {
      return;
    }
    // Milliseconds between two timestamps; the mask handles wraparound of the valid bits.
    const auto elapsedMs = [&](uint64_t from, uint64_t to) {
      return static_cast<double>((to - from) & timestampMask) * timestampPeriodNs * 1e-6;
    };
    for (uint32_t stamp = 1; stamp < GPU_TIMESTAMP_COUNT; ++stamp) {
      gpuTimeSumsMs[stamp] += elapsedMs(ticks[stamp - 1u], ticks[stamp]);
    }
    gpuTimeSumsMs[0] += elapsedMs(ticks[TIMESTAMP_FRAME_START], ticks[GPU_TIMESTAMP_COUNT - 1u]);
    ++gpuTimeSamples;

    const auto now = std::chrono::steady_clock::now();
    if (now - gpuTimeWindowStart < std::chrono::seconds(1)) {
      return;
    }
    char line[512];
    int length = std::snprintf(line, sizeof(line), "GPU ms (avg of %u frames): total %.3f |", gpuTimeSamples,
                               gpuTimeSumsMs[0] / gpuTimeSamples);
    for (uint32_t stamp = 1; stamp < GPU_TIMESTAMP_COUNT && length > 0 && length < static_cast<int>(sizeof(line));
         ++stamp) {
      length += std::snprintf(line + length, sizeof(line) - static_cast<size_t>(length), " %s %.3f",
                              GPU_TIMESTAMP_NAMES[stamp], gpuTimeSumsMs[stamp] / gpuTimeSamples);
    }
    std::clog << line << '\n';
    gpuTimeSumsMs = {};
    gpuTimeSamples = 0;
    gpuTimeWindowStart = now;
  }

  /** Create the pipeline cache, seeded from PIPELINE_CACHE_FILE when its header matches this GPU and driver. */
  void createPipelineCache() {
    std::vector<char> data;
    if (std::ifstream file(PIPELINE_CACHE_FILE, std::ios::binary | std::ios::ate); file) {
      data.resize(static_cast<size_t>(file.tellg()));
      file.seekg(0);
      file.read(data.data(), static_cast<std::streamsize>(data.size()));
    }
    // Header version one: length, version, vendor ID, device ID, cache UUID. Data from another GPU or driver is
    // dropped here (drivers must ignore it too, but some have not).
    const vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    const auto readU32 = [&](size_t offset) {
      uint32_t value;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      return value;
    };
    constexpr size_t headerSize = 16u + VK_UUID_SIZE;
    if (data.size() < headerSize || readU32(4) != static_cast<uint32_t>(vk::PipelineCacheHeaderVersion::eOne) ||
        readU32(8) != properties.vendorID || readU32(12) != properties.deviceID ||
        std::memcmp(data.data() + 16, properties.pipelineCacheUUID.data(), VK_UUID_SIZE) != 0) {
      data.clear();
    }
    pipelineCache = vk::raii::PipelineCache(
        device, vk::PipelineCacheCreateInfo{.initialDataSize = data.size(), .pInitialData = data.data()});
  }

  /** Write the pipeline cache to PIPELINE_CACHE_FILE (via a temporary file, so an interrupted save never truncates it). */
  void savePipelineCache() {
    if (pipelineCache == nullptr) {
      return;
    }
    const std::vector<uint8_t> data = pipelineCache.getData();
    const std::string temporaryPath = std::string(PIPELINE_CACHE_FILE) + ".tmp";
    {
      std::ofstream file(temporaryPath, std::ios::binary | std::ios::trunc);
      file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
      if (!file) {
        return;
      }
    }
    std::rename(temporaryPath.c_str(), PIPELINE_CACHE_FILE);
  }

  /**
   * Read a whole binary file.
   *
   * @return its bytes; throws if it can't be opened, is empty, or the read fails.
   */
  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("failed to open file: " + filename);
    }

    const auto fileSize = file.tellg();
    if (fileSize <= 0) {
      throw std::runtime_error("file is empty or could not be sized: " + filename);
    }
    std::vector<char> buffer(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()))) {
      throw std::runtime_error("failed to read file: " + filename);
    }
    return buffer;
  }
#pragma endregion
#pragma endregion
};

void Renderer::run() {
  VulkanEngine app;
  app.run();
}
