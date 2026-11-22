#include "renderer.hpp"
#include "block_registry.hpp"
#include "camera.hpp"
#include "chunk_slots.hpp"
#include "clipmap_culler.hpp"
#include "cpu_world.hpp"
#include "lod.hpp"
#include "player.hpp"
#include "terrain_cpu.hpp"
#include "vegetation.hpp"
#include "world.hpp"

#include <algorithm>
#include <array>
#include <assert.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float3.hpp>
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

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
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
  uint32_t clusterCount; // QuadClusters the task shader walks: 16 per meshed chunk
  uint32_t historyValid;
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
static_assert(offsetof(UniformBufferObject, clusterCount) == 136);
static_assert(offsetof(UniformBufferObject, historyValid) == 140);
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
  std::array<float, 4> tint; // every LOD (blocks.txt `tint`)
};
static_assert(sizeof(GpuBlockFace) == 48);
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
};
constexpr uint32_t GEOMETRY_PASS_OPAQUE = 0u;
constexpr uint32_t GEOMETRY_PASS_TRANSLUCENT = 1u; // blended water: no back-face culling
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

class HelloTriangleApplication {
public:
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
  uint32_t queueIndex = ~0;
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
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;
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
  vk::raii::Pipeline selectedTexturePipeline = nullptr;
  vk::raii::Pipeline crosshairPipeline = nullptr; // overlayPipelineLayout, inverting blend
  vk::raii::PipelineLayout outlinePipelineLayout = nullptr;
  vk::raii::Pipeline outlinePipeline = nullptr;
  // Hi-Z occlusion (mesh path): farthest-depth pyramid of last frame's depth, read by taskMain (hzb.slang).
  vk::raii::DescriptorSetLayout hzbSetLayout = nullptr;
  vk::raii::PipelineLayout hzbPipelineLayout = nullptr;
  vk::raii::Pipeline hzbBuildLowPipeline = nullptr;     // resolved depth -> mips 0..4
  vk::raii::Pipeline hzbBuildHighPipeline = nullptr;    // mip 4 -> mips 5..last
  vk::raii::DescriptorPool hzbDescriptorPool = nullptr; // own pool: rebuilt with the swapchain-sized resources
  std::vector<vk::raii::DescriptorSet> hzbSets;         // per swapchain image: its depth + every mip
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
  // Reverse-Z depth (1 at NEAR_PLANE, 0 at FAR_PLANE) of view depth d: linear in 1 / d.
  static constexpr float reverseZDepth(float d) {
    return NEAR_PLANE * (FAR_PLANE - d) / (d * (FAR_PLANE - NEAR_PLANE));
  }
#pragma endregion

#pragma region Image State
  std::vector<vk::raii::Image> multisampleColorImages;
  std::vector<vk::raii::DeviceMemory> multisampleColorImageMemories;
  std::vector<vk::raii::ImageView> multisampleColorImageViews;
  std::vector<vk::raii::Image> sceneColorImages;
  std::vector<vk::raii::DeviceMemory> sceneColorImageMemories;
  std::vector<vk::raii::ImageView> sceneColorImageViews;
  std::vector<vk::raii::Image> multisampleMotionImages;
  std::vector<vk::raii::DeviceMemory> multisampleMotionImageMemories;
  std::vector<vk::raii::ImageView> multisampleMotionImageViews;
  std::vector<vk::raii::Image> motionImages;
  std::vector<vk::raii::DeviceMemory> motionImageMemories;
  std::vector<vk::raii::ImageView> motionImageViews;
  std::array<vk::raii::Image, 2> taaHistoryImages{nullptr, nullptr};
  std::array<vk::raii::DeviceMemory, 2> taaHistoryImageMemories{nullptr, nullptr};
  std::array<vk::raii::ImageView, 2> taaHistoryImageViews{nullptr, nullptr};
  std::vector<vk::raii::Image> depthImages;
  std::vector<vk::raii::DeviceMemory> depthImageMemories;
  std::vector<vk::raii::ImageView> depthImageViews;
  // Single-sample MIN resolve of depthImages (farthest sample per pixel under reverse-Z), written by the opaque pass;
  // the water fog reads it, and recordHzb builds the pyramid from it. Reading the MSAA depth directly forced a ~40 us
  // decompression.
  std::vector<vk::raii::Image> resolvedDepthImages;
  std::vector<vk::raii::DeviceMemory> resolvedDepthImageMemories;
  std::vector<vk::raii::ImageView> resolvedDepthImageViews;

  static constexpr uint32_t MAX_TEXTURE_COUNT = 1024;
  uint32_t textureDescriptorCapacity = 0;
  std::vector<vk::raii::Image> textureImages; // one per animation frame
  // Per registry texture index: first bindless slot and frame count (createTextureImages).
  std::vector<uint32_t> textureSlots;
  std::vector<uint32_t> textureFrames;
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
  vk::raii::Buffer treeLodTableBuffer = nullptr; // chunk compute binding 19 (vegetation::buildTreeLodTable)
  vk::raii::DeviceMemory treeLodTableBufferMemory = nullptr;
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
    glm::vec4 boxMin; // world space; w = line width in pixels
    glm::vec4 boxMax;
  };
  static_assert(sizeof(OutlineParams) == 32);
  // QuadCluster in chunk_compute.slang / slang.slang: one per meshColumns group, 16 groups per chunk.
  static constexpr vk::DeviceSize QUAD_CLUSTER_SIZE = 48u;
  static constexpr uint32_t QUAD_CLUSTERS_PER_CHUNK = 16u;
  // Capacity is largest at the narrowest supported FOV, not the widest FOV.
  static constexpr uint32_t CLIPMAP_KEEP_ALIVE_MULTIPLIER = 4u;
  static constexpr uint32_t MAX_HORIZONTAL_CHUNK_COUNT =
      calculateClipmapCapacity(TARGET_SCREEN_PERCENTAGE, MIN_SUPPORTED_TAN_HALF_FOV, CHUNK_SIZE,
                               world::kRenderDistance) *
      CLIPMAP_KEEP_ALIVE_MULTIPLIER;
  static constexpr uint32_t VERTICAL_CHUNK_COUNT = 2u * static_cast<uint32_t>(world::kVerticalRenderDistance) + 1u;
  // Work slots (one per meshed vertical chunk, 16 QuadClusters each), handed out on the GPU by allocateWork. Columns
  // only mesh the chunks their heights reach (plus one each way), a few on average. The pool starts at
  // WORK_POOL_INITIAL_CAPACITY and doubles (growWorkPool) whenever allocateWork runs dry, up to the most the clipmap
  // could ever need.
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
  // CPU square centers of clipmap jobs submitted after the applied one (sequence, center). Their squares stay pinned
  // in cpuWorld so whichever job finishes can be applied without a hole; dropped once a newer result is applied.

  //! ONLY HAPPENS IN -O0:
  //! KNOWN ISSUE: flying at full freecam speed (1000 blocks/s), the drawn CPU square still lags behind the player and
  //! GPU LOD terrain shows under them (red in the X debug view). CPU generation itself keeps up (a headless test
  //! stays ~1 chunk behind even in debug); the lag is in the clipmap hand-over (applied clipmap vs advertised square).
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

  // Compact once dead quads pass a share of the live ones, or once the buffer is three-quarters used while compacting
  // would actually free space. The vertex path still pays its fixed per-vertex cost for dead quads (it draws every
  // quad in the buffer), so it compacts at 1/4; the mesh path never draws them, so it waits for 1/1. Compaction only
  // copies the live quads (no remeshing), so it costs a fraction of the full rebuild it replaces; a full rebuild is
  // left for an overflowed QuadBuffer, whose clusters lost quads that only remeshing can restore.
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
  // descriptor set. Chunks get a cluster slot (16 clusters) and a quad range.
  //
  // Loaded is not drawn: a column is drawn only inside the square the applied clipmap leaves out
  // (lastAppliedClipmapView->cpuRegionCenter), so CPU and GPU terrain never overlap or leave a gap, even while the
  // CPU holds several squares during a hand-over. Hidden chunks keep their quads but have zeroed clusters (mesh
  // path); the vertex path draws visible chunks one by one.
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
    int32_t cy;
    uint32_t slot;
    uint32_t quadOffset;
    uint32_t quadCount;
    std::array<cpuworld::QuadCluster, cpuworld::kClustersPerChunk> clusters;            // rebased onto quadOffset
    std::array<cpuworld::QuadCluster, cpuworld::kClustersPerChunk> translucentClusters; // rebased too
    uint32_t opaqueQuadCount; // quads [0, opaqueQuadCount) are opaque, the rest translucent (vertex path)
  };
  struct CpuColumnGpu {
    int32_t cx;
    int32_t cz;
    bool visible;
    std::vector<CpuChunkAllocation> chunks;
  };
  std::unordered_map<uint64_t, CpuColumnGpu> cpuColumns;
  SlotAllocator cpuChunkSlots;
  RangeAllocator cpuQuadRanges{CPU_MAX_QUADS};
  std::deque<cpuworld::ColumnMeshEvent> cpuPendingEvents; // waiting for staging space
  // Center offered to the clipmap: the CPU's ready center, once all of its geometry is uploaded.
  std::optional<cpuworld::ChunkXZ> cpuAdvertisedCenter;
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
  // RAII: members are destroyed in reverse declaration order, so every set allocated from
  // descriptorPool must be declared after it (sets are then freed before the pool is destroyed).
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
  Camera camera;
  // Block breaking (left mouse): once per click, then every BLOCK_BREAK_REPEAT_SECONDS while held (creative-style).
  static constexpr float BLOCK_REACH = 5.0f;
  static constexpr double BLOCK_BREAK_REPEAT_SECONDS = 0.25;
  bool breakClickPending = false;
  double nextBreakTime = 0.0;
  // Block placing (right mouse), same timing; keys 1-9 pick the block (blockRegistry.blocks in file order).
  static constexpr double BLOCK_PLACE_REPEAT_SECONDS = 0.25;
  bool placeClickPending = false;
  double nextPlaceTime = 0.0;
  size_t selectedBlockIndex = 0;
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
    camera.initialize(window);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

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

  static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
      app->breakClickPending = true; // a click shorter than a frame still breaks
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
      app->placeClickPending = true;
    }
  }

  static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
    app->camera.handleKey(window, key, action);
    app->player.handleKey(key, action);
    if (action != GLFW_PRESS) {
      return;
    }

    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
      const size_t index = static_cast<size_t>(key - GLFW_KEY_1);
      if (index < app->blockRegistry.blocks.size()) {
        app->selectedBlockIndex = index;
        std::clog << "Selected block " << app->blockRegistry.blocks[index].name << '\n';
      }
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
    if (key == GLFW_KEY_X) {
      app->terrainView = static_cast<TerrainView>((static_cast<int>(app->terrainView) + 1) % 4);
      std::clog << "Terrain view: " << terrainViewName(app->terrainView) << '\n';
    }
    if (key == GLFW_KEY_V) {
      app->useVertexPulling = !app->useVertexPulling;
      std::clog << "Quad path: " << (app->useVertexPulling ? "vertex shader (vertex pulling)" : "mesh shader") << '\n';
    }
    if (key == GLFW_KEY_G) {
      app->lodFreezeRequested = !app->lodFreezeRequested;
      std::clog << "LOD / chunk updates " << (app->lodFreezeRequested ? "frozen" : "unfrozen") << '\n';
    }
  }
#pragma endregion

#pragma region Renderer Initialization
  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
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
    createTreeLodTableBuffer();
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
  }
#pragma endregion

#pragma region Main Loop
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
  void cleanupSwapChain() {
    graphicsPipeline = nullptr;
    vertexPullingPipeline = nullptr;
    wireframePipeline = nullptr;
    wireframeVertexPullingPipeline = nullptr;
    waterPipeline = nullptr;
    waterVertexPullingPipeline = nullptr;
    taaPipeline = nullptr;
    overlayPipeline = nullptr;
    crosshairPipeline = nullptr;
    outlinePipeline = nullptr;
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
    sceneColorImageViews.clear();
    sceneColorImages.clear();
    sceneColorImageMemories.clear();
    multisampleMotionImageViews.clear();
    multisampleMotionImages.clear();
    multisampleMotionImageMemories.clear();
    motionImageViews.clear();
    motionImages.clear();
    motionImageMemories.clear();
    multisampleColorImageViews.clear();
    multisampleColorImages.clear();
    multisampleColorImageMemories.clear();
    hzbSets.clear();
    hzbDescriptorPool = nullptr;
    hzbMipViews.clear();
    hzbView = nullptr;
    hzbImage = nullptr;
    hzbImageMemory = nullptr;
    hzbImageInitialized = false;
    hzbValid = false;
    depthImageViews.clear();
    resolvedDepthImageViews.clear();
    resolvedDepthImages.clear();
    resolvedDepthImageMemories.clear();
    depthImages.clear();
    depthImageMemories.clear();
    swapChainImageViews.clear();
    swapChain = nullptr;
    taaHistoryValid = false;
    taaHistoryInitialized = false;
    taaHistoryWriteIndex = 0;
    previousMvpValid = false;
  }

  void cleanup() {
    cleanupSwapChain();
    surface = nullptr;
    glfwDestroyWindow(window);
    glfwTerminate();
  }
#pragma endregion

#pragma region Swapchain Recreation
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

    renderFinishedSemaphores.clear();
    cleanupSwapChain();
    createSwapChain();
    createRenderFinishedSemaphores();
    createImageViews();
    createGraphicsPipeline();
    createTaaPipeline();
    createOverlayPipeline();
    createDepthResources();
    createTaaDescriptorSets();
    createHzbResources();
  }
#pragma endregion
#pragma endregion

#pragma region Vulkan Setup
#pragma region Surface Setup
  void createInstance() {
    constexpr vk::ApplicationInfo appInfo{.pApplicationName = "Hello Triangle",
                                          .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                          .pEngineName = "No Engine",
                                          .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                          .apiVersion = vk::ApiVersion14};

    // Get the required layers
    std::vector<char const *> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layerProperties](auto const &requiredLayer) {
      return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty) {
        return strcmp(layerProperty.layerName, requiredLayer) == 0;
      });
    });
    if (unsupportedLayerIt != requiredLayers.end()) {
      throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
    }

    // Get the required extensions.
    auto requiredExtensions = getRequiredInstanceExtensions();

    // Check if the required extensions are supported by the Vulkan
    // implementation.
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

  void createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
  }
#pragma endregion

#pragma region Device Selection
  bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice) {
    // Check if the physicalDevice supports the Vulkan 1.3 API version
    bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

    // Check if any of the queue families support both graphics and presentation
    // to our surface
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    uint32_t qfpIndex = 0;
    bool supportsGraphicsAndPresent =
        std::ranges::any_of(queueFamilies, [&physicalDevice, &surface = this->surface, &qfpIndex](auto const &qfp) {
          bool const suitable = (qfp.queueFlags & vk::QueueFlagBits::eGraphics) &&
                                (qfp.queueFlags & vk::QueueFlagBits::eCompute) &&
                                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface);
          qfpIndex++;
          return suitable;
        });

    // Check if all required physicalDevice extensions are available
    auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions =
        std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
          return std::ranges::any_of(
              availableDeviceExtensions, [requiredDeviceExtension](auto const &availableDeviceExtension) {
                return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
              });
        });

    // Check if the physicalDevice supports the required features
    auto features =
        physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
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
    const auto &limits = physicalDevice.getProperties().limits;
    const bool supportsFourSampleRendering =
        (limits.framebufferColorSampleCounts & sampleCount) && (limits.framebufferDepthSampleCounts & sampleCount);
    const auto motionFeatures = physicalDevice.getFormatProperties(MOTION_FORMAT).optimalTilingFeatures;
    const auto requiredMotionFeatures =
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage;
    const bool supportsMotionFormat = (motionFeatures & requiredMotionFeatures) == requiredMotionFeatures;
    const auto historyFeatures = physicalDevice.getFormatProperties(TAA_HISTORY_FORMAT).optimalTilingFeatures;
    const auto requiredHistoryFeatures =
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage;
    const bool supportsHistoryFormat = (historyFeatures & requiredHistoryFeatures) == requiredHistoryFeatures;

    // Return true if the physicalDevice meets all the criteria
    return supportsVulkan1_3 && supportsGraphicsAndPresent && supportsAllRequiredExtensions &&
           supportsRequiredFeatures && supportsFourSampleRendering && supportsMotionFormat && supportsHistoryFormat;
  }

  void pickPhysicalDevice() {
    std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    auto const devIter = std::ranges::find_if(
        physicalDevices, [&](auto const &physicalDevice) { return isDeviceSuitable(physicalDevice); });
    if (devIter == physicalDevices.end()) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
    physicalDevice = *devIter;
  }
#pragma endregion

#pragma region Logical Device
  void createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports both
    // graphics and present
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
          (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
          physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
        // found a queue family that supports both graphics and present
        queueIndex = qfpIndex;
        break;
      }
    }
    if (queueIndex == ~0) {
      throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
    }

    // query for Vulkan 1.3 features
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

    // create a Device
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
         // This frame's opaque depth (resolved by the opaque pass) for the water fog. Only fragWater uses it, and
         // only the translucent sets have it written (updateFrameDescriptorSets).
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
  void createGraphicsPipeline() {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

    // Separate module on purpose: with the task and mesh entry points in one SPIR-V module (two
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

    // Cull mode is dynamic so the LOD freeze (G) can turn off hardware backface culling: the frozen view's
    // shader-side culling then decides alone which faces are drawn.
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
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    // Identical state; vertex input stays empty because the vertex shader pulls from QuadBuffer itself.
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 2;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = vertexPullingStages;
    vertexPullingPipeline =
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    rasterizer.polygonMode = vk::PolygonMode::eLine;
    wireframeVertexPullingPipeline =
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 3;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = shaderStages;
    wireframePipeline =
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    // Water: fragWater (adds the water fog, reading the opaque pass's depth), blended over what is drawn, keeping the
    // terrain's depth and motion.
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
    waterPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().stageCount = 2;
    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().pStages = vertexPullingStages;
    waterVertexPullingPipeline =
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

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
    taaPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  // Text box (alpha blend) and crosshair (inverting blend: 1 - destination); both use OverlayParams push constants.
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
    selectedTexturePipeline =
        createOverlayStylePipeline(shaderModule, "selectedTextureVertexMain", "selectedTextureFragmentMain",
                                   vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha);
    crosshairPipeline = createOverlayStylePipeline(shaderModule, "crosshairVertexMain", "crosshairFragmentMain",
                                                   vk::BlendFactor::eOneMinusDstColor, vk::BlendFactor::eZero);
    createOutlinePipeline();
  }

  // Scene-pass pipeline (MSAA, color + motion, depth test without writes) using the scene descriptor set's uniform
  // buffer, so the outline is jittered and reprojected like the terrain and TAA treats it the same way.
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
        .stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = sizeof(OutlineParams)};
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
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  // Last draw of the scene pass (viewport and scissor are still the scene's): the targeted block's outline.
  void recordOutline(vk::raii::CommandBuffer &commandBuffer) {
    if (!outlineTarget) {
      return;
    }
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *outlinePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *outlinePipelineLayout, 0,
                                     *descriptorSets[frameIndex], nullptr);
    // A hair larger than the block, so its edges sit just outside the faces they outline.
    constexpr float grow = 0.002f;
    const glm::vec3 block{static_cast<float>((*outlineTarget)[0]), static_cast<float>((*outlineTarget)[1]),
                          static_cast<float>((*outlineTarget)[2])};
    const float lineWidth = std::max(1.0f, std::round(static_cast<float>(swapChainExtent.height) / 400.0f));
    const OutlineParams params{.boxMin = glm::vec4(block - grow, lineWidth),
                               .boxMax = glm::vec4(block + 1.0f + grow, 0.0f)};
    commandBuffer.pushConstants(*outlinePipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(params), &params);
    commandBuffer.draw(72, 1, 0, 0);
  }

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
    return vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  // Glyph index in overlay.slang's FONT table; characters it doesn't have become spaces.
  static uint8_t overlayGlyphIndex(char c) {
    static constexpr std::string_view glyphs = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:.+-/()";
    if (c >= 'a' && c <= 'z') {
      c = static_cast<char>(c - 'a' + 'A');
    }
    const size_t index = glyphs.find(c);
    return index == std::string_view::npos ? 0u : static_cast<uint8_t>(index);
  }

  // Top-right debug text: FPS, the active geometry path, terrain view and player block coordinates. Drawn on top of the
  // presented image (after TAA).
  void recordOverlay(vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) {
    const std::array<std::string, OVERLAY_LINES> lines{
        "FPS " + std::to_string(static_cast<int>(std::lround(displayedFps))),
        std::string(useVertexPulling ? "VERTEX " : "MESH ") + (player.waitingForTerrain()                 ? "LOADING"
                                                               : player.mode() == Player::Mode::Spectator ? "FREECAM"
                                                               : player.mode() == Player::Mode::Flying    ? "FLYING"
                                                               : player.swimming()                        ? "SWIMMING"
                                                                                                          : "WALKING"),
        std::string("VIEW ") + terrainViewName(terrainView),
        "XYZ " + std::to_string(static_cast<int64_t>(std::floor(player.feetPosition().x))) + " " +
            std::to_string(static_cast<int64_t>(std::floor(player.feetPosition().y))) + " " +
            std::to_string(static_cast<int64_t>(std::floor(player.feetPosition().z)))};
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
      const std::string &line = lines[row];
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
                                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                sizeof(params), &params);
    commandBuffer.draw(6, 1, 0, 0);
    // Bottom-right: show the side texture for the block selected with keys 1-9.
    if (selectedBlockIndex < blockRegistry.blocks.size()) {
      // First frame of the side texture (frame strips occupy consecutive bindless slots).
      const uint32_t sideTexture = textureSlots[blockRegistry.blocks[selectedBlockIndex].faces[0].texture];
      OverlayParams textureParams{
          .screenSize = {static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height)},
          .scale = params.scale,
          .columns = sideTexture,
          .text = {}};
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *selectedTexturePipeline);
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *overlayPipelineLayout, 0,
                                       *descriptorSets[frameIndex], nullptr);
      commandBuffer.pushConstants(*overlayPipelineLayout,
                                  vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                  sizeof(textureParams), &textureParams);
      commandBuffer.draw(6, 1, 0, 0);
    }
    // Same layout and push constants. Hidden in freecam, where blocks can't be broken.
    if (player.mode() != Player::Mode::Spectator) {
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *crosshairPipeline);
      commandBuffer.draw(18, 1, 0, 0);
    }
    commandBuffer.endRendering();
  }

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
    auto create = [&](const char *entry) {
      return vk::raii::Pipeline(
          device, nullptr,
          vk::ComputePipelineCreateInfo{
              .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = entry},
              .layout = hzbPipelineLayout});
    };
    hzbBuildLowPipeline = create("hzbBuildLow");
    hzbBuildHighPipeline = create("hzbBuildHigh");
  }

  // Swapchain-sized: the pyramid, its views and descriptor sets, and the task shader's binding 7.
  void createHzbResources() {
    auto halfNextPowerOfTwo = [](uint32_t v) {
      uint32_t p = 1;
      while (p < v) {
        p <<= 1u;
      }
      return std::max(p / 2u, 1u);
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

    const uint32_t setCount = static_cast<uint32_t>(swapChainImages.size());
    const std::array<vk::DescriptorPoolSize, 2> poolSizes{
        {{.type = vk::DescriptorType::eSampledImage, .descriptorCount = setCount},
         {.type = vk::DescriptorType::eStorageImage, .descriptorCount = MAX_HZB_MIPS * setCount}}};
    hzbDescriptorPool = vk::raii::DescriptorPool(
        device, vk::DescriptorPoolCreateInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                             .maxSets = setCount,
                                             .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                             .pPoolSizes = poolSizes.data()});
    std::vector<vk::DescriptorSetLayout> layouts(setCount, *hzbSetLayout);
    hzbSets = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
        .descriptorPool = *hzbDescriptorPool, .descriptorSetCount = setCount, .pSetLayouts = layouts.data()});

    std::array<vk::DescriptorImageInfo, MAX_HZB_MIPS> mipInfos{};
    for (uint32_t mip = 0; mip < MAX_HZB_MIPS; ++mip) {
      mipInfos[mip] = {.imageView = *hzbMipViews[std::min(mip, hzbMipCount - 1u)],
                       .imageLayout = vk::ImageLayout::eGeneral};
    }
    for (size_t i = 0; i < hzbSets.size(); ++i) {
      const vk::DescriptorImageInfo depthInfo{.imageView = *resolvedDepthImageViews[i],
                                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
      const std::array<vk::WriteDescriptorSet, 2> writes{{{.dstSet = hzbSets[i],
                                                           .dstBinding = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eSampledImage,
                                                           .pImageInfo = &depthInfo},
                                                          {.dstSet = hzbSets[i],
                                                           .dstBinding = 1,
                                                           .descriptorCount = MAX_HZB_MIPS,
                                                           .descriptorType = vk::DescriptorType::eStorageImage,
                                                           .pImageInfo = mipInfos.data()}}};
      device.updateDescriptorSets(writes, {});
    }

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

  // After the scene pass: depth (MSAA) -> Hi-Z mip 0 -> ... -> last mip, for the next frame's taskMain.
  void recordHzb(vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) {
    const vk::ImageSubresourceRange allMips{vk::ImageAspectFlagBits::eColor, 0, hzbMipCount, 0, 1};
    // The resolved depth is already SHADER_READ_ONLY and visible to compute (barrier after the opaque pass); the
    // water pass only read it since.
    const std::array<vk::ImageMemoryBarrier2, 2> before{
        {{.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
          .srcAccessMask = vk::AccessFlagBits2::eNone,
          .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
          .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
          .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
          .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
          .image = *resolvedDepthImages[imageIndex],
          .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}},
         // This frame's taskMain read the pyramid; finish that before overwriting it.
         {.srcStageMask =
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
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *hzbPipelineLayout, 0, *hzbSets[imageIndex],
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

  void createChunkComputePipeline() {
    std::array<vk::DescriptorSetLayoutBinding, 20> bindings{};
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
      return vk::raii::Pipeline(device, nullptr,
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
  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       .queueFamilyIndex = queueIndex};
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }
#pragma endregion

#pragma region Depth Image
  void createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    sceneColorImages.reserve(swapChainImages.size());
    sceneColorImageMemories.reserve(swapChainImages.size());
    sceneColorImageViews.reserve(swapChainImages.size());
    multisampleColorImages.reserve(swapChainImages.size());
    multisampleColorImageMemories.reserve(swapChainImages.size());
    multisampleColorImageViews.reserve(swapChainImages.size());
    multisampleMotionImages.reserve(swapChainImages.size());
    multisampleMotionImageMemories.reserve(swapChainImages.size());
    multisampleMotionImageViews.reserve(swapChainImages.size());
    motionImages.reserve(swapChainImages.size());
    motionImageMemories.reserve(swapChainImages.size());
    motionImageViews.reserve(swapChainImages.size());
    depthImages.reserve(swapChainImages.size());
    depthImageMemories.reserve(swapChainImages.size());
    depthImageViews.reserve(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); ++i) {
      auto [sceneColorImage, sceneColorImageMemory] = createImage(
          swapChainExtent.width, swapChainExtent.height, swapChainSurfaceFormat.format, vk::ImageTiling::eOptimal,
          vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
              vk::ImageUsageFlagBits::eTransferSrc,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
      sceneColorImages.emplace_back(std::move(sceneColorImage));
      sceneColorImageMemories.emplace_back(std::move(sceneColorImageMemory));
      sceneColorImageViews.emplace_back(
          createImageView(sceneColorImages.back(), swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));

      auto [colorImage, colorImageMemory] = createImage(
          swapChainExtent.width, swapChainExtent.height, swapChainSurfaceFormat.format, vk::ImageTiling::eOptimal,
          vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, 1, RENDER_SAMPLE_COUNT);
      multisampleColorImages.emplace_back(std::move(colorImage));
      multisampleColorImageMemories.emplace_back(std::move(colorImageMemory));
      multisampleColorImageViews.emplace_back(createImageView(
          multisampleColorImages.back(), swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));

      auto [motionImage, motionImageMemory] =
          createImage(swapChainExtent.width, swapChainExtent.height, MOTION_FORMAT, vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);
      motionImages.emplace_back(std::move(motionImage));
      motionImageMemories.emplace_back(std::move(motionImageMemory));
      motionImageViews.emplace_back(
          createImageView(motionImages.back(), MOTION_FORMAT, vk::ImageAspectFlagBits::eColor));

      auto [multisampleMotionImage, multisampleMotionImageMemory] = createImage(
          swapChainExtent.width, swapChainExtent.height, MOTION_FORMAT, vk::ImageTiling::eOptimal,
          vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, 1, RENDER_SAMPLE_COUNT);
      multisampleMotionImages.emplace_back(std::move(multisampleMotionImage));
      multisampleMotionImageMemories.emplace_back(std::move(multisampleMotionImageMemory));
      multisampleMotionImageViews.emplace_back(
          createImageView(multisampleMotionImages.back(), MOTION_FORMAT, vk::ImageAspectFlagBits::eColor));

      // Never sampled (Hi-Z and the water fog read the resolve below), so the driver can keep it compressed.
      auto [depthImage, depthImageMemory] =
          createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, 1,
                      RENDER_SAMPLE_COUNT);
      depthImages.emplace_back(std::move(depthImage));
      depthImageMemories.emplace_back(std::move(depthImageMemory));
      depthImageViews.emplace_back(createImageView(depthImages.back(), depthFormat, vk::ImageAspectFlagBits::eDepth));

      auto [resolvedDepthImage, resolvedDepthImageMemory] =
          createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);
      resolvedDepthImages.emplace_back(std::move(resolvedDepthImage));
      resolvedDepthImageMemories.emplace_back(std::move(resolvedDepthImageMemory));
      resolvedDepthImageViews.emplace_back(
          createImageView(resolvedDepthImages.back(), depthFormat, vk::ImageAspectFlagBits::eDepth));
    }

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

  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features) {
    for (const auto format : candidates) {
      vk::FormatProperties props = physicalDevice.getFormatProperties(format);
      if (((tiling == vk::ImageTiling::eLinear) && ((props.linearTilingFeatures & features) == features)) ||
          ((tiling == vk::ImageTiling::eOptimal) && ((props.optimalTilingFeatures & features) == features))) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage);
  }
#pragma endregion

#pragma region Texture Setup
  // One 16x16 RGBA frame (64 bytes per row) into a mipmapped texture.
  void createTextureImage(const stbi_uc *pixels, vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory) {
    constexpr int texWidth = 16;
    constexpr int texHeight = 16;
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

    // A texture file is 16 px wide and 16 * N tall: N animation frames, top first. Each frame is its own texture in
    // the bindless array, at consecutive slots (textureSlots / textureFrames, by registry texture index).
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
        createTextureImage(pixels + static_cast<size_t>(frame) * 16u * 16u * 4u, textureImages.back(),
                           textureImageMemories.back());
      }
      stbi_image_free(pixels);
    }
  }

  // Uploads the per-face block table once; the fragment shader reads it by block ID and normal.
  void createBlockFaceBuffer() {
    // Indexed by block ID, which has a gap between the opaque (0..) and non-opaque (512..) ranges; gaps stay zero.
    std::vector<GpuBlockFace> faces((blockRegistry.maxId() + 1u) * blocks::kFaceCount, GpuBlockFace{});
    for (const blocks::Block &block : blockRegistry.blocks) {
      for (size_t f = 0; f < blocks::kFaceCount; ++f) {
        const blocks::Face &face = block.faces[f];
        const uint32_t lodTexture = face.hasLodTexture ? face.lodTexture : face.texture;
        const uint32_t frameMs =
            static_cast<uint32_t>(std::lround(std::clamp(block.frameTime, 0.001f, 60.0f) * 1000.0f));
        faces[block.id * blocks::kFaceCount + f] = {
            .texture = textureSlots[face.texture],
            .lodTexture = textureSlots[lodTexture],
            .flags = (face.randomRotation ? BLOCK_FACE_RANDOM_ROTATION : 0u) |
                     (face.hasLodTexture ? BLOCK_FACE_HAS_LOD_TEXTURE : 0u) |
                     (block.fluid != blocks::Fluid::None ? BLOCK_FACE_FLUID : 0u),
            .animation = textureFrames[face.texture] | (textureFrames[lodTexture] << 8u) | (frameMs << 16u),
            .lodTint = {block.lodTint[0], block.lodTint[1], block.lodTint[2], 1.0f},
            .tint = {block.tint[0], block.tint[1], block.tint[2], 1.0f}};
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

  void createTreeLodTableBuffer() {
    const std::vector<uint32_t> table = vegetation::buildTreeLodTable();
    const vk::DeviceSize size = table.size() * sizeof(uint32_t);
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data = stagingBufferMemory.mapMemory(0, size);
    std::memcpy(data, table.data(), static_cast<size_t>(size));
    stagingBufferMemory.unmapMemory();

    std::tie(treeLodTableBuffer, treeLodTableBufferMemory) =
        createBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyBuffer(stagingBuffer, treeLodTableBuffer, size);
  }

  void createTextureImageViews() {
    textureImageViews.reserve(textureImages.size());
    for (const auto &image : textureImages) {
      textureImageViews.emplace_back(
          createImageView(*image, TEXTURE_FORMAT, vk::ImageAspectFlagBits::eColor, TEXTURE_MIP_LEVELS));
    }
  }

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

    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                     .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
    vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(imageMemory, 0);

    return {std::move(image), std::move(imageMemory)};
  }
#pragma endregion

#pragma region Upload Helpers
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
  std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                                   vk::MemoryPropertyFlags properties) {
    vk::BufferCreateInfo bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
    vk::raii::Buffer buffer = vk::raii::Buffer(device, bufferInfo);
    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                     .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
    vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
    buffer.bindMemory(*bufferMemory, 0);
    return {std::move(buffer), std::move(bufferMemory)};
  }

  void createUniformBuffers() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
      auto [buffer, bufferMem] =
          createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      uniformBuffers.emplace_back(std::move(buffer));
      uniformBuffersMemory.emplace_back(std::move(bufferMem));
      uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
    }
  }

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

    constexpr uint32_t HEIGHT_STRIDE = CHUNK_SIZE + 2u;
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

    // Static {0,1,2, 0,2,3} + 4q pattern for VERTEX_QUADS_PER_INSTANCE quads; written once, never changes.
    {
      const vk::DeviceSize indexBufferSize = sizeof(uint16_t) * 6u * VERTEX_QUADS_PER_INSTANCE;
      auto [indexBufferValue, indexBufferMemoryValue] =
          createBuffer(indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      quadIndexBuffer = std::move(indexBufferValue);
      quadIndexBufferMemory = std::move(indexBufferMemoryValue);
      auto *indices = static_cast<uint16_t *>(quadIndexBufferMemory.mapMemory(0, indexBufferSize));
      for (uint32_t quad = 0; quad < VERTEX_QUADS_PER_INSTANCE; ++quad) {
        const auto base = static_cast<uint16_t>(quad * 4u);
        const std::array<uint16_t, 6> quadIndices{
            base, static_cast<uint16_t>(base + 1u), static_cast<uint16_t>(base + 2u),
            base, static_cast<uint16_t>(base + 2u), static_cast<uint16_t>(base + 3u)};
        std::memcpy(indices + quad * 6u, quadIndices.data(), sizeof(quadIndices));
      }
      quadIndexBufferMemory.unmapMemory();
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

  // Everything sized by workPoolCapacity (bindings 5, 6, 13, 16, 17 and 18). Contents start undefined: a full rebuild
  // (clears + initWorkPool) always follows.
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

  // allocateWork ran out of work slots: grow the pool to fit `demand` slots (with headroom, and at least double) and
  // remesh everything into it. Rare (the pool keeps its size afterwards), so it simply waits for the GPU to go idle
  // before replacing the buffers.
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
      for (size_t b = 0; b < computeBindings.size(); ++b) {
        infos[b] = {.buffer = computeBindings[b].second, .offset = 0, .range = vk::WholeSize};
        writes[b] = {.dstSet = chunkComputeDescriptorSets[i],
                     .dstBinding = computeBindings[b].first,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                     .pBufferInfo = &infos[b]};
      }
      // The GPU clipmap's graphics sets read the clusters too (task shader, binding 6).
      const size_t opaque = computeBindings.size();
      infos[opaque] = {.buffer = *quadClusterBuffer, .offset = 0, .range = vk::WholeSize};
      writes[opaque] = {.dstSet = descriptorSets[i],
                        .dstBinding = 6,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &infos[opaque]};
      infos[opaque + 1] = {.buffer = *waterClusterBuffer, .offset = 0, .range = vk::WholeSize};
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
    // Start with no clusters.
    auto commandBuffer = beginSingleTimeCommands();
    commandBuffer.fillBuffer(*cpuClusterBuffer, 0, vk::WholeSize, 0u);
    commandBuffer.fillBuffer(*cpuTranslucentClusterBuffer, 0, vk::WholeSize, 0u);
    commandBuffer.fillBuffer(*cpuQuadCountBuffer, 0, vk::WholeSize, 0u);
    endSingleTimeCommands(std::move(commandBuffer));
  }

  // Breaks (left) and places (right) on click / hold, then picks the block to outline (after editing, so the
  // outline moves on at once).
  void updateBlockInteraction(const glm::vec3 &eye) {
    const bool clicked = std::exchange(breakClickPending, false);
    const bool placeClicked = std::exchange(placeClickPending, false);
    outlineTarget.reset();
    cpuWorld->tickFluids(glfwGetTime()); // ocean flowing into dug cells, one block per flow step
    if (player.mode() == Player::Mode::Spectator) {
      return;
    }
    const glm::vec3 &front = camera.front();
    const auto target = [&] {
      return cpuWorld->raycast({eye.x, eye.y, eye.z}, {front.x, front.y, front.z}, BLOCK_REACH);
    };
    std::optional<cpuworld::BlockHit> hit = target();
    const double now = glfwGetTime();
    const bool held = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (hit && camera.cursorCaptured() && (clicked || (held && now >= nextBreakTime))) {
      nextBreakTime = now + BLOCK_BREAK_REPEAT_SECONDS;
      cpuWorld->breakBlock(hit->block[0], hit->block[1], hit->block[2]);
      hit = target();
    }
    const bool placeHeld = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (hit && camera.cursorCaptured() && (placeClicked || (placeHeld && now >= nextPlaceTime)) &&
        hit->normal != cpuworld::BlockXYZ{}) {
      nextPlaceTime = now + BLOCK_PLACE_REPEAT_SECONDS;
      const cpuworld::BlockXYZ cell{hit->block[0] + hit->normal[0], hit->block[1] + hit->normal[1],
                                    hit->block[2] + hit->normal[2]};
      if (!player.overlapsBlock(cell[0], cell[1], cell[2])) {
        const auto id = static_cast<uint16_t>(blockRegistry.blocks[selectedBlockIndex].id);
        if (cpuWorld->placeBlock(cell[0], cell[1], cell[2], id)) {
          hit = target();
        }
      }
    }
    if (hit) {
      outlineTarget = hit->block;
    }
  }

  void createCpuWorld() {
    cpuworld::TerrainBlockIds ids;
    for (size_t layer = 0; layer < ids.top.size(); ++layer) {
      ids.top[layer] = static_cast<uint16_t>(terrainLayerBlocks[layer]);
      ids.filler[layer] = static_cast<uint16_t>(terrainLayerFillerBlocks[layer]);
      ids.deep[layer] = static_cast<uint16_t>(terrainLayerDeepBlocks[layer]);
    }
    ids.ocean = static_cast<uint16_t>(blockRegistry.idOf("ocean"));
    ids.log = static_cast<uint16_t>(treeLogBlock);
    ids.leaves = static_cast<uint16_t>(treeLeavesBlock);
    oceanBlockId = ids.ocean;
    cpuWorld = std::make_unique<cpuworld::CpuWorld>(ids);
    const int32_t spawnHeight =
        std::max(static_cast<int32_t>(std::floor(terrain::sampleHeight(0, 0))), world::kMinTerrainHeight);
    player.spawn({0.5, static_cast<double>(spawnHeight), 0.5});
  }

  static uint64_t cpuColumnKey(int32_t cx, int32_t cz) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32u) | static_cast<uint32_t>(cz);
  }

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

  // Per frame, after updateVisibleClipmap: moves the CPU square with the player, shows exactly the columns of the
  // applied clipmap's square, turns finished / unloaded columns into this frame's transfers (staging[frame]), and
  // offers the ready square to the clipmap once it is uploaded.
  void updateCpuWorld(const glm::vec3 &eye, uint32_t frame) {
    const cpuworld::ChunkXZ desired{cpuworld::floorDiv(static_cast<int32_t>(std::floor(eye.x)), CHUNK_SIZE),
                                    cpuworld::floorDiv(static_cast<int32_t>(std::floor(eye.z)), CHUNK_SIZE)};
    const std::optional<cpuworld::ChunkXZ> applied =
        lastAppliedClipmapView ? lastAppliedClipmapView->cpuRegionCenter : std::nullopt;
    const auto shouldShow = [&](int32_t cx, int32_t cz) {
      return applied && cpuworld::CpuWorld::inSquare(*applied, cx, cz);
    };
    std::vector<cpuworld::ChunkXZ> pinned;
    for (const auto &center : {applied, cpuAdvertisedCenter}) {
      if (center) {
        pinned.push_back(*center);
      }
    }
    for (const auto &[sequence, center] : inFlightCpuCenters) {
      pinned.push_back(center);
    }
    cpuWorld->update(desired, pinned);
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
    // written (768 bytes per chunk), so it always fits in the staging buffer.
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
      const auto release = [&](const CpuChunkAllocation &chunk) {
        cpuClearedSlots.push_back(chunk.slot);
        cpuQuadRanges.release(chunk.quadOffset, chunk.quadCount);
        cpuChunkSlots.release(chunk.slot);
        cpuQuadCountDirty = true;
      };
      const auto upload = [&](CpuColumnGpu &column, cpuworld::ChunkMesh &mesh) {
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
              release(chunk);
              return true;
            });
            upload(column, mesh);
          }
        }
      } else {
        if (existing != cpuColumns.end()) {
          for (const CpuChunkAllocation &chunk : existing->second.chunks) {
            release(chunk);
          }
          cpuColumns.erase(existing);
        }
        if (!event.removed) {
          CpuColumnGpu column{event.cx, event.cz, shouldShow(event.cx, event.cz), {}};
          for (cpuworld::ChunkMesh &mesh : event.chunks) {
            upload(column, mesh);
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

  // Clears (hidden / freed clusters), then copies from this frame's staging buffer, before any of this frame's draws.
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
      const std::array<uint32_t, 2> count{cpuQuadRanges.highWater(), 0u};
      commandBuffer.updateBuffer<uint32_t>(*cpuQuadCountBuffer, 0, count);
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
    // Synchronous fallback only when the applied set is beyond the stale tolerance (first frame, FOV/LOD change,
    // teleport, a fast turn on the frustum-culled path). Normal movement keeps the previous set until the async
    // result lands, even when the camera outruns the grace distance meanwhile (fast flight, slow debug builds):
    // rebuilding the clipmap on this thread then stalled the frame for several ms on every async round trip.
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
  // Submit a new async cull once the camera leaves the culling margin (2 chunks / 5 degrees). Keep using the
  // previous set while that job is in flight, up to the stale tolerance (12 chunks / 7.5 degrees); past it the main
  // thread builds the set itself. The position limit is loose because the set covers the whole area around the
  // camera: a lagging one only has slightly coarser LODs near the camera and a few chunks missing at the far
  // horizon. The direction limit stays tight, because the frustum-culled set has nothing outside the view.
  // A new CPU square needs a new clipmap (submit), but the old one stays valid meanwhile because the CPU keeps the
  // old square loaded (stale cover ignores it).
  static constexpr ClipmapCoverTolerance kClipmapSubmitCover{static_cast<float>(CHUNK_SIZE) * 2.0f, 0.9961947f, true};
  static constexpr ClipmapCoverTolerance kClipmapStaleCover{static_cast<float>(CHUNK_SIZE) * 12.0f, 0.9914449f, false};

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
    // Without frustum culling, and unzoomed (same LOD distances inside and outside the frustum), the result doesn't
    // depend on the view direction, so turning never forces a rebuild.
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

    // Sets per frame with the same layout: the GPU-meshed clipmap, the CPU-meshed near chunks, and the near chunks'
    // translucent clusters (same buffers but binding 6).
    for (size_t setIndex = 0; setIndex < SCENE_SET_KINDS * MAX_FRAMES_IN_FLIGHT; setIndex++) {
      const size_t i = setIndex % MAX_FRAMES_IN_FLIGHT;
      // Kinds: 0 GPU clipmap, 1 CPU chunks, 2 CPU translucent, 3 GPU translucent.
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
      // Binding 7 (Hi-Z) is written by writeHzbDescriptors, which runs again whenever the swapchain is rebuilt.
      device.updateDescriptorSets(descriptorWrites, {});
    }

    std::vector<vk::DescriptorSetLayout> computeLayouts(MAX_FRAMES_IN_FLIGHT, chunkComputeDescriptorSetLayout);
    chunkComputeDescriptorSets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{.descriptorPool = descriptorPool,
                                      .descriptorSetCount = static_cast<uint32_t>(computeLayouts.size()),
                                      .pSetLayouts = computeLayouts.data()});

    const vk::DeviceSize coordinateBufferSize = sizeof(world::ChunkLod) * MAX_HORIZONTAL_CHUNK_COUNT;
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      std::array<vk::DescriptorBufferInfo, 20> bufferInfos{{
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
          {.buffer = treeLodTableBuffer, .offset = 0, .range = vk::WholeSize},
      }};
      std::array<vk::WriteDescriptorSet, 20> writes{};
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

  void createTaaDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, taaDescriptorSetLayout);
    taaDescriptorSets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{.descriptorPool = descriptorPool,
                                      .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                                      .pSetLayouts = layouts.data()});
  }

  void updateFrameDescriptorSets(uint32_t frame, uint32_t imageIndex) {
    const uint32_t historyReadIndex = 1u - taaHistoryWriteIndex;
    const std::array<vk::DescriptorImageInfo, 4> imageInfos{
        {{.imageView = sceneColorImageViews[imageIndex], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
         {.imageView = taaHistoryImageViews[historyReadIndex], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
         {.imageView = motionImageViews[imageIndex], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
         {.sampler = taaSampler}}};
    std::array<vk::WriteDescriptorSet, 4> writes{};
    for (uint32_t binding = 0; binding < writes.size(); ++binding) {
      writes[binding] = {.dstSet = taaDescriptorSets[frame],
                         .dstBinding = binding,
                         .dstArrayElement = 0,
                         .descriptorCount = 1,
                         .descriptorType =
                             binding == 3u ? vk::DescriptorType::eSampler : vk::DescriptorType::eSampledImage,
                         .pImageInfo = &imageInfos[binding]};
    }
    device.updateDescriptorSets(writes, {});

    // Translucent scene sets: this swapchain image's opaque depth, for the water fog (fragWater).
    const vk::DescriptorImageInfo depthInfo{.imageView = resolvedDepthImageViews[imageIndex],
                                            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    const std::array<vk::DescriptorSet, 2> translucentSets{*cpuTranslucentDescriptorSets[frame],
                                                           *gpuTranslucentDescriptorSets[frame]};
    std::array<vk::WriteDescriptorSet, 2> depthWrites{};
    for (size_t i = 0; i < depthWrites.size(); ++i) {
      depthWrites[i] = {.dstSet = translucentSets[i],
                        .dstBinding = 8,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eSampledImage,
                        .pImageInfo = &depthInfo};
    }
    device.updateDescriptorSets(depthWrites, {});
  }

#pragma endregion
#pragma endregion

#pragma region Command Setup
#pragma region Single-Time Submit
  vk::raii::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
    vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo);

    return std::move(commandBuffer);
  }

  void endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
  }
#pragma endregion

#pragma region Buffer Helpers
  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size) {
    vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
    commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size = size});
    endSingleTimeCommands(std::move(commandCopyBuffer));
  }

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }
#pragma endregion

#pragma region Command Allocation
  void createCommandBuffers() {
    commandBuffers.clear();
    vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool,
                                            .level = vk::CommandBufferLevel::ePrimary,
                                            .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }
#pragma endregion

#pragma region Command Recording
  void recordCommandBuffer(uint32_t imageIndex) {
    auto &commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});
    recordCpuUploads(commandBuffer);
    recordChunkCompute(commandBuffer);

    const auto colorOutputStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    const auto colorWrite = vk::AccessFlagBits2::eColorAttachmentWrite;
    transition_image_layout(*multisampleColorImages[imageIndex], vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite,
                            vk::PipelineStageFlagBits2::eTopOfPipe, colorOutputStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(*sceneColorImages[imageIndex], vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite,
                            vk::PipelineStageFlagBits2::eTopOfPipe, colorOutputStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(*multisampleMotionImages[imageIndex], vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite,
                            vk::PipelineStageFlagBits2::eTopOfPipe, colorOutputStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(*motionImages[imageIndex], vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite,
                            vk::PipelineStageFlagBits2::eTopOfPipe, colorOutputStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(
        *depthImages[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);
    // Depth resolves run in the color attachment output stage with color attachment access. Every frame resolves it:
    // the water pass reads it, and the Hi-Z build too when it runs. Last read by those on this swapchain image.
    transition_image_layout(*resolvedDepthImages[imageIndex], vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eDepthAttachmentOptimal, {}, colorWrite,
                            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
                            colorOutputStage, vk::ImageAspectFlagBits::eDepth);
    transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {}, colorWrite,
                            vk::PipelineStageFlagBits2::eTopOfPipe, colorOutputStage, vk::ImageAspectFlagBits::eColor);
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

    // The scene draws in two passes over the same MSAA targets. The opaque pass resolves its depth (farthest sample)
    // into resolvedDepth and stores the MSAA targets; the water pass loads them, reads that depth (water fog), and
    // resolves color and motion for TAA.
    std::array<vk::RenderingAttachmentInfo, 2> sceneAttachments{
        {{.imageView = multisampleColorImageViews[imageIndex],
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = clearColor},
         {.imageView = multisampleMotionImageViews[imageIndex],
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)}}};

    vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = depthImageViews[imageIndex],
                                                       .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                                                       .resolveMode =
                                                           vk::ResolveModeFlagBits::eMin, // farthest sample (reverse-Z)
                                                       .resolveImageView = *resolvedDepthImageViews[imageIndex],
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
    // Negative height flips Y for Vulkan; see Uniform buffers.
    commandBuffer.setViewport(0, vk::Viewport(0.0f, static_cast<float>(swapChainExtent.height),
                                              static_cast<float>(swapChainExtent.width),
                                              -static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIndex],
                                     nullptr);
    const bool debugView = terrainView == TerrainView::Debug;
    uint32_t geometryPass = GEOMETRY_PASS_OPAQUE;
    const auto pushGeometry = [&](uint32_t clusterCount, uint32_t debugTint, uint32_t quadBase = 0u,
                                  uint32_t quadCount = 0u) {
      const GeometryPushConstants push{clusterCount, debugView ? debugTint : DEBUG_TINT_OFF, quadBase, quadCount,
                                       geometryPass};
      commandBuffer.pushConstants(*pipelineLayout, GEOMETRY_PUSH_STAGES, 0, sizeof(push), &push);
    };
    // 32 clusters per task group, spread over x and y to stay under the 65535 group limit per dimension.
    const auto drawClusters = [&](uint32_t clusterCount, uint32_t debugTint) {
      if (clusterCount == 0u) {
        return;
      }
      pushGeometry(clusterCount, debugTint);
      const uint32_t taskGroups = (clusterCount + 31u) / 32u;
      commandBuffer.drawMeshTasksEXT(std::min(taskGroups, 65535u), (taskGroups + 65534u) / 65535u, 1);
    };
    if (useVertexPulling) {
      commandBuffer.bindIndexBuffer(*quadIndexBuffer, 0, vk::IndexType::eUint16);
    }
    // GPU-meshed clipmap (LOD terrain, with the hole the CPU square fills).
    if (terrainView != TerrainView::RealOnly) {
      if (useVertexPulling) {
        pushGeometry(0u, DEBUG_TINT_LOD);
        commandBuffer.drawIndexedIndirect(*vertexDrawArgumentsBuffer, 0, 1, sizeof(vk::DrawIndexedIndirectCommand));
      } else {
        // Work slots are handed out on the GPU: finalizeWork wrote the group count from their high water. Clusters past
        // it (the folded y/z grid overshoots) are zero, so the bound is just the buffer.
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
        drawClusters(cpuChunkSlots.highWater() * QUAD_CLUSTERS_PER_CHUNK, DEBUG_TINT_CPU);
      }
    }
    commandBuffer.endRendering();

    // Opaque pass -> water pass: the depth resolve becomes a shader input (the Hi-Z build reads it later too), and the
    // MSAA targets' stores happen before the water pass loads and writes them.
    const vk::ImageMemoryBarrier2 opaqueDepthReady{.srcStageMask = colorOutputStage,
                                                   .srcAccessMask = colorWrite,
                                                   .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader |
                                                                   vk::PipelineStageFlagBits2::eComputeShader,
                                                   .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                                                   .oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                                                   .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                                   .image = *resolvedDepthImages[imageIndex],
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
        {{.imageView = multisampleColorImageViews[imageIndex],
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .resolveMode = vk::ResolveModeFlagBits::eAverage,
          .resolveImageView = sceneColorImageViews[imageIndex],
          .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eLoad,
          .storeOp = vk::AttachmentStoreOp::eDontCare},
         {.imageView = multisampleMotionImageViews[imageIndex],
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .resolveMode = vk::ResolveModeFlagBits::eAverage,
          .resolveImageView = motionImageViews[imageIndex],
          .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eLoad,
          .storeOp = vk::AttachmentStoreOp::eDontCare}}};
    // Depth-tested, never written: nothing after this pass reads the MSAA depth.
    const vk::RenderingAttachmentInfo waterDepthAttachment{.imageView = depthImageViews[imageIndex],
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

    // Translucent pass (ocean), after every opaque draw: blended over them, depth-tested against them.
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
        drawClusters(cpuChunkSlots.highWater() * QUAD_CLUSTERS_PER_CHUNK, DEBUG_TINT_CPU);
      }
      geometryPass = GEOMETRY_PASS_OPAQUE;
    }
    recordOutline(commandBuffer);
    commandBuffer.endRendering();
    if (!useVertexPulling && hzbOcclusionEnabled && !frozenLodView) {
      recordHzb(commandBuffer, imageIndex);
    }

    const auto fragmentStage = vk::PipelineStageFlagBits2::eFragmentShader;
    transition_image_layout(*sceneColorImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal, colorWrite, vk::AccessFlagBits2::eShaderRead,
                            colorOutputStage, fragmentStage, vk::ImageAspectFlagBits::eColor);
    transition_image_layout(*motionImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
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
                                     *taaDescriptorSets[frameIndex], nullptr);
    const TaaParams taaParams{
        {1.0f / static_cast<float>(swapChainExtent.width), 1.0f / static_cast<float>(swapChainExtent.height)},
        (TAA_ENABLED ? 2u : 0u) | (TAA_ENABLED && taaHistoryValid ? 1u : 0u),
        TAA_DEPTH_REJECTION_THRESHOLD};
    commandBuffer.pushConstants(*taaPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(taaParams),
                                &taaParams);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    transition_image_layout(*taaHistoryImages[taaHistoryWriteIndex], vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal, colorWrite, vk::AccessFlagBits2::eShaderRead,
                            colorOutputStage, fragmentStage, vk::ImageAspectFlagBits::eColor);
    recordOverlay(commandBuffer, imageIndex);
    transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::ePresentSrcKHR, colorWrite, {}, colorOutputStage,
                            vk::PipelineStageFlagBits2::eBottomOfPipe, vk::ImageAspectFlagBits::eColor);
    taaHistoryInitialized = true;
    taaHistoryValid = true;
    taaHistoryWriteIndex = historyReadIndex;
    commandBuffer.end();
  }

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
    commandBuffer.pushConstants(*chunkComputePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                sizeof(computeParams), &computeParams);
    // One item per Z group, split into batches under the device's Z group limit (65535 or more), each pushing its
    // first item as zBase. Ends with zBase back at 0.
    const uint32_t maxGroupsZ = std::min(physicalDevice.getProperties().limits.maxComputeWorkGroupCount[2], 65535u);
    const auto dispatchBatchedZ = [&](uint32_t groupsX, uint32_t groupsY, uint32_t items) {
      for (uint32_t base = 0; base < items; base += maxGroupsZ) {
        if (base != 0u) {
          commandBuffer.pushConstants(*chunkComputePipelineLayout, vk::ShaderStageFlagBits::eCompute,
                                      offsetof(ChunkComputeParams, zBase), sizeof(uint32_t), &base);
        }
        commandBuffer.dispatch(groupsX, groupsY, std::min(items - base, maxGroupsZ));
      }
      if (items > maxGroupsZ) {
        const uint32_t zero = 0u;
        commandBuffer.pushConstants(*chunkComputePipelineLayout, vk::ShaderStageFlagBits::eCompute,
                                    offsetof(ChunkComputeParams, zBase), sizeof(uint32_t), &zero);
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
      // Over the work slots below the old high water, as of the last finalizeWork: the ones past the new one are
      // empty now, so they add nothing.
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
      // One 8x8-column tile (one QuadCluster) per group. Two or four tiles per group measured the same or slower:
      // the pass is bound by registers and latency, not by the per-SM group limit.
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
  void createSyncObjects() {
    assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

    createRenderFinishedSemaphores();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
      inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
  }

  void createRenderFinishedSemaphores() {
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }
  }

#pragma endregion
#pragma endregion

#pragma region Frame Rendering
#pragma region Camera Update
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
    updateBlockInteraction(eye); // before updateCpuWorld, which uploads the remeshed chunks this frame

    const glm::mat4 model = glm::mat4(1.0f);
    const bool zoomed = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
    // LODs and the clipmap use the base FOV; the sprint widening (player.fovMultiplier) only touches the projection,
    // so its animation doesn't rebuild the clipmap every frame.
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
    const ClipmapView clipmapView{{eye.x, eye.y, eye.z},
                                  {camera.front().x, camera.front().y, camera.front().z},
                                  {cameraRight.x, cameraRight.y, cameraRight.z},
                                  {cameraUp.x, cameraUp.y, cameraUp.z},
                                  lodDistanceSquared,
                                  baseLodDistanceSquared,
                                  fov,
                                  aspectRatio,
                                  useVertexPulling,
                                  cpuAdvertisedCenter};
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
    // Reverse-Z (near and far swapped: depth 1 at the near plane, 0 at the far one). With the float depth buffer this
    // keeps roughly constant relative precision out to the far plane; standard Z lost all of it past a few thousand
    // blocks, and the ocean surface fought with the sea floor below it on coarse LODs. Explicitly the zero-to-one
    // variant: pch.hpp includes glm before GLM_FORCE_DEPTH_ZERO_TO_ONE above, so glm::perspective is -1..1, which
    // with near and far swapped puts everything past the near plane at negative depth (clipped).
    glm::mat4 proj = glm::perspectiveRH_ZO(
        projectionFov, static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), farPlane,
        nearPlane);
    const uint32_t jitterSample = (taaJitterIndex++ % 8u) + 1u;
    const float jitterX = TAA_ENABLED ? (halton(jitterSample, 2u) - 0.5f) * TAA_JITTER_PIXELS : 0.0f;
    const float jitterY = TAA_ENABLED ? (halton(jitterSample, 3u) - 0.5f) * TAA_JITTER_PIXELS : 0.0f;
    proj[2][0] -= 2.0f * jitterX / static_cast<float>(swapChainExtent.width);
    proj[2][1] += 2.0f * jitterY / static_cast<float>(swapChainExtent.height);
    ubo.mvp = proj * view * model;
    ubo.previousMvp = previousMvpValid ? previousMvp : ubo.mvp;
    ubo.renderSize = {static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height)};
    // Slots are final for this frame here (the remesh, if any, is recorded below in the same frame).
    ubo.clusterCount = workPoolCapacity * QUAD_CLUSTERS_PER_CHUNK;
    ubo.historyValid = TAA_ENABLED && taaHistoryValid ? 1u : 0u;
    ubo.lodAoLighteningPerLevel = LOD_AO_LIGHTENING_PER_LEVEL;
    ubo.nearPlane = nearPlane;
    ubo.farPlane = farPlane;
    // w: seconds for texture animation and fake waves, wrapped so a float keeps millisecond precision.
    ubo.cameraPosition = glm::vec4(eye, static_cast<float>(std::fmod(glfwGetTime(), 3600.0)));
    // z: 1 while the camera is inside an ocean cell (underwater fog in fragMain).
    cameraUnderwater =
        cpuWorld->blockId(static_cast<int32_t>(std::floor(eye.x)), static_cast<int32_t>(std::floor(eye.y)),
                          static_cast<int32_t>(std::floor(eye.z))) == oceanBlockId;
    // Border fog end: the clipmap's radius, less how far the applied clipmap may lag behind the camera (stale cover)
    // and a few chunks of slack, so the clipmap's edge is always inside opaque fog.
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

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    previousMvp = ubo.mvp;
    previousMvpValid = true;
  }
#pragma endregion

#pragma region Frame Submit
  void drawFrame() {
    // Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are
    // indexed by frameIndex,
    //       while renderFinishedSemaphores is indexed by imageIndex
    vk::Result result;
    uint32_t imageIndex;
    {
      const auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
      if (fenceResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for fence!");
      }
      if (quadCountReadbackPending[frameIndex]) {
        quadCountReadbackPending[frameIndex] = false;
        const uint32_t appended = quadCountReadbackMapped[frameIndex][0];
        const uint32_t live = quadCountReadbackMapped[frameIndex][1];
        const uint32_t freeSlots = quadCountReadbackMapped[frameIndex][2];
        const uint32_t slotHighWater = quadCountReadbackMapped[frameIndex][3];
        const uint32_t workPoolOverflow = quadCountReadbackMapped[frameIndex][5];
        const uint32_t liveSlots = workPoolCapacity > freeSlots ? workPoolCapacity - freeSlots : 0u;
        // Retired slots leave holes the draw and compaction still walk; packing them (with the quads) once they
        // pass a quarter of the live slots keeps that bounded.
        const bool slotsSparse = slotHighWater > liveSlots + liveSlots / 4u + 1024u;
        const uint32_t dead = appended > live ? appended - live : 0u;
        const uint32_t deadLimit = useVertexPulling ? live / 4u : live;
        const bool garbageHeavy = dead > deadLimit && dead > REBUILD_MIN_DEAD_QUADS;
        const bool nearlyFull = appended > MAX_QUAD_COUNT / 4u * 3u && live < MAX_QUAD_COUNT / 2u;
        if (quadCountReadbackEpoch[frameIndex] != quadCountEpoch) {
          // Recorded before the last compaction or rebuild: these counts no longer describe QuadBuffer.
        } else if (workPoolOverflow != 0u) {
          // allocateWork ran out of work slots and dropped chunks: a bigger pool, and a rebuild to fill it. Failed pops
          // still took their count off the stack top, so capacity - top (top gone negative) is everything asked for.
          const int64_t top = static_cast<int32_t>(freeSlots);
          growWorkPool(static_cast<uint64_t>(std::max<int64_t>(int64_t{workPoolCapacity} - top, 0)));
        } else if (appended > MAX_QUAD_COUNT) {
          // Quads past the end were dropped (holes). A rebuild only helps if the set got smaller since; say so once,
          // since a set that doesn't fit rebuilds every frame.
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
    updateFrameDescriptorSets(frameIndex, imageIndex);

    // Only reset the fence if we are submitting work
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
    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined,
    // eErrorOutOfDateKHR can be checked as a result here and does not need to
    // be caught by an exception.
    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else {
      // There are no other success codes than eSuccess; on any error code,
      // presentKHR already threw an exception.
      assert(result == vk::Result::eSuccess);
    }
    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
  }
#pragma endregion
#pragma endregion

#pragma region Vulkan Utilities
#pragma region Shader Module
  [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const {
    vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(),
                                          .pCode = reinterpret_cast<const uint32_t *>(code.data())};
    vk::raii::ShaderModule shaderModule{device, createInfo};

    return shaderModule;
  }
#pragma endregion

#pragma region Swapchain Configuration
  static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
  }

  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    assert(!availableFormats.empty());
    const auto formatIt = std::ranges::find_if(availableFormats, [](const auto &format) {
      return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
  }

  static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes) {
    assert(std::ranges::any_of(availablePresentModes,
                               [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(availablePresentModes,
                               [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; })
               ? vk::PresentModeKHR::eMailbox
               : vk::PresentModeKHR::eFifo;
  }

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

#pragma region File Loading
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
  HelloTriangleApplication app;
  app.run();
}
