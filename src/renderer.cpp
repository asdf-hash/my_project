#include "renderer.hpp"
#include "camera.hpp"
#include "clipmap_culler.hpp"
#include "lod.hpp"
#include "world.hpp"

#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float3.hpp>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr vk::SampleCountFlagBits RENDER_SAMPLE_COUNT = vk::SampleCountFlagBits::e2;
constexpr float BASE_FOV_RADIANS = glm::radians(70.0f);
constexpr vk::Format TAA_HISTORY_FORMAT = vk::Format::eR16G16B16A16Sfloat;
constexpr uint32_t MAX_BLOCK_TYPES = 10;
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
  uint32_t quadCount;
  uint32_t historyValid;
  float lodAoLighteningPerLevel;
  alignas(16) glm::vec4 cameraPosition;
  alignas(16) std::array<std::array<uint32_t, 4>, MAX_BLOCK_TYPES * 2> blockFaceTextures{};
};
static_assert(offsetof(UniformBufferObject, previousMvp) == 64);
static_assert(offsetof(UniformBufferObject, renderSize) == 128);
static_assert(offsetof(UniformBufferObject, quadCount) == 136);
static_assert(offsetof(UniformBufferObject, historyValid) == 140);
static_assert(offsetof(UniformBufferObject, lodAoLighteningPerLevel) == 144);
static_assert(offsetof(UniformBufferObject, cameraPosition) == 160);
static_assert(offsetof(UniformBufferObject, blockFaceTextures) == 176);
static_assert(sizeof(UniformBufferObject) == 496);

struct ChunkComputeParams {
  uint32_t chunkCount;
  uint32_t horizontalChunkCount;
};

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
  vk::raii::DescriptorSetLayout taaDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout taaPipelineLayout = nullptr;
  vk::raii::Pipeline taaPipeline = nullptr;
  vk::raii::DescriptorSetLayout chunkComputeDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout chunkComputePipelineLayout = nullptr;
  vk::raii::Pipeline generateHeightsPipeline = nullptr;
  vk::raii::Pipeline meshColumnsPipeline = nullptr;
  vk::raii::Pipeline finalizeMeshPipeline = nullptr;
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

  static constexpr uint32_t MAX_TEXTURE_COUNT = 1024;
  uint32_t textureDescriptorCapacity = 0;
  std::vector<vk::raii::Image> textureImages;
  std::vector<vk::raii::DeviceMemory> textureImageMemories;
  std::vector<vk::raii::ImageView> textureImageViews;
  std::array<std::array<uint32_t, 6>, MAX_BLOCK_TYPES> blockTextureIndices{};
  vk::raii::Sampler textureSampler = nullptr;
  vk::raii::Sampler taaSampler = nullptr;
#pragma endregion

#pragma region Buffer State
  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  static constexpr uint32_t MAX_QUAD_COUNT = 1u << 22u;
  // Capacity is largest at the narrowest supported FOV, not the widest FOV.
  static constexpr uint32_t CLIPMAP_KEEP_ALIVE_MULTIPLIER = 4u;
  static constexpr uint32_t MAX_HORIZONTAL_CHUNK_COUNT =
      calculateClipmapCapacity(TARGET_SCREEN_PERCENTAGE, MIN_SUPPORTED_TAN_HALF_FOV, CHUNK_SIZE,
                               world::kRenderDistance) *
      CLIPMAP_KEEP_ALIVE_MULTIPLIER;
  static constexpr uint32_t VERTICAL_CHUNK_COUNT = 2u * static_cast<uint32_t>(world::kVerticalRenderDistance) + 1u;
  std::vector<world::ChunkLod> visibleChunkSet;
  std::vector<std::array<int32_t, 2>> visibleChunkWork;
  ClipmapCuller clipmapCuller;
  std::optional<ClipmapView> lastSubmittedClipmapView;
  std::optional<ClipmapView> lastAppliedClipmapView;
  std::uint64_t latestSubmittedClipmapSequence = 0;
  std::uint64_t lastAppliedClipmapSequence = 0;
  uint32_t chunkComputeCount = 0;
  bool chunkBuffersDirty = true;
  std::array<float, MAX_LOD> lodDistanceSquared =
      initializeLodDistanceSquared(TARGET_SCREEN_PERCENTAGE, BASE_FOV_RADIANS, {WIDTH, HEIGHT});
  float cachedLodFov = BASE_FOV_RADIANS;
  vk::Extent2D cachedLodExtent{WIDTH, HEIGHT};

  vk::raii::Buffer quadBuffer = nullptr;
  vk::raii::DeviceMemory quadBufferMemory = nullptr;
  vk::raii::Buffer chunkOffsetBuffer = nullptr;
  vk::raii::DeviceMemory chunkOffsetBufferMemory = nullptr;
  vk::raii::Buffer chunkHeightBuffer = nullptr;
  vk::raii::DeviceMemory chunkHeightBufferMemory = nullptr;
  vk::raii::Buffer quadCountBuffer = nullptr;
  vk::raii::DeviceMemory quadCountBufferMemory = nullptr;
  vk::raii::Buffer meshDrawArgumentsBuffer = nullptr;
  vk::raii::DeviceMemory meshDrawArgumentsBufferMemory = nullptr;
  std::vector<vk::raii::Buffer> chunkCoordinateBuffers;
  std::vector<vk::raii::DeviceMemory> chunkCoordinateBufferMemories;
  std::vector<void *> chunkCoordinateBuffersMapped;
  std::vector<vk::raii::Buffer> chunkWorkBuffers;
  std::vector<vk::raii::DeviceMemory> chunkWorkBufferMemories;
  std::vector<void *> chunkWorkBuffersMapped;
#pragma endregion

#pragma region Descriptor State
  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;
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
  Camera camera;
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
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    camera.initialize(window);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
    app->camera.handleKey(window, key, action);
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
    createChunkComputePipeline();
    createCommandPool();
    createDepthResources();
    createTextureImages();
    createTextureImageViews();
    createTextureSampler();
    createTaaSampler();
    createUniformBuffers();
    createChunkBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createTaaDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }
#pragma endregion

#pragma region Main Loop
  void mainLoop() {
    using Clock = std::chrono::steady_clock;
    constexpr auto frameDuration =
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / 90.0));
    auto nextFrame = Clock::now();

    while (!glfwWindowShouldClose(window)) {
      // Fps Limit
      nextFrame += frameDuration;
      glfwPollEvents();
      drawFrame();
      std::this_thread::sleep_until(nextFrame);

      const auto now = Clock::now();
      if (now > nextFrame + frameDuration) {
        nextFrame = now;
      }
    }

    device.waitIdle();
  }
#pragma endregion

#pragma region Cleanup
  void cleanupSwapChain() {
    graphicsPipeline = nullptr;
    taaPipeline = nullptr;
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
    depthImageViews.clear();
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
    surface = nullptr; // god knows why linux throws a seg fault, same thing
                       // happened on the wgpu project too, always gotta force
                       // release furface before glfw windows.
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

    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createGraphicsPipeline();
    createTaaPipeline();
    createDepthResources();
    createTaaDescriptorSets();
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
            {.features = {.independentBlend = true, .samplerAnisotropy = true, .shaderInt64 = true}},
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

    std::array<vk::DescriptorSetLayoutBinding, 6> bindings{
        {{.binding = 0,
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eMeshEXT},
         {.binding = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment},
         {.binding = 2,
          .descriptorType = vk::DescriptorType::eSampler,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eFragment},
         {.binding = 3,
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eMeshEXT},
         {.binding = 4,
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eMeshEXT},
         {.binding = 11,
          .descriptorType = vk::DescriptorType::eSampledImage,
          .descriptorCount = textureDescriptorCapacity,
          .stageFlags = vk::ShaderStageFlagBits::eFragment}}};

    std::array<vk::DescriptorBindingFlags, 6> bindingFlags{};
    bindingFlags[5] =
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

    vk::PipelineShaderStageCreateInfo meshShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eMeshEXT, .module = shaderModule, .pName = "meshMain"};
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
    vk::PipelineShaderStageCreateInfo shaderStages[] = {meshShaderStageInfo, fragShaderStageInfo};

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
                                                         .depthCompareOp = vk::CompareOp::eLess,
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

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                                                    .pDynamicStates = dynamicStates.data()};

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0};
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::Format depthFormat = findDepthFormat();

    const std::array<vk::Format, 2> colorFormats{swapChainSurfaceFormat.format, MOTION_FORMAT};
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = 2,
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

  void createChunkComputePipeline() {
    std::array<vk::DescriptorSetLayoutBinding, 7> bindings{};
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
    meshColumnsPipeline = createComputePipeline("meshColumns");
    finalizeMeshPipeline = createComputePipeline("finalizeMeshDispatch");
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

      auto [depthImage, depthImageMemory] =
          createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                      vk::MemoryPropertyFlagBits::eDeviceLocal, 1, RENDER_SAMPLE_COUNT);
      depthImages.emplace_back(std::move(depthImage));
      depthImageMemories.emplace_back(std::move(depthImageMemory));
      depthImageViews.emplace_back(createImageView(depthImages.back(), depthFormat, vk::ImageAspectFlagBits::eDepth));
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
  void createTextureImage(const char *path, vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory) {
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
      throw std::runtime_error(std::string("failed to load texture image: ") + path);
    }
    if (texWidth != 16 || texHeight != 16) {
      stbi_image_free(pixels);
      throw std::runtime_error(std::string("block texture must be 16x16: ") + path);
    }
    const auto formatFeatures = physicalDevice.getFormatProperties(TEXTURE_FORMAT).optimalTilingFeatures;
    constexpr auto requiredBlitFeatures = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst |
                                          vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
    if ((formatFeatures & requiredBlitFeatures) != requiredBlitFeatures) {
      stbi_image_free(pixels);
      throw std::runtime_error("texture format does not support linear mipmap blits");
    }
    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();

    stbi_image_free(pixels);

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
    constexpr std::array<const char *, MAX_BLOCK_TYPES> blockDefinitionPaths = {
        "blocks/grass.block", "blocks/dirt.block",     "blocks/stone.block", "blocks/sand.block",
        "blocks/snow.block",  "blocks/gravel.block",   "blocks/stone.block", "blocks/stone.block",
        "blocks/stone.block", "blocks/deepslate.block"};
    constexpr std::array<const char *, 6> normalNames = {"positive_x", "negative_x", "positive_y",
                                                         "negative_y", "positive_z", "negative_z"};
    std::vector<std::string> texturePaths;
    std::unordered_map<std::string, uint32_t> textureIndices;

    const auto trim = [](std::string value) {
      const auto first = value.find_first_not_of(" \t\r\n");
      if (first == std::string::npos) {
        return std::string{};
      }
      const auto last = value.find_last_not_of(" \t\r\n");
      return value.substr(first, last - first + 1);
    };

    for (size_t blockType = 0; blockType < blockDefinitionPaths.size(); ++blockType) {
      std::ifstream definition(blockDefinitionPaths[blockType]);
      if (!definition) {
        throw std::runtime_error(std::string("failed to open block definition: ") + blockDefinitionPaths[blockType]);
      }

      std::array<bool, 6> definedNormals{};
      std::string line;
      size_t lineNumber = 0;
      while (std::getline(definition, line)) {
        ++lineNumber;
        const auto commentStart = line.find('#');
        if (commentStart != std::string::npos) {
          line.erase(commentStart);
        }
        line = trim(std::move(line));
        if (line.empty()) {
          continue;
        }

        const auto separator = line.find('=');
        if (separator == std::string::npos) {
          throw std::runtime_error(std::string("invalid block definition at ") + blockDefinitionPaths[blockType] + ":" +
                                   std::to_string(lineNumber));
        }
        const std::string normalName = trim(line.substr(0, separator));
        const std::string textureName = trim(line.substr(separator + 1));
        const auto normal = std::find_if(normalNames.begin(), normalNames.end(),
                                         [&normalName](const char *name) { return normalName == name; });
        if (normal == normalNames.end() || textureName.empty()) {
          throw std::runtime_error(std::string("invalid normal or texture at ") + blockDefinitionPaths[blockType] +
                                   ":" + std::to_string(lineNumber));
        }
        const size_t normalIndex = static_cast<size_t>(normal - normalNames.begin());
        if (definedNormals[normalIndex]) {
          throw std::runtime_error(std::string("duplicate normal in block definition: ") +
                                   blockDefinitionPaths[blockType]);
        }
        definedNormals[normalIndex] = true;

        const std::string texturePath = "textures/" + textureName;
        auto [textureIndex, inserted] = textureIndices.emplace(texturePath, static_cast<uint32_t>(texturePaths.size()));
        if (inserted) {
          texturePaths.push_back(texturePath);
        }
        blockTextureIndices[blockType][normalIndex] = textureIndex->second;
      }
      if (!std::all_of(definedNormals.begin(), definedNormals.end(), [](bool defined) { return defined; })) {
        throw std::runtime_error(std::string("block definition must specify all six normals: ") +
                                 blockDefinitionPaths[blockType]);
      }
    }

    if (texturePaths.size() > textureDescriptorCapacity) {
      throw std::runtime_error("texture descriptor capacity is too small for the loaded textures");
    }

    textureImages.reserve(texturePaths.size());
    textureImageMemories.reserve(texturePaths.size());
    for (const std::string &path : texturePaths) {
      textureImages.emplace_back(nullptr);
      textureImageMemories.emplace_back(nullptr);
      createTextureImage(path.c_str(), textureImages.back(), textureImageMemories.back());
    }
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

    constexpr uint32_t HEIGHT_STRIDE = CHUNK_SIZE + 2u;
    const vk::DeviceSize heightBufferSize =
        sizeof(int32_t) * MAX_HORIZONTAL_CHUNK_COUNT * HEIGHT_STRIDE * HEIGHT_STRIDE;
    auto [heightBufferValue, heightBufferMemoryValue] = createBuffer(
        heightBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    chunkHeightBuffer = std::move(heightBufferValue);
    chunkHeightBufferMemory = std::move(heightBufferMemoryValue);

    auto [quadCountBufferValue, quadCountBufferMemoryValue] =
        createBuffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    quadCountBuffer = std::move(quadCountBufferValue);
    quadCountBufferMemory = std::move(quadCountBufferMemoryValue);

    auto [drawArgumentsBufferValue, drawArgumentsBufferMemoryValue] =
        createBuffer(sizeof(vk::DrawMeshTasksIndirectCommandEXT),
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    meshDrawArgumentsBuffer = std::move(drawArgumentsBufferValue);
    meshDrawArgumentsBufferMemory = std::move(drawArgumentsBufferMemoryValue);

    const vk::DeviceSize coordinateBufferSize = sizeof(world::ChunkLod) * MAX_HORIZONTAL_CHUNK_COUNT;
    const vk::DeviceSize workBufferSize =
        sizeof(std::array<int32_t, 2>) * MAX_HORIZONTAL_CHUNK_COUNT * VERTICAL_CHUNK_COUNT;

    for (size_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
      auto [coordinateBuffer, coordinateBufferMemory] =
          createBuffer(coordinateBufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      chunkCoordinateBuffers.emplace_back(std::move(coordinateBuffer));
      chunkCoordinateBufferMemories.emplace_back(std::move(coordinateBufferMemory));
      chunkCoordinateBuffersMapped.emplace_back(
          chunkCoordinateBufferMemories.back().mapMemory(0, coordinateBufferSize));
      std::memset(chunkCoordinateBuffersMapped.back(), 0, static_cast<size_t>(coordinateBufferSize));
      std::memcpy(chunkCoordinateBuffersMapped.back(), visibleChunkSet.data(),
                  visibleChunkSet.size() * sizeof(world::ChunkLod));

      auto [workBuffer, workBufferMemory] =
          createBuffer(workBufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      chunkWorkBuffers.emplace_back(std::move(workBuffer));
      chunkWorkBufferMemories.emplace_back(std::move(workBufferMemory));
      chunkWorkBuffersMapped.emplace_back(chunkWorkBufferMemories.back().mapMemory(0, workBufferSize));
      std::memset(chunkWorkBuffersMapped.back(), 0, static_cast<size_t>(workBufferSize));
      std::memcpy(chunkWorkBuffersMapped.back(), visibleChunkWork.data(),
                  visibleChunkWork.size() * sizeof(visibleChunkWork[0]));
    }
  }

  void updateVisibleClipmap(const ClipmapView &view) {
    if (!lastSubmittedClipmapView || *lastSubmittedClipmapView != view) {
      latestSubmittedClipmapSequence = clipmapCuller.submit(view);
      lastSubmittedClipmapView = view;
    }
    if (auto completion = clipmapCuller.takeCompletion(); completion &&
                                                          completion->sequence > lastAppliedClipmapSequence &&
                                                          clipmapViewCanCover(completion->view, view)) {
      lastAppliedClipmapSequence = completion->sequence;
      lastAppliedClipmapView = completion->view;
      if (completion->result.chunks != visibleChunkSet || completion->result.work != visibleChunkWork) {
        visibleChunkSet = std::move(completion->result.chunks);
        visibleChunkWork = std::move(completion->result.work);
        chunkComputeCount = static_cast<uint32_t>(visibleChunkWork.size());
        chunkBuffersDirty = true;
      }
    }
    if (!lastAppliedClipmapView || !clipmapViewCanCover(*lastAppliedClipmapView, view)) {
      ClipmapResult result = buildVisibleClipmap(view);
      lastAppliedClipmapSequence = latestSubmittedClipmapSequence;
      lastAppliedClipmapView = view;
      if (result.chunks != visibleChunkSet || result.work != visibleChunkWork) {
        visibleChunkSet = std::move(result.chunks);
        visibleChunkWork = std::move(result.work);
        chunkComputeCount = static_cast<uint32_t>(visibleChunkWork.size());
        chunkBuffersDirty = true;
      }
    }
  }

  static bool clipmapViewCanCover(const ClipmapView &builtView, const ClipmapView &currentView) {
    constexpr float maxPositionDelta = static_cast<float>(CHUNK_SIZE) * 0.5f;
    constexpr float minForwardDot = 0.9961947f;
    const float dx = builtView.position[0] - currentView.position[0];
    const float dy = builtView.position[1] - currentView.position[1];
    const float dz = builtView.position[2] - currentView.position[2];
    const float positionDeltaSquared = dx * dx + dy * dy + dz * dz;
    const float forwardDot = builtView.forward[0] * currentView.forward[0] +
                             builtView.forward[1] * currentView.forward[1] +
                             builtView.forward[2] * currentView.forward[2];
    return positionDeltaSquared <= maxPositionDelta * maxPositionDelta && forwardDot >= minForwardDot &&
           builtView.lodDistanceSquared == currentView.lodDistanceSquared &&
           builtView.verticalFovRadians == currentView.verticalFovRadians &&
           builtView.aspectRatio == currentView.aspectRatio;
  }
#pragma endregion

#pragma region Descriptor Setup
  void createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 4> poolSize{
        {{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 16u * MAX_FRAMES_IN_FLIGHT},
         {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT},
         {.type = vk::DescriptorType::eSampledImage,
          .descriptorCount = MAX_FRAMES_IN_FLIGHT * static_cast<uint32_t>(textureImageViews.size())},
         {.type = vk::DescriptorType::eSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT}}};
    vk::DescriptorPoolCreateInfo poolInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                                                   vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
                                          .maxSets = 3u * MAX_FRAMES_IN_FLIGHT,
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

    std::vector<vk::DescriptorImageInfo> textureImageInfos;
    textureImageInfos.reserve(textureImageViews.size());
    for (const auto &imageView : textureImageViews) {
      textureImageInfos.push_back(
          {.sampler = nullptr, .imageView = imageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal});
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo uniformBufferInfo{
          .buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};
      vk::DescriptorBufferInfo quadBufferInfo{
          .buffer = quadBuffer, .offset = 0, .range = sizeof(uint64_t) * MAX_QUAD_COUNT};
      vk::DescriptorBufferInfo chunkOffsetBufferInfo{
          .buffer = chunkOffsetBuffer, .offset = 0, .range = sizeof(uint64_t) * MAX_QUAD_COUNT};
      vk::DescriptorBufferInfo quadCountBufferInfo{.buffer = quadCountBuffer, .offset = 0, .range = sizeof(uint32_t)};
      vk::DescriptorImageInfo samplerInfo{
          .sampler = textureSampler, .imageView = nullptr, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

      std::array<vk::WriteDescriptorSet, 6> descriptorWrites{{
          {.dstSet = descriptorSets[i],
           .dstBinding = 0,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &quadBufferInfo},
          {.dstSet = descriptorSets[i],
           .dstBinding = 1,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eUniformBuffer,
           .pBufferInfo = &uniformBufferInfo},
          {.dstSet = descriptorSets[i],
           .dstBinding = 2,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eSampler,
           .pImageInfo = &samplerInfo},
          {.dstSet = descriptorSets[i],
           .dstBinding = 3,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &chunkOffsetBufferInfo},
          {.dstSet = descriptorSets[i],
           .dstBinding = 11,
           .dstArrayElement = 0,
           .descriptorCount = static_cast<uint32_t>(textureImageInfos.size()),
           .descriptorType = vk::DescriptorType::eSampledImage,
           .pImageInfo = textureImageInfos.data()},
          {.dstSet = descriptorSets[i],
           .dstBinding = 4,
           .dstArrayElement = 0,
           .descriptorCount = 1,
           .descriptorType = vk::DescriptorType::eStorageBuffer,
           .pBufferInfo = &quadCountBufferInfo},
      }};
      device.updateDescriptorSets(descriptorWrites, {});
    }

    std::vector<vk::DescriptorSetLayout> computeLayouts(MAX_FRAMES_IN_FLIGHT, chunkComputeDescriptorSetLayout);
    chunkComputeDescriptorSets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{.descriptorPool = descriptorPool,
                                      .descriptorSetCount = static_cast<uint32_t>(computeLayouts.size()),
                                      .pSetLayouts = computeLayouts.data()});

    const vk::DeviceSize coordinateBufferSize = sizeof(world::ChunkLod) * MAX_HORIZONTAL_CHUNK_COUNT;
    const vk::DeviceSize workBufferSize =
        sizeof(std::array<int32_t, 2>) * MAX_HORIZONTAL_CHUNK_COUNT * VERTICAL_CHUNK_COUNT;
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      std::array<vk::DescriptorBufferInfo, 7> bufferInfos{{
          {.buffer = chunkHeightBuffer, .offset = 0, .range = sizeof(int32_t) * MAX_HORIZONTAL_CHUNK_COUNT * 34u * 34u},
          {.buffer = chunkCoordinateBuffers[i], .offset = 0, .range = coordinateBufferSize},
          {.buffer = quadBuffer, .offset = 0, .range = sizeof(uint64_t) * MAX_QUAD_COUNT},
          {.buffer = chunkOffsetBuffer, .offset = 0, .range = sizeof(uint64_t) * MAX_QUAD_COUNT},
          {.buffer = quadCountBuffer, .offset = 0, .range = sizeof(uint32_t)},
          {.buffer = meshDrawArgumentsBuffer, .offset = 0, .range = sizeof(vk::DrawMeshTasksIndirectCommandEXT)},
          {.buffer = chunkWorkBuffers[i], .offset = 0, .range = workBufferSize},
      }};
      std::array<vk::WriteDescriptorSet, 7> writes{};
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

  void updateTaaDescriptorSet(uint32_t frame, uint32_t imageIndex) {
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
      if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
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

    vk::ClearValue clearColor = vk::ClearColorValue(0.55f, 0.72f, 0.9f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    std::array<vk::RenderingAttachmentInfo, 2> sceneAttachments{
        {{.imageView = multisampleColorImageViews[imageIndex],
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .resolveMode = vk::ResolveModeFlagBits::eAverage,
          .resolveImageView = sceneColorImageViews[imageIndex],
          .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = clearColor},
         {.imageView = multisampleMotionImageViews[imageIndex],
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .resolveMode = vk::ResolveModeFlagBits::eAverage,
          .resolveImageView = motionImageViews[imageIndex],
          .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)}}};

    vk::RenderingAttachmentInfo depthAttachmentInfo = {.imageView = depthImageViews[imageIndex],
                                                       .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                                                       .loadOp = vk::AttachmentLoadOp::eClear,
                                                       .storeOp = vk::AttachmentStoreOp::eStore,
                                                       .clearValue = clearDepth};

    vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
                                       .layerCount = 1,
                                       .colorAttachmentCount = static_cast<uint32_t>(sceneAttachments.size()),
                                       .pColorAttachments = sceneAttachments.data(),
                                       .pDepthAttachment = &depthAttachmentInfo};

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    // Negative height flips Y for Vulkan; see Uniform buffers.
    commandBuffer.setViewport(0, vk::Viewport(0.0f, static_cast<float>(swapChainExtent.height),
                                              static_cast<float>(swapChainExtent.width),
                                              -static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIndex],
                                     nullptr);
    commandBuffer.drawMeshTasksIndirectEXT(*meshDrawArgumentsBuffer, 0, 1, sizeof(vk::DrawMeshTasksIndirectCommandEXT));
    commandBuffer.endRendering();

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
    transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::ePresentSrcKHR, colorWrite, {}, colorOutputStage,
                            vk::PipelineStageFlagBits2::eBottomOfPipe, vk::ImageAspectFlagBits::eColor);
    taaHistoryInitialized = true;
    taaHistoryValid = true;
    taaHistoryWriteIndex = historyReadIndex;
    commandBuffer.end();
  }

  void recordChunkCompute(vk::raii::CommandBuffer &commandBuffer) {
    if (!chunkBuffersDirty) {
      return;
    }

    vk::MemoryBarrier2 beforeWrite{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eMeshShaderEXT |
                        vk::PipelineStageFlagBits2::eDrawIndirect,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                         vk::AccessFlagBits2::eIndirectCommandRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &beforeWrite});
    commandBuffer.fillBuffer(*quadCountBuffer, 0, sizeof(uint32_t), 0);

    vk::MemoryBarrier2 counterResetBarrier{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                           .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                           .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                           .dstAccessMask =
                                               vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
    commandBuffer.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &counterResetBarrier});

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *chunkComputePipelineLayout, 0,
                                     *chunkComputeDescriptorSets[frameIndex], nullptr);
    const ChunkComputeParams computeParams{chunkComputeCount, static_cast<uint32_t>(visibleChunkSet.size())};
    commandBuffer.pushConstants(*chunkComputePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                sizeof(computeParams), &computeParams);

    if (computeParams.horizontalChunkCount != 0u) {
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *generateHeightsPipeline);
      commandBuffer.dispatch(5, 5, computeParams.horizontalChunkCount);

      vk::MemoryBarrier2 heightReadyBarrier{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                            .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
      commandBuffer.pipelineBarrier2(
          vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &heightReadyBarrier});
    }

    if (chunkComputeCount != 0u) {
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *meshColumnsPipeline);
      commandBuffer.dispatch(4, 4, chunkComputeCount);
    }

    vk::MemoryBarrier2 counterReadyBarrier{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                           .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                           .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                           .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
    commandBuffer.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &counterReadyBarrier});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *finalizeMeshPipeline);
    commandBuffer.dispatch(1, 1, 1);

    vk::MemoryBarrier2 meshReadyBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eMeshShaderEXT | vk::PipelineStageFlagBits2::eDrawIndirect,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eIndirectCommandRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &meshReadyBarrier});
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

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
      inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
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
    static auto lastTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
    lastTime = currentTime;

    UniformBufferObject ubo{};
    const glm::mat4 view = camera.update(window, deltaTime);

    const glm::mat4 model = glm::mat4(1.0f);
    const bool zoomed = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
    const float fov = zoomed ? 2.0f * std::atan(std::tan(BASE_FOV_RADIANS * 0.5f) * 0.5f) : BASE_FOV_RADIANS;
    if (fov != cachedLodFov || swapChainExtent.width != cachedLodExtent.width ||
        swapChainExtent.height != cachedLodExtent.height) {
      lodDistanceSquared =
          initializeLodDistanceSquared(TARGET_SCREEN_PERCENTAGE, fov, {swapChainExtent.width, swapChainExtent.height});
      cachedLodFov = fov;
      cachedLodExtent = swapChainExtent;
    }
    const glm::vec3 worldUp{0.0f, 1.0f, 0.0f};
    const glm::vec3 cameraRight = glm::normalize(glm::cross(camera.front(), worldUp));
    const glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, camera.front()));
    const float aspectRatio = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
    const ClipmapView clipmapView{{camera.position().x, camera.position().y, camera.position().z},
                                  {camera.front().x, camera.front().y, camera.front().z},
                                  {cameraRight.x, cameraRight.y, cameraRight.z},
                                  {cameraUp.x, cameraUp.y, cameraUp.z},
                                  lodDistanceSquared,
                                  fov,
                                  aspectRatio};
    updateVisibleClipmap(clipmapView);
    if (chunkBuffersDirty) {
      std::memcpy(chunkCoordinateBuffersMapped[currentImage], visibleChunkSet.data(),
                  visibleChunkSet.size() * sizeof(world::ChunkLod));
      std::memcpy(chunkWorkBuffersMapped[currentImage], visibleChunkWork.data(),
                  visibleChunkWork.size() * sizeof(visibleChunkWork[0]));
    }

    glm::mat4 proj = glm::perspective(
        fov, static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 1000000.0f);
    const uint32_t jitterSample = (taaJitterIndex++ % 8u) + 1u;
    const float jitterX = TAA_ENABLED ? (halton(jitterSample, 2u) - 0.5f) * TAA_JITTER_PIXELS : 0.0f;
    const float jitterY = TAA_ENABLED ? (halton(jitterSample, 3u) - 0.5f) * TAA_JITTER_PIXELS : 0.0f;
    proj[2][0] -= 2.0f * jitterX / static_cast<float>(swapChainExtent.width);
    proj[2][1] += 2.0f * jitterY / static_cast<float>(swapChainExtent.height);
    ubo.mvp = proj * view * model;
    ubo.previousMvp = previousMvpValid ? previousMvp : ubo.mvp;
    ubo.renderSize = {static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height)};
    ubo.quadCount = 0;
    ubo.historyValid = TAA_ENABLED && taaHistoryValid ? 1u : 0u;
    ubo.lodAoLighteningPerLevel = LOD_AO_LIGHTENING_PER_LEVEL;
    ubo.cameraPosition = glm::vec4(camera.position(), 0.0f);
    for (size_t blockType = 0; blockType < blockTextureIndices.size(); ++blockType) {
      for (size_t normal = 0; normal < blockTextureIndices[blockType].size(); ++normal) {
        ubo.blockFaceTextures[blockType * 2 + normal / 4][normal % 4] = blockTextureIndices[blockType][normal];
      }
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
    auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined,
    // eErrorOutOfDateKHR can be checked as a result here and does not need to
    // be caught by an exception.
    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    }
    // On other success codes than eSuccess and eSuboptimalKHR we just throw an
    // exception. On any error code, aquireNextImage already threw an exception.
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
      assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
      throw std::runtime_error("failed to acquire swap chain image!");
    }
    updateUniformBuffer(frameIndex);
    updateTaaDescriptorSet(frameIndex, imageIndex);

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
    queue.submit(submitInfo, *inFlightFences[frameIndex]);

    const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
                                            .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                                            .swapchainCount = 1,
                                            .pSwapchains = &*swapChain,
                                            .pImageIndices = &imageIndex};
    result = queue.presentKHR(presentInfoKHR);
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
      throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
  }
#pragma endregion
#pragma endregion
};

void Renderer::run() {
  HelloTriangleApplication app;
  app.run();
}
