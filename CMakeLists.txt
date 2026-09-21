cmake_minimum_required (VERSION 3.29)

project (engine)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan" OFF)

if(ENABLE_CPP20_MODULE)
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

find_package (glfw3 REQUIRED)
find_package (Vulkan 1.4.335 REQUIRED)

if(ENABLE_CPP20_MODULE)
  add_library(VulkanCppModule)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)

  target_compile_definitions(VulkanCppModule
    PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  )
  target_include_directories(VulkanCppModule
    PUBLIC "${Vulkan_INCLUDE_DIR}"
  )
  target_link_libraries(VulkanCppModule
    PUBLIC Vulkan::Vulkan
  )

  set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

  if(MSVC)
    target_compile_options(VulkanCppModule PRIVATE
      /std:c++latest
      /permissive-
      /Zc:__cplusplus
      /EHsc
      /Zc:preprocessor
      /translateInclude
    )
  endif()

  target_sources(VulkanCppModule
    PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS "${Vulkan_INCLUDE_DIR}"
    FILES "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
  )

  target_sources(VulkanCppModule
    PRIVATE "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
  )
else()
  add_library(VulkanCppModule INTERFACE)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)
  target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
  target_compile_definitions(VulkanCppModule
    INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  )
endif()

# Find Slang Compiler (slangc)
find_program (SLANGC_COMPILER "slangc" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*" OR ${CMAKE_GENERATOR} MATCHES "Ninja Multi-Config")
  set (SHADERS_DIR ${CMAKE_BINARY_DIR}/$<CONFIG>/shaders)
else()
  set (SHADERS_DIR ${CMAKE_BINARY_DIR}/shaders)
endif()

# Find all Slang and HLSL files meant for compilation
file (GLOB SLANG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.slang" "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.hlsl")

# Include path where legacy libraries (e.g., FastNoiseLite.hlsl) live
set (SHADER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")

add_custom_target (engine_shaders)
set (ALL_SPV_TARGETS "")

foreach (SHADER_FILE ${SLANG_FILES})
  get_filename_component (SHADER_NAME ${SHADER_FILE} NAME_WE)

  # Slang compiles all stages into a single SPIR-V file matching main.cpp ("shaders/slang.spv" or "${SHADER_NAME}.spv")
  set (COMPILED_SPV "${SHADERS_DIR}/${SHADER_NAME}.spv")

  add_custom_command (
    OUTPUT ${COMPILED_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    COMMAND ${SLANGC_COMPILER} ${SHADER_FILE}
            -target spirv
            -I ${SHADER_INCLUDE_DIR}
            -o ${COMPILED_SPV}
    DEPENDS ${SHADER_FILE}
    VERBATIM
  )

  list (APPEND ALL_SPV_TARGETS ${COMPILED_SPV})
endforeach ()

if (ALL_SPV_TARGETS)
  add_custom_target (engine_shader_build DEPENDS ${ALL_SPV_TARGETS})
  add_dependencies (engine_shaders engine_shader_build)
endif ()

add_executable (engine src/main.cpp)

set_target_properties (engine PROPERTIES CXX_STANDARD 20)
target_link_libraries (engine Vulkan::cppm glfw)
target_include_directories (engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(ENABLE_CPP20_MODULE)
  target_compile_definitions(engine PRIVATE USE_CPP20_MODULES=1)
endif()

target_compile_definitions(engine PRIVATE "VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS")

if(WIN32 AND ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
  set_target_properties(engine PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

add_custom_command(
    TARGET engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/textures
        $<TARGET_FILE_DIR:engine>/textures
    COMMENT "Copying texture folder to build directory"
)

add_dependencies (engine engine_shaders)