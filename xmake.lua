set_project("engine")
set_languages("c++20")
set_toolchains("clang")

add_rules("mode.debug", "mode.release")
set_defaultmode("release")
-- Release: -O3 explicitly (xmake's default for release, but not on every platform).
if is_mode("release") then
    set_optimize("fastest")
end

-- lld links much faster than GNU ld (the debug binary is ~10 MB). Swap for "-fuse-ld=mold" if mold is installed.
add_ldflags("-fuse-ld=lld")

-- keep build/compile_commands.json fresh for clangd
add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})

-- Compile each .slang file to <targetdir>/shaders/<name>.spv (the path the renderer loads).
rule("slang")
    set_extensions(".slang")
    on_build_file(function (target, sourcefile, opt)
        import("core.project.depend")
        import("utils.progress")

        local outdir = path.join(target:targetdir(), "shaders")
        local outfile = path.join(outdir, path.basename(sourcefile) .. ".spv")
        local incdir = path.directory(sourcefile)

        -- Depend only on the files this shader actually #includes (recursively), so editing
        -- one .hlsl does not recompile every shader.
        local deps = {sourcefile}
        local seen = {}
        local function scan(file)
            if seen[file] or not os.isfile(file) then return end
            seen[file] = true
            table.insert(deps, file)
            for inc in io.readfile(file):gmatch('#include%s*"([^"]+)"') do
                scan(path.join(path.directory(file), inc))
            end
        end
        for inc in io.readfile(sourcefile):gmatch('#include%s*"([^"]+)"') do
            scan(path.join(incdir, inc))
        end

        depend.on_changed(function ()
            progress.show(opt.progress, "${color.build.object}compiling.slang %s", sourcefile)
            os.mkdir(outdir)
            os.vrunv("slangc", {sourcefile, "-target", "spirv", "-fvk-use-entrypoint-name", "-I", incdir, "-o", outfile})
        end, {
            files = deps,
            dependfile = target:dependfile(outfile),
            changed = not os.isfile(outfile)
        })
    end)
rule_end()

-- CPU terrain generation + meshing, shared by the engine and the tests so it is compiled once.
target("cpu_world")
    set_kind("static")
    add_files("src/cpu_world.cpp", "src/cpu_mesher.cpp")
    -- No FMA contraction in the terrain port: AVX2 and scalar builds then give bit-identical heights.
    add_files("src/terrain_cpu.cpp", {cxflags = "-ffp-contract=off"})
    -- simd.hpp; without these flags it falls back to scalar lanes. Public so users of the headers match.
    add_vectorexts("avx2", "fma", {public = true})
    add_syslinks("pthread", {public = true})

target("engine")
    set_kind("binary")
    add_deps("cpu_world")
    add_files("src/main.cpp", "src/renderer.cpp", "src/camera.cpp", "src/block_registry.cpp", "src/player.cpp",
              "src/stb_image_impl.cpp")
    -- vulkan.hpp/glm/glfw parsed once instead of on every renderer.cpp edit (~5 s of its compile time).
    set_pcxxheader("src/pch.hpp")
    add_rules("slang")
    add_files("src/shaders/*.slang")

    -- system libraries from pacman (glfw, vulkan-icd-loader); glm/stb/vulkan-hpp are header-only in /usr/include
    add_links("glfw", "vulkan")
    add_defines(
        "VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1",
        "VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1",
        "VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS",
        -- unused parts of vulkan.hpp; cuts its parse time
        "VULKAN_HPP_NO_SPACESHIP_OPERATOR",
        "VULKAN_HPP_NO_SMART_HANDLE")

    -- MangoHud overlay for `xmake run` (Vulkan implicit layer; needs the mangohud package)
    set_runenv("MANGOHUD", "1")

    -- runtime assets are loaded relative to the working directory (xmake run uses targetdir)
    after_build(function (target)
        os.cp("$(projectdir)/src/textures", target:targetdir())
        os.cp("$(projectdir)/src/blocks.txt", target:targetdir())
    end)

-- CPU terrain checks: mesher vs brute force, height checksum, timings. `xmake build terrain_tests && xmake run terrain_tests`
target("terrain_tests")
    set_kind("binary")
    set_default(false)
    add_deps("cpu_world")
    add_files("tests/terrain_tests.cpp", "src/block_registry.cpp")
