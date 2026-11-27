set_project("engine")
set_languages("c++20")
-- Linux: system clang. Windows cross: mingw-w64-gcc (pacman), `xmake f -p mingw -a x86_64`.
if not is_plat("mingw") then
    set_toolchains("clang")
end

add_rules("mode.debug", "mode.release")
set_defaultmode("release")
-- -O3 explicitly: not xmake's release default on every platform.
if is_mode("release") then
    set_optimize("fastest")
    -- LTO: inlines cpu_world calls (e.g. CpuWorld::isCollidable) across the static lib.
    set_policy("build.optimization.lto", true)
end

-- lld links much faster than GNU ld. Not on mingw: GCC LTO objects (GIMPLE) need GNU ld's LTO plugin.
if not is_plat("mingw") then
    add_ldflags("-fuse-ld=lld")
else
    -- Static libstdc++/libgcc/winpthreads: no mingw DLLs needed next to the .exe.
    add_ldflags("-static")
    -- Workaround for GCC bug 54412: Win64 stacks are only 16-byte aligned, yet GCC emits aligned 32-byte AVX moves
    -- for stack locals. The assembler rewrites them as unaligned; also on the link line, where LTO generates code.
    add_cxflags("-Wa,-muse-unaligned-vector-move", {force = true})
    add_ldflags("-Wa,-muse-unaligned-vector-move", {force = true})
end

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

        -- Depend only on this shader's (recursive) #includes, so editing one .hlsl doesn't recompile every shader.
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
    add_files("src/cpu_world.cpp", "src/cpu_mesher.cpp", "src/cpu_light.cpp")
    -- No FMA contraction in the terrain port: AVX2 and scalar builds then give bit-identical heights.
    add_files("src/terrain_cpu.cpp", {cxflags = "-ffp-contract=off"})
    -- simd.hpp falls back to scalar lanes without this. Public so users of the headers match.
    -- x86-64-v3 = AVX2 + FMA + BMI1/2 + LZCNT (tzcnt/blsr in the mesher bit loops): Haswell or newer.
    add_cxflags("-march=x86-64-v3", {public = true, force = true})
    if not is_plat("mingw") then
        add_syslinks("pthread", {public = true})
    end

target("engine")
    set_kind("binary")
    add_deps("cpu_world")
    add_files("src/main.cpp", "src/renderer.cpp", "src/camera.cpp", "src/block_registry.cpp", "src/player.cpp",
              "src/stb_image_impl.cpp")
    -- vulkan.hpp/glm/glfw parsed once, not on every renderer.cpp edit (~5 s).
    set_pcxxheader("src/pch.hpp")
    add_rules("slang")
    add_files("src/shaders/*.slang")

    -- Linux: pacman glfw, glm, stb, vulkan-headers in /usr/include.
    -- mingw: AUR mingw-w64-glfw + pacman mingw-w64-vulkan-headers. Header-only glm and stb have no mingw package, so
    -- they are symlinked into their own include dir (adding /usr/include would pull in Linux libc headers).
    -- raii::Context loads the Vulkan loader at runtime, so no -lvulkan.
    if not is_plat("mingw") then
        add_links("glfw")
    else
        add_links("glfw3")
        add_syslinks("gdi32")
        on_config(function (target)
            local dir = path.join(os.projectdir(), "build", "mingw-include")
            for _, name in ipairs({"glm", "stb"}) do
                local link = path.join(dir, name)
                if not os.exists(link) then
                    os.mkdir(dir)
                    os.ln(path.join("/usr/include", name), link)
                end
            end
            target:add("sysincludedirs", dir)
        end)
    end
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
