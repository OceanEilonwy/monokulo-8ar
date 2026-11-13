#!/bin/sh
# Regenerates scan.comp, scan.spv and scan.metal. Only needed when
# gen_shader.py changes; all three are checked in, so building the crate needs
# no shader tools. glslc and spirv-cross come with the Vulkan SDK; CI uses the
# version named in .github/workflows/ci.yml and checks the three match.
set -e
cd "$(dirname "$0")"
python3 gen_shader.py
glslc --target-env=vulkan1.0 -O -o scan.spv scan.comp
# Metal's copy of the same shader, for src/metal.rs, from SPIR-V compiled
# again without -O: the optimiser inlines every call and strips the names,
# which made 6 MB of MSL whose buffers can't be told apart. Metal's compiler
# optimises it. MSL 2.0 runs on every GPU that iOS 13 and macOS 10.15 do.
glslc --target-env=vulkan1.0 -o scan-msl.tmp scan.comp
spirv-cross --msl --msl-version 20000 --output scan.metal scan-msl.tmp
rm scan-msl.tmp
