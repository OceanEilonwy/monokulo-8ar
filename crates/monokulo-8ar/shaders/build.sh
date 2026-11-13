#!/bin/sh
# Regenerates scan.comp, scan.spv and scan.metal. Only needed when
# gen_shader.py changes; all three are checked in, so building the crate needs
# no shader tools. glslc and spirv-cross come with the Vulkan SDK; CI uses the
# version named in .github/workflows/ci.yml and checks the three match.
set -e
cd "$(dirname "$0")"
python3 gen_shader.py
glslc --target-env=vulkan1.0 -O -o scan.spv scan.comp
# Metal's copy of the same shader, for src/metal.rs. MSL 2.0 runs on every
# Apple GPU that iOS 13 and macOS 10.15 run on.
spirv-cross --msl --msl-version 20000 --output scan.metal scan.spv
