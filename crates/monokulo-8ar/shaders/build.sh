#!/bin/sh
# Regenerates scan.comp and scan.spv. Only needed when gen_shader.py changes;
# both are checked in, so building the crate needs no shader tools.
cd "$(dirname "$0")" && python3 gen_shader.py && glslc --target-env=vulkan1.0 -O -o scan.spv scan.comp
