#!/bin/sh
# Regenerates scan.comp and scan.spv; both are kept beside this script.
cd "$(dirname "$0")" && python3 gen_shader.py && glslc --target-env=vulkan1.0 -O -o scan.spv scan.comp
