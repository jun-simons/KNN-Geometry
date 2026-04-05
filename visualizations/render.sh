#!/bin/bash
# render.sh — build kdtree_viz, generate .dot, and render to SVG + PNG
# Run from the visualizations/ directory

set -e

BUILD_DIR="build"

echo "--- Building ---"
mkdir -p "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null
cmake --build "$BUILD_DIR" --config Release

echo "--- Generating .dot ---"
"$BUILD_DIR/kdtree_viz" kdtree.dot

echo "--- Rendering ---"
DOT=$(command -v dot \
    || find /opt/homebrew /usr/local -name "dot" -type f 2>/dev/null | head -1)

if [ -z "$DOT" ]; then
    echo "graphviz not found — install with: brew install graphviz"
    echo "Then run:  dot -Tsvg kdtree.dot -o kdtree.svg"
    exit 1
fi
echo "Using dot: $DOT"

"$DOT" -Tsvg kdtree.dot -o kdtree.svg
"$DOT" -Tpng kdtree.dot -o kdtree.png

echo "--- Done ---"
echo "Output: kdtree.svg  kdtree.png"
