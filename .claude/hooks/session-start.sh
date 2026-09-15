#!/bin/bash
# Install everything needed to build fresnel and run its tests and linters.
#
# fresnel is a C++ extension module: there is nothing to test until it is
# compiled, so this configures and builds it as well as installing dependencies.
set -euo pipefail

# Only for Claude Code on the web. A local checkout has its own toolchain,
# usually pixi (see BUILDING.rst).
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# ---------------------------------------------------------------------------
# System libraries
#
# BUILDING.rst installs these with pixi, but pixi.sh is not reachable from the
# web sandbox, so take them from apt instead. Guarded on a header that only
# libembree-dev provides, because SessionStart also fires on resume and compact.
# ---------------------------------------------------------------------------
if [ ! -f /usr/include/embree4/rtcore.h ]; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends \
        build-essential \
        clang-format \
        cmake \
        libembree-dev \
        libqhull-dev \
        libtbb-dev \
        ninja-build
fi

# Random123 supplies the counter based RNG that both tracers use. The other two
# submodules are documentation examples and a vendored qhull that the apt
# package makes unnecessary, so leave them uncloned.
git submodule update --init --depth 1 extern/random123

# ---------------------------------------------------------------------------
# Python packages
#
# In a virtualenv rather than the system interpreter: Ubuntu's python3-numpy is
# built for a different CPython minor version than the image's python3 and
# fails to import.
# ---------------------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet numpy pillow pytest pybind11 ruff

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
    -Dpybind11_DIR="$(.venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    > /dev/null
ninja -C build

# ---------------------------------------------------------------------------
# Session environment
#
# The build tree holds the compiled modules next to copies of the Python
# sources, so it is what `import fresnel` needs to find. Appended only once,
# since this hook runs again on resume and compact.
# ---------------------------------------------------------------------------
if ! grep -qs "fresnel-session-start" "$CLAUDE_ENV_FILE" 2>/dev/null; then
    {
        echo "# fresnel-session-start"
        echo "export PATH=\"$PWD/.venv/bin:\$PATH\""
        echo "export PYTHONPATH=\"$PWD/build\""
    } >> "$CLAUDE_ENV_FILE"
fi
