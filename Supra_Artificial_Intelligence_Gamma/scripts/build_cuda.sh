#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Build CUDA + NCCL backend.
#
# Optional environment variables:
#
#   BUILD_DIR=/path/to/build
#   BUILD_TYPE=Release
#   JOBS=8
#
#   CUDAToolkit_ROOT=/usr/local/cuda
#   CUDACXX=/usr/local/cuda/bin/nvcc
#
#   NCCL_ROOT=/usr
#   NCCL_INCLUDE_DIR=/usr/include
#   NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so
#
# Usage:
#   chmod +x build_cuda.sh
#   ./build_cuda.sh
#
# Example with custom CUDA/NCCL installation:
#   CUDAToolkit_ROOT=/usr/local/cuda-12.4 \
#   NCCL_ROOT=/opt/nccl \
#   ./build_cuda.sh
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-cuda}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [[ -n "${JOBS:-}" ]]; then
    BUILD_JOBS="${JOBS}"
else
    BUILD_JOBS="$(nproc)"
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake was not found in PATH" >&2
    exit 1
fi

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "error: mpicxx was not found in PATH" >&2
    echo "Install or load an MPI implementation such as Open MPI or MPICH." >&2
    exit 1
fi

# Resolve CUDA compiler. CUDACXX takes precedence when explicitly set.
if [[ -z "${CUDACXX:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        export CUDACXX="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDACXX="/usr/local/cuda/bin/nvcc"
    else
        echo "error: nvcc was not found." >&2
        echo "Set CUDACXX=/path/to/nvcc or add CUDA/bin to PATH." >&2
        exit 1
    fi
fi

if [[ ! -x "${CUDACXX}" ]]; then
    echo "error: CUDACXX does not point to an executable: ${CUDACXX}" >&2
    exit 1
fi

# Infer a CUDA root only when the caller did not provide one.
if [[ -z "${CUDAToolkit_ROOT:-}" ]]; then
    export CUDAToolkit_ROOT="$(
        cd "$(dirname "${CUDACXX}")/.." && pwd
    )"
fi

if [[ ! -f "${CUDAToolkit_ROOT}/include/cuda_runtime.h" ]]; then
    echo "error: cuda_runtime.h not found under:" >&2
    echo "  ${CUDAToolkit_ROOT}/include" >&2
    echo "Set CUDAToolkit_ROOT to the CUDA Toolkit root." >&2
    exit 1
fi

# NCCL location:
# - NCCL_ROOT can be passed for a custom NCCL installation.
# - Standard Debian/Ubuntu installation locations are used as fallbacks.
NCCL_ROOT="${NCCL_ROOT:-}"

if [[ -n "${NCCL_INCLUDE_DIR:-}" ]]; then
    NCCL_INCLUDE_DIR_VALUE="${NCCL_INCLUDE_DIR}"
elif [[ -n "${NCCL_ROOT}" && -f "${NCCL_ROOT}/include/nccl.h" ]]; then
    NCCL_INCLUDE_DIR_VALUE="${NCCL_ROOT}/include"
elif [[ -f /usr/include/nccl.h ]]; then
    NCCL_INCLUDE_DIR_VALUE="/usr/include"
elif [[ -f /usr/local/include/nccl.h ]]; then
    NCCL_INCLUDE_DIR_VALUE="/usr/local/include"
else
    echo "error: nccl.h was not found." >&2
    echo "Set NCCL_ROOT=/path/to/nccl or NCCL_INCLUDE_DIR=/path/to/include." >&2
    exit 1
fi

if [[ -n "${NCCL_LIBRARY:-}" ]]; then
    NCCL_LIBRARY_VALUE="${NCCL_LIBRARY}"
elif [[ -n "${NCCL_ROOT}" && -f "${NCCL_ROOT}/lib/libnccl.so" ]]; then
    NCCL_LIBRARY_VALUE="${NCCL_ROOT}/lib/libnccl.so"
elif [[ -n "${NCCL_ROOT}" && -f "${NCCL_ROOT}/lib64/libnccl.so" ]]; then
    NCCL_LIBRARY_VALUE="${NCCL_ROOT}/lib64/libnccl.so"
elif [[ -f /usr/lib/x86_64-linux-gnu/libnccl.so ]]; then
    NCCL_LIBRARY_VALUE="/usr/lib/x86_64-linux-gnu/libnccl.so"
elif [[ -f /usr/local/lib/libnccl.so ]]; then
    NCCL_LIBRARY_VALUE="/usr/local/lib/libnccl.so"
elif [[ -f /usr/local/lib64/libnccl.so ]]; then
    NCCL_LIBRARY_VALUE="/usr/local/lib64/libnccl.so"
else
    echo "error: libnccl.so was not found." >&2
    echo "Set NCCL_ROOT=/path/to/nccl or NCCL_LIBRARY=/path/to/libnccl.so." >&2
    exit 1
fi

if [[ ! -f "${NCCL_INCLUDE_DIR_VALUE}/nccl.h" ]]; then
    echo "error: nccl.h not found in ${NCCL_INCLUDE_DIR_VALUE}" >&2
    exit 1
fi

if [[ ! -f "${NCCL_LIBRARY_VALUE}" ]]; then
    echo "error: NCCL library not found: ${NCCL_LIBRARY_VALUE}" >&2
    exit 1
fi

echo "== CUDA/NCCL configuration =="
echo "Project root : ${PROJECT_ROOT}"
echo "Build dir    : ${BUILD_DIR}"
echo "Build type   : ${BUILD_TYPE}"
echo "Build jobs   : ${BUILD_JOBS}"
echo "MPI compiler : $(command -v mpicxx)"
echo "CUDACXX      : ${CUDACXX}"
echo "CUDA root    : ${CUDAToolkit_ROOT}"
echo "NCCL include : ${NCCL_INCLUDE_DIR_VALUE}"
echo "NCCL library : ${NCCL_LIBRARY_VALUE}"
echo

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
    -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
    -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}" \
    -DNCCL_ROOT="${NCCL_ROOT}" \
    -DNCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR_VALUE}" \
    -DNCCL_LIBRARY="${NCCL_LIBRARY_VALUE}" \
    -DENABLE_CUDA=ON \
    -DENABLE_HIP=OFF \
    -DENABLE_NCCL=ON \
    -DENABLE_RCCL=OFF

cmake --build "${BUILD_DIR}" \
    --config "${BUILD_TYPE}" \
    --parallel "${BUILD_JOBS}"

echo
echo "Build successful."
echo "Build directory: ${BUILD_DIR}"