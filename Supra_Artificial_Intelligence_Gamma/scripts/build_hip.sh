#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Build HIP + RCCL backend.
#
# Optional environment variables:
#
#   BUILD_DIR=/path/to/build
#   BUILD_TYPE=Release
#   JOBS=8
#
#   ROCM_PATH=/opt/rocm
#   HIP_PATH=/opt/rocm
#   CMAKE_PREFIX_PATH=/opt/rocm
#
#   RCCL_ROOT=/opt/rocm
#   RCCL_INCLUDE_DIR=/opt/rocm/include
#   RCCL_LIBRARY=/opt/rocm/lib/librccl.so
#
# Usage:
#   chmod +x build_hip.sh
#   ./build_hip.sh
#
# Custom ROCm example:
#   ROCM_PATH=/opt/rocm-6.3 \
#   RCCL_ROOT=/opt/rccl \
#   ./build_hip.sh
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-hip}"
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
    echo "Install or load Open MPI / MPICH before building." >&2
    exit 1
fi

# ROCm is conventionally installed under /opt/rocm. The caller may override
# ROCM_PATH, HIP_PATH, or CMAKE_PREFIX_PATH for versioned/custom installs.
ROCM_PATH="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"

if [[ ! -d "${ROCM_PATH}" ]]; then
    echo "error: ROCm installation directory not found: ${ROCM_PATH}" >&2
    echo "Set ROCM_PATH=/path/to/rocm." >&2
    exit 1
fi

# Keep the compiler explicit where AMD's compiler is installed. This project
# currently uses HIP Runtime APIs from C++ sources; setting a ROCm compiler
# still avoids accidentally mixing a system clang with ROCm headers/libraries.
if [[ -z "${CXX:-}" ]]; then
    if [[ -x "${ROCM_PATH}/bin/amdclang++" ]]; then
        export CXX="${ROCM_PATH}/bin/amdclang++"
    elif [[ -x "${ROCM_PATH}/llvm/bin/clang++" ]]; then
        export CXX="${ROCM_PATH}/llvm/bin/clang++"
    elif command -v hipcc >/dev/null 2>&1; then
        export CXX="$(command -v hipcc)"
    else
        echo "error: no ROCm C++ compiler was found." >&2
        echo "Expected amdclang++, clang++, or hipcc under ${ROCM_PATH}." >&2
        echo "Set CXX=/path/to/amdclang++ or install the ROCm compiler stack." >&2
        exit 1
    fi
fi

if [[ ! -x "${CXX}" ]]; then
    echo "error: CXX does not point to an executable: ${CXX}" >&2
    exit 1
fi

# A HIP runtime header is sufficient for this project because the backend
# implementation files call hipMalloc/hipMemcpyAsync/etc. but contain no
# __global__ kernels.
if [[ -f "${HIP_PATH}/include/hip/hip_runtime.h" ]]; then
    HIP_INCLUDE_DIR_VALUE="${HIP_PATH}/include"
elif [[ -f "${ROCM_PATH}/include/hip/hip_runtime.h" ]]; then
    HIP_INCLUDE_DIR_VALUE="${ROCM_PATH}/include"
else
    echo "error: hip/hip_runtime.h was not found." >&2
    echo "Checked:" >&2
    echo "  ${HIP_PATH}/include/hip/hip_runtime.h" >&2
    echo "  ${ROCM_PATH}/include/hip/hip_runtime.h" >&2
    echo "Set ROCM_PATH or HIP_PATH to your ROCm installation." >&2
    exit 1
fi

# RCCL keeps NCCL-compatible header/symbol names:
#   include/nccl.h
#   lib/librccl.so
RCCL_ROOT="${RCCL_ROOT:-${ROCM_PATH}}"

if [[ -n "${RCCL_INCLUDE_DIR:-}" ]]; then
    RCCL_INCLUDE_DIR_VALUE="${RCCL_INCLUDE_DIR}"
elif [[ -f "${RCCL_ROOT}/include/nccl.h" ]]; then
    RCCL_INCLUDE_DIR_VALUE="${RCCL_ROOT}/include"
elif [[ -f "${ROCM_PATH}/include/nccl.h" ]]; then
    RCCL_INCLUDE_DIR_VALUE="${ROCM_PATH}/include"
elif [[ -f /usr/include/nccl.h ]]; then
    RCCL_INCLUDE_DIR_VALUE="/usr/include"
else
    echo "error: RCCL header nccl.h was not found." >&2
    echo "Set RCCL_ROOT=/path/to/rccl or RCCL_INCLUDE_DIR=/path/to/include." >&2
    exit 1
fi

if [[ -n "${RCCL_LIBRARY:-}" ]]; then
    RCCL_LIBRARY_VALUE="${RCCL_LIBRARY}"
elif [[ -f "${RCCL_ROOT}/lib/librccl.so" ]]; then
    RCCL_LIBRARY_VALUE="${RCCL_ROOT}/lib/librccl.so"
elif [[ -f "${RCCL_ROOT}/lib64/librccl.so" ]]; then
    RCCL_LIBRARY_VALUE="${RCCL_ROOT}/lib64/librccl.so"
elif [[ -f "${ROCM_PATH}/lib/librccl.so" ]]; then
    RCCL_LIBRARY_VALUE="${ROCM_PATH}/lib/librccl.so"
elif [[ -f "${ROCM_PATH}/lib64/librccl.so" ]]; then
    RCCL_LIBRARY_VALUE="${ROCM_PATH}/lib64/librccl.so"
elif [[ -f /usr/lib/librccl.so ]]; then
    RCCL_LIBRARY_VALUE="/usr/lib/librccl.so"
else
    echo "error: RCCL library librccl.so was not found." >&2
    echo "Set RCCL_ROOT=/path/to/rccl or RCCL_LIBRARY=/path/to/librccl.so." >&2
    exit 1
fi

if [[ ! -f "${RCCL_INCLUDE_DIR_VALUE}/nccl.h" ]]; then
    echo "error: nccl.h was not found in ${RCCL_INCLUDE_DIR_VALUE}" >&2
    exit 1
fi

if [[ ! -f "${RCCL_LIBRARY_VALUE}" ]]; then
    echo "error: RCCL library not found: ${RCCL_LIBRARY_VALUE}" >&2
    exit 1
fi

# Let CMake find ROCm config packages, including HIP's package config.
# Preserve caller-provided prefixes rather than overwriting them.
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    CMAKE_PREFIX_PATH_VALUE="${ROCM_PATH};${HIP_PATH};${CMAKE_PREFIX_PATH}"
else
    CMAKE_PREFIX_PATH_VALUE="${ROCM_PATH};${HIP_PATH}"
fi

echo "== HIP/RCCL configuration =="
echo "Project root : ${PROJECT_ROOT}"
echo "Build dir    : ${BUILD_DIR}"
echo "Build type   : ${BUILD_TYPE}"
echo "Build jobs   : ${BUILD_JOBS}"
echo "MPI compiler : $(command -v mpicxx)"
echo "C++ compiler : ${CXX}"
echo "ROCm path    : ${ROCM_PATH}"
echo "HIP path     : ${HIP_PATH}"
echo "HIP include  : ${HIP_INCLUDE_DIR_VALUE}"
echo "RCCL include : ${RCCL_INCLUDE_DIR_VALUE}"
echo "RCCL library : ${RCCL_LIBRARY_VALUE}"
echo

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH_VALUE}" \
    -DROCM_PATH="${ROCM_PATH}" \
    -DHIP_PATH="${HIP_PATH}" \
    -DRCCL_ROOT="${RCCL_ROOT}" \
    -DRCCL_INCLUDE_DIR="${RCCL_INCLUDE_DIR_VALUE}" \
    -DRCCL_LIBRARY="${RCCL_LIBRARY_VALUE}" \
    -DENABLE_CUDA=OFF \
    -DENABLE_HIP=ON \
    -DENABLE_NCCL=OFF \
    -DENABLE_RCCL=ON

cmake --build "${BUILD_DIR}" \
    --config "${BUILD_TYPE}" \
    --parallel "${BUILD_JOBS}"

echo
echo "Build successful."
echo "Build directory: ${BUILD_DIR}"