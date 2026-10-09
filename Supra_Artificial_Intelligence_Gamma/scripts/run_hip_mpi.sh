#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run the HIP + RCCL MPI application.
#
# Defaults:
#   - 4 MPI ranks: rank 0 gateway + ranks 1..3 GPU workers
#   - binary:    ./build-hip/distributed_llama
#   - model:     ./model.gguf
#   - topology:  ./model_topology.json
#   - HTTP port: 8080 (configured in main.cpp)
#
# Environment overrides:
#
#   NP=4
#   BINARY=./build-hip/distributed_llama
#   MODEL=./model.gguf
#   TOPOLOGY=./model_topology.json
#
#   ROCM_PATH=/opt/rocm
#   RCCL_ROOT=/opt/rocm
#
#   MPI_LAUNCHER=mpirun
#   MPI_EXTRA_ARGS="--bind-to none --map-by slot"
#
#   NCCL_DEBUG=INFO
#   NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET
#   NCCL_SOCKET_IFNAME=ib0
#
# Usage:
#   chmod +x run_hip_mpi.sh
#   ./run_hip_mpi.sh
#
# Examples:
#   NP=2 ./run_hip_mpi.sh
#   NP=4 MODEL=/models/model.gguf ./run_hip_mpi.sh
#   NP=4 NCCL_DEBUG=INFO ./run_hip_mpi.sh
#   NP=4 NCCL_SOCKET_IFNAME=ib0 ./run_hip_mpi.sh
#
# RCCL per-rank logs:
#   NP=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
#   NCCL_DEBUG_FILE=rccl.%h.%p.log ./run_hip_mpi.sh
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NP="${NP:-4}"
BINARY="${BINARY:-${PROJECT_ROOT}/build-hip/distributed_llama}"
MODEL="${MODEL:-${PROJECT_ROOT}/model.gguf}"
TOPOLOGY="${TOPOLOGY:-${PROJECT_ROOT}/model_topology.json}"

MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:---bind-to none --map-by slot}"

ROCM_PATH="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
RCCL_ROOT="${RCCL_ROOT:-${ROCM_PATH}}"

# RCCL uses NCCL-compatible environment-variable names.
#
# ERROR is RCCL's default. WARN is a practical default for application runs;
# use INFO for startup/network diagnostics.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,BOOTSTRAP,NET}"

add_library_path() {
    local directory="$1"

    if [[ -n "${directory}" && -d "${directory}" ]]; then
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            export LD_LIBRARY_PATH="${directory}:${LD_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="${directory}"
        fi
    fi
}

if [[ -d "${ROCM_PATH}" ]]; then
    add_library_path "${ROCM_PATH}/lib"
    add_library_path "${ROCM_PATH}/lib64"
fi

if [[ -n "${RCCL_ROOT}" ]]; then
    add_library_path "${RCCL_ROOT}/lib"
    add_library_path "${RCCL_ROOT}/lib64"
fi

if ! command -v "${MPI_LAUNCHER}" >/dev/null 2>&1; then
    echo "error: MPI launcher not found: ${MPI_LAUNCHER}" >&2
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "error: executable not found or not executable:" >&2
    echo "  ${BINARY}" >&2
    echo "Build first with ./build_hip.sh or set BINARY=/path/to/executable." >&2
    exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "error: GGUF model file not found:" >&2
    echo "  ${MODEL}" >&2
    echo "Set MODEL=/path/to/model.gguf." >&2
    exit 1
fi

if [[ ! -f "${TOPOLOGY}" ]]; then
    echo "error: topology JSON file not found:" >&2
    echo "  ${TOPOLOGY}" >&2
    echo "Set TOPOLOGY=/path/to/model_topology.json." >&2
    exit 1
fi

if ! [[ "${NP}" =~ ^[0-9]+$ ]] || [[ "${NP}" -lt 2 ]]; then
    echo "error: NP must be an integer >= 2." >&2
    echo "Need rank 0 for the gateway and at least rank 1 for a worker." >&2
    exit 1
fi

if [[ ! -d "${ROCM_PATH}" ]]; then
    echo "error: ROCm directory does not exist:" >&2
    echo "  ${ROCM_PATH}" >&2
    echo "Set ROCM_PATH=/path/to/rocm." >&2
    exit 1
fi

if [[ ! -f "${ROCM_PATH}/include/hip/hip_runtime.h" ]]; then
    echo "warning: hip_runtime.h was not found at:" >&2
    echo "  ${ROCM_PATH}/include/hip/hip_runtime.h" >&2
    echo "The binary may still work with a custom ROCm installation, but verify ROCM_PATH." >&2
fi

# rocminfo is not always available in containers, so this is a warning only.
if command -v rocminfo >/dev/null 2>&1; then
    GPU_COUNT="$(
        rocminfo 2>/dev/null |
        grep -E '^[[:space:]]*Name:[[:space:]]+gfx' |
        wc -l ||
        true
    )"

    if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] &&
       [[ "${GPU_COUNT}" -gt 0 ]] &&
       [[ "${GPU_COUNT}" -lt $((NP - 1)) ]]; then
        echo "warning: requested ${NP} MPI ranks = $((NP - 1)) GPU workers," >&2
        echo "but rocminfo reports only ${GPU_COUNT} visible GPU agent(s)." >&2
        echo "Your current rank-to-device mapping may reuse GPUs." >&2
    fi
else
    echo "warning: rocminfo not found; cannot verify visible AMD GPUs." >&2
fi

# Leave automatic RCCL network selection enabled by default. For a multi-node
# run, specify the actual NIC explicitly:
#
#   NCCL_SOCKET_IFNAME=ib0 ./run_hip_mpi.sh
#
# Exact matching can be requested with an '=' prefix:
#
#   NCCL_SOCKET_IFNAME='=eno1' ./run_hip_mpi.sh
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export NCCL_SOCKET_IFNAME
fi

# If provided by the caller, pass these RCCL transport/tuning variables through
# to every MPI rank. Do not set them blindly; topology is platform-specific.
for variable_name in \
    NCCL_IB_DISABLE \
    NCCL_IB_HCA \
    NCCL_P2P_DISABLE \
    NCCL_NET_GDR_LEVEL \
    NCCL_DEBUG_FILE
do
    if [[ -n "${!variable_name:-}" ]]; then
        export "${variable_name}"
    fi
done

MPI_ARGS=(
    "-np" "${NP}"
)

# Open MPI deliberately refuses root launches by default. This is useful for
# containers but should not be the normal deployment configuration.
if [[ "$(id -u)" -eq 0 ]]; then
    MPI_ARGS+=("--allow-run-as-root")
fi

# Expand administrator/user-provided Open MPI flags intentionally.
if [[ -n "${MPI_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${MPI_EXTRA_ARGS} )
    MPI_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "== HIP/RCCL MPI launch =="
echo "Binary        : ${BINARY}"
echo "MPI launcher  : ${MPI_LAUNCHER}"
echo "MPI ranks     : ${NP}"
echo "Gateway rank  : 0"
echo "Worker ranks  : 1..$((NP - 1))"
echo "Model         : ${MODEL}"
echo "Topology      : ${TOPOLOGY}"
echo "ROCm path     : ${ROCM_PATH}"
echo "RCCL root     : ${RCCL_ROOT}"
echo "NCCL_DEBUG    : ${NCCL_DEBUG}"
echo "NCCL_SUBSYS   : ${NCCL_DEBUG_SUBSYS}"
echo

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    echo "RCCL interface: ${NCCL_SOCKET_IFNAME}"
fi

if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
    echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
fi

if [[ -n "${NCCL_IB_HCA:-}" ]]; then
    echo "NCCL_IB_HCA: ${NCCL_IB_HCA}"
fi

if [[ -n "${NCCL_P2P_DISABLE:-}" ]]; then
    echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE}"
fi

if [[ -n "${NCCL_DEBUG_FILE:-}" ]]; then
    echo "NCCL_DEBUG_FILE: ${NCCL_DEBUG_FILE}"
fi

echo
echo "Gateway endpoint after startup:"
echo "  http://127.0.0.1:8080/v1/chat/completions"
echo

exec "${MPI_LAUNCHER}" "${MPI_ARGS[@]}" \
    "${BINARY}" \
    --model "${MODEL}" \
    --topology "${TOPOLOGY}"