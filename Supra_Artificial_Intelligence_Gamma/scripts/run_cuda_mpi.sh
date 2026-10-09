#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run the CUDA + NCCL MPI application.
#
# Defaults:
#   - 4 MPI ranks: rank 0 gateway + ranks 1..3 GPU workers
#   - binary:    ./build-cuda/distributed_llama
#   - model:     ./model.gguf
#   - topology:  ./model_topology.json
#   - HTTP port: 8080 (configured in main.cpp)
#
# Environment overrides:
#
#   NP=4
#   BINARY=./build-cuda/distributed_llama
#   MODEL=./model.gguf
#   TOPOLOGY=./model_topology.json
#
#   MPI_LAUNCHER=mpirun
#   MPI_EXTRA_ARGS="--bind-to none --map-by slot"
#
#   NCCL_DEBUG=INFO
#   NCCL_DEBUG_SUBSYS=INIT,NET
#   NCCL_SOCKET_IFNAME=eth0
#   NCCL_IB_DISABLE=0
#
#   NCCL_ROOT=/opt/nccl
#   CUDA_HOME=/usr/local/cuda
#
# Usage:
#   chmod +x run_cuda_mpi.sh
#   ./run_cuda_mpi.sh
#
# Examples:
#   NP=2 ./run_cuda_mpi.sh
#   NP=4 MODEL=/models/chat.gguf ./run_cuda_mpi.sh
#   NP=4 NCCL_SOCKET_IFNAME=ib0 ./run_cuda_mpi.sh
#   NP=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET ./run_cuda_mpi.sh
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NP="${NP:-4}"
BINARY="${BINARY:-${PROJECT_ROOT}/build-cuda/distributed_llama}"
MODEL="${MODEL:-${PROJECT_ROOT}/model.gguf}"
TOPOLOGY="${TOPOLOGY:-${PROJECT_ROOT}/model_topology.json}"

MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:---bind-to none --map-by slot}"

# Conservative default for first NCCL tests:
#   WARN gives failures without flooding logs.
# Set NCCL_DEBUG=INFO when diagnosing communicator/network setup.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

# CUDA dynamic library paths. Usually unnecessary after a normal system
# installation, but helpful for custom /opt installs.
CUDA_HOME="${CUDA_HOME:-${CUDAToolkit_ROOT:-/usr/local/cuda}}"
NCCL_ROOT="${NCCL_ROOT:-}"

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

if [[ -d "${CUDA_HOME}" ]]; then
    add_library_path "${CUDA_HOME}/lib64"
    add_library_path "${CUDA_HOME}/lib"
fi

if [[ -n "${NCCL_ROOT}" ]]; then
    add_library_path "${NCCL_ROOT}/lib"
    add_library_path "${NCCL_ROOT}/lib64"
fi

if ! command -v "${MPI_LAUNCHER}" >/dev/null 2>&1; then
    echo "error: MPI launcher not found: ${MPI_LAUNCHER}" >&2
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "error: executable not found or not executable:" >&2
    echo "  ${BINARY}" >&2
    echo "Build first with ./build_cuda.sh or set BINARY=/path/to/executable." >&2
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

# Helpful early check. It does not guarantee that one process maps to one GPU,
# but it detects missing NVIDIA driver tooling on normal installations.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"

    if [[ "${GPU_COUNT}" -lt $((NP - 1)) ]]; then
        echo "warning: requested ${NP} MPI ranks = $((NP - 1)) GPU workers," >&2
        echo "but nvidia-smi reports only ${GPU_COUNT} visible GPU(s)." >&2
        echo "The current world-rank modulo mapping may reuse a GPU." >&2
    fi
else
    echo "warning: nvidia-smi not found; cannot verify visible GPUs." >&2
fi

# NCCL_SOCKET_IFNAME is intentionally not set by default. NCCL's automatic
# selection is usually appropriate on one node. Set it explicitly for
# multi-node runs, e.g. NCCL_SOCKET_IFNAME=ib0 or NCCL_SOCKET_IFNAME=eno1.
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export NCCL_SOCKET_IFNAME
fi

# Leave InfiniBand transport enabled by default. For Ethernet-only systems,
# use: NCCL_IB_DISABLE=1 ./run_cuda_mpi.sh
if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
    export NCCL_IB_DISABLE
fi

# Diagnostic fallback only. Do not set permanently because this disables
# direct GPU peer-to-peer transport and can reduce performance.
if [[ -n "${NCCL_P2P_DISABLE:-}" ]]; then
    export NCCL_P2P_DISABLE
fi

MPI_ARGS=(
    "-np" "${NP}"
)

# Open MPI refuses root by default. Add this flag only when the script is
# actually running under root, e.g. inside a container. MPI documentation
# recommends using a non-root user in normal deployments.
if [[ "$(id -u)" -eq 0 ]]; then
    MPI_ARGS+=("--allow-run-as-root")
fi

# Split user-provided flags intentionally using shell expansion. These flags
# are controlled by the local administrator/user of the script.
if [[ -n "${MPI_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${MPI_EXTRA_ARGS} )
    MPI_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "== CUDA/NCCL MPI launch =="
echo "Binary        : ${BINARY}"
echo "MPI launcher  : ${MPI_LAUNCHER}"
echo "MPI ranks     : ${NP}"
echo "Gateway rank  : 0"
echo "Worker ranks  : 1..$((NP - 1))"
echo "Model         : ${MODEL}"
echo "Topology      : ${TOPOLOGY}"
echo "NCCL_DEBUG    : ${NCCL_DEBUG}"
echo "NCCL_SUBSYS   : ${NCCL_DEBUG_SUBSYS}"
echo "CUDA_HOME     : ${CUDA_HOME}"
echo

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    echo "NCCL interface: ${NCCL_SOCKET_IFNAME}"
fi

if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
    echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
fi

if [[ -n "${NCCL_P2P_DISABLE:-}" ]]; then
    echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE}"
fi

echo
echo "Gateway endpoint after startup:"
echo "  http://127.0.0.1:8080/v1/chat/completions"
echo

exec "${MPI_LAUNCHER}" "${MPI_ARGS[@]}" \
    "${BINARY}" \
    --model "${MODEL}" \
    --topology "${TOPOLOGY}"