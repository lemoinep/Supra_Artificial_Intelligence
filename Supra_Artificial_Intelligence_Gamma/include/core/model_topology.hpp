#pragma once

#include "gpu/gpu_collectives.hpp"

#include <mpi.h>

#include <map>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Description of one logical model group.
//
// `ranks` uses MPI_COMM_WORLD ranks.
// `mpi_comm` contains only ranks that belong to this model.
// `nccl_comm` is backed by NCCL in CUDA builds and RCCL in HIP builds.
// -----------------------------------------------------------------------------
struct ModelGroup {
    // Logical model identifier, for example: "default", "chat", or "code".
    std::string model_id;

    // MPI_COMM_WORLD ranks belonging to this model group.
    std::vector<int> ranks;

    // MPI sub-communicator containing precisely `ranks`, or MPI_COMM_NULL for
    // a process that does not belong to this group.
    MPI_Comm mpi_comm = MPI_COMM_NULL;

    // GPU collective communicator for `mpi_comm`, or nullptr for a process
    // outside this group.
    ncclComm_t nccl_comm = nullptr;

    // Rank and size inside mpi_comm. Invalid / zero outside the group.
    int mpi_rank_local = -1;
    int mpi_size_local = 0;
};

// -----------------------------------------------------------------------------
// Mapping between logical model IDs and distributed MPI/GPU model groups.
//
// The topology owns the MPI and NCCL/RCCL communicators that it creates.
// It must therefore not be copied or moved.
// -----------------------------------------------------------------------------
class ModelTopology {
public:
    ModelTopology() = default;

    ~ModelTopology() {
        // Expected lifecycle:
        // ModelTopology is destroyed before MPI_Finalize() in main.cpp.
        finalize();
    }

    ModelTopology(const ModelTopology&) = delete;
    ModelTopology& operator=(const ModelTopology&) = delete;

    ModelTopology(ModelTopology&&) = delete;
    ModelTopology& operator=(ModelTopology&&) = delete;

    // Initialize a fallback topology:
    // rank 0 is the gateway; ranks [1, world_size) form the "default" group.
    void init_static(int world_rank, int world_size);

    // Initialize groups from a JSON configuration file.
    //
    // Example:
    // {
    //   "models": [
    //     { "model_id": "default", "ranks": [1, 2] },
    //     { "model_id": "chat",    "ranks": [1] },
    //     { "model_id": "code",    "ranks": [2] }
    //   ]
    // }
    void load_from_json(const std::string& json_path,
                        int world_rank,
                        int world_size);

    // Return true if an MPI_COMM_WORLD rank belongs to model_id.
    [[nodiscard]] bool rank_belongs(const std::string& model_id,
                                    int world_rank) const;

    // Get the local view of a logical model group, or nullptr if model_id is
    // unknown. For a known group that does not contain this process, mpi_comm
    // is MPI_COMM_NULL and nccl_comm is nullptr.
    [[nodiscard]] const ModelGroup* get_group(
        const std::string& model_id) const;

    // Destroy every communicator owned by the topology.
    //
    // Safe to call multiple times.
    void finalize() noexcept;

private:
    void create_group_comms(ModelGroup& group,
                            int world_rank,
                            int world_size);

    std::map<std::string, ModelGroup> groups_;
};

