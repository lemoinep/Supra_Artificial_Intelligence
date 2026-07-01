#pragma once

#include <string>
#include <vector>
#include <map>
#include <nccl.h>
#include <mpi.h>

// Description of a model's GPU group
struct ModelGroup {
    std::string          model_id;       // logical model name
    std::vector<int>     ranks;          // MPI ranks belonging to this model
    MPI_Comm             mpi_comm;       // MPI sub-communicator (or MPI_COMM_NULL)
    ncclComm_t           nccl_comm;      // NCCL communicator (or nullptr)
    int                  mpi_rank_local; // rank within mpi_comm (or -1)
    int                  mpi_size_local; // size of mpi_comm (or 0)

    ModelGroup()
        : mpi_comm(MPI_COMM_NULL),
          nccl_comm(nullptr),
          mpi_rank_local(-1),
          mpi_size_local(0) {}
};

// Topology mapping from model_id to ModelGroup
class ModelTopology {
public:
    ModelTopology() = default;

    // Initialize from static policy (fallback).
    void init_static(int world_rank, int world_size);

    // Initialize from a JSON config file describing model groups.
    // JSON format:
    // {
    //   "models": [
    //     { "model_id": "default", "ranks": [1,2,3] },
    //     { "model_id": "chat", "ranks": [1,2] },
    //     ...
    //   ]
    // }
    void load_from_json(const std::string& json_path,
                        int world_rank,
                        int world_size);

    // Query: does this rank belong to given model group?
    bool rank_belongs(const std::string& model_id,
                      int world_rank) const;

    // Get ModelGroup (const &), or nullptr if unknown
    const ModelGroup* get_group(const std::string& model_id) const;

    // Destroy communicators (MPI+NCCL)
    void finalize();

private:
    void create_group_comms(ModelGroup& group,
                            int world_rank,
                            int world_size);

private:
    std::map<std::string, ModelGroup> groups_;
};