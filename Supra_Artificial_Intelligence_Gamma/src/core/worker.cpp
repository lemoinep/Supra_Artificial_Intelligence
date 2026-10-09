#include "worker.hpp"

#include "core/model_topology.hpp"
#include "gpu/gpu_runtime.hpp"
#include "llama_backend.hpp"

#include <mpi.h>

#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr int kGatewayRank = 0;

enum MessageTag : int {
    kTagRequestId = 100,
    kTagModelLength = 101,
    kTagModelData = 102,
    kTagPromptLength = 103,
    kTagPromptData = 104,

    kTagResponseId = 200,
    kTagResponseLength = 201,
    kTagResponseData = 202,
};

void check_mpi(int result, const char* what) {
    if (result == MPI_SUCCESS) {
        return;
    }

    char error_text[MPI_MAX_ERROR_STRING] {};
    int error_length = 0;

    MPI_Error_string(
        result,
        error_text,
        &error_length);

    throw std::runtime_error(
        std::string(what) +
        " failed: " +
        std::string(
            error_text,
            static_cast<std::size_t>(error_length)));
}

int receive_int_from_gateway(int tag, MPI_Status* status) {
    int value = 0;

    check_mpi(
        MPI_Recv(&value,
                 1,
                 MPI_INT,
                 kGatewayRank,
                 tag,
                 MPI_COMM_WORLD,
                 status),
        "MPI_Recv(integer from gateway)");

    return value;
}

std::string receive_string_from_gateway(int length_tag,
                                        int data_tag,
                                        MPI_Status* status,
                                        const char* field_name) {
    const int length =
        receive_int_from_gateway(length_tag, status);

    if (length < 0) {
        throw std::runtime_error(
            std::string("Worker received negative ") +
            field_name +
            " length");
    }

    constexpr std::size_t kMaximumMessageBytes =
        16U * 1024U * 1024U;

    if (static_cast<std::size_t>(length) > kMaximumMessageBytes) {
        throw std::runtime_error(
            std::string("Worker received oversized ") +
            field_name);
    }

    std::string value(
        static_cast<std::size_t>(length),
        '\0');

    if (length > 0) {
        check_mpi(
            MPI_Recv(value.data(),
                     length,
                     MPI_CHAR,
                     kGatewayRank,
                     data_tag,
                     MPI_COMM_WORLD,
                     status),
            "MPI_Recv(string data from gateway)");
    }

    return value;
}

void send_response_to_gateway(int request_id,
                              const std::string& response) {
    if (response.size() >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Worker response is too large for MPI protocol");
    }

    const int response_length =
        static_cast<int>(response.size());

    check_mpi(
        MPI_Send(&request_id,
                 1,
                 MPI_INT,
                 kGatewayRank,
                 kTagResponseId,
                 MPI_COMM_WORLD),
        "MPI_Send(response request ID)");

    check_mpi(
        MPI_Send(&response_length,
                 1,
                 MPI_INT,
                 kGatewayRank,
                 kTagResponseLength,
                 MPI_COMM_WORLD),
        "MPI_Send(response length)");

    if (response_length > 0) {
        check_mpi(
            MPI_Send(response.data(),
                     response_length,
                     MPI_CHAR,
                     kGatewayRank,
                     kTagResponseData,
                     MPI_COMM_WORLD),
            "MPI_Send(response data)");
    }
}

const ModelGroup& require_worker_group(
    const ModelTopology& topology,
    const std::string& model_id,
    int world_rank) {
    const ModelGroup* group =
        topology.get_group(model_id);

    if (group == nullptr) {
        throw std::runtime_error(
            "Unknown model_id in worker request: '" +
            model_id +
            "'");
    }

    if (group->mpi_comm == MPI_COMM_NULL ||
        group->mpi_rank_local < 0 ||
        !topology.rank_belongs(model_id, world_rank)) {
        throw std::runtime_error(
            "Worker rank " +
            std::to_string(world_rank) +
            " does not belong to model group '" +
            model_id +
            "'");
    }

    return *group;
}

}  // namespace

void worker_loop(const AppConfig& cfg,
                 const ModelTopology& topology) {
    if (cfg.world_rank <= 0) {
        throw std::invalid_argument(
            "worker_loop must only run on MPI worker ranks >= 1");
    }

    // Device selection and GPU communicator initialization must already have
    // completed in main.cpp before ModelTopology was constructed.
    if (cfg.verbose) {
        std::cout << "[Worker] Rank "
                  << cfg.world_rank
                  << " started with backend "
                  << gpu_backend_name()
                  << '\n';
    }

    std::map<std::string, std::unique_ptr<LlamaBackend>> backends;

    auto get_backend_for_model =
        [&](const std::string& model_id) -> LlamaBackend& {
            auto it = backends.find(model_id);

            if (it != backends.end()) {
                return *it->second;
            }

            LlamaBackendConfig backend_config =
                cfg.llama_cfg;

            // This mapping should eventually come from topology JSON or
            // explicit AppConfig model metadata rather than hard-coded names.
            if (model_id == "code") {
                backend_config.model_path = "code_model.gguf";
            } else if (model_id == "chat") {
                backend_config.model_path = "chat_model.gguf";
            }

            auto backend =
                std::make_unique<LlamaBackend>(backend_config);

            LlamaBackend& result = *backend;

            backends.emplace(
                model_id,
                std::move(backend));

            return result;
        };

    const GenerationParams gen_params {
        .max_tokens = 128,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .repeat_penalty = 1.0f,
    };

    while (true) {
        MPI_Status status {};

        const int request_id =
            receive_int_from_gateway(
                kTagRequestId,
                &status);

        if (request_id < 0) {
            if (cfg.verbose) {
                std::cout << "[Worker] Rank "
                          << cfg.world_rank
                          << " received shutdown.\n";
            }

            break;
        }

        const std::string model_id =
            receive_string_from_gateway(
                kTagModelLength,
                kTagModelData,
                &status,
                "model ID");

        const std::string prompt =
            receive_string_from_gateway(
                kTagPromptLength,
                kTagPromptData,
                &status,
                "prompt");

        std::string response;

        try {
            const ModelGroup& group =
                require_worker_group(
                    topology,
                    model_id,
                    cfg.world_rank);

            (void) group;

            if (prompt.rfind(":load_lora", 0) == 0) {
                throw std::runtime_error(
                    "LoRA loading is a model-group collective operation. "
                    "The gateway must dispatch this command to every rank "
                    "in the model group, rather than to a single worker.");
            }

            if (cfg.verbose) {
                std::cout << "[Worker] Rank "
                          << cfg.world_rank
                          << " handling request "
                          << request_id
                          << " for model '"
                          << model_id
                          << "'.\n";
            }

            LlamaBackend& backend =
                get_backend_for_model(model_id);

            response = backend.generate(
                prompt,
                gen_params);
        } catch (const std::exception& error) {
            std::cerr << "[Worker] Rank "
                      << cfg.world_rank
                      << " request "
                      << request_id
                      << " failed: "
                      << error.what()
                      << '\n';

            response =
                std::string("[error] ") +
                error.what();
        }

        send_response_to_gateway(
            request_id,
            response);
    }
}