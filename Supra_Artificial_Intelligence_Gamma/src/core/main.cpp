#include "api_http.hpp"
#include "config.hpp"
#include "core/model_topology.hpp"
#include "gpu/gpu_runtime.hpp"
#include "scheduler_threadsafe.hpp"
#include "worker.hpp"

#include "httplib.h"

#include <mpi.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

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

int string_size_for_mpi(const std::string& value,
                        const char* field_name) {
    if (value.size() >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string(field_name) +
            " is too large for MPI protocol");
    }

    return static_cast<int>(value.size());
}

void send_request_to_worker(const AssignedRequest& assigned) {
    const int model_length =
        string_size_for_mpi(
            assigned.req.model_id,
            "model ID");

    const int prompt_length =
        string_size_for_mpi(
            assigned.req.prompt,
            "prompt");

    const int worker_rank =
        assigned.worker_rank;

    check_mpi(
        MPI_Send(&assigned.req.id,
                 1,
                 MPI_INT,
                 worker_rank,
                 kTagRequestId,
                 MPI_COMM_WORLD),
        "MPI_Send(request ID)");

    check_mpi(
        MPI_Send(&model_length,
                 1,
                 MPI_INT,
                 worker_rank,
                 kTagModelLength,
                 MPI_COMM_WORLD),
        "MPI_Send(model ID length)");

    if (model_length > 0) {
        check_mpi(
            MPI_Send(assigned.req.model_id.data(),
                     model_length,
                     MPI_CHAR,
                     worker_rank,
                     kTagModelData,
                     MPI_COMM_WORLD),
            "MPI_Send(model ID)");
    }

    check_mpi(
        MPI_Send(&prompt_length,
                 1,
                 MPI_INT,
                 worker_rank,
                 kTagPromptLength,
                 MPI_COMM_WORLD),
        "MPI_Send(prompt length)");

    if (prompt_length > 0) {
        check_mpi(
            MPI_Send(assigned.req.prompt.data(),
                     prompt_length,
                     MPI_CHAR,
                     worker_rank,
                     kTagPromptData,
                     MPI_COMM_WORLD),
            "MPI_Send(prompt)");
    }
}

std::string receive_response_from_worker(int expected_request_id,
                                         int worker_rank,
                                         int* received_request_id) {
    MPI_Status status {};

    int response_request_id = 0;

    check_mpi(
        MPI_Recv(&response_request_id,
                 1,
                 MPI_INT,
                 worker_rank,
                 kTagResponseId,
                 MPI_COMM_WORLD,
                 &status),
        "MPI_Recv(response request ID)");

    int response_length = 0;

    check_mpi(
        MPI_Recv(&response_length,
                 1,
                 MPI_INT,
                 worker_rank,
                 kTagResponseLength,
                 MPI_COMM_WORLD,
                 &status),
        "MPI_Recv(response length)");

    constexpr int kMaximumResponseBytes =
        64 * 1024 * 1024;

    if (response_length < 0 ||
        response_length > kMaximumResponseBytes) {
        throw std::runtime_error(
            "Worker returned an invalid response length");
    }

    std::string response(
        static_cast<std::size_t>(response_length),
        '\0');

    if (response_length > 0) {
        check_mpi(
            MPI_Recv(response.data(),
                     response_length,
                     MPI_CHAR,
                     worker_rank,
                     kTagResponseData,
                     MPI_COMM_WORLD,
                     &status),
            "MPI_Recv(response data)");
    }

    if (response_request_id != expected_request_id) {
        throw std::runtime_error(
            "Worker response request ID does not match dispatched request");
    }

    *received_request_id = response_request_id;

    return response;
}

void send_shutdown_to_workers(int world_size) {
    const int exit_request_id = -1;

    for (int rank = 1; rank < world_size; ++rank) {
        check_mpi(
            MPI_Send(&exit_request_id,
                     1,
                     MPI_INT,
                     rank,
                     kTagRequestId,
                     MPI_COMM_WORLD),
            "MPI_Send(worker shutdown)");
    }
}

void dispatcher_loop(ThreadSafeScheduler& scheduler,
                     api::ResponseStore& response_store,
                     std::atomic<bool>& shutdown,
                     int world_size) {
    try {
        while (!shutdown.load(std::memory_order_acquire)) {
            const auto assigned =
                scheduler.try_schedule_next();

            if (!assigned.has_value()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(5));

                continue;
            }

            const int worker_rank =
                assigned->worker_rank;

            try {
                send_request_to_worker(*assigned);

                int response_request_id = 0;

                const std::string response =
                    receive_response_from_worker(
                        assigned->req.id,
                        worker_rank,
                        &response_request_id);

                response_store.put(
                    response_request_id,
                    response);
            } catch (...) {
                // The worker must become schedulable again even when an MPI
                // exchange fails. The outer handler then ends the gateway.
                scheduler.set_worker_busy(
                    worker_rank,
                    false);

                throw;
            }

            scheduler.set_worker_busy(
                worker_rank,
                false);
        }

        send_shutdown_to_workers(world_size);
    } catch (const std::exception& error) {
        std::cerr << "[Gateway dispatcher] Fatal error: "
                  << error.what()
                  << '\n';

        shutdown.store(
            true,
            std::memory_order_release);

        // MPI_Abort is intentional: a worker may otherwise remain blocked in
        // MPI_Recv while the gateway dispatcher has failed.
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int mpi_thread_level = MPI_THREAD_SINGLE;

    const int init_result =
        MPI_Init_thread(
            &argc,
            &argv,
            MPI_THREAD_MULTIPLE,
            &mpi_thread_level);

    if (init_result != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed\n";
        return EXIT_FAILURE;
    }

    int world_rank = 0;
    int world_size = 1;

    try {
        check_mpi(
            MPI_Comm_rank(
                MPI_COMM_WORLD,
                &world_rank),
            "MPI_Comm_rank");

        check_mpi(
            MPI_Comm_size(
                MPI_COMM_WORLD,
                &world_size),
            "MPI_Comm_size");

        if (mpi_thread_level < MPI_THREAD_MULTIPLE) {
            throw std::runtime_error(
                "MPI implementation does not provide MPI_THREAD_MULTIPLE; "
                "this gateway uses MPI from a dispatcher thread.");
        }

        AppConfig cfg =
            parse_app_config(
                argc,
                argv,
                world_rank,
                world_size);

        print_app_config(cfg);

        // GPU selection must happen before ModelTopology initializes an
        // NCCL/RCCL communicator. Rank 0 is a CPU gateway and does not select
        // a GPU in the current architecture.
        if (!cfg.is_gateway) {
            const int device =
                select_device_for_rank(
                    cfg.world_rank);

            if (cfg.verbose) {
                print_device_info(device);

                std::cout << "[Worker] Rank "
                          << cfg.world_rank
                          << " using device "
                          << device
                          << '\n';
            }
        }

        ModelTopology topology;

        if (!cfg.topology_path.empty()) {
            topology.load_from_json(
                cfg.topology_path,
                world_rank,
                world_size);
        } else {
            topology.init_static(
                world_rank,
                world_size);
        }

        if (!cfg.is_gateway) {
            worker_loop(cfg, topology);

            topology.finalize();

            check_mpi(
                MPI_Finalize(),
                "MPI_Finalize");

            return EXIT_SUCCESS;
        }

        ThreadSafeScheduler scheduler(world_size);
        api::ResponseStore response_store;

        std::atomic<bool> shutdown {false};
        std::atomic<int> next_request_id {1};

        std::thread dispatcher(
            dispatcher_loop,
            std::ref(scheduler),
            std::ref(response_store),
            std::ref(shutdown),
            world_size);

        httplib::Server server;

        api::setup_http_routes(
            server,
            scheduler,
            response_store,
            next_request_id);

        std::cout << "[Gateway] HTTP server listening on "
                  << "0.0.0.0:8080\n";

        const bool listen_ok =
            server.listen("0.0.0.0", 8080);

        if (!listen_ok) {
            std::cerr << "[Gateway] HTTP server stopped or failed "
                      << "to bind 0.0.0.0:8080\n";
        }

        shutdown.store(
            true,
            std::memory_order_release);

        dispatcher.join();

        topology.finalize();

        check_mpi(
            MPI_Finalize(),
            "MPI_Finalize");

        return listen_ok
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "[Rank "
                  << world_rank
                  << "] Fatal error: "
                  << error.what()
                  << '\n';

        MPI_Abort(
            MPI_COMM_WORLD,
            EXIT_FAILURE);

        return EXIT_FAILURE;
    }
}