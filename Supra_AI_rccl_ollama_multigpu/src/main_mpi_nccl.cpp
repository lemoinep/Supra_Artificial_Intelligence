#include "config.hpp"
#include "model_topology.hpp"
#include "worker.hpp"
#include "scheduler_threadsafe.hpp"
#include "api_http.hpp"

#include "httplib.h"

#include <mpi.h>
#include <thread>
#include <iostream>
#include <atomic>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        AppConfig cfg = parse_app_config(argc, argv, world_rank, world_size);
        print_app_config(cfg);

        ModelTopology topology;
        if (!cfg.topology_path.empty()) {
            topology.load_from_json(cfg.topology_path, world_rank, world_size);
        } else {
            topology.init_static(world_rank, world_size);
        }

        if (cfg.is_gateway) {
            ThreadSafeScheduler scheduler(world_size);
            api::ResponseStore resp_store;
            std::atomic<bool> shutdown{false};

            std::thread dispatcher([&]() {
                while (!shutdown.load()) {
                    bool sd = shutdown.load();
                    auto ar_opt = scheduler.try_schedule_next();
                    if (!ar_opt.has_value()) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(5));
                        continue;
                    }
                    auto ar = *ar_opt;

                    int model_len  = (int)ar.req.model_id.size();
                    int prompt_len = (int)ar.req.prompt.size();

                    MPI_Send(&ar.req.id, 1, MPI_INT, ar.worker_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(&model_len, 1, MPI_INT, ar.worker_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(ar.req.model_id.data(), model_len, MPI_CHAR,
                             ar.worker_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(&prompt_len, 1, MPI_INT, ar.worker_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(ar.req.prompt.data(), prompt_len, MPI_CHAR,
                             ar.worker_rank, 0, MPI_COMM_WORLD);

                    int resp_id  = 0;
                    int resp_len = 0;
                    MPI_Status st;
                    MPI_Recv(&resp_id, 1, MPI_INT, ar.worker_rank, 0,
                             MPI_COMM_WORLD, &st);
                    MPI_Recv(&resp_len, 1, MPI_INT, ar.worker_rank, 0,
                             MPI_COMM_WORLD, &st);

                    std::string resp(resp_len, '\0');
                    if (resp_len > 0) {
                        MPI_Recv(resp.data(), resp_len, MPI_CHAR,
                                 ar.worker_rank, 0, MPI_COMM_WORLD, &st);
                    }

                    resp_store.put(resp_id, resp);
                    scheduler.set_worker_busy(ar.worker_rank, false);
                }

                // envoyer signal d'arrêt aux workers
                int exit_id = -1;
                for (int r = 1; r < world_size; ++r) {
                    MPI_Send(&exit_id, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                }
            });

            // HTTP server
            httplib::Server srv;
            int next_request_id = 1;
            api::setup_http_routes(srv, scheduler, resp_store, next_request_id);

            std::thread http_thread([&]() {
                std::cout << "[Gateway] HTTP server listening on 0.0.0.0:8080\n";
                srv.listen("0.0.0.0", 8080);
                shutdown.store(true);
            });

            // Optionnel: CLI local simple (prompts en console)
            std::thread cli_thread([&]() {
                std::string line;
                while (std::getline(std::cin, line)) {
                    if (line == ":quit" || line == ":exit") {
                        srv.stop();
                        break;
                    }
                    if (line.empty()) continue;

                    Request r;
                    r.id         = next_request_id++;
                    r.model_id   = "default";
                    r.prompt     = line;
                    r.priority   = 1;
                    r.est_tokens = 128;
                    scheduler.enqueue(r);
                }
            });

            http_thread.join();
            shutdown.store(true);
            dispatcher.join();
            cli_thread.join();

            topology.finalize();
        } else {
            worker_loop(cfg, topology);
            topology.finalize();
        }
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << world_rank << "] Fatal error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}