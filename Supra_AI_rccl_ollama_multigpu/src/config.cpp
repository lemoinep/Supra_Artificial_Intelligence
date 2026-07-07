#include "config.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>

static int parse_int_arg(int argc, char** argv, int i) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + argv[i]);
    }
    return std::atoi(argv[i + 1]);
}

static std::string parse_str_arg(int argc, char** argv, int i) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + argv[i]);
    }
    return std::string(argv[i + 1]);
}

AppConfig parse_app_config(int argc, char** argv,
                           int world_rank, int world_size)
{
    AppConfig cfg;
    cfg.world_rank = world_rank;
    cfg.world_size = world_size;

    // Defaults
    cfg.llama_cfg.model_path      = "model.gguf";
    cfg.llama_cfg.n_ctx           = 4096;
    cfg.llama_cfg.n_gpu_layers    = 35;
    cfg.llama_cfg.seed            = 0;
    cfg.llama_cfg.gpu_layer_split_mode = 0;
    cfg.llama_cfg.use_flash_attn  = true;
    cfg.llama_cfg.verbose         = (world_rank == 0);

    cfg.is_gateway = (world_rank == 0);
    cfg.verbose    = cfg.llama_cfg.verbose;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            cfg.llama_cfg.model_path = parse_str_arg(argc, argv, i);
            ++i;
        } else if (std::strcmp(argv[i], "--n_ctx") == 0) {
            cfg.llama_cfg.n_ctx = parse_int_arg(argc, argv, i);
            ++i;
        } else if (std::strcmp(argv[i], "--n_gpu_layers") == 0 ||
                   std::strcmp(argv[i], "--ngl") == 0) {
            cfg.llama_cfg.n_gpu_layers = parse_int_arg(argc, argv, i);
            ++i;
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            cfg.llama_cfg.seed = parse_int_arg(argc, argv, i);
            ++i;
        } else if (std::strcmp(argv[i], "--no-flash-attn") == 0) {
            cfg.llama_cfg.use_flash_attn = false;
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            cfg.llama_cfg.verbose = false;
        } else if (std::strcmp(argv[i], "--topology") == 0) {
            cfg.topology_path = parse_str_arg(argc, argv, i);
            ++i;
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) {
                std::cout
                    << "Usage: " << argv[0] << " [options]\n"
                    << "Options:\n"
                    << "  --model PATH           Path to GGUF model (default: model.gguf)\n"
                    << "  --n_ctx N              Context length (default: 4096)\n"
                    << "  --n_gpu_layers N       Number of layers on GPU (default: 35)\n"
                    << "  --seed N               RNG seed (default: 0 -> random)\n"
                    << "  --no-flash-attn        Disable flash attention\n"
                    << "  --quiet                Less verbose logging\n"
                    << "  --topology path.json   Model topology JSON\n"
                    << "  --help, -h             Show this help\n";
            }
            std::exit(0);
        } else {
            if (world_rank == 0) {
                std::cerr << "[Config] Warning: unknown option '"
                          << argv[i] << "'\n";
            }
        }
    }

    cfg.is_gateway = (world_rank == 0);
    cfg.verbose    = cfg.llama_cfg.verbose;

    return cfg;
}

void print_app_config(const AppConfig& cfg) {
    if (!cfg.verbose || cfg.world_rank != 0) return;

    std::cout << "[AppConfig] world_size=" << cfg.world_size
              << ", world_rank=" << cfg.world_rank << "\n";
    std::cout << "[AppConfig] model_path=" << cfg.llama_cfg.model_path
              << ", n_ctx=" << cfg.llama_cfg.n_ctx
              << ", n_gpu_layers=" << cfg.llama_cfg.n_gpu_layers
              << ", seed=" << cfg.llama_cfg.seed
              << ", flash_attn=" << (cfg.llama_cfg.use_flash_attn ? "on" : "off")
              << "\n";
    if (!cfg.topology_path.empty()) {
        std::cout << "[AppConfig] topology_path=" << cfg.topology_path << "\n";
    }
}