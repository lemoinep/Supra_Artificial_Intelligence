#include "config.hpp"

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

int parse_int_arg(int argc, char** argv, int index) {
    if (index + 1 >= argc) {
        throw std::runtime_error(
            std::string("Missing value for ") + argv[index]);
    }

    const std::string_view text(argv[index + 1]);

    if (text.empty()) {
        throw std::runtime_error(
            std::string("Invalid empty integer value for ") + argv[index]);
    }

    int value = 0;

    const auto [end, error] = std::from_chars(
        text.data(),
        text.data() + text.size(),
        value);

    if (error == std::errc::invalid_argument ||
        end != text.data() + text.size()) {
        throw std::runtime_error(
            std::string("Invalid integer value '") +
            std::string(text) +
            "' for " +
            argv[index]);
    }

    if (error == std::errc::result_out_of_range) {
        throw std::runtime_error(
            std::string("Integer value out of range for ") +
            argv[index] +
            ": " +
            std::string(text));
    }

    return value;
}

std::string parse_str_arg(int argc, char** argv, int index) {
    if (index + 1 >= argc) {
        throw std::runtime_error(
            std::string("Missing value for ") + argv[index]);
    }

    const char* value = argv[index + 1];

    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(
            std::string("Empty value for ") + argv[index]);
    }

    return value;
}

void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --model PATH           Path to GGUF model "
           "(default: model.gguf)\n"
        << "  --n_ctx N              Context length, N > 0 "
           "(default: 4096)\n"
        << "  --n_gpu_layers N       Number of GPU layers, N >= -1 "
           "(default: 35)\n"
        << "  --ngl N                Alias for --n_gpu_layers\n"
        << "  --seed N               RNG seed, 0 selects a random seed "
           "(default: 0)\n"
        << "  --no-flash-attn        Disable flash attention\n"
        << "  --quiet                Reduce application logging\n"
        << "  --topology PATH        Model topology JSON file\n"
        << "  --help, -h             Show this help\n";
}

void validate_config(const AppConfig& cfg) {
    if (cfg.world_size <= 0) {
        throw std::invalid_argument(
            "world_size must be greater than zero");
    }

    if (cfg.world_rank < 0 || cfg.world_rank >= cfg.world_size) {
        throw std::invalid_argument(
            "world_rank must be in [0, world_size)");
    }

    if (cfg.llama_cfg.model_path.empty()) {
        throw std::invalid_argument(
            "model path must not be empty");
    }

    if (cfg.llama_cfg.n_ctx <= 0) {
        throw std::invalid_argument(
            "--n_ctx must be greater than zero");
    }

    if (cfg.llama_cfg.n_gpu_layers < -1) {
        throw std::invalid_argument(
            "--n_gpu_layers must be greater than or equal to -1");
    }
}

}  // namespace

AppConfig parse_app_config(int argc,
                           char** argv,
                           int world_rank,
                           int world_size) {
    AppConfig cfg;

    cfg.world_rank = world_rank;
    cfg.world_size = world_size;

    // Defaults.
    cfg.llama_cfg.model_path = "model.gguf";
    cfg.llama_cfg.n_ctx = 4096;
    cfg.llama_cfg.n_gpu_layers = 35;
    cfg.llama_cfg.seed = 0;
    cfg.llama_cfg.gpu_layer_split_mode = 0;
    cfg.llama_cfg.use_flash_attn = true;
    cfg.llama_cfg.verbose = (world_rank == 0);

    cfg.is_gateway = (world_rank == 0);
    cfg.verbose = cfg.llama_cfg.verbose;

    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);

        if (option == "--model") {
            cfg.llama_cfg.model_path = parse_str_arg(argc, argv, i);
            ++i;
        } else if (option == "--n_ctx") {
            cfg.llama_cfg.n_ctx = parse_int_arg(argc, argv, i);
            ++i;
        } else if (option == "--n_gpu_layers" || option == "--ngl") {
            cfg.llama_cfg.n_gpu_layers = parse_int_arg(argc, argv, i);
            ++i;
        } else if (option == "--seed") {
            cfg.llama_cfg.seed = parse_int_arg(argc, argv, i);
            ++i;
        } else if (option == "--no-flash-attn") {
            cfg.llama_cfg.use_flash_attn = false;
        } else if (option == "--quiet") {
            cfg.llama_cfg.verbose = false;
        } else if (option == "--topology") {
            cfg.topology_path = parse_str_arg(argc, argv, i);
            ++i;
        } else if (option == "--help" || option == "-h") {
            if (world_rank == 0) {
                print_usage(argv[0]);
            }

            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(
                std::string("Unknown option: ") + argv[i] +
                ". Use --help to list supported options.");
        }
    }

    cfg.is_gateway = (world_rank == 0);
    cfg.verbose = cfg.llama_cfg.verbose;

    validate_config(cfg);

    return cfg;
}

void print_app_config(const AppConfig& cfg) {
    if (!cfg.verbose || cfg.world_rank != 0) {
        return;
    }

    std::cout << "[AppConfig] world_size=" << cfg.world_size
              << ", world_rank=" << cfg.world_rank
              << ", gateway=" << (cfg.is_gateway ? "yes" : "no")
              << '\n';

    std::cout << "[AppConfig] model_path=" << cfg.llama_cfg.model_path
              << ", n_ctx=" << cfg.llama_cfg.n_ctx
              << ", n_gpu_layers=" << cfg.llama_cfg.n_gpu_layers
              << ", seed=" << cfg.llama_cfg.seed
              << ", flash_attn="
              << (cfg.llama_cfg.use_flash_attn ? "on" : "off")
              << '\n';

    if (!cfg.topology_path.empty()) {
        std::cout << "[AppConfig] topology_path="
                  << cfg.topology_path
                  << '\n';
    }
}