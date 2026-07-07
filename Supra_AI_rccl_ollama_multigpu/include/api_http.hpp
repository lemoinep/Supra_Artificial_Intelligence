#pragma once

#include "scheduler_threadsafe.hpp"
#include <unordered_map>
#include <mutex>
#include <optional>
#include <string>

namespace httplib {
class Server;
}

namespace api {

struct ResponseStore {
    std::mutex m;
    std::unordered_map<int, std::string> responses;

    void put(int id, const std::string& resp) {
        std::lock_guard<std::mutex> lock(m);
        responses[id] = resp;
    }

    std::optional<std::string> get(int id) {
        std::lock_guard<std::mutex> lock(m);
        auto it = responses.find(id);
        if (it == responses.end()) return std::nullopt;
        return it->second;
    }
};

// Setup HTTP routes for:
//  - POST /v1/chat/completions
//  - GET  /v1/request/{id}
//  - POST /admin/lora/load
//
// next_request_id is incremented for each new request.
void setup_http_routes(httplib::Server& srv,
                       ThreadSafeScheduler& scheduler,
                       ResponseStore& resp_store,
                       int& next_request_id);

} // namespace api