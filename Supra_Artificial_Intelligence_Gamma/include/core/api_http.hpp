#pragma once

#include "core/scheduler_threadsafe.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace httplib {
class Server;
}

namespace api {

// -----------------------------------------------------------------------------
// Thread-safe in-memory storage for completed request responses.
//
// The gateway dispatcher writes a result after an MPI worker finishes.
// HTTP GET handlers read the result using the request ID.
// -----------------------------------------------------------------------------
class ResponseStore {
public:
    ResponseStore() = default;

    ResponseStore(const ResponseStore&) = delete;
    ResponseStore& operator=(const ResponseStore&) = delete;

    // Store or replace the response associated with request_id.
    void put(int request_id, std::string response) {
        std::lock_guard<std::mutex> lock(mutex_);
        responses_[request_id] = std::move(response);
    }

    // Return a copy of the response, or std::nullopt when it is unavailable.
    [[nodiscard]] std::optional<std::string> get(int request_id) const {
        std::lock_guard<std::mutex> lock(mutex_);

        const auto it = responses_.find(request_id);

        if (it == responses_.end()) {
            return std::nullopt;
        }

        return it->second;
    }

    // Remove one completed response. Useful after a client has consumed it.
    bool erase(int request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return responses_.erase(request_id) != 0;
    }

    // Clear all stored responses, for shutdown or tests.
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        responses_.clear();
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return responses_.size();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<int, std::string> responses_;
};

// Configure gateway HTTP routes:
//
//   POST /v1/chat/completions
//   GET  /v1/request/{id}
//   POST /admin/lora/load
//
// next_request_id is atomically incremented for every submitted request.
void setup_http_routes(httplib::Server& server,
                       ThreadSafeScheduler& scheduler,
                       ResponseStore& response_store,
                       std::atomic<int>& next_request_id);

} // namespace api