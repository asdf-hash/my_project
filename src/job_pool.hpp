#pragma once

// Small fork-join pool: parallelFor splits [0, count) into chunks that the workers and the calling thread take
// until none remain, then returns once every worker is done. Nothing outlives a call.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class JobPool {
public:
  explicit JobPool(unsigned threads) {
    for (unsigned i = 0; i < threads; ++i) {
      workers.emplace_back([this](std::stop_token stop) { workerMain(stop); });
    }
  }
  ~JobPool() {
    {
      std::lock_guard lock(mutex);
      stopping = true;
    }
    start.notify_all();
    workers.clear(); // joins
  }
  JobPool(const JobPool &) = delete;
  JobPool &operator=(const JobPool &) = delete;

  unsigned threadCount() const { return static_cast<unsigned>(workers.size()); }

  /**
   * Run fn(begin, end) over [0, count) in chunks of `grain`, on the workers and this thread.
   */
  void parallelFor(std::size_t count, std::size_t grain, const std::function<void(std::size_t, std::size_t)> &fn) {
    grain = std::max<std::size_t>(grain, 1);
    if (workers.empty() || count <= grain) {
      if (count != 0) {
        fn(0, count);
      }
      return;
    }
    {
      std::lock_guard lock(mutex);
      job = &fn;
      jobCount = count;
      jobGrain = grain;
      next.store(0, std::memory_order_relaxed);
      active = static_cast<unsigned>(workers.size());
      ++generation;
    }
    start.notify_all();
    runChunks(fn, count, grain);
    std::unique_lock lock(mutex);
    done.wait(lock, [&] { return active == 0; });
    job = nullptr;
  }

private:
  void runChunks(const std::function<void(std::size_t, std::size_t)> &fn, std::size_t count, std::size_t grain) {
    for (;;) {
      const std::size_t begin = next.fetch_add(grain, std::memory_order_relaxed);
      if (begin >= count) {
        return;
      }
      fn(begin, std::min(begin + grain, count));
    }
  }

  void workerMain(std::stop_token) {
    std::uint64_t seen = 0;
    for (;;) {
      const std::function<void(std::size_t, std::size_t)> *fn;
      std::size_t count;
      std::size_t grain;
      {
        std::unique_lock lock(mutex);
        start.wait(lock, [&] { return stopping || generation != seen; });
        if (stopping) {
          return;
        }
        seen = generation;
        fn = job;
        count = jobCount;
        grain = jobGrain;
      }
      runChunks(*fn, count, grain);
      {
        std::lock_guard lock(mutex);
        if (--active == 0) {
          done.notify_one();
        }
      }
    }
  }

  std::mutex mutex;
  std::condition_variable start;
  std::condition_variable done;
  const std::function<void(std::size_t, std::size_t)> *job = nullptr;
  std::size_t jobCount = 0;
  std::size_t jobGrain = 1;
  std::atomic<std::size_t> next{0};
  unsigned active = 0;
  std::uint64_t generation = 0;
  bool stopping = false;
  std::vector<std::jthread> workers; // last: joined before the state above is destroyed
};
