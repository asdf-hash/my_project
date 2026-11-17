#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>

class Profiler {
public:
  enum class Metric : std::size_t {
    Frame,
    SyncAndAcquire,
    UniformUpdate,
    ClipmapUpdate,
    ClipmapBuild,
    ClipmapFallback,
    CommandRecording,
    SubmitAndPresent,
    GpuCompute,
    GpuComputeUpdate,
    GpuHeightGeneration,
    GpuMeshGeneration,
    GpuFinalize,
    GpuScene,
    GpuTaa,
    Count
  };

  class Scope {
  public:
    explicit Scope(Metric metric) noexcept : metric_(metric), enabled_(Profiler::instance().enabled()) {
      if (enabled_) {
        start_ = Clock::now();
      }
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    ~Scope() {
      if (enabled_) {
        Profiler::instance().recordNanoseconds(
            metric_, static_cast<std::uint64_t>(
                         std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count()));
      }
    }

  private:
    using Clock = std::chrono::steady_clock;
    Metric metric_;
    bool enabled_;
    Clock::time_point start_;
  };

  static Profiler &instance() {
    static Profiler profiler;
    return profiler;
  }

  bool enabled() const noexcept { return enabled_.load(std::memory_order_relaxed); }

  void setEnabled(bool enabled) {
    if (enabled_.load(std::memory_order_relaxed) == enabled) {
      return;
    }
    if (!enabled) {
      enabled_.store(false, std::memory_order_relaxed);
    }
    clearMetrics();
    lastReport_ = Clock::now();
    if (enabled) {
      enabled_.store(true, std::memory_order_relaxed);
    }
  }

  void recordNanoseconds(Metric metric, std::uint64_t nanoseconds) noexcept {
    if (!enabled()) {
      return;
    }
    auto &stats = metrics_[static_cast<std::size_t>(metric)];
    stats.count.fetch_add(1, std::memory_order_relaxed);
    stats.totalNanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
    auto bucket = nanoseconds / histogramBucketNanoseconds;
    if (bucket >= histogramBucketCount) {
      bucket = histogramBucketCount - 1;
    }
    stats.histogram[bucket].fetch_add(1, std::memory_order_relaxed);

    auto maximum = stats.maxNanoseconds.load(std::memory_order_relaxed);
    while (maximum < nanoseconds &&
           !stats.maxNanoseconds.compare_exchange_weak(maximum, nanoseconds, std::memory_order_relaxed)) {
    }
  }

  void reportIfDue() {
    if (!enabled()) {
      return;
    }
    const auto now = Clock::now();
    const auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastReport_);
    if (interval < reportInterval_) {
      return;
    }

    lastReport_ = now;
    std::clog << "[profile " << interval.count() << "ms]";
    for (std::size_t i = 0; i < metrics_.size(); ++i) {
      const auto count = metrics_[i].count.exchange(0, std::memory_order_relaxed);
      const auto totalNanoseconds = metrics_[i].totalNanoseconds.exchange(0, std::memory_order_relaxed);
      const auto maxNanoseconds = metrics_[i].maxNanoseconds.exchange(0, std::memory_order_relaxed);
      if (count == 0) {
        continue;
      }

      const double averageMilliseconds =
          static_cast<double>(totalNanoseconds) / static_cast<double>(count) / 1'000'000.0;
      const double maximumMilliseconds = static_cast<double>(maxNanoseconds) / 1'000'000.0;
      const auto percentileIndex = (count * 95u + 99u) / 100u;
      std::uint64_t samplesBefore = 0;
      std::size_t percentileBucket = 0;
      std::array<std::uint64_t, histogramBucketCount> histogram{};
      for (std::size_t bucket = 0; bucket < histogramBucketCount; ++bucket) {
        histogram[bucket] = metrics_[i].histogram[bucket].exchange(0, std::memory_order_relaxed);
      }
      for (; percentileBucket < histogramBucketCount; ++percentileBucket) {
        samplesBefore += histogram[percentileBucket];
        if (samplesBefore >= percentileIndex) {
          break;
        }
      }
      const double percentileMilliseconds =
          static_cast<double>((percentileBucket + 1) * histogramBucketNanoseconds) / 1'000'000.0;
      std::clog << ' ' << metricNames_[i] << '=' << averageMilliseconds << '/' << percentileMilliseconds << '/'
                << maximumMilliseconds << "ms(" << count << ')';
    }
    std::clog << '\n' << std::flush;
  }

private:
  using Clock = std::chrono::steady_clock;

  struct MetricStats {
    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint64_t> totalNanoseconds{0};
    std::atomic<std::uint64_t> maxNanoseconds{0};
    std::array<std::atomic<std::uint64_t>, 256> histogram{};
  };

  Profiler() : lastReport_(Clock::now()) {}

  void clearMetrics() noexcept {
    for (auto &stats : metrics_) {
      stats.count.store(0, std::memory_order_relaxed);
      stats.totalNanoseconds.store(0, std::memory_order_relaxed);
      stats.maxNanoseconds.store(0, std::memory_order_relaxed);
      for (auto &bucket : stats.histogram) {
        bucket.store(0, std::memory_order_relaxed);
      }
    }
  }

  static constexpr std::chrono::milliseconds reportInterval_{5000};
  static constexpr std::uint64_t histogramBucketNanoseconds = 250'000;
  static constexpr std::size_t histogramBucketCount = 256;
  static constexpr std::array metricNames_{"frame",          "sync/acquire", "uniform",   "clipmap",
                                           "clipmap-build", "clipmap-fallback", "record", "submit/present",
                                           "gpu-compute",   "gpu-compute-update", "gpu-heights", "gpu-meshing",
                                           "gpu-finalize", "gpu-scene", "gpu-taa"};
  std::array<MetricStats, static_cast<std::size_t>(Metric::Count)> metrics_{};
  std::atomic<bool> enabled_{true};
  Clock::time_point lastReport_;
};