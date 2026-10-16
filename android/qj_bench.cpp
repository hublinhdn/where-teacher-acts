// where-teacher-acts: on-device latency of the deployed ONNX graphs, ONNX Runtime CPU execution provider (Android, arm64).
//
// Mirrors pipeline/27_bench_latency.py, which times the same graphs on the desktop CPU: per graph and thread
// setting a new session (intra-op threads t, inter-op threads 1, all graph optimizations), one random
// 1 x 3 x size x size float input, `warmup` untimed runs and `iters` timed runs; one CSV row per graph and
// thread setting with the median, 95th percentile (linear interpolation, as numpy), mean, minimum and
// maximum of the timed runs. One invocation is one round over every graph; the host script
// (run_android_bench.py) runs the rounds, logs the device temperature between them and aggregates.
//
// With --search it also times the gallery ranking of the desktop benchmark: scores of one 512-d query
// against 9804 x 512 random references and a full ascending argsort, 20 untimed then `iters` timed
// repetitions, reported as the mean per repetition (the desktop reports the same with numpy).
//
// With --peak-memory it prints, after all runs, the peak resident set size of the process (VmHWM, kB) as a
// last row; the host script runs one process per graph for this, so each value belongs to one graph.
//
// usage: qj_bench [--threads 1,4] [--warmup 30] [--iters 200] [--size 224] [--search] [--peak-memory] graph.onnx ...
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Stats {
    double median, p95, mean, min, max;
};

double percentile(const std::vector<double>& sorted, double q) {
    const double idx = q * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (idx - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s{};
    s.median = percentile(v, 0.5);
    s.p95 = percentile(v, 0.95);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    s.min = v.front();
    s.max = v.back();
    return s;
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

std::string csv_safe(std::string s) {
    std::replace(s.begin(), s.end(), ',', ';');
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
    return s;
}

long peak_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return std::atol(line.c_str() + 6);
    }
    return -1;
}

double time_search(int iters) {
    const int n = 9804, d = 512;
    std::mt19937 g0(0), g1(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> gallery(static_cast<size_t>(n) * d), query(d), scores(n);
    for (auto& x : gallery) x = nd(g0);
    for (auto& x : query) x = nd(g1);
    std::vector<int> order(n);
    auto once = [&]() {
        for (int i = 0; i < n; ++i) {
            const float* row = gallery.data() + static_cast<size_t>(i) * d;
            float acc = 0.f;
            for (int k = 0; k < d; ++k) acc += row[k] * query[k];
            scores[i] = acc;
        }
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] < scores[b]; });
    };
    for (int i = 0; i < 20; ++i) once();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) once();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> threads{1, 4};
    int warmup = 30, iters = 200, size = 224;
    bool search = false, peak_memory = false;
    std::vector<std::string> graphs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) {
            threads = parse_list(argv[++i]);
        } else if (a == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (a == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (a == "--size" && i + 1 < argc) {
            size = std::atoi(argv[++i]);
        } else if (a == "--search") {
            search = true;
        } else if (a == "--peak-memory") {
            peak_memory = true;
        } else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        } else {
            graphs.push_back(a);
        }
    }
    if (threads.empty() || warmup < 0 || iters < 1 || size < 1) {
        std::fprintf(stderr, "bad arguments\n");
        return 2;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qj_bench");
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> input(static_cast<size_t>(3) * size * size);
    for (auto& x : input) x = nd(rng);
    const std::vector<int64_t> shape{1, 3, size, size};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::printf("graph,threads,status,median_ms,p95_ms,mean_ms,min_ms,max_ms,warmup,iters,error\n");
    std::fflush(stdout);
    for (const auto& path : graphs) {
        for (const int t : threads) {
            try {
                Ort::SessionOptions so;
                so.SetIntraOpNumThreads(t);
                so.SetInterOpNumThreads(1);
                so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                Ort::Session session(env, path.c_str(), so);
                Ort::AllocatorWithDefaultOptions alloc;
                Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, alloc);
                const size_t n_out = session.GetOutputCount();
                std::vector<Ort::AllocatedStringPtr> out_owned;
                std::vector<const char*> out_names;
                for (size_t k = 0; k < n_out; ++k) {
                    out_owned.push_back(session.GetOutputNameAllocated(k, alloc));
                    out_names.push_back(out_owned.back().get());
                }
                const char* in_names[] = {in_name.get()};
                Ort::Value x = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(),
                                                               shape.size());
                for (int i = 0; i < warmup; ++i) {
                    session.Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names.data(), n_out);
                }
                std::vector<double> lat;
                lat.reserve(static_cast<size_t>(iters));
                for (int i = 0; i < iters; ++i) {
                    const auto t0 = std::chrono::steady_clock::now();
                    auto y = session.Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names.data(), n_out);
                    const auto t1 = std::chrono::steady_clock::now();
                    lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
                const Stats s = summarize(lat);
                std::printf("%s,%d,OK,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,\n", path.c_str(), t, s.median, s.p95, s.mean,
                            s.min, s.max, warmup, iters);
            } catch (const std::exception& e) {
                std::printf("%s,%d,FAILED,,,,,,%d,%d,%s\n", path.c_str(), t, warmup, iters, csv_safe(e.what()).c_str());
            }
            std::fflush(stdout);
        }
    }
    if (search) {
        std::printf("gallery_search_9804x512,1,OK,%.4f,,,,,20,%d,\n", time_search(iters), iters);
        std::fflush(stdout);
    }
    if (peak_memory) {
        std::printf("peak_rss_kb,0,OK,%ld,,,,,%d,%d,\n", peak_rss_kb(), warmup, iters);
        std::fflush(stdout);
    }
    return 0;
}
