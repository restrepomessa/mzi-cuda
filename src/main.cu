#include "../include/types.hpp"
#include "../include/cpu_simulator.hpp"
#include "../include/cuda_simulator.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <iomanip>

struct BenchmarkResult {
    int N;
    int B;
    double cpu_time_ms;
    float gpu_kernel_time_ms;
    double speedup;
    double mse;
    double max_err;
};

float random_float(float min, float max) {
    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}

BenchmarkResult run_experiment(int N, int B) {
    int num_mzi = N * (N / 2);

    std::vector<float> phases_theta(num_mzi);
    std::vector<float> phases_phi(num_mzi);
    for (int i = 0; i < num_mzi; i++) {
        phases_theta[i] = random_float(0.0f, 2.0f * 3.14159265f);
        phases_phi[i]   = random_float(0.0f, 2.0f * 3.14159265f);
    }

    std::vector<ComplexCPU> h_input(B * N);
    for (int i = 0; i < B * N; i++) {
        h_input[i] = ComplexCPU(random_float(-1.0f, 1.0f), random_float(-1.0f, 1.0f));
    }

    double cpu_time_ms = 0.0;
    std::vector<ComplexCPU> out_cpu;
    if (B <= 100000) {
        auto t0 = std::chrono::high_resolution_clock::now();
        out_cpu = CPUSimulator::propagate(N, B, h_input, phases_theta, phases_phi);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
        int sample_batch = 10000;
        std::vector<ComplexCPU> sub_in(sample_batch * N);
        for (int k = 0; k < sample_batch * N; k++) {
            sub_in[k] = h_input[k];
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        CPUSimulator::propagate(N, sample_batch, sub_in, phases_theta, phases_phi);
        auto t1 = std::chrono::high_resolution_clock::now();
        double sub_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_time_ms = sub_time * ((double)B / sample_batch);
    }

    float gpu_kernel_time_ms = 0.0f;
    std::vector<ComplexCPU> out_gpu = CUDASimulator::propagate(
        N, B, h_input, phases_theta, phases_phi, gpu_kernel_time_ms
    );

    double mse = 0.0;
    double max_err = 0.0;
    if (!out_cpu.empty()) {
        for (size_t i = 0; i < out_cpu.size(); i++) {
            double dr = out_cpu[i].real() - out_gpu[i].real();
            double di = out_cpu[i].imag() - out_gpu[i].imag();
            double err = dr * dr + di * di;
            mse += err;
            if (std::sqrt(err) > max_err) {
                max_err = std::sqrt(err);
            }
        }
        mse = mse / out_cpu.size();
    }

    double speedup = 0.0;
    if (gpu_kernel_time_ms > 0.0f) {
        speedup = cpu_time_ms / gpu_kernel_time_ms;
    }

    BenchmarkResult res;
    res.N = N;
    res.B = B;
    res.cpu_time_ms = cpu_time_ms;
    res.gpu_kernel_time_ms = gpu_kernel_time_ms;
    res.speedup = speedup;
    res.mse = mse;
    res.max_err = max_err;
    return res;
}

int main() {
    srand(42);

    std::vector<BenchmarkResult> results;

    int batches[] = {1000, 5000, 10000, 50000, 100000, 500000};
    int num_batches = sizeof(batches) / sizeof(batches[0]);

    std::cout << "\n----- 1: Batch Size Variation (N=8) -----\n";
    for (int i = 0; i < num_batches; i++) {
        int b = batches[i];
        std::cout << "Executing N=8, Batch=" << std::setw(6) << b << "... " << std::flush;
        BenchmarkResult res = run_experiment(8, b);
        results.push_back(res);
        std::cout << "CPU: " << std::fixed << std::setprecision(2) << res.cpu_time_ms << " ms | "
                  << "GPU: " << res.gpu_kernel_time_ms << " ms | "
                  << "Speedup: " << res.speedup << "x\n";
    }

    int ports[] = {4, 8, 16, 32};
    int num_ports = sizeof(ports) / sizeof(ports[0]);

    std::cout << "\n----- 2: Port Number Variation (N=8, Batch=50000) -----\n";
    for (int i = 0; i < num_ports; i++) {
        int n = ports[i];
        std::cout << "Executing N=" << std::setw(2) << n << ", Batch=50000... " << std::flush;
        BenchmarkResult res = run_experiment(n, 50000);
        results.push_back(res);
        std::cout << "CPU: " << std::fixed << std::setprecision(2) << res.cpu_time_ms << " ms | "
                  << "GPU: " << res.gpu_kernel_time_ms << " ms | "
                  << "Speedup: " << res.speedup << "x\n";
    }

    std::ofstream csv("results_benchmark.csv");
    csv << "N,Batch,CPU_Time_ms,GPU_Time_ms,Speedup,MSE,MaxError\n";
    for (size_t i = 0; i < results.size(); i++) {
        csv << results[i].N << "," 
            << results[i].B << "," 
            << results[i].cpu_time_ms << ","
            << results[i].gpu_kernel_time_ms << "," 
            << results[i].speedup << ","
            << results[i].mse << "," 
            << results[i].max_err << "\n";
    }
    csv.close();

    return 0;
}