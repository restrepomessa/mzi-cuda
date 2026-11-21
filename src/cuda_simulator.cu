#include "../include/cuda_simulator.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__device__ cuFloatComplex complex_mult(cuFloatComplex a, cuFloatComplex b) {
    cuFloatComplex res;
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res;
}

__device__ cuFloatComplex complex_add(cuFloatComplex a, cuFloatComplex b) {
    cuFloatComplex res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
}

__global__ void clements_propagate_kernel(
    int N, int B,
    const cuFloatComplex* d_in,
    cuFloatComplex* d_out,
    const float* d_theta,
    const float* d_phi) 
{
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= B) {
        return;
    }

    cuFloatComplex state[64];
    for (int i = 0; i < N; i++) {
        state[i] = d_in[b * N + i];
    }

    int mzi_per_layer = N / 2;

    for (int layer = 0; layer < N; layer++) {
        bool is_even = (layer % 2 == 0);
        int mzi_count = is_even ? (N / 2) : ((N - 1) / 2);

        for (int m = 0; m < mzi_count; m++) {
            int p_top = is_even ? (2 * m) : (2 * m + 1);
            int p_bot = p_top + 1;

            int phase_idx = layer * mzi_per_layer + m;
            float theta = d_theta[phase_idx];
            float phi   = d_phi[phase_idx];

            float half_th = theta * 0.5f;
            float s = sinf(half_th);
            float c = cosf(half_th);

            cuFloatComplex phase_common;
            phase_common.x = -s;
            phase_common.y = c;

            cuFloatComplex phase_phi;
            phase_phi.x = cosf(phi);
            phase_phi.y = sinf(phi);

            cuFloatComplex s_cpx = make_cuFloatComplex(s, 0.0f);
            cuFloatComplex c_cpx = make_cuFloatComplex(c, 0.0f);
            cuFloatComplex neg_s_cpx = make_cuFloatComplex(-s, 0.0f);

            cuFloatComplex common_phi = complex_mult(phase_common, phase_phi);
            cuFloatComplex t00 = complex_mult(common_phi, s_cpx);
            cuFloatComplex t01 = complex_mult(common_phi, c_cpx);
            cuFloatComplex t10 = complex_mult(phase_common, c_cpx);
            cuFloatComplex t11 = complex_mult(phase_common, neg_s_cpx);

            cuFloatComplex in_top = state[p_top];
            cuFloatComplex in_bot = state[p_bot];

            cuFloatComplex out_top = complex_add(complex_mult(t00, in_top), complex_mult(t01, in_bot));
            cuFloatComplex out_bot = complex_add(complex_mult(t10, in_top), complex_mult(t11, in_bot));

            state[p_top] = out_top;
            state[p_bot] = out_bot;
        }
    }

    for (int i = 0; i < N; i++) {
        d_out[b * N + i] = state[i];
    }
}

std::vector<ComplexCPU> CUDASimulator::propagate(
    int N, int B,
    const std::vector<ComplexCPU>& h_input,
    const std::vector<float>& h_theta,
    const std::vector<float>& h_phi,
    float& kernel_time_ms) 
{
    int total_states = B * N;
    int total_phases = N * (N / 2);
    size_t state_bytes = total_states * sizeof(cuFloatComplex);
    size_t phase_bytes = total_phases * sizeof(float);

    cuFloatComplex* d_in = NULL;
    cuFloatComplex* d_out = NULL;
    float* d_theta = NULL;
    float* d_phi = NULL;

    cudaMalloc((void**)&d_in, state_bytes);
    cudaMalloc((void**)&d_out, state_bytes);
    cudaMalloc((void**)&d_theta, phase_bytes);
    cudaMalloc((void**)&d_phi, phase_bytes);

    cudaMemcpy(d_in, &h_input[0], state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, &h_theta[0], phase_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, &h_phi[0], phase_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (B + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    clements_propagate_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, B, d_in, d_out, d_theta, d_phi);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    std::vector<ComplexCPU> h_out(total_states);
    cudaMemcpy(&h_out[0], d_out, state_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_theta);
    cudaFree(d_phi);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return h_out;
}