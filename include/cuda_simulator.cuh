#pragma once
#include "types.hpp"
#include <vector>

class CUDASimulator {
public:
    static std::vector<ComplexCPU> propagate(
        int N, int B,
        const std::vector<ComplexCPU>& h_input,
        const std::vector<float>& h_theta,
        const std::vector<float>& h_phi,
        float& kernel_time_ms);
};