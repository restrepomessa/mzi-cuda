#pragma once
#include "types.hpp"
#include <cmath>
#include <vector>

class CPUSimulator {
public:
    static std::vector<ComplexCPU> propagate(
        int N, int B,
        const std::vector<ComplexCPU>& batch_input,
        const std::vector<float>& phases_theta,
        const std::vector<float>& phases_phi) 
    {
        std::vector<ComplexCPU> state = batch_input;
        int mzi_per_layer = N / 2;

        for (int layer = 0; layer < N; layer++) {
            bool is_even = (layer % 2 == 0);
            int mzi_count = is_even ? (N / 2) : ((N - 1) / 2);

            for (int b = 0; b < B; b++) {
                for (int m = 0; m < mzi_count; m++) {
                    int p_top = is_even ? (2 * m) : (2 * m + 1);
                    int p_bot = p_top + 1;

                    int phase_idx = layer * mzi_per_layer + m;
                    float theta = phases_theta[phase_idx];
                    float phi   = phases_phi[phase_idx];

                    float half_theta = theta * 0.5f;
                    float s = std::sin(half_theta);
                    float c = std::cos(half_theta);

                    ComplexCPU phase_common(-s, c);
                    ComplexCPU phase_phi(std::cos(phi), std::sin(phi));

                    ComplexCPU t00 = phase_common * phase_phi * s;
                    ComplexCPU t01 = phase_common * phase_phi * c;
                    ComplexCPU t10 = phase_common * c;
                    ComplexCPU t11 = phase_common * (-s);

                    int idx_top = b * N + p_top;
                    int idx_bot = b * N + p_bot;

                    ComplexCPU in_top = state[idx_top];
                    ComplexCPU in_bot = state[idx_bot];

                    state[idx_top] = t00 * in_top + t01 * in_bot;
                    state[idx_bot] = t10 * in_top + t11 * in_bot;
                }
            }
        }
        return state;
    }
};