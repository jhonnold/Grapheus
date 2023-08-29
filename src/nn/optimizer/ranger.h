#pragma once

#include "operations/operations.h"
#include "optimizer.h"

namespace nn {

struct Ranger : public Optimizer {

    std::vector<data::DenseMatrix<float>> first_moment {};
    std::vector<data::DenseMatrix<float>> second_moment {};
    std::vector<data::DenseMatrix<float>> slow_buffer {};

    float                                 beta1           = 0.9;
    float                                 beta2           = 0.999;
    float                                 eps             = 1e-7;
    float                                 alpha           = 0.5;
    int                                   k               = 6;
    int                                   N_sma_threshold = 6;

    Ranger(const std::vector<OptimizerEntry>& entries,
           float                              beta1,
           float                              beta2,
           float                              eps,
           float                              alpha,
           int                                k,
           int                                N_sma_threshold)
        : Optimizer(entries)
        , beta1(beta1)
        , beta2(beta2)
        , eps(eps)
        , alpha(alpha)
        , k(k)
        , N_sma_threshold(N_sma_threshold) {
        first_moment.reserve(1024);
        second_moment.reserve(1024);
        slow_buffer.reserve(1024);
    }

    protected:
    void add_field(OptimizerEntry& entry) override {
        first_moment.emplace_back(entry.m_reference->values.m, entry.m_reference->values.n);
        second_moment.emplace_back(entry.m_reference->values.m, entry.m_reference->values.n);
        slow_buffer.emplace_back(entry.m_reference->values.m, entry.m_reference->values.n);

        first_moment.back().malloc<data::BOTH>();
        second_moment.back().malloc<data::BOTH>();
        slow_buffer.back().malloc<data::BOTH>();
    }

    void step(OptimizerEntry& entry, int idx, float lr) override {
        float beta2_t           = powf(beta2, step_);
        float N_sma_max         = 2.0 / (1.0 - beta2) - 1.0;
        float N_sma             = N_sma_max - 2 * step_ * beta2_t / (1.0 - beta2_t);

        bool  nsma              = N_sma >= N_sma_threshold;
        bool  store_slow_buffer = (step_ % k) == 0;

        float step_size;
        if (nsma) {
            step_size = lr
                        * sqrtf((1.0 - beta2_t) * (N_sma - 4.0) / (N_sma_max - 4.0) * (N_sma - 2.0)
                                / N_sma * N_sma_max / (N_sma_max - 2.0))
                        / (1.0 - powf(beta1, step_));
        } else {
            step_size = lr * (1.0 - powf(beta1, step_));
        }

        operations::ranger<data::GPU>(entry.m_reference->values,
                                      entry.m_reference->gradients,
                                      first_moment[idx],
                                      second_moment[idx],
                                      slow_buffer[idx],
                                      step_size,
                                      alpha,
                                      beta1,
                                      beta2,
                                      eps,
                                      nsma,
                                      store_slow_buffer,
                                      entry.m_min,
                                      entry.m_max);
    }
};

}    // namespace nn