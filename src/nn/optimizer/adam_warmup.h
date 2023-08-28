#pragma once

#include "operations/operations.h"
#include "optimizer.h"

namespace nn {

struct AdamWarmup : public Adam {

    int warmup = 0;

    AdamWarmup(const std::vector<OptimizerEntry>& entries,
               float                              beta1,
               float                              beta2,
               float                              eps,
               int                                warmup)
        : Adam(entries, beta1, beta2, eps)
        , warmup(warmup) {}

    protected:
    void step(OptimizerEntry& entry, int idx, float lr) override {
        operations::adam_w<data::GPU>(entry.m_reference->values,
                                      entry.m_reference->gradients,
                                      first_moment[idx],
                                      second_moment[idx],
                                      lr,
                                      beta1,
                                      beta2,
                                      eps,
                                      warmup,
                                      step_,
                                      entry.m_min,
                                      entry.m_max,
                                      entry.m_lasso,
                                      entry.m_ridge);
    }
};

}    // namespace nn