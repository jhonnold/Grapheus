#pragma once

#include "adam.h"

namespace operations {

// clang-format off
template<data::Device DEV>
void adam_w(      data::DenseMatrix<float>& values,
            const data::DenseMatrix<float>& gradients,
                  data::DenseMatrix<float>& first_moment,
                  data::DenseMatrix<float>& second_moment,
            float                           lr,
            float                           beta1,
            float                           beta2,
            float                           eps,
            int                             warmup,
            int                             step,
            // not directly related to adam but any optimizer should set use this too ^^
            float min,
            float max,
            float lasso,
            float ridge) {
    // clang-format on

    ASSERT(first_moment.ld == second_moment.ld);
    ASSERT(values.ld == gradients.ld);
    ASSERT(values.m == gradients.m && gradients.m == first_moment.m
           && first_moment.m == second_moment.m);
    ASSERT(values.n == gradients.n && gradients.n == first_moment.n
           && first_moment.n == second_moment.n);

    ASSERT(values.first<DEV>());
    ASSERT(gradients.first<DEV>());
    ASSERT(first_moment.first<DEV>());
    ASSERT(second_moment.first<DEV>());

    float bc1 = 1.0f - powf(beta1, step);
    float bc2 = 1.0f - powf(beta2, step);
    float slr = lr;
    if (warmup > step)
        slr = 1e-8 + step * lr / warmup;

    slr *= sqrtf(bc2) / bc1;

    if (data::is_gpu(DEV)) {
        int block_size_x;
        int block_size_y;
        if (values.m > 128) {
            block_size_x = 1;
            block_size_y = 32;
        } else if (values.m > 8) {
            block_size_x = 32;
            block_size_y = 8;
        } else {
            block_size_x = 512;
            block_size_y = 1;
        };

        dim3 block(block_size_x, block_size_y);
        dim3 grid(std::ceil((float) values.n / block_size_x),
                  std::ceil((float) values.m / block_size_y));
        adam_kernel<<<grid, block>>>(values.first<DEV>(),
                                     gradients.first<DEV>(),
                                     first_moment.first<DEV>(),
                                     second_moment.first<DEV>(),
                                     values.m,
                                     values.n,
                                     values.ld,
                                     first_moment.ld,
                                     slr,
                                     beta1,
                                     beta2,
                                     eps,
                                     min,
                                     max,
                                     lasso,
                                     ridge);
    } else {
        ERROR(false);
    }
}

}    // namespace operations