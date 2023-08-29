#pragma once

namespace operations {

__global__ void ranger_kernel(float* __restrict__ values,
                              float* __restrict__ gradients,
                              float* __restrict__ first_moment,
                              float* __restrict__ second_moment,
                              float* __restrict__ slow_buffer,
                              size_t m,
                              size_t n,
                              size_t ldv,
                              size_t ldm,
                              float  step_size,
                              float  alpha,
                              float  beta1,
                              float  beta2,
                              float  eps,
                              bool   n_sma,
                              bool   store_slow_buffer,
                              float  m_min,
                              float  m_max) {
    // clang-format on

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= n || idy >= m)
        return;

    size_t idv = MATRIX_INDEX(ldv, idy, idx);
    size_t idm = MATRIX_INDEX(ldm, idy, idx);

    // moments
    first_moment[idm]  = beta1 * first_moment[idm] + (1 - beta1) * gradients[idv];
    second_moment[idm] = beta2 * second_moment[idm] + (1 - beta2) * gradients[idv] * gradients[idv];

    float delta = n_sma ?
        step_size * first_moment[idm] / (sqrtf(second_moment[idm]) + eps) :
        step_size * first_moment[idm];

    values[idv] -= delta;

    if (store_slow_buffer) {
        slow_buffer[idm] += alpha * (values[idv] - slow_buffer[idm]);
        values[idv] = slow_buffer[idm];
    }

    values[idv]    = max(m_min, min(m_max, values[idv]));
    gradients[idv] = 0;
}

// clang-format off
template<data::Device DEV>
void ranger(    data::DenseMatrix<float>& values,
          const data::DenseMatrix<float>& gradients,
                data::DenseMatrix<float>& first_moment,
                data::DenseMatrix<float>& second_moment,
                data::DenseMatrix<float>& slow_buffer,
          float                           step_size,
          float                           alpha, // NOT LR IN RANGER
          float                           beta1,
          float                           beta2,
          float                           eps,
          bool n_sma,
          bool store_slow_buffer,
          float min,
          float max) {
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
        ranger_kernel<<<grid, block>>>(values.first<DEV>(),
                                       gradients.first<DEV>(),
                                       first_moment.first<DEV>(),
                                       second_moment.first<DEV>(),
                                       slow_buffer.first<DEV>(),
                                       values.m,
                                       values.n,
                                       values.ld,
                                       first_moment.ld,
                                       step_size,
                                       alpha,
                                       beta1,
                                       beta2,
                                       eps,
                                       n_sma,
                                       store_slow_buffer,
                                       min,
                                       max);
    } else {
        ERROR(false);
    }
}

}    // namespace operations