#pragma once

#include "affine.h"

#include <iostream>

namespace nn {

struct AffineMulti : public nn::Affine {

    AffineMulti(Layer* prev, size_t size, size_t counts)
        : Affine(prev, size * counts) {
        for (size_t i = size; i < size * counts; i++)
            for (size_t j = 0; j < prev->size; j++)
                weights.values(i, j) = weights.values(i % size, j);

        weights.values >> data::GPU;
    }
};
}    // namespace nn
