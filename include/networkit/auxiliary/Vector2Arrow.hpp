/*
 * Vector2Arrow.hpp
 *
 *  Created on: 06.23.2026
 *      Author: Ian Chen (ianchen3)
 */

#ifndef NETWORKIT_AUXILIARY_VECTOR2_ARROW_HPP_
#define NETWORKIT_AUXILIARY_VECTOR2_ARROW_HPP_

#include <memory>
#include <vector>
#include <arrow/buffer.h>

namespace Aux {

template <typename T, typename Arr>
std::shared_ptr<Arr> vectorToArrow(std::vector<T> vec) {
    if (vec.empty())
        return nullptr;

    int64_t length = vec.size();
    int64_t byte_size = length * sizeof(T);

    auto *heap_vec = new std::vector<T>(std::move(vec));
    auto buffer = std::shared_ptr<arrow::Buffer>(
        new arrow::Buffer(reinterpret_cast<const uint8_t *>(heap_vec->data()), byte_size),
        [heap_vec](arrow::Buffer *b) {
            delete b;
            delete heap_vec;
        });

    return std::make_shared<Arr>(length, buffer);
}

} // namespace Aux

#endif // NETWORKIT_AUXILIARY_VECTOR2_ARROW_HPP_
