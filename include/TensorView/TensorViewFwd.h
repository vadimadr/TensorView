#pragma once

#include <cstddef>

namespace tensor_view {

struct implicit_broadcast{};
struct explicit_broadcast{};
struct disable_broadcast{};


template<class T, size_t ndim, class BroadcastPolicy = implicit_broadcast>
class TensorView;

template<class T, size_t ndim, class BroadcastPolicy = implicit_broadcast>
class Tensor;

} // namespace