#pragma once

#include <cstddef>
#include "Traits.h"

namespace tensor_view {

template<size_t N, size_t K>
class CalculateOffsetImpl {
public:
    template<typename V, typename... Vs>
    static size_t calculate(size_t offset, const size_t* stride, V i, Vs&& ... vs) {
        return CalculateOffsetImpl<N - 1, K>::calculate(offset + stride[K - N] * i, stride, vs...);
    }
};

template<size_t K>
class CalculateOffsetImpl<0, K> {
public:
    template<typename V>
    static size_t calculate(size_t offset, const size_t* stride, V i) {
        return offset + stride[K] * i;
    }
};

void calculate_strides(const size_t* shapes, size_t* strides, size_t ndim) {
    size_t prod = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = prod;
        prod *= shapes[i];
    }
}


namespace detail {

template<class TensorViewLhs, class TensorViewRhs>
bool check_shapes(TensorViewLhs first, TensorViewRhs second, implicit_broadcast) {
    size_t ndims = std::min(TensorViewLhs::NumDims, TensorViewRhs::NumDims);
    for (int i = 0; i < ndims; ++i) {
        size_t size_lhs = first.size(TensorViewLhs::NumDims - i - 1);
        size_t size_rhs = second.size(TensorViewRhs::NumDims - i - 1);
        if (size_lhs != size_rhs && size_lhs != 1 && size_rhs != 1) {
            return false;
        }
    }
    return true;
}

template<class TensorViewLhs, class TensorViewRhs>
bool check_shapes(TensorViewLhs first, TensorViewRhs second, explicit_broadcast) {
    if (TensorViewLhs::NumDims != TensorViewRhs::NumDims) {
        return false;
    }
    return check_shapes(first, second, implicit_broadcast{});
}

template<class TensorViewLhs, class TensorViewRhs>
bool check_shapes(TensorViewLhs first, TensorViewRhs second, disable_broadcast) {
    if (TensorViewLhs::NumDims != TensorViewRhs::NumDims) {
        return false;
    }
    for (int i = 0; i < TensorViewLhs::NumDims; ++i) {
        if (first.size(i) != second.size(i)) {
            return false;
        }
    }
    return true;
}

} // detail

template<class TensorViewLhs, class TensorViewRhs>
bool check_shapes(TensorViewLhs first, TensorViewRhs second) {
    using LhsType = TensorViewChecked<TensorViewLhs>;
    using RhsType = TensorViewChecked<TensorViewRhs>;
    typename RhsType::BroadcastPolicyTag broadcast_tag;
    return detail::check_shapes(first, second, broadcast_tag);
}

template<class TensorViewLhs, class TensorViewRhs>
bool is_trivial_layout(const TensorViewLhs& lhs, const TensorViewRhs& rhs) {
    return TensorViewLhs::NumDims == TensorViewRhs::NumDims &&
           lhs.is_contiguous() &&
           rhs.is_contiguous() &&
           std::array<size_t, TensorViewLhs::NumDims>(lhs.shape()) ==
           std::array<size_t, TensorViewRhs::NumDims>(rhs.shape());
}

} // namespace