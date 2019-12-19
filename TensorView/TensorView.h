#pragma once

#include <cstddef>
#include <algorithm>
#include <iostream>

#include "Dims.h"
#include "TensorViewFwd.h"
#include "Traits.h"
#include "Operations.h"
#include "TensorIO.h"

namespace tensor_view {

template<class TTensorView, std::enable_if_t<is_tensor_view_v<TTensorView>, int> = 0>
std::ostream& operator<<(std::ostream& stream, const TTensorView& t);

template<class T, size_t ndim, class BroadcastPolicy>
class TensorView {
public:
    using Type = TensorView<T, ndim, BroadcastPolicy>;
    using ValueType = T;
    using ShapeType = const size_t*;
    using BroadcastPolicyTag = BroadcastPolicy;
    static constexpr size_t NumDims = ndim;

    TensorView(T* data_ptr, const size_t shape[ndim]) :
            data_ptr_(data_ptr) {
        std::copy(shape, shape + ndim, shape_);
        calculate_strides(shape_, stride_, ndim);
    }

    TensorView(T* data_ptr, const size_t shape[ndim], const size_t stride[ndim]) :
            data_ptr_(data_ptr) {
        std::copy(shape, shape + ndim, shape_);
        std::copy(stride, stride + ndim, stride_);
    }

    template<class BinaryOp, enable_if_t<is_binary_op_v<BinaryOp>, int> = 0>
    Type& operator=(const BinaryOp& op) {
        op.apply(*this);
    }

    template<typename ...TInds, std::enable_if_t<sizeof...(TInds) == ndim, int> = 0>
    T& at(TInds&& ... inds) {
        /* Returns specific element of a tensor view */
        size_t offset = CalculateOffsetImpl<sizeof... (TInds) - 1, ndim - 1>::calculate(0, stride_, inds...);
        return data_ptr_[offset];
    }

    template<typename ...TInds, std::enable_if_t<sizeof...(TInds) < ndim, int> = 0>
    TensorView<T, ndim - sizeof...(TInds), BroadcastPolicyTag> at(TInds&& ... inds) {
        /* Returns "sub-view" of a tensor, i.e. TensorView with the first coordinates set to inds */
        const size_t NInds = sizeof...(TInds);
        const size_t new_ndims = ndim - NInds;
        size_t offset = CalculateOffsetImpl<NInds - 1, NInds - 1>::calculate(0, stride_, inds...);
        return TensorView<T, new_ndims, BroadcastPolicyTag>(data_ptr_ + offset, shape_ + NInds, stride_ + NInds);
    }

    template<typename ...TInds, std::enable_if_t<sizeof...(TInds) == ndim, int> = 0>
    T& operator()(TInds&& ... inds) {
        return at(inds...);
    }

    template<typename ...TInds, std::enable_if_t<sizeof...(TInds) < ndim, int> = 0>
    TensorView<T, ndim - sizeof...(TInds), BroadcastPolicyTag> operator()(TInds&& ... inds) {
        return at(inds...);
    };

    bool is_contiguous() {
        size_t prod = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            if (stride_[i] != prod) {
                return false;
            }
            prod *= shape_[i];
        }
        return true;
    }

    template<class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    void assign_(TensorViewRhs rhs) {
        ElementWiseInplaceOp<Type, TensorViewRhs>::impl([&rhs](auto& a, auto& b) {
            return b;
        }, *this, rhs);
    }

    template<class... Ts>
    Type permute(Ts... ts) {
        static_assert(sizeof...(Ts) == NumDims, "Number of arguments to permute function must be NumDims");

        std::array<size_t, NumDims> permute_inds{{ts...}};
        std::array<size_t, NumDims> shape;
        std::array<size_t, NumDims> strides;

        for (int i = 0; i < NumDims; ++i) {
            shape[i] = shape_[permute_inds[i]];
            strides[i] = stride_[permute_inds[i]];
        }

        return Type(data_ptr_, shape.data(), strides.data());
    }

    template<class... Ts>
    TensorView<ValueType, sizeof...(Ts), BroadcastPolicyTag> reshape(Ts... ts) {
        TV_ASSERT(is_contiguous(), "Tensor for reshape must be contiguous")

        size_t total_size_orig = std::accumulate(shape_, shape_ + NumDims, 1, std::multiplies<>());

        std::array<int, sizeof...(Ts)> shape{{ts...}};
        std::array<size_t, sizeof...(Ts)> shape_post;

        int placeholder_dim_ind = -1;
        size_t total_size_result = 1;
        for (int i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                TV_ASSERT(placeholder_dim_ind == -1, "Only one dimension can be inferred")
                placeholder_dim_ind = i;
                continue;
            }
            total_size_result *= shape[i];
            shape_post[i] = shape[i];
        }
        if (placeholder_dim_ind != -1) {
            shape_post[placeholder_dim_ind] = total_size_orig / total_size_result;
            total_size_result = total_size_orig;
        }
        TV_ASSERT_DEBUG(total_size_orig == total_size_result, "Trying to reshape to invalid shape")
        return {data_ptr_, shape_post.data()};
    }


    ShapeType stride() const {
        return stride_;
    }

    ShapeType shape() const {
        return shape_;
    }

    size_t size(size_t dim) const {
        return shape_[dim];
    }

    T* data() {
        return data_ptr_;
    }


    ValueType max() const {
        ValueType val{};
        AllReduceImpl<ndim>::impl([](const ValueType& a, const ValueType& b) {
            return std::max(a, b);
        }, *this, val);
        return val;
    }

    template<class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    auto operator+(const TensorViewRhs& rhs) {
        return make_binary_op(std::plus<ValueType>(), *this, rhs);
    }

    friend std::ostream& operator<<<Type>(std::ostream&, const Type&);

private:
    T* data_ptr_;
    size_t shape_[ndim];
    size_t stride_[ndim];

    int deduce_maxw() const {
        size_t maxw = 1;
        AllReduceImpl<ndim>::impl([](const size_t& a, const ValueType& b) {
            auto i = print_element(b);
            return std::max(a, i.size());
        }, *this, maxw);
        return maxw;
    }
};


template<class T, size_t ndim, class U>
TensorView<T, ndim> make_view(T* data, U (&& shape)[ndim]) {
    size_t shape_[ndim];
    for (int i = 0; i < ndim; ++i) {
        shape_[i] = static_cast<size_t>(shape[i]);
    }
    return TensorView<T, ndim>(data, shape_);
}

template<class TTensorView, std::enable_if_t<is_tensor_view_v<TTensorView>, int> = 0>
std::ostream& operator<<(std::ostream& stream, const TTensorView& t) {
    const size_t ndim = TTensorView::NumDims;

    int maxw = t.deduce_maxw();
    stream << "TensorView<" << typeid(typename TTensorView::ValueType).name() << ", " << ndim << "> shape: [";
    for (int i = 0; i < ndim; ++i) {
        std::cout << t.shape_[i] << (i < ndim - 1 ? ", " : "");
    }
    stream << "], data: \n";
    TensorPrinter<ndim>::print(stream, t, 1, maxw);
    stream << std::endl;
}

} // namespace