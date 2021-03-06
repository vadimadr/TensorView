#pragma once

#include <cstddef>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <functional>

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

    TensorView() : data_ptr_(nullptr) {}

    TensorView(T* data_ptr, const size_t* shape) :
            data_ptr_(data_ptr) {
        std::copy(shape, shape + ndim, shape_);
        calculate_strides(shape_, stride_, ndim);
    }

    TensorView(T* data_ptr, const size_t* shape, const size_t* stride) :
            data_ptr_(data_ptr) {
        std::copy(shape, shape + ndim, shape_);
        std::copy(stride, stride + ndim, stride_);
    }

    template<class TDeferredOperation, enable_if_t<is_operation_v<TDeferredOperation>, int> = 0>
    Type& operator=(const TDeferredOperation& op) {
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
    const T& at(TInds&& ... inds) const {
        /* Returns specific element of a tensor view */
        size_t offset = CalculateOffsetImpl<sizeof... (TInds) - 1, ndim - 1>::calculate(0, stride_, inds...);
        return data_ptr_[offset];
    }

    template<typename ...TInds, std::enable_if_t<sizeof...(TInds) < ndim, int> = 0>
    const TensorView<T, ndim - sizeof...(TInds), BroadcastPolicyTag> at(TInds&& ... inds) const {
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

    template<typename ...TInds, std::enable_if_t<sizeof...(TInds) == ndim, int> = 0>
    const T& operator()(TInds&& ... inds) const {
        return at(inds...);
    }

    template<typename ...TInds, std::enable_if_t<sizeof...(TInds) < ndim, int> = 0>
    const TensorView<T, ndim - sizeof...(TInds), BroadcastPolicyTag> operator()(TInds&& ... inds) const {
        return at(inds...);
    };

    bool is_contiguous() const {
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
    void assign_(const TensorViewRhs& rhs) {
        ElementWiseInplaceOp<Type, TensorViewRhs>::impl([&rhs](auto& a, auto& b) {
            return b;
        }, *this, rhs);
    }

    void assign_(ValueType value) {
        map_([value](const ValueType& val) { return value; });
    }

    template<class... Ts>
    Type permute(Ts... ts) const {
        static_assert(sizeof...(Ts) == NumDims, "Number of arguments to permute function must be NumDims");

        std::array<size_t, NumDims> permute_inds{{static_cast<size_t>(ts)...}};
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

        size_t total_size_orig = num_elements();

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

    TensorView<ValueType, NumDims + 1, BroadcastPolicy> unsqueeze(size_t dim = NumDims - 1) {
        TV_ASSERT(is_contiguous(), "Tensor for unsqueeze must be contiguous")
        size_t new_dims[NumDims + 1];
        for (int i = 0; i < dim; ++i) {
            new_dims[i] = size(i);
        }
        for (int j = dim + 1; j < NumDims + 1; ++j) {
            new_dims[j] = size(j - 1);
        }
        new_dims[dim] = 1;
        return {data_ptr_, new_dims};
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

    const T* data() const {
        return data_ptr_;
    }

    bool empty() const {
        return data_ptr_ == nullptr;
    }

    size_t num_elements() const {
        return std::accumulate(shape_, shape_ + NumDims, 1, std::multiplies<>());
    }

    ValueType max() const {
        const ValueType& (& max_fn)(const ValueType&, const ValueType&) = std::max<ValueType>;
        return reduce(max_fn);
    }

    template<class TensorViewDst, enable_if_t<is_tensor_view_v<TensorViewDst>, int> = 0>
    auto max(TensorViewDst& dst, size_t axis) const {
        const ValueType& (& max_fn)(const ValueType&, const ValueType&) = std::max<ValueType>;
        return reduce(max_fn, dst, axis, std::numeric_limits<typename TensorViewDst::ValueType>::min());
    }

    ValueType sum() const {
        return reduce(std::plus<ValueType>());
    }

    template<class TensorViewDst, enable_if_t<is_tensor_view_v<TensorViewDst>, int> = 0>
    auto sum(TensorViewDst& dst, size_t axis) {
        return reduce(std::plus<typename TensorViewDst::ValueType>(), dst, axis, 0);
    }


    template<class Func>
    Type& map_(Func&& f) {
        UnaryInplaceOp<Type>::impl(std::forward<Func>(f), *this);
        return *this;
    }

    template<class Func>
    auto map(Func&& f) const {
        return make_unary_op(std::forward<Func>(f), *this);
    }

    template<class Func, class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    Type& map_(Func&& f, const TensorViewRhs& rhs) {
        ElementWiseInplaceOp<Type, TensorViewRhs>::impl(std::forward<Func>(f), *this, rhs);
        return *this;
    }

    template<class Func, class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    auto map(Func&& f, const TensorViewRhs& rhs) const {
        return make_reduce_operation(std::forward<Func>(f), *this, rhs);
    }


    template<class Func, class TResult = ValueType>
    TResult reduce(Func&& f, TResult initial_value = TResult{}) const {
        size_t trivial_dim = find_first_trivial_dim(*this);
        TResult result = initial_value;
        AllReduceImpl<ndim>::impl(std::forward<Func>(f), *this, result, trivial_dim, initial_value);
        return result;
    }

    template<class Func, class TensorViewDst, enable_if_t<is_tensor_view_v<TensorViewDst>, int> = 0>
    void reduce(Func&& f,
                TensorViewDst dst,
                size_t axis,
                typename TensorViewDst::ValueType initial_value = typename TensorViewDst::ValueType{}) const {
        static_assert(NumDims == TensorViewDst::NumDims + 1, "Incorrect number of dims of destination tensor");
        // todo: check shapes (all but `axis` must be the same)
        dst.assign_(initial_value);
        ReduceDim<NumDims, NumDims - 1>::impl(std::forward<Func>(f), *this, dst, NumDims - axis);
    }

    template<class Func, class TInitial = ValueType>
    auto reduce(Func&& f, size_t axis, TInitial initial_value) const {
        return make_reduce_operation(std::forward<Func>(f), *this, axis, initial_value);
    }


    template<class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    auto operator+(const TensorViewRhs& rhs) {
        return map(std::plus<ValueType>(), rhs);
    }

    template<class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    Type& operator+=(const TensorViewRhs& rhs) {
        return map_(std::plus<ValueType>(), rhs);
    }

    template<class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    Type& operator/=(const TensorViewRhs& rhs) {
        return map_(std::divides<ValueType>(), rhs);
    }

    template<class TensorViewRhs, enable_if_t<is_tensor_view_v<TensorViewRhs>, int> = 0>
    Type& operator-=(const TensorViewRhs& rhs) {
        return map_(std::minus<ValueType>(), rhs);
    }

    Type& operator*=(ValueType c) {
        ValueType c_cast = static_cast<ValueType>(c);
        using namespace std::placeholders;
        return map_(std::bind(std::multiplies<ValueType>(), c_cast, _1));
    }

    Type& operator/=(ValueType c) {
        ValueType c_cast = static_cast<ValueType>(c);
        using namespace std::placeholders;
        return map_(std::bind(std::divides<ValueType>(), c_cast, _1));
    }

    auto operator*(ValueType c) {
        ValueType c_cast = static_cast<ValueType>(c);
        using namespace std::placeholders;
        return map(std::bind(std::multiplies<ValueType>(), c_cast, _1));
    }

    friend std::ostream& operator<<<Type>(std::ostream&, const Type&);

protected:
    T* data_ptr_;
    size_t shape_[ndim];
    size_t stride_[ndim];

    int deduce_maxw() const {
        return reduce([](const size_t& a, const ValueType& b) {
            auto i = print_element(b);
            return std::max(a, i.size());
        }, 1);
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

/*template<class T, size_t ndim, class U>
const TensorView<T, ndim> make_view(const T* data, U (&& shape)[ndim]) {
    size_t shape_[ndim];
    for (int i = 0; i < ndim; ++i) {
        shape_[i] = static_cast<size_t>(shape[i]);
    }
    return TensorView<T, ndim>(data, shape_);
}*/

template<class TTensorView, std::enable_if_t<is_tensor_view_v<TTensorView>, int>>
std::ostream& operator<<(std::ostream& stream, const TTensorView& t) {
    const size_t ndim = TTensorView::NumDims;

    int maxw = t.deduce_maxw();
    stream << "TensorView<" << typeid(typename TTensorView::ValueType).name() << ", " << ndim << "> shape: [";
    for (int i = 0; i < ndim; ++i) {
        stream << t.shape_[i] << (i < ndim - 1 ? ", " : "");
    }
    stream << "], data:\n";
    TensorPrinter<ndim>::print(stream, t, 1, maxw);
    stream << '\n';
}

template<class T, size_t ndim, class BroadcastPolicy>
constexpr const size_t TensorView<T, ndim, BroadcastPolicy>::NumDims;

} // namespace