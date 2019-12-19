#pragma once

#include <algorithm>
#include <iostream>

#include "TensorViewFwd.h"
#include "Traits.h"
#include "Utils.h"

namespace tensor_view {

template<class TensorViewLhs, class TensorViewRhs>
class BroadcastTensors {
    using LhsType = TensorViewChecked<TensorViewLhs>;
    using RhsType = TensorViewChecked<TensorViewRhs>;

    static constexpr size_t ResultNdims = std::max(LhsType::NumDims, RhsType::NumDims);
public:
    using ResultType = TensorView<typename LhsType::ValueType, ResultNdims, explicit_broadcast>;

    static ResultType impl(TensorViewLhs first, TensorViewRhs second) {
        size_t shape[ResultNdims];
        size_t stride[ResultNdims];
        int k = 0;
        // pad front dimensions with "ones"
        for (int i = 0; i < ResultNdims - RhsType::NumDims; ++i) {
            stride[k] = 0;
            shape[k++] = 1;
        }
        // copy the rest dimensions
        for (int i = 0; i < RhsType::NumDims; ++i) {
            stride[k] = second.stride()[i];
            shape[k++] = second.size(i);
        }
        return ResultType(second.data(), shape);
    }
};

template<size_t N>
class ElementWiseOpImpl {
public:
    template<class F, class TensorViewLhs, class TensorViewRhs, class TensorViewDst>
    static void impl(F f, TensorViewLhs first, TensorViewRhs second, TensorViewDst dst) {
        for (int i = 0; i < dst.size(0); ++i) {
            auto sub_view_first = first.at(first.size(0) == 1 ? 0 : i);
            auto sub_view_second = second.at(second.size(0) == 1 ? 0 : i);
            auto sub_view_dst = dst.at(i);
            ElementWiseOpImpl<N - 1>::impl(f, sub_view_first, sub_view_second, sub_view_dst);
        }
    }
};


template<>
class ElementWiseOpImpl<1> {
public:
    template<class F, class TensorViewLhs, class TensorViewRhs, class TensorViewDst>
    static void impl(F f, TensorViewLhs first, TensorViewRhs second, TensorViewDst dst) {
        for (int i = 0; i < dst.size(0); ++i) {
            const typename TensorViewLhs::ValueType& elem_first = first.at(first.size(0) == 1 ? 0 : i);
            const typename TensorViewRhs::ValueType& elem_second = second.at(second.size(0) == 1 ? 0 : i);
            typename TensorViewDst::ValueType& elem_dst = dst.at(i);
            elem_dst = f(elem_first, elem_second);
        }
    }
};


template<class TensorViewLhs, class TensorViewRhs>
class ElementWiseInplaceOp {
public:
    using LhsType = TensorViewChecked<TensorViewLhs>;
    using RhsType = TensorViewChecked<TensorViewRhs>;


    template<class F>
    static void impl(F f, TensorViewLhs first, TensorViewRhs second) {
        static_assert(LhsType::NumDims >= RhsType::NumDims, "Lhs tensor ndim must be greater or equal than rhs' one");
        TV_ASSERT(check_shapes(first, second), "Shapes of input tensors are not compatible")
        auto second_broadcasted = BroadcastTensors<TensorViewLhs, TensorViewRhs>::impl(first, second);
        ElementWiseOpImpl<LhsType::NumDims>::impl(f, first, second_broadcasted, first);
    }
};

template<class TensorViewLhs, class TensorViewRhs, class TFunc>
class ElementWiseOperation {
public:
    using LhsType = TensorViewChecked<TensorViewLhs>;
    using RhsType = TensorViewChecked<TensorViewRhs>;

    TensorViewLhs lhs_;
    TensorViewRhs rhs_;
    TFunc func_;

    template<class TensorViewDst>
    void apply(TensorViewDst& dst) const {
        static_assert(TensorViewDst::NumDims == TensorViewLhs::NumDims, "Incorrect number of dims of dst tensor");
        TV_ASSERT(check_shapes(dst, lhs_), "Incorrect shape of destination tensor")
        TV_ASSERT(check_shapes(dst, rhs_), "Incorrect shape of destination tensor")

        ElementWiseOpImpl<TensorViewDst::NumDims>::impl(func_, lhs_, rhs_, dst);
    }

    ElementWiseOperation(const TensorViewLhs& lhs, const TensorViewRhs& rhs, TFunc f) :
            lhs_(lhs),
            rhs_(rhs),
            func_(f) {}
};

template<class F, class TensorViewLhs, class TensorViewRhs>
ElementWiseOperation<
        typename BroadcastTensors<TensorViewLhs, TensorViewRhs>::ResultType,
        typename BroadcastTensors<TensorViewRhs, TensorViewLhs>::ResultType,
        F>
make_binary_op(const F& f, const TensorViewLhs& first, const TensorViewRhs& second) {
    TV_ASSERT(check_shapes(first, second), "Shapes of input tensors are not compatible")
    auto first_broadcasted = BroadcastTensors<TensorViewLhs, TensorViewRhs>::impl(first, second);
    auto second_broadcasted = BroadcastTensors<TensorViewRhs, TensorViewLhs>::impl(second, first);

    static_assert(decltype(first_broadcasted)::NumDims == decltype(second_broadcasted)::NumDims,
                  "Incorrect number of dims after broadcast");
    return {first_broadcasted, second_broadcasted, f};
}


template<size_t N>
class AllReduceImpl {
public:
    template<class F, class TTensorViewType, class T>
    static void impl(F f, TTensorViewType view, T& t) {
        for (int i = 0; i < view.size(0); ++i) {
            auto sub_view = view.at(i);
            AllReduceImpl<N - 1>::impl(f, sub_view, t);
        }
    }
};

template<>
class AllReduceImpl<1> {
public:
    template<class F, class TTensorViewType, class T>
    static void impl(F f, TTensorViewType view, T& t) {
        for (int i = 0; i < view.size(0); ++i) {
            typename TTensorViewType::ValueType at = view.at(i);
            t = f(t, at);
        }
    }
};


namespace detail {
template<class T>
struct is_binary_op : std::false_type {
};

template<class T1, class T2, class T3>
struct is_binary_op<ElementWiseOperation<T1, T2, T3>> : std::true_type {
};
}

template<class T>
struct is_binary_op {
    static constexpr bool const value = detail::is_binary_op<std::decay_t<T>>::value;
};

template<class T>
constexpr bool is_binary_op_v = is_binary_op<T>::value;

} // namespace