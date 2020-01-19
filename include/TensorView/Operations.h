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
        return ResultType(second.data(), shape, stride);
    }
};

template<size_t N>
class ElementWiseOpImpl {
public:
    template<class F, class TensorViewLhs, class TensorViewRhs, class TensorViewDst>
    static void impl(F f, TensorViewLhs first, TensorViewRhs second, TensorViewDst dst, size_t trivial_dim) {
        if (N == trivial_dim) {
            std::transform(first.data(), first.data() + first.num_elements(), second.data(), dst.data(), f);
            return;
        }
        for (int i = 0; i < dst.size(0); ++i) {
            auto sub_view_first = first.at(first.size(0) == 1 ? 0 : i);
            auto sub_view_second = second.at(second.size(0) == 1 ? 0 : i);
            auto sub_view_dst = dst.at(i);
            ElementWiseOpImpl<N - 1>::impl(f, sub_view_first, sub_view_second, sub_view_dst, trivial_dim);
        }
    }
};


template<>
class ElementWiseOpImpl<1> {
public:
    template<class F, class TensorViewLhs, class TensorViewRhs, class TensorViewDst>
    static void impl(F&& f, TensorViewLhs first, TensorViewRhs second, TensorViewDst dst, size_t trivial_dim) {
        if (trivial_dim == 1) {
            std::transform(first.data(), first.data() + first.num_elements(), second.data(), dst.data(), f);
            return;
        }
        for (int i = 0; i < dst.size(0); ++i) {
            const typename TensorViewLhs::ValueType& elem_first = first.at(first.size(0) == 1 ? 0 : i);
            const typename TensorViewRhs::ValueType& elem_second = second.at(second.size(0) == 1 ? 0 : i);
            typename TensorViewDst::ValueType& elem_dst = dst.at(i);
            elem_dst = f(elem_first, elem_second);
        }
    }
};

template<size_t N>
class UnaryOpImpl {
public:
    template<class F, class TensorViewSrc, class TensorViewDst>
    static void impl(F f, TensorViewSrc src, TensorViewDst dst, size_t trivial_dim) {
        if (trivial_dim == N) {
            std::transform(src.data(), src.data() + src.num_elements(), dst.data(), f);
            return;
        }
        for (int i = 0; i < dst.size(0); ++i) {
            auto sub_view_src = src.at(src.size(0) == 1 ? 0 : i);
            auto sub_view_dst = dst.at(i);
            UnaryOpImpl<N - 1>::impl(f, sub_view_src, sub_view_dst, trivial_dim);
        }
    }
};


template<>
class UnaryOpImpl<1> {
public:
    template<class F, class TensorViewSrc, class TensorViewDst>
    static void impl(F f, TensorViewSrc src, TensorViewDst dst, size_t trivial_dim) {
        if (trivial_dim == 1) {
            std::transform(src.data(), src.data() + src.num_elements(), dst.data(), f);
            return;
        }
        for (int i = 0; i < dst.size(0); ++i) {
            const typename TensorViewSrc::ValueType& elem_src = src.at(src.size(0) == 1 ? 0 : i);
            typename TensorViewDst::ValueType& elem_dst = dst.at(i);
            elem_dst = f(elem_src);
        }
    }
};


template<class TensorViewLhs, class TensorViewRhs>
class ElementWiseInplaceOp {
public:
    using LhsType = TensorViewChecked<TensorViewLhs>;
    using RhsType = TensorViewChecked<TensorViewRhs>;


    template<class F>
    static void impl(F&& f, TensorViewLhs first, TensorViewRhs second) {
        static_assert(LhsType::NumDims >= RhsType::NumDims, "Lhs tensor ndim must be greater or equal than rhs' one");
        TV_ASSERT(check_shapes(first, second), "Shapes of input tensors are not compatible")
        auto second_broadcasted = BroadcastTensors<TensorViewLhs, TensorViewRhs>::impl(first, second);
        size_t trivial_dim = find_first_trivial_dim(first, second_broadcasted);
        ElementWiseOpImpl<LhsType::NumDims>::impl(std::forward<F>(f), first, second_broadcasted, first, trivial_dim);
    }
};

template<class TTensorView>
class UnaryInplaceOp {
public:
    template<class F>
    static void impl(F f, TTensorView first) {
        size_t trivial_dim = find_first_trivial_dim(first, first);
        UnaryOpImpl<TTensorView::NumDims>::impl(f, first, first, trivial_dim);
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
        size_t trivial_dim = std::min(find_first_trivial_dim(lhs_, rhs_), find_first_trivial_dim(lhs_, dst));
        ElementWiseOpImpl<TensorViewDst::NumDims>::impl(func_, lhs_, rhs_, dst, trivial_dim);
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
make_reduce_operation(const F& f, const TensorViewLhs& first, const TensorViewRhs& second) {
    TV_ASSERT(check_shapes(first, second), "Shapes of input tensors are not compatible")
    auto first_broadcasted = BroadcastTensors<TensorViewLhs, TensorViewRhs>::impl(first, second);
    auto second_broadcasted = BroadcastTensors<TensorViewRhs, TensorViewLhs>::impl(second, first);

    static_assert(decltype(first_broadcasted)::NumDims == decltype(second_broadcasted)::NumDims,
                  "Incorrect number of dims after broadcast");
    return {first_broadcasted, second_broadcasted, f};
}

template<class TensorViewSrc, class TFunc>
class UnaryOperation {
public:
    using SrcType = TensorViewChecked<TensorViewSrc>;

    TensorViewSrc src_;
    TFunc func_;

    template<class TensorViewDst>
    void apply(TensorViewDst& dst) const {
        static_assert(TensorViewDst::NumDims == TensorViewSrc::NumDims, "Incorrect number of dims of dst tensor");
        TV_ASSERT(check_shapes(dst, src_), "Incorrect shape of destination tensor")
        size_t trivial_dim = find_first_trivial_dim(src_, dst);
        UnaryOpImpl<TensorViewDst::NumDims>::impl(func_, src_, dst, trivial_dim);
    }

    UnaryOperation(const TensorViewSrc& src, TFunc f) :
            src_(src),
            func_(f) {}
};


template<class F, class TensorViewSrc>
UnaryOperation<TensorViewSrc, F>
make_unary_op(F&& f, const TensorViewSrc& first) {
    return {first, std::forward<F>(f)};
}

template<class TensorViewLhs, class TInitial, class TFunc>
class ReduceOperation {
public:
    using SrcType = TensorViewChecked<TensorViewLhs>;

    TensorViewLhs src_;
    TFunc func_;
    size_t axis_;
    TInitial initial_;

    template<class TensorViewDst>
    void apply(TensorViewDst& dst) const {
        static_assert(TensorViewDst::NumDims + 1 == TensorViewLhs::NumDims, "Incorrect number of dims of dst tensor");
        auto initial = static_cast<typename TensorViewDst::ValueType>(initial_);
        src_.reduce(func_, dst, axis_, initial);
    }

    ReduceOperation(const TensorViewLhs& lhs, size_t axis, TFunc f, TInitial initial) :
            src_(lhs),
            axis_(axis),
            func_(f),
            initial_(initial) {}
};

template<class F, class TensorViewSrc, class TInitial>
ReduceOperation<TensorViewSrc, TInitial, F>
make_reduce_operation(F&& f, const TensorViewSrc& src, size_t axis, TInitial initial_value) {
    return {src, axis, std::forward<F>(f), initial_value};
}


template<size_t N>
class AllReduceImpl {
public:
    template<class F, class TTensorViewType, class T>
    static void impl(F&& f, TTensorViewType&& view, T& t, size_t trivial_dim, T initial_value) {
        if (trivial_dim == N) {
            t = std::accumulate(view.data(), view.data() + view.num_elements(), initial_value, f);
            return;
        }
        for (int i = 0; i < view.size(0); ++i) {
            auto sub_view = view.at(i);
            AllReduceImpl<N - 1>::impl(std::forward<F>(f), sub_view, t, trivial_dim, initial_value);
        }
    }
};

template<>
class AllReduceImpl<1> {
public:
    template<class F, class TTensorView, class T>
    static void impl(F&& f, TTensorView view, T& t, size_t trivial_dim, T initial_value) {
        if (trivial_dim == 1) {
            t = std::accumulate(view.data(), view.data() + view.num_elements(), initial_value, f);
            return;
        }
        for (int i = 0; i < view.size(0); ++i) {
            typename TTensorView::ValueType at = view.at(i);
            t = f(t, at);
        }
    }
};

template<size_t N, size_t M>
class ReduceDim {
public:
    template<class F, class TTensorViewSrc, class TTensorViewDst>
    static void impl(F&& f, const TTensorViewSrc& src, TTensorViewDst dst, size_t reduce_dim) {
        for (int i = 0; i < src.size(0); ++i) {
            if (reduce_dim == N) {
                ReduceDim<N - 1, M>::impl(f, src.at(i), dst, reduce_dim);
            } else {
                ReduceDim<N - 1, M - 1>::impl(f, src.at(i), dst.at(i), reduce_dim);
            }
        }
    }
};

template<size_t N>
class ReduceDim<N, N> {
public:
    template<class F, class TTensorViewSrc, class TTensorViewDst>
    static void impl(F&& f, const TTensorViewSrc& src, TTensorViewDst dst, size_t reduce_dim) {
        for (int i = 0; i < src.size(0); ++i) {
            ReduceDim<N - 1, N - 1>::impl(f, src.at(i), dst.at(i), reduce_dim);
        }
    }
};

template<>
class ReduceDim<1, 0> {
public:
    template<class F, class TTensorViewSrc, class TResult>
    static void impl(F&& f, const TTensorViewSrc& src, TResult& dst, size_t reduce_dim) {
        for (int i = 0; i < src.size(0); ++i) {
            dst = f(dst, src.at(i));
        }
    }
};

template<>
class ReduceDim<1, 1> {
public:
    template<class F, class TTensorViewSrc, class TTensorViewDst>
    static void impl(F&& f, const TTensorViewSrc& src, TTensorViewDst dst, size_t reduce_dim) {
        for (int i = 0; i < src.size(0); ++i) {
            dst.at(i) = f(dst.at(i), src.at(i));
        }
    }
};


namespace detail {
template<class T>
struct is_operation : std::false_type {
};

template<class T1, class T2, class T3>
struct is_operation<ElementWiseOperation<T1, T2, T3>> : std::true_type {
};

template<class T1, class T2, class T3>
struct is_operation<ReduceOperation<T1, T2, T3>> : std::true_type {
};

template<class T1, class T2>
struct is_operation<UnaryOperation<T1, T2>> : std::true_type {
};
}

template<class T>
struct is_operation {
    static constexpr bool const value = detail::is_operation<std::decay_t<T>>::value;
};

template<class T>
constexpr bool is_operation_v = is_operation<T>::value;

} // namespace