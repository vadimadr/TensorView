#pragma once

#include <type_traits>
#include <cstddef>

#include "TensorViewFwd.h"

namespace tensor_view {
namespace detail {

template<class T>
struct is_tensor_view_impl : std::false_type {
};
template<class T, size_t nd, class Tag>
struct is_tensor_view_impl<TensorView<T, nd, Tag>> : std::true_type {
};

}

template<class T>
struct is_tensor_view {
    static constexpr bool const value = detail::is_tensor_view_impl<std::decay_t<T> >::value;
};

template<class T>
constexpr bool is_tensor_view_v = is_tensor_view<T>::value;


template<class TInput>
struct TensorViewChecked {
    static_assert(is_tensor_view_v<TInput>, "Input type must be an instance of TensorView");
    using InputType = std::decay_t<TInput>;
    using ValueType = typename InputType::ValueType;
    using BroadcastPolicyTag = typename InputType::BroadcastPolicyTag;
    static constexpr size_t NumDims = InputType::NumDims;
};

} // namespace