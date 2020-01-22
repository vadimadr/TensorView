#pragma once

#include <vector>
#include <array>
#include <TensorView/TensorView.h>


namespace tensor_view {

template<class T, class ...Ts>
size_t product(T t, Ts ...ts) {
    std::array<T, sizeof...(ts) + 1> data{t, ts...};

    size_t res = 1;
    for (auto& t: data) {
        res *= t;
    }
    return res;
}

size_t product(const size_t* dims, size_t nd) {
    size_t res = 1;
    for (int i = 0; i < nd; ++i) {
        res *= dims[i];
    }
    return res;
}

template<class T, size_t ndim, class BroadcastPolicy>
class Tensor : public TensorView<T, ndim, BroadcastPolicy> {
public:
    template<typename ...TDims, std::enable_if_t<sizeof...(TDims) == ndim, int> = 0>
    Tensor(TDims... dims) : data_(product(dims...)) {
        std::array<size_t, ndim> a_dims{{static_cast<size_t>(dims)Cm    ...}};
        std::copy(a_dims.begin(), a_dims.end(), this->shape_);
        calculate_strides(this->shape_, this->stride_, ndim);
    }

    Tensor(const size_t* dims) :
            data_(product(dims, ndim)) {
        this->data_ptr_ = data_.data();
        std::copy(dims, dims + ndim, this->shape_);
        calculate_strides(this->shape_, this->stride_, ndim);
    }

private:
    std::vector<T> data_;

};

} // namespace tensor_view