#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <array>
#include <initializer_list>
#include <type_traits>


#include "TensorView/TensorView.h"


using std::enable_if_t;
using namespace tensor_view;

template<bool cond, class T = int>
using enable_if_ = typename std::enable_if<cond, T>::type;

template<class T, enable_if_<std::is_floating_point<T>::value> = 0>
void check_is_tensor_view(T t) {
    std::cout << "float" << std::endl;
}

template<class T, enable_if_<std::is_integral<T>::value> = 0>
void check_is_tensor_view(T t) {
    std::cout << "int" << std::endl;
}

template<class T, enable_if_<is_tensor_view<T>::value> = 0>
void check_is_tensor_view(T t) {
    using Type = TensorViewChecked<T>;

    std::cout << "tensor view" << std::endl;
    std::cout << "Num dims: " << Type::NumDims << std::endl;
}


int main() {
    std::vector<float> v(2 * 3 * 4);
    std::vector<float> u(2 * 3 * 4);
    std::iota(v.begin(), v.end(), 0);

    const int x[3] = {1, 2, 3};
    int y[3];

    for (int i = 0; i < 3; ++i) {
        y[i] = x[i];
    }

    auto view = make_view(v.data(), {6, 4});
    auto u_view = make_view(u.data(), {6, 4});


    std::cout << view << std::endl;
    std::cout << u_view << std::endl;
    u_view = view + view;
    std::cout << u_view << std::endl;


    return 0;
}