#pragma once

#include "TensorViewFwd.h"
#include "Traits.h"

#include <iostream>
#include <iomanip>

#ifndef TENSORIO_MAX_ELEMENTS_WRAP
#define TENSORIO_MAX_ELEMENTS_WRAP 15
#endif

#ifndef TENSORIO_WRAPPER_NUM_ELEMENTS
#define TENSORIO_WRAPPER_NUM_ELEMENTS 3
#endif

namespace tensor_view {

const size_t LINE_WRAP = 80;
const size_t ELEMENTS_WRAP = TENSORIO_MAX_ELEMENTS_WRAP;
const size_t WRAPPER_NUM_ELEMENTS = TENSORIO_WRAPPER_NUM_ELEMENTS;

using std::enable_if_t;

template<int I>
struct rank : rank<I - 1> {
};
template<>
struct rank<0> {
};


template<class T>
std::string print_element_impl(const T& t, rank<0>) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

template<class T, enable_if_t<std::is_floating_point<T>::value, int> = 0>
std::string print_element_impl(const T& t, rank<1>) {
    std::stringstream ss;
    ss << std::setprecision(3);
    ss << t;
    return ss.str();
}

template<class T>
std::string print_element(const T& t) {
    return print_element_impl(t, rank<1>{});
}


inline void print_margin(std::ostream& stream, int margin) {
    for (int j = 0; j < margin; ++j) {
        stream << ' ';
    }
}

inline  void print_line_breaks(std::ostream& stream, int n) {
    while (n--) stream << '\n';
}


template<size_t N>
class TensorPrinter {
public:
    template<class SubViewType>
    static void print(std::ostream& stream, SubViewType view, int margin, int maxw) {
        stream << '[';
        for (int i = 0; i < view.size(0) - 1; ++i) {
            if (view.size(0) > ELEMENTS_WRAP && i == WRAPPER_NUM_ELEMENTS) {
                stream << "...,";
                print_line_breaks(stream, N - 1);
                print_margin(stream, margin);
                break;
            }
            TensorPrinter<N - 1>::print(stream, view.at(i), margin + 1, maxw);
            stream << ',';

            print_line_breaks(stream, N - 1);
            print_margin(stream, margin);
        }
        if (view.size(0) > ELEMENTS_WRAP) {
            for (int i = view.size(0) - WRAPPER_NUM_ELEMENTS; i < view.size(0) - 1; ++i) {
                TensorPrinter<N - 1>::print(stream, view.at(i), margin + 1, maxw);
                stream << ',';
                print_line_breaks(stream, N - 1);
                print_margin(stream, margin);
            }
        }
        TensorPrinter<N - 1>::print(stream, view.at(view.size(0) - 1), margin + 1, maxw);
        stream << ']';
    }
};

template<>
class TensorPrinter<1> {
public:
    template<class SubViewType>
    static void print(std::ostream& stream, SubViewType view, int margin, int maxw) {
        stream << '[';
        for (int i = 0; i < view.size(0) - 1; ++i) {
            if (view.size(0) > ELEMENTS_WRAP && i == WRAPPER_NUM_ELEMENTS) {
                stream << "..., ";
                break;
            }
            stream << std::setw(maxw) << print_element(view.at(i)) << ", ";
        }
        if (view.size(0) > ELEMENTS_WRAP) {
            for (int i = view.size(0) - WRAPPER_NUM_ELEMENTS; i < view.size(0) - 1; ++i) {
                stream << std::setw(maxw) << print_element(view.at(i)) << ", ";
            }
        }
        stream << std::setw(maxw) << print_element(view.at(view.size(0) - 1)) << ']';
    }
};

} // namespace