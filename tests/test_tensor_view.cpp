#include <numeric>
#include <vector>


#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "TensorView/TensorView.h"
#include "TensorView/Tensor.h"


template<class TTensorView>
std::vector<size_t> get_size(const TTensorView& view) {
    auto ndim = TTensorView::NumDims;
    auto shape = view.shape();
    return std::vector<size_t>(shape, shape + ndim);
}

template<class TTensorView>
std::vector<size_t> get_stride(const TTensorView& view) {
    auto ndim = TTensorView::NumDims;
    auto strides = view.stride();
    return std::vector<size_t>(strides, strides + ndim);
}

namespace {

using namespace tensor_view;
using ::testing::ElementsAreArray;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::ElementsAre;
using ::testing::StrEq;

class Creation : public testing::Test {
protected:
    virtual void SetUp() {
        data_ = std::vector<float>(12);
        std::iota(data_.begin(), data_.end(), 0);
    }

    std::vector<float> data_;
};


TEST_F(Creation, default_constructor) {
    TensorView<float, 2> view;

    EXPECT_THAT(view.data(), Eq(nullptr));
    EXPECT_THAT(view.empty(), Eq(true));
}

TEST_F(Creation, assignment) {
    TensorView<float, 2> view;
    view = make_view(data_.data(), {3, 4});

    EXPECT_EQ(view(1, 1), 5);
    EXPECT_THAT(view.empty(), Eq(false));
    EXPECT_THAT(get_size(view), ElementsAre(3, 4));
    EXPECT_THAT(get_stride(view), ElementsAre(4, 1));
}

TEST_F(Creation, make_view_simple) {
    auto view = make_view(data_.data(), {3, 4});

    EXPECT_EQ(view(1, 1), 5);
    EXPECT_THAT(view.empty(), Eq(false));
    EXPECT_THAT(get_size(view), ElementsAre(3, 4));
    EXPECT_THAT(get_stride(view), ElementsAre(4, 1));
}

TEST_F(Creation, make_view_1d) {
    auto view = make_view(data_.data(), {12});

    EXPECT_EQ(view(5), 5);
    EXPECT_THAT(get_size(view), ElementsAre(12));
    EXPECT_THAT(get_stride(view), ElementsAre(1));
}

TEST_F(Creation, make_view_3d) {
    auto view = make_view(data_.data(), {3, 2, 2});

    EXPECT_EQ(view(1, 1, 1), 7);
    EXPECT_THAT(get_size(view), ElementsAre(3, 2, 2));
    EXPECT_THAT(get_stride(view), ElementsAre(4, 2, 1));
}

TEST_F(Creation, constructor) {
    const size_t shape[] = {3, 4};
    TensorView<float, 2> view(data_.data(), shape);

    EXPECT_EQ(view(1, 1), 5);
    EXPECT_THAT(get_size(view), ElementsAre(3, 4));
    EXPECT_THAT(get_stride(view), ElementsAre(4, 1));
}

TEST_F(Creation, constructor_with_stride) {
    const size_t shape[] = {3, 4};
    const size_t stride[] = {4, 1};
    TensorView<float, 2> view(data_.data(), shape, stride);

    EXPECT_EQ(view(1, 1), 5);
    EXPECT_THAT(get_size(view), ElementsAre(3, 4));
    EXPECT_THAT(get_stride(view), ElementsAre(4, 1));
}

TEST_F(Creation, non_default_stride) {
    const size_t shape[] = {3, 2};
    const size_t stride[] = {4, 2};
    TensorView<float, 2> view(data_.data(), shape, stride);

    EXPECT_EQ(view(1, 1), 6);
    EXPECT_THAT(get_size(view), ElementsAre(3, 2));
    EXPECT_THAT(get_stride(view), ElementsAre(4, 2));
}


class BasicOperations : public Creation {
protected:
    void SetUp() override {
        Creation::SetUp();
        view = make_view(data_.data(), {3, 2, 2});
    }

    TensorView<float, 3> view;
};

TEST_F(BasicOperations, indexing) {
    EXPECT_THAT(view.at(1, 1, 1), Eq(7));
    EXPECT_THAT(view(1, 1, 1), Eq(7));
}

TEST_F(BasicOperations, index_subview) {
    auto subview1 = view.at(1);
    auto subview1_1 = subview1.at(1);
    auto subview2 = view.at(1, 1);

    EXPECT_THAT(get_size(subview1), ElementsAre(2, 2));
    EXPECT_THAT(subview1_1.at(1), Eq(7));
    EXPECT_THAT(get_size(subview1_1), ElementsAre(2));
    EXPECT_THAT(get_size(subview2), ElementsAre(2));
    EXPECT_THAT(subview2.at(1), Eq(7));
}

TEST_F(BasicOperations, assignment) {
    view.at(1, 1, 1) = 42;
    EXPECT_THAT(view.at(1, 1, 1), Eq(42));
}

TEST_F(BasicOperations, stream_output) {
    std::stringstream ss;
    ss << view;
    std::string expected =
            R"""(TensorView<f, 3> shape: [3, 2, 2], data:
[[[ 0,  1],
  [ 2,  3]],

 [[ 4,  5],
  [ 6,  7]],

 [[ 8,  9],
  [10, 11]]]
)""";

    EXPECT_THAT(ss.str(), StrEq(expected));
}


TEST_F(BasicOperations, permute) {
    std::vector<float> v_result(12);
    auto view_result = make_view(v_result.data(), {2, 2, 3});
    auto view_permute = view.permute(1, 2, 0);
    view_result.assign_(view_permute);


    EXPECT_THAT(view_permute(1, 1, 1), Eq(7));
    EXPECT_THAT(view_result(1, 1, 1), Eq(7));
    EXPECT_THAT(view_permute.is_contiguous(), Eq(false));
    EXPECT_THAT(view_result.is_contiguous(), Eq(true));
    std::vector<float> expected = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    EXPECT_THAT(v_result, ElementsAreArray(expected));
}

TEST_F(BasicOperations, reverse_permute) {
    auto view_permuted = view.permute(1, 2, 0);
    auto view_double_permuted = view_permuted.permute(2, 0, 1);

    EXPECT_THAT(view_double_permuted.is_contiguous(), Eq(true));
    EXPECT_THAT(get_size(view_double_permuted), ElementsAre(3, 2, 2));
}

TEST_F(BasicOperations, reshpae) {
    std::vector<float> v_result(12);
    auto view_result = make_view(v_result.data(), {4, 3});
    view_result.assign_(view.reshape(4, 3));

    EXPECT_THAT(v_result, ElementsAreArray(data_));
    EXPECT_THAT(get_size(view.reshape(4, 3)), ElementsAre(4, 3));
    EXPECT_THAT(view.reshape(4, 3).is_contiguous(), Eq(true));
}

TEST_F(BasicOperations, max) {
    auto max = view.max();

    EXPECT_THAT(max, Eq(11));
}

TEST_F(BasicOperations, max_subview) {
    auto max = view(0).max();

    EXPECT_THAT(max, Eq(3));
}

TEST_F(BasicOperations, max_permute_subview) {
    auto max = view.permute(1, 2, 0).at(0).max();

    EXPECT_THAT(max, Eq(9));
}

class ModifyingData : public Creation {
protected:

protected:
    virtual void SetUp() {
        Creation::SetUp();
        data2_ = std::vector<float>(12);
        std::iota(data2_.begin(), data2_.end(), 10);
        view = make_view(data_.data(), {3, 2, 2});
        view2 = make_view(data2_.data(), {3, 2, 2});
    }

    TensorView<float, 3> view, view2;
    std::vector<float> data2_;
};


TEST_F(ModifyingData, assign) {
    view.assign_(view2);

    EXPECT_THAT(data_, ElementsAreArray(data2_));
    EXPECT_THAT(get_size(view), ElementsAre(3, 2, 2));
}

TEST_F(ModifyingData, add) {
    view = view + view2;

    std::vector<float> expected = {10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    EXPECT_THAT(view.data(), Eq(data_.data()));
    EXPECT_THAT(data_, ElementsAreArray(expected));
}

TEST_F(ModifyingData, add_inplace) {
    view += view2;

    std::vector<float> expected = {10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    EXPECT_THAT(view.data(), Eq(data_.data()));
    EXPECT_THAT(data_, ElementsAreArray(expected));
}

TEST_F(ModifyingData, add_subview) {
    view(0) = view(1) + view2(2);


    std::vector<float> expected = {22, 24, 26, 28, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_THAT(view.data(), Eq(data_.data()));
    EXPECT_THAT(data_, ElementsAreArray(expected));
}


TEST_F(ModifyingData, add_broadcasted) {
    view = view + view2(2);

    std::vector<float> expected = {18, 20, 22, 24, 22, 24, 26, 28, 26, 28, 30, 32};
    EXPECT_THAT(view.data(), Eq(data_.data()));
    EXPECT_THAT(data_, ElementsAreArray(expected));
}


TEST_F(ModifyingData, add_permuted) {
    std::vector<float> data_result(12);
    auto view_result = make_view(data_result.data(), {2, 2, 3});
    auto view_p = view.permute(2, 1, 0);
    auto view2_p = view2.permute(1, 2, 0);

    view_result = view_p + view2_p;

    std::vector<float> expected = {10, 18, 26, 13, 21, 29, 13, 21, 29, 16, 24, 32};
    EXPECT_THAT(view_result.data(), Eq(data_result.data()));
    EXPECT_THAT(data_result, ElementsAreArray(expected));
}

TEST_F(ModifyingData, inplace_mul) {
    view *= 2;

    std::vector<float> expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    EXPECT_THAT(data_, ElementsAreArray(expected));
}

TEST_F(ModifyingData, inplace_mul_permuted) {
    auto view_permuted = view.permute(2, 1, 0);
    view_permuted *= 2;

    std::vector<float> expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    EXPECT_THAT(data_, ElementsAreArray(expected));
}

TEST_F(ModifyingData, inplace_mul_permuted2) {
    auto view_permuted = view.permute(1, 0, 2);
    view_permuted *= 2;

    std::vector<float> expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    EXPECT_THAT(data_, ElementsAreArray(expected));
}

TEST_F(ModifyingData, mul_by_const_permuted) {
    std::vector<float> data_result(12);
    auto view_result_permuted = make_view(data_result.data(), {2, 2, 3});

    view_result_permuted = view.permute(2, 1, 0) * 2;
    view2.assign_(view_result_permuted.permute(2, 1, 0));

    std::vector<float> expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    EXPECT_THAT(data2_, ElementsAreArray(expected));
}

class ReduceOperation : public BasicOperations {
};


TEST_F(ReduceOperation, all_reduce_sum) {
    auto sum = view.reduce([&](auto x, auto y) {
        return x + y;
    });

    EXPECT_THAT(sum, Eq(66));
}

TEST_F(ReduceOperation, all_reduce_sum_transposed) {
    auto view_transposed = view.permute(2, 1, 0);
    auto sum = view_transposed.reduce([&](auto x, auto y) {
        return x + y;
    });

    EXPECT_THAT(sum, Eq(66));
}

TEST_F(ReduceOperation, all_reduce_prod_initial_value) {
    auto view_transposed = view(1, 0);
    auto prod = view_transposed.reduce([&](auto x, auto y) {
        return x * y;
    }, 1);

    EXPECT_THAT(prod, Eq(20));
}

template<class T>
struct Foo {
    template<class K, enable_if_t<is_tensor_view_v<K>, int> = 0>
    static int bar(K k) {
        return 0;
    }

    static int bar(T t) {
        return 1;
    }
};

TEST_F(ReduceOperation, foobar) {
    int j = 0;
    EXPECT_THAT(Foo<int>::bar(j), Eq(1));
    EXPECT_THAT(Foo<int>::bar(view), Eq(0));
}

TEST_F(ReduceOperation, reduce_axis0_sum) {
    std::vector<float> dst_data(2 * 2);
    auto dst_view = make_view(dst_data.data(), {2, 2});

    view.reduce([&](float x, float y) {
        return x + y;
    }, dst_view, 0);

    std::vector<float> data_expected = {12, 15, 18, 21};
    EXPECT_THAT(dst_data, ElementsAreArray(data_expected));
}

TEST_F(ReduceOperation, reduce_axis1_sum) {
    std::vector<float> dst_data(3 * 2);
    auto dst_view = make_view(dst_data.data(), {3, 2});

    view.reduce([&](float x, float y) {
        return x + y;
    }, dst_view, 1);

    std::vector<float> data_expected = {2, 4, 10, 12, 18, 20};
    EXPECT_THAT(dst_data, ElementsAreArray(data_expected));
}

TEST_F(ReduceOperation, reduce_axis2_sum) {
    std::vector<float> dst_data(3 * 2);
    auto dst_view = make_view(dst_data.data(), {3, 2});

    view.reduce([&](float x, float y) {
        return x + y;
    }, dst_view, 2);

    std::vector<float> data_expected = {1, 5, 9, 13, 17, 21};
    EXPECT_THAT(dst_data, ElementsAreArray(data_expected));
}

TEST_F(ReduceOperation, reduce_axis_min) {
    std::vector<float> dst_data(3 * 2);
    auto dst_view = make_view(dst_data.data(), {3, 2});

    view.reduce([&](float x, float y) {
        return std::min(x, y);
    }, dst_view, 1, std::numeric_limits<float>::max());

    std::vector<float> data_expected = {0, 1, 4, 5, 8, 9};
    EXPECT_THAT(dst_data, ElementsAreArray(data_expected));
}

TEST_F(ReduceOperation, reduce_axis1_sum_deferred) {
    std::vector<float> dst_data(3 * 2);
    auto dst_view = make_view(dst_data.data(), {3, 2});

    dst_view = view.reduce([&](float x, float y) {
        return x + y;
    }, 1, 0);

    std::vector<float> data_expected = {2, 4, 10, 12, 18, 20};
    EXPECT_THAT(dst_data, ElementsAreArray(data_expected));
}


class OwningTensor : public testing::Test {
};

TEST_F(OwningTensor, create_constructor_parameter_pack) {
    Tensor<float, 3> tensor(4,5,6);

    EXPECT_THAT(tensor.size(0), Eq(4));
    EXPECT_THAT(tensor.size(1), Eq(5));
    EXPECT_THAT(tensor.size(2), Eq(6));
}
TEST_F(OwningTensor, create_constructor_array) {
    size_t shape[] = {4,5,6};
    Tensor<float, 3> tensor(shape);

    EXPECT_THAT(tensor.size(0), Eq(4));
    EXPECT_THAT(tensor.size(1), Eq(5));
    EXPECT_THAT(tensor.size(2), Eq(6));
}



}