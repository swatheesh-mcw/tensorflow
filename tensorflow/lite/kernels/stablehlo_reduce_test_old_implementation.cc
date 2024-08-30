/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
         //
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
// #include "tensorflow/lite/kernels/stablehlo_reduce_window_test_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace reduce {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

// TF_LITE_ENSURE* family of macros require a context to be passed, which we do
// not have when building the model.
#define REDUCE_WINDOW_ENSURE_OK(expr)                        \
  do {                                                       \
    if (TfLiteStatus status = (expr); status != kTfLiteOk) { \
      ABSL_LOG(ERROR) << #expr " failed.\n";                 \
      return status;                                         \
    }                                                        \
  } while (false)

// Returns kTfLiteError if the expression evaluates to false.
#define REDUCE_WINDOW_ENSURE_IMPL(expr, msg) \
  do {                                       \
    if (!(expr)) {                           \
      ABSL_LOG(ERROR) << #msg " failed.\n";  \
      return kTfLiteError;                   \
    }                                        \
  } while (false)

#define REDUCE_WINDOW_ENSURE(expr) REDUCE_WINDOW_ENSURE_IMPL((expr), #expr)

#define REDUCE_WINDOW_ENSURE_EQ(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) == (b), #a " == " #b)
#define REDUCE_WINDOW_ENSURE_NE(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) != (b), #a " != " #b)
#define REDUCE_WINDOW_ENSURE_GE(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) >= (b), #a " >= " #b)
#define REDUCE_WINDOW_ENSURE_LE(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) <= (b), #a " <= " #b)
#define REDUCE_WINDOW_ENSURE_GT(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) > (b), #a " > " #b)
#define REDUCE_WINDOW_ENSURE_LT(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) < (b), #a " < " #b)
#define REDUCE_WINDOW_ENSURE_UNREACHABLE(msg) \
  REDUCE_WINDOW_ENSURE_IMPL(false, msg)

// Maps the native C++ types to the corresponding TFLite tensor type enum
// values.
template <class T>
struct TensorTypeFor;

#define TENSOR_TYPE_ASSOC(CPP_TYPE, TENSORTYPE_VALUE)     \
  template <>                                             \
  struct TensorTypeFor<CPP_TYPE> {                        \
    static constexpr TensorType value = TENSORTYPE_VALUE; \
  };

TENSOR_TYPE_ASSOC(int8_t, TensorType_INT8);
TENSOR_TYPE_ASSOC(int16_t, TensorType_INT16);
TENSOR_TYPE_ASSOC(int32_t, TensorType_INT32);
TENSOR_TYPE_ASSOC(int64_t, TensorType_INT64);
TENSOR_TYPE_ASSOC(uint8_t, TensorType_UINT8);
TENSOR_TYPE_ASSOC(uint16_t, TensorType_UINT16);
TENSOR_TYPE_ASSOC(uint32_t, TensorType_UINT32);
TENSOR_TYPE_ASSOC(uint64_t, TensorType_UINT64);
TENSOR_TYPE_ASSOC(float, TensorType_FLOAT32);
static_assert(sizeof(float) == 4, "float type is expected to be 32 bit long");
TENSOR_TYPE_ASSOC(double, TensorType_FLOAT64);
static_assert(sizeof(double) == 8, "double type is expected to be 64 bit long");

enum class BodyFunction {
  kUnset,
  kUnsupported,
  kAdd,
  kMul,
  kMax,
  kMin,
  kAll,
  kAny
};

std::ostream& operator<<(std::ostream& os, const BodyFunction& f) {
  switch (f) {
    case BodyFunction::kUnset:
      return os << "unset";
    case BodyFunction::kUnsupported:
      return os << "unsupported";
    case BodyFunction::kAdd:
      return os << "add";
    case BodyFunction::kMul:
      return os << "mul";
    case BodyFunction::kMax:
      return os << "max";
    case BodyFunction::kMin:
      return os << "min";
    case BodyFunction::kAll:
      return os << "all";
    case BodyFunction::kAny:
      return os << "any";
  }
  return os;
}

template <class T>
class ReduceOpModel : public SingleOpModel {
  static constexpr TensorType kTensorType = TensorTypeFor<T>::value;

 public:
  // Sets the input tensor shape and data.
  //
  // If the data isn't provided, the buffer is filled with `iota`.
  void SetInput(std::vector<T> input, absl::Span<const int64_t> shape) {
    input_shape_.assign(shape.begin(), shape.end());
    input_data_ = input;
  }

  void SetDimensions(absl::Span<const int64_t> dimensions) {
    dimensions_.assign(dimensions.begin(), dimensions.end());
  }

  void SetInitValue(const T& val) { init_value_ = val; }

  void SetBody(const BodyFunction func) { body_function_ = func; }

  TfLiteStatus Build() {
    constexpr int kBodySubGraphIndex = 1;

    REDUCE_WINDOW_ENSURE(!input_shape_.empty());
    REDUCE_WINDOW_ENSURE_NE(body_function_, BodyFunction::kUnset);
    REDUCE_WINDOW_ENSURE_NE(body_function_, BodyFunction::kUnsupported);

    input_tensor_id_ =
        AddInput({kTensorType,
                  std::vector<int>(input_shape_.begin(), input_shape_.end())});
    init_value_tensor_id_ = AddConstInput(kTensorType, {init_value_}, {1});
    output_tensor_id_ = AddOutput(kTensorType);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_REDUCE,
                 BuiltinOptions2_StablehloReduceOptions,
                 CreateStablehloReduceOptions(
                     builder_, builder_.CreateVector(dimensions_), kBodySubGraphIndex)
                     .Union());

    BuildInterpreter(
        /*input_shapes=*/{std::vector<int>(input_shape_.begin(),
                                           input_shape_.end())},
        /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
        /*apply_delegate=*/true, /*allocate_and_delegate=*/false,
        /*use_simple_allocator=*/false);

    int body_subgraph_index;
    AddSubgraphs(1, &body_subgraph_index);
    REDUCE_WINDOW_ENSURE_EQ(body_subgraph_index, kBodySubGraphIndex);
    switch (body_function_) {
      case BodyFunction::kAdd:
        subgraph_builder_.BuildAddSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kMul:
        subgraph_builder_.BuildMulSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kMax:
        subgraph_builder_.BuildMaximumSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kMin:
        subgraph_builder_.BuildMinimumSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kAll:
        subgraph_builder_.BuildLogicalAndSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kAny:
        subgraph_builder_.BuildLogicalOrSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      default:
        REDUCE_WINDOW_ENSURE_UNREACHABLE("Unhandled body function enum value.");
    }

    AllocateAndDelegate(/*apply_delegate=*/true);

    PopulateTensor(input_tensor_id_, input_data_);
    return kTfLiteOk;
  }

  TfLiteStatus BuildAndInvoke() {
    REDUCE_WINDOW_ENSURE_OK(Build());
    return Invoke();
  }

  absl::Span<const T> GetOutputData() {
    return absl::Span<const T>(interpreter_->typed_tensor<T>(output_tensor_id_),
                               GetTensorSize(output_tensor_id_));
  }

  absl::Span<const int> GetOutputShape() {
    const TfLiteIntArray& shape =
        *(interpreter_->tensor(output_tensor_id_)->dims);
    return absl::Span<const int>(shape.data, shape.size);
  }

  const std::vector<T>& GetInput() const { return input_data_; }

  const std::vector<int64_t>& GetInputShape() const { return input_shape_; }

  const std::vector<int64_t>& GetDimensions() const {
    return dimensions_;
  }

  const T& GetInitValue() const { return init_value_; }

  const BodyFunction& GetBodyFunction() const { return body_function_; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ReduceOpModel& model) {
    using Adapt = ReduceOpModel::VectorOutputAdapter;
    os << "input dimensions: {" << Adapt{model.GetInputShape()} << "}\n";
    os << "  dimensions: {" << Adapt{model.GetDimensions()}
       << "}\n";
    os << "  init value: " << +model.GetInitValue() << "\n";
    os << "  body function: " << model.GetBodyFunction() << "\n";
    return os;
  }

 protected:
  struct VectorOutputAdapter {
    const std::vector<int64_t>& data;
    friend std::ostream& operator<<(std::ostream& os,
                                    const VectorOutputAdapter& vec) {
      if (!vec.data.empty()) {
        os << +vec.data[0];
        for (size_t i = 1; i < vec.data.size(); ++i) {
          os << ", " << +vec.data[i];
        }
      }
      return os;
    }
  };

  int input_tensor_id_ = -1;
  int init_value_tensor_id_ = -1;
  int output_tensor_id_ = -1;
  std::vector<T> input_data_;
  T init_value_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> dimensions_;
  BodyFunction body_function_{};
  subgraph_test_util::SubgraphBuilder subgraph_builder_;
};

template <class StorageType>
class StablehloReduceTest : public testing::Test {};

using TestList =
    testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(StablehloReduceTest, TestList);

TYPED_TEST(StablehloReduceTest, Identity) {
  ReduceOpModel<TypeParam> model;
  model.SetInput({0,1,2,3,4,5},{1,6});
  model.SetDimensions({0});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kMul);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(15));
}

}  // namespace
}  // namespace reduce_window
}  // namespace tflite
