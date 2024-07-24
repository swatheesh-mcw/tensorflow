/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <initializer_list>
#include <iostream>
#include <vector>

#include "Eigen/Core"
#include "subgraph_test_util.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

template <typename T>
tflite::TensorType GetTTEnum();

template <>
tflite::TensorType GetTTEnum<Eigen::half>() {
  return tflite::TensorType_FLOAT16;
}

template <>
tflite::TensorType GetTTEnum<Eigen::bfloat16>() {
  return tflite::TensorType_BFLOAT16;
}

template <>
tflite::TensorType GetTTEnum<float>() {
  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<double>() {
  return tflite::TensorType_FLOAT64;
}

template <>
tflite::TensorType GetTTEnum<int8_t>() {
  return tflite::TensorType_INT8;
}

template <>
tflite::TensorType GetTTEnum<int16_t>() {
  return tflite::TensorType_INT16;
}

template <>
tflite::TensorType GetTTEnum<int32_t>() {
  return tflite::TensorType_INT32;
}

template <>
tflite::TensorType GetTTEnum<int64_t>() {
  return tflite::TensorType_INT64;
}

template <>
tflite::TensorType GetTTEnum<bool>() {
  return tflite::TensorType_BOOL;
}

template <typename T>
TfLiteType GetTfLiteEnum();

template <>
TfLiteType GetTfLiteEnum<Eigen::half>() {
  return kTfLiteFloat16;
}

template <>
TfLiteType GetTfLiteEnum<Eigen::bfloat16>() {
  return kTfLiteBFloat16;
}

template <>
TfLiteType GetTfLiteEnum<float>() {
  return kTfLiteFloat32;
}

template <>
TfLiteType GetTfLiteEnum<double>() {
  return kTfLiteFloat64;
}

template <>
TfLiteType GetTfLiteEnum<int8_t>() {
  return kTfLiteInt8;
}

template <>
TfLiteType GetTfLiteEnum<int16_t>() {
  return kTfLiteInt16;
}

template <>
TfLiteType GetTfLiteEnum<int32_t>() {
  return kTfLiteInt32;
}

template <>
TfLiteType GetTfLiteEnum<int64_t>() {
  return kTfLiteInt64;
}

template <>
TfLiteType GetTfLiteEnum<bool>() {
  return kTfLiteBool;
}

using subgraph_test_util::ReduceFunction;
using testing::ElementsAre;
using testing::ElementsAreArray;

class ReduceOpModel : public SingleOpModel {
 public:
  ReduceOpModel(const TensorData& inputs, const TensorData& init_value,
                const TensorData& outputs,
                const TfLiteStablehloReduceParams& params,
                ReduceFunction reduce_function, TfLiteType type) {
    inputs_ = AddInput(SymmetricInt16Scaling(inputs));
    init_value_ = AddInput(SymmetricInt16Scaling(init_value));
    outputs_ = AddOutput(SymmetricInt16Scaling(outputs));
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_REDUCE,
        BuiltinOptions2_StablehloReduceOptions,
        CreateStablehloReduceOptions(
            builder_,
            builder_.CreateVector(params.dimensions, params.num_dimensions), 1)
            .Union());
    BuildInterpreter({GetShape(inputs_), GetShape(init_value_)},
                     /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);

    AddSubgraphs(1, nullptr);
    subgraph_builder_.BuildReduceSubgraph(interpreter_->subgraph(1),
                                          reduce_function, type);

    AllocateAndDelegate(true);
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(inputs_, data);
  }

  template <typename T>
  void SetInitValue(std::initializer_list<T> data) {
    PopulateTensor<T>(init_value_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(outputs_);
  }

  template <typename QuantizedType>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<QuantizedType>(
        this->template ExtractVector<QuantizedType>(this->outputs_),
        GetScale(this->outputs_), GetZeroPoint(this->outputs_));
  }

  TensorData SymmetricInt16Scaling(TensorData tensor) {
    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range [int_type::min+1, int_type::max] to ensure a null
    // zero-point.
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
    return tensor;
  }

  int input() { return inputs_; }
  int init_value() { return init_value_; }
  std::vector<int> GetOutputShape() { return GetTensorShape(outputs_); }

 protected:
  Subgraph* subgraph_;
  int inputs_;
  int init_value_;
  int outputs_;
  ReduceFunction reduce_function;
  subgraph_test_util::SubgraphBuilder subgraph_builder_;
};

// for quantized, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0 * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <typename Int>
class StablehloReduceTestInt : public ::testing::Test {
 public:
  using IntType = Int;
};

using TestTypes = ::testing::Types<int16_t, int32_t, int64_t>;

TYPED_TEST_SUITE(StablehloReduceTestInt, TestTypes);

TYPED_TEST(StablehloReduceTestInt, ReduceIntAddExample1) {
  using Int = typename TestFixture::IntType;
  ReduceFunction reduce_function = ReduceFunction::kADD;
  TfLiteStablehloReduceParams params = {{2},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Int>(), {1, 2, 6}}, {GetTTEnum<Int>(), {1}},
                      {GetTTEnum<Int>(), {}}, params, reduce_function,
                      GetTfLiteEnum<Int>());

  model.SetInput<Int>({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
  model.SetInitValue<Int>({10});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Int> expected_values = {31, 31};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(model.GetOutput<Int>(), ElementsAreArray(expected_values));
}

TYPED_TEST(StablehloReduceTestInt, ReduceIntMulExample1) {
  using Int = typename TestFixture::IntType;
  ReduceFunction reduce_function = ReduceFunction::kMUL;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Int>(), {1, 2, 6}}, {GetTTEnum<Int>(), {1}},
                      {GetTTEnum<Int>(), {}}, params, reduce_function,
                      GetTfLiteEnum<Int>());

  model.SetInput<Int>({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
  model.SetInitValue<Int>({1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Int> expected_values = {1, 4, 9, 16, 25, 36};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Int>(), ElementsAreArray(expected_values));
}

TYPED_TEST(StablehloReduceTestInt, ReduceIntMinExample1) {
  using Int = typename TestFixture::IntType;
  ReduceFunction reduce_function = ReduceFunction::kMIN;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Int>(), {1, 2, 6}}, {GetTTEnum<Int>(), {1}},
                      {GetTTEnum<Int>(), {}}, params, reduce_function,
                      GetTfLiteEnum<Int>());

  model.SetInput<Int>({1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15});
  model.SetInitValue<Int>({1000});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Int> expected_values = {1, 2, 3, 4, 5, 6};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Int>(), ElementsAreArray(expected_values));
}

TYPED_TEST(StablehloReduceTestInt, ReduceIntMaxExample1) {
  using Int = typename TestFixture::IntType;
  ReduceFunction reduce_function = ReduceFunction::kMAX;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Int>(), {1, 2, 6}}, {GetTTEnum<Int>(), {1}},
                      {GetTTEnum<Int>(), {}}, params, reduce_function,
                      GetTfLiteEnum<Int>());

  model.SetInput<Int>({1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15});
  model.SetInitValue<Int>({0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Int> expected_values = {10, 11, 12, 13, 14, 15};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Int>(), ElementsAreArray(expected_values));
}

template <typename Float>
class StablehloReduceTestFloat : public ::testing::Test {
 public:
  using FloatType = Float;
};

using FloatTestTypes = ::testing::Types<float>;

TYPED_TEST_SUITE(StablehloReduceTestFloat, FloatTestTypes);

TYPED_TEST(StablehloReduceTestFloat, ReduceFloatAddExample1) {
  using Float = typename TestFixture::FloatType;
  ReduceFunction reduce_function = ReduceFunction::kADD;
  TfLiteStablehloReduceParams params = {{2},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Float>(), {1, 2, 6}},
                      {GetTTEnum<Float>(), {1}}, {GetTTEnum<Float>(), {}},
                      params, reduce_function, GetTfLiteEnum<Float>());

  model.SetInput<Float>({Float(1), Float(2), Float(3), Float(4), Float(5),
                         Float(6), Float(1), Float(2), Float(3), Float(4),
                         Float(5), Float(6)});
  model.SetInitValue<Float>({Float(10)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Float> expected_values = {Float(31), Float(31)};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(model.GetOutput<Float>(), ElementsAreArray(expected_values));
}

TYPED_TEST(StablehloReduceTestFloat, ReduceFloatMulExample1) {
  using Float = typename TestFixture::FloatType;
  ReduceFunction reduce_function = ReduceFunction::kMUL;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Float>(), {1, 2, 6}},
                      {GetTTEnum<Float>(), {1}}, {GetTTEnum<Float>(), {}},
                      params, reduce_function, GetTfLiteEnum<Float>());

  model.SetInput<Float>({Float(1), Float(2), Float(3), Float(4), Float(5),
                         Float(6), Float(1), Float(2), Float(3), Float(4),
                         Float(5), Float(6)});
  model.SetInitValue<Float>({Float(1)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Float> expected_values = {Float(1),  Float(4),  Float(9),
                                        Float(16), Float(25), Float(36)};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Float>(), ElementsAreArray(expected_values));
}

TYPED_TEST(StablehloReduceTestFloat, ReduceFloatMinExample1) {
  using Float = typename TestFixture::FloatType;
  ReduceFunction reduce_function = ReduceFunction::kMIN;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Float>(), {1, 2, 6}},
                      {GetTTEnum<Float>(), {1}}, {GetTTEnum<Float>(), {}},
                      params, reduce_function, GetTfLiteEnum<Float>());

  model.SetInput<Float>({Float(1), Float(2), Float(3), Float(4), Float(5),
                         Float(6), Float(10), Float(11), Float(12), Float(13),
                         Float(14), Float(15)});
  model.SetInitValue<Float>({float(1000)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Float> expected_values = {Float(1), Float(2), Float(3),
                                        Float(4), Float(5), Float(6)};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Float>(), ElementsAreArray(expected_values));
}

TYPED_TEST(StablehloReduceTestFloat, ReduceFloatMaxExample1) {
  using Float = typename TestFixture::FloatType;
  ReduceFunction reduce_function = ReduceFunction::kMAX;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Float>(), {1, 2, 6}},
                      {GetTTEnum<Float>(), {1}}, {GetTTEnum<Float>(), {}},
                      params, reduce_function, GetTfLiteEnum<Float>());

  model.SetInput<Float>({Float(1), Float(2), Float(3), Float(4), Float(5),
                         Float(6), Float(10), Float(11), Float(12), Float(13),
                         Float(14), Float(15)});
  model.SetInitValue<Float>({Float(0)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Float> expected_values = {Float(10), Float(11), Float(12),
                                        Float(13), Float(14), Float(15)};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Float>(), ElementsAreArray(expected_values));
}

template <typename Bool>
class StablehloReduceTestBool : public ::testing::Test {
 public:
  using BoolType = Bool;
};

using BoolTestTypes = ::testing::Types<bool>;

TYPED_TEST_SUITE(StablehloReduceTestBool, BoolTestTypes);

TYPED_TEST(StablehloReduceTestBool, ReduceBoolAnyExample1) {
  using Bool = typename TestFixture::BoolType;
  ReduceFunction reduce_function = ReduceFunction::kANY;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Bool>(), {1, 2, 6}}, {GetTTEnum<Bool>(), {1}},
                      {GetTTEnum<Bool>(), {}}, params, reduce_function,
                      GetTfLiteEnum<Bool>());

  model.SetInput<Bool>({true, true, true, true, true, true, false, false, false,
                        false, false, false});
  model.SetInitValue<Bool>({false});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Bool> expected_values = {true, true, true, true, true, true};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Bool>(), ElementsAreArray(expected_values));
}

TYPED_TEST(StablehloReduceTestBool, ReduceBoolAllExample1) {
  using Bool = typename TestFixture::BoolType;
  ReduceFunction reduce_function = ReduceFunction::kALL;
  TfLiteStablehloReduceParams params = {{1, 0},  // dimensions
                                        2,       // num_dimensions
                                        1};      // body_subgraph_index
  ReduceOpModel model({GetTTEnum<Bool>(), {1, 2, 6}}, {GetTTEnum<Bool>(), {1}},
                      {GetTTEnum<Bool>(), {}}, params, reduce_function,
                      GetTfLiteEnum<Bool>());

  model.SetInput<Bool>({true, true, true, true, true, true, false, false, false,
                        false, false, false});
  model.SetInitValue<Bool>({false});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<Bool> expected_values = {false, false, false,
                                       false, false, false};
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetOutput<Bool>(), ElementsAreArray(expected_values));
}

template <typename QuantizedInt>
class StablehloReduceQuantizedTestInt : public ::testing::Test {
 public:
  using QuantizedIntType = QuantizedInt;
};

using QuantizedTestTypes = ::testing::Types<int8_t>;

TYPED_TEST_SUITE(StablehloReduceQuantizedTestInt, QuantizedTestTypes);

TYPED_TEST(StablehloReduceQuantizedTestInt, ReduceQuantizedIntProdExample1) {
  using QuantizedInt = typename TestFixture::QuantizedIntType;
  float kQuantizedTolerance = GetTolerance<QuantizedInt>(0.0f, 128.0f);
  ReduceFunction reduce_function = ReduceFunction::kMUL;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<QuantizedInt>(), {1, 2, 6}, 0.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {1}, 0.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {}, 0.0f, 128.0f}, params,
                      reduce_function, GetTfLiteEnum<QuantizedInt>());
  model.QuantizeAndPopulate<QuantizedInt>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<QuantizedInt>(model.init_value(), {1.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(
      model.GetDequantizedOutput<QuantizedInt>(),
      ElementsAreArray(ArrayFloatNear(
          {10.0f, 22.0f, 36.0f, 52.0f, 70.0f, 90.0f}, kQuantizedTolerance)));
}

TYPED_TEST(StablehloReduceQuantizedTestInt, ReduceQuantizedIntAddExample1) {
  using QuantizedInt = typename TestFixture::QuantizedIntType;
  float kQuantizedTolerance = GetTolerance<QuantizedInt>(0.0f, 128.0f);
  ReduceFunction reduce_function = ReduceFunction::kADD;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<QuantizedInt>(), {1, 2, 6}, 0.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {1}, 0.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {}, 0.0f, 128.0f}, params,
                      reduce_function, GetTfLiteEnum<QuantizedInt>());
  model.QuantizeAndPopulate<QuantizedInt>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<QuantizedInt>(model.init_value(), {0.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(
      model.GetDequantizedOutput<QuantizedInt>(),
      ElementsAreArray(ArrayFloatNear(
          {11.0f, 13.0f, 15.0f, 17.0f, 19.0f, 21.0f}, kQuantizedTolerance)));
}

TYPED_TEST(StablehloReduceQuantizedTestInt, ReduceQuantizedIntMinExample1) {
  using QuantizedInt = typename TestFixture::QuantizedIntType;
  float kQuantizedTolerance = GetTolerance<QuantizedInt>(-127.0f, 128.0f);
  ReduceFunction reduce_function = ReduceFunction::kMIN;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<QuantizedInt>(), {1, 2, 6}, -127.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {1}, -127.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {}, -127.0f, 128.0f}, params,
                      reduce_function, GetTfLiteEnum<QuantizedInt>());
  model.QuantizeAndPopulate<QuantizedInt>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<QuantizedInt>(model.init_value(), {24.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(model.GetDequantizedOutput<QuantizedInt>(),
              ElementsAreArray(ArrayFloatNear(
                  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, kQuantizedTolerance)));
}

TYPED_TEST(StablehloReduceQuantizedTestInt, ReduceQuantizedIntMaxExample1) {
  using QuantizedInt = typename TestFixture::QuantizedIntType;
  float kQuantizedTolerance = GetTolerance<QuantizedInt>(0.0f, 128.0f);
  ReduceFunction reduce_function = ReduceFunction::kMAX;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<QuantizedInt>(), {1, 2, 6}, 0.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {1}, 0.0f, 128.0f},
                      {GetTTEnum<QuantizedInt>(), {}, 0.0f, 128.0f}, params,
                      reduce_function, GetTfLiteEnum<QuantizedInt>());
  model.QuantizeAndPopulate<QuantizedInt>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<QuantizedInt>(model.init_value(), {0.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(
      model.GetDequantizedOutput<QuantizedInt>(),
      ElementsAreArray(ArrayFloatNear(
          {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f}, kQuantizedTolerance)));
}

TEST(ReduceOpModelTest, ReduceQuantizedInt16AddExample1) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-127.0f, 127.0f);
  ReduceFunction reduce_function = ReduceFunction::kADD;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<int16_t>(), {1, 2, 6}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {1}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {}, -127.0f, 127.0f}, params,
                      reduce_function, GetTfLiteEnum<int16_t>());
  model.QuantizeAndPopulate<int16_t>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<int16_t>(model.init_value(), {0.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(
      model.GetDequantizedOutput<int16_t>(),
      ElementsAreArray(ArrayFloatNear(
          {11.0f, 13.0f, 15.0f, 17.0f, 19.0f, 21.0f}, kQuantizedTolerance)));
}

TEST(ReduceOpModelTest, ReduceQuantizedInt16ProdExample1) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-127.0f, 127.0f);
  ReduceFunction reduce_function = ReduceFunction::kMUL;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<int16_t>(), {1, 2, 6}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {1}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {}, -127.0f, 127.0f}, params,
                      reduce_function, GetTfLiteEnum<int16_t>());
  model.QuantizeAndPopulate<int16_t>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<int16_t>(model.init_value(), {1.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(
      model.GetDequantizedOutput<int16_t>(),
      ElementsAreArray(ArrayFloatNear(
          {10.0f, 22.0f, 36.0f, 52.0f, 70.0f, 90.0f}, kQuantizedTolerance)));
}

TEST(ReduceOpModelTest, ReduceQuantizedInt16MinExample1) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-127.0f, 127.0f);
  ReduceFunction reduce_function = ReduceFunction::kMIN;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<int16_t>(), {1, 2, 6}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {1}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {}, -127.0f, 127.0f}, params,
                      reduce_function, GetTfLiteEnum<int16_t>());
  model.QuantizeAndPopulate<int16_t>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<int16_t>(model.init_value(), {24.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(model.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, kQuantizedTolerance)));
}

TEST(ReduceOpModelTest, ReduceQuantizedInt16MaxExample1) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-127.0f, 127.0f);
  ReduceFunction reduce_function = ReduceFunction::kMAX;
  TfLiteStablehloReduceParams params = {{1},  // dimensions
                                        1,    // num_dimensions
                                        1};   // body_subgraph_index
  ReduceOpModel model({GetTTEnum<int16_t>(), {1, 2, 6}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {1}, -127.0f, 127.0f},
                      {GetTTEnum<int16_t>(), {}, -127.0f, 127.0f}, params,
                      reduce_function, GetTfLiteEnum<int16_t>());
  model.QuantizeAndPopulate<int16_t>(
      model.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f});
  model.QuantizeAndPopulate<int16_t>(model.init_value(), {0.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6}));
  EXPECT_THAT(
      model.GetDequantizedOutput<int16_t>(),
      ElementsAreArray(ArrayFloatNear(
          {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f}, kQuantizedTolerance)));
}

}  // namespace
}  // namespace tflite