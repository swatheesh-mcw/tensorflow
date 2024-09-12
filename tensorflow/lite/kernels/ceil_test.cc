/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class CeilOpModel : public SingleOpModel {
 public:
  CeilOpModel(std::initializer_list<int> input_shape, TensorType input_type) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_CEIL, BuiltinOptions_NONE, 0);
    BuildInterpreter({
        input_shape,
    });
  }
  CeilOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(SymmetricInt16Scaling(input));
    output_ = AddOutput(SymmetricInt16Scaling(output));
    std::cout << "Before builtin \n\n ";
    SetBuiltinOp(BuiltinOperator_CEIL, BuiltinOptions_NONE, 0);
    // std::cout << "After builtin " << GetTensorShape(input_) << std::endl;
    BuildInterpreter({
        GetShape(input_),
    });
    std::cout << "After Interpreter \n\n ";
  }
  int input() { return input_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
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

 protected:
  int input_;
  int output_;
};

template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-7.0, 7.0);
  std::vector<float> input = {-6.2, -5.3, 0, 1.2, 4.4, 6.0};
  std::vector<float> result = {-6.0, -5.0, 0, 2.0, 5.0, 6.0};

  CeilOpModel model({tensor_type, {6}, -7.0, 7.0},
                    {tensor_type, {6}, -7.0, 7.0});
  model.QuantizeAndPopulate<integer_dtype>(model.input(), input);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(result, kQuantizedTolerance)));
}

TEST(QuantizedCeilOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedCeilOpModel, QuantizedTestsNoActivationInt16) {
  QuantizedTestsNoActivation<TensorType_INT16, int16_t>();
}

TEST(QuantizedCeilOpModel, QuantizedTestsNoActivationInt8DifferentMinMax) {
  float kQuantizedTolerance = GetTolerance<int8_t>(-20.0, 15.0);
  std::vector<float> input = {-6.4, -12.0, 0, -19.0, 4.8, 15.0};
  std::vector<float> result = {-6.0, -12.0, 0, -19.0, 5.0, 15.0};

  CeilOpModel model({TensorType_INT8, {6}, -19.0, 15.0},
                    {TensorType_INT8, {6}, -19.0, 15.0});
  model.QuantizeAndPopulate<int8_t>(model.input(), input);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(model.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(result, kQuantizedTolerance)));
}

TEST(CeilOpTest, SingleDim) {
  CeilOpModel model({2}, TensorType_FLOAT32);
  model.PopulateTensor<float>(model.input(), {8.5, 0.0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({9, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

// TEST(CeilOpTest, MultiDims) {
//   CeilOpModel model({2, 1, 1, 5}, TensorType_FLOAT32);
//   std::vector<float> input;
//   // if (AllowFp16PrecisionForFp32()) {
//   //   input = {
//   //       0.01, 8.01, 0.99, 9.99, 0.5, -0.01, -8.01, -0.99, -9.99, -0.5,
//   //   };
//   // } else {
//     input = {
//         0.0001,  8.0001,  0.9999,  9.9999,  0.5,
//         -0.0001, -8.0001, -0.9999, -9.9999, -0.5,
//     };
//   // }
//   model.PopulateTensor<float>(model.input(), input);
//   ASSERT_EQ(model.Invoke(), kTfLiteOk);
//   EXPECT_THAT(model.GetOutput(),
//               Pointwise(FloatingPointEq(), {1, 9, 1, 10, 1, 0, -8, 0, -9, 0}));
//   EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 5}));
// }

}  // namespace
}  // namespace tflite