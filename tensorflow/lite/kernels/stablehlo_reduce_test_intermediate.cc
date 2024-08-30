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
#include <vector>
#include <iostream>

#include "Eigen/Core"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using subgraph_test_util::ConditionDirection;
using testing::ElementsAre;
using testing::ElementsAreArray;

class StablehloReduceOpModel : public SingleOpModel {
 public:
  StablehloReduceOpModel(const TensorData& inputs,
              const TfLiteStablehloSortParams& params,
              ConditionDirection condition_direction, TfLiteType type) {
    inputs_ = AddInput(inputs);
    SetBuiltinOp(BuiltinOperator_STABLEHLO_REDUCE,
                 BuiltinOptions2_StablehloReduceOptions,
                 CreateStablehloReduceOptions(builder_, params.dimension,0)
                     .Union());
    BuildInterpreter({GetShape(inputs)},
                     /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);

    int* dummy = nullptr;
    AddSubgraphs(1, dummy);
    subgraph_builder_.BuildConditionSubgraph(interpreter_->subgraph(1),
                                              condition_direction, type);
    AllocateAndDelegate(true);
  }

  template <typename T>
  void SetInput(std::vector<T> data) {
    PopulateTensor<T>(inputs_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(outputs_);
  }

 protected:
  Subgraph* subgraph_;
  int inputs_;
  int outputs_;
  ConditionDirection condition_direction;
  subgraph_test_util::SubgraphBuilder subgraph_builder_;
};

template <>
void StablehloReduceOpModel::SetInput<Eigen::half>(
    int index, std::initializer_list<Eigen::half> data) {
  PopulateTensor<Eigen::half>(inputs_[index], data);
}
template <>
void StablehloReduceOpModel::SetInput<Eigen::bfloat16>(
    int index, std::initializer_list<Eigen::bfloat16> data) {
  PopulateTensor<Eigen::bfloat16>(inputs_[index], data);
}

TEST(StablehloReduceOpTest, SortInt32) {
  std::cout<<"Came here";
  ConditionDirection condition_direction = ConditionDirection::kADD;
  TfLiteStablehloReduceParams params = {{0}, 1};
  int num_inputs = 1;
  std::vector<TensorData> inputs(num_inputs,
                                 {TensorType_INT32, {2, 2, 2, 2, 2, 2}});
  StablehloReduceOpModel model(inputs, params, condition_direction, num_inputs,
                    kTfLiteInt32);
  model.SetInput<int>(
      0, {7, 2, 5, 9, 3, 6, 1, 8, 10, 4,  7, 3, 2, 6, 4, 5, 8, 1, 3,  10, 6, 9,
          2, 7, 5, 4, 9, 3, 8, 2, 1,  6,  2, 5, 7, 1, 3, 8, 4, 6, 10, 2,  9, 5,
          7, 6, 8, 3, 1, 7, 4, 3, 5,  10, 6, 9, 2, 8, 3, 4, 7, 1, 5,  6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<int> expected_values_0 = {
      7, 2, 9, 5, 6, 3, 8, 1, 10, 4, 7, 3, 6, 2, 5, 4, 8, 1, 10, 3, 9, 6,
      7, 2, 5, 4, 9, 3, 8, 2, 6,  1, 5, 2, 7, 1, 8, 3, 6, 4, 10, 2, 9, 5,
      7, 6, 8, 3, 7, 1, 4, 3, 10, 5, 9, 6, 8, 2, 4, 3, 7, 1, 6,  5,
  };
  EXPECT_THAT(model.GetOutput<int>(0), ElementsAreArray(expected_values_0));
}

// TEST(StablehloSortOpTest, SortFloat32) {
//   ComparisonDirection comparison_direction = ComparisonDirection::kLE;
//   TfLiteStablehloSortParams params = {0, false, 1};
//   int num_inputs = 2;
//   int lhs = 2;
//   int rhs = 3;
//   std::vector<TensorData> inputs(num_inputs, {TensorType_FLOAT32, {3, 3, 2}});
//   SortOpModel model(inputs, params, comparison_direction, num_inputs, lhs, rhs,
//                     kTfLiteFloat32);
//   model.SetInput<float>(
//       0, {8.865, 3.267, 1.964, 9.764, 11.212, 5.597, 6.111, 4.276, 10.919,
//           1.427, 3.123, 2.179, 2.433, 7.935, 5.791, 6.308, 9.822, 10.118});
//   model.SetInput<float>(
//       1, {10.499, 7.248, 3.621, 4.862, 5.305, 8.484, 2.257, 1.739, 9.438, 6.794,
//           11.596, 12.000, 4.070, 2.950, 8.015, 11.740, 7.611, 3.194});
//   ASSERT_EQ(model.Invoke(), kTfLiteOk);
//   std::vector<float> expected_values_0 = {
//       2.433000e+00, 7.935000e+00, 1.964000e+00, 9.76399993,   1.121200e+01,
//       2.179000e+00, 8.86499977,   3.267000e+00, 5.791000e+00, 1.427000e+00,
//       9.82199954,   1.011800e+01, 6.111000e+00, 4.276000e+00, 1.091900e+01,
//       6.308000e+00, 3.123000e+00, 5.597000e+00};
//   std::vector<float> expected_values_1 = {
//       4.070000e+00, 2.950000e+00, 3.621000e+00, 4.862000e+00, 5.305000e+00,
//       1.200000e+01, 1.049900e+01, 7.248000e+00, 8.015000e+00, 6.794000e+00,
//       7.611000e+00, 3.194000e+00, 2.257000e+00, 1.739000e+00, 9.43799972,
//       1.174000e+01, 1.159600e+01, 8.484000e+00};
//   EXPECT_THAT(model.GetOutput<float>(0), ElementsAreArray(expected_values_0));
//   EXPECT_THAT(model.GetOutput<float>(1), ElementsAreArray(expected_values_1));
// }

// TEST(StablehloSortOpTest, SortInt32UnstableSort) {
//   ComparisonDirection comparison_direction = ComparisonDirection::kLE;
//   TfLiteStablehloSortParams params = {1, false, 1};
//   int num_inputs = 3;
//   int lhs = 0;
//   int rhs = 1;
//   std::vector<TensorData> inputs(num_inputs, {TensorType_INT32, {2, 3}});
//   SortOpModel model(inputs, params, comparison_direction, num_inputs, lhs, rhs,
//                     kTfLiteInt32);
//   model.SetInput<int>(0, {1, 2, 3, 3, 2, 1});
//   model.SetInput<int>(1, {3, 2, 1, 1, 2, 3});
//   model.SetInput<int>(2, {1, 2, 3, 3, 2, 1});
//   ASSERT_EQ(model.Invoke(), kTfLiteOk);
//   std::vector<int> expected_values_0 = {1, 2, 3, 1, 2, 3};
//   std::vector<int> expected_values_1 = {3, 2, 1, 3, 2, 1};
//   std::vector<int> expected_values_2 = {1, 2, 3, 1, 2, 3};
//   EXPECT_THAT(model.GetOutput<int>(0), ElementsAreArray(expected_values_0));
//   EXPECT_THAT(model.GetOutput<int>(1), ElementsAreArray(expected_values_1));
//   EXPECT_THAT(model.GetOutput<int>(2), ElementsAreArray(expected_values_2));
// }

// TEST(StablehloSortOpTest, SortQuantized) {
//   ComparisonDirection comparison_direction = ComparisonDirection::kLE;
//   TfLiteStablehloSortParams params = {1, true, 1};
//   int num_inputs = 2;
//   int lhs = 0;
//   int rhs = 1;
//   std::vector<TensorData> inputs(num_inputs, {TensorType_INT8, {2, 3}, -127.f, 127.f, 1.5, 0});
//   SortOpModel model(inputs, params, comparison_direction, num_inputs, lhs, rhs, kTfLiteInt8);
//   model.SetInput<int8_t>(0, {1, 2, 3, 3, 2, 1});
//   model.SetInput<int8_t>(1, {3, 2, 1, 1, 2, 3});
//   ASSERT_EQ(model.Invoke(), kTfLiteOk);
//   std::vector<int8_t> expected_values_0 = {1, 2, 3, 1, 2, 3};
//   std::vector<int8_t> expected_values_1 = {3, 2, 1, 3, 2, 1};
//   EXPECT_THAT(model.GetOutput<int8_t>(0), ElementsAreArray(expected_values_0));
//   EXPECT_THAT(model.GetOutput<int8_t>(1), ElementsAreArray(expected_values_1));
// }

}  // namespace
}  // namespace tflite
