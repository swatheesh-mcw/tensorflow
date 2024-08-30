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

#include <algorithm>
#include <cmath>

#include <cstdint>
#include <limits>
#include <numeric>

#include "Eigen/Core"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_reduce {

constexpr int kInputTensor = 0;
constexpr int kInitValue = 1;
constexpr int kOutputTensor = 0;
constexpr int kMaxStablehloReduceRank = 8;

// Gets offset of index if reducing on axis. When reducing, the flattened offset
// will not change, if the input index changes on the given axis. For example,
// if you have a 3D tensor and you are reducing to 2D by eliminating axis 0,
// then index (0, 1, 2) and index (1, 1, 2) will map to the same flattened
// offset.
// This function is modified from  ReducedOutputOffset function from
// tensorflow/lite/kernels/internal/types.h
inline size_t StablehloReducedOutputOffset(const int num_dims, const int* dims,
                                           const int* index,
                                           const int64_t num_axis,
                                           const int64_t* axis) {
  if (num_dims == 0) {
    return 0;
  }

  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx) {
    // if we need to skip this axis
    bool is_axis = false;
    for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
      if (idx == axis[axis_idx]) {
        is_axis = true;
        break;
      }
    }
    if (!is_axis) {
      offset = offset * static_cast<size_t>(dims[idx]) +
               static_cast<size_t>(index[idx]);
    }
  }
  return offset;
}

// This method parses the input 'axis' to remove duplicates and handle negative
// values, and returns a valid 'out_axis'
// This function is modified from  ResolveAxis function from
// tensorflow/lite/kernels/internal/reference/reduce.h
inline bool StablehloResolveAxis(const int num_dims, const int64_t* axis,
                                 const int64_t num_axis, int* out_axis,
                                 int64_t* out_num_axis) {
  *out_num_axis = 0;  // Just in case.
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    return true;
  }

  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    // Handle negative index. A positive index 'p_idx' can be represented as a
    // negative index 'n_idx' as: n_idx = p_idx-num_dims
    // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]  */
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    TFLITE_DCHECK(current >= 0 && current < num_dims);
    if (current < 0 || current >= num_dims) {
      return false;
    }
    bool is_dup = false;
    for (int j = 0; j < *out_num_axis; ++j) {
      if (out_axis[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

template <typename T>
inline bool InitTensorDataForReduce(const int* dims, const int num_dims,
                                    const T init_value, T* data) {
  size_t num_elements = 1;
  for (int idx = 0; idx < num_dims; ++idx) {
    size_t current = static_cast<size_t>(dims[idx]);
    // Overflow prevention.
    if (current > 0 &&
        num_elements > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_elements *= current;
  }
  for (size_t idx = 0; idx < num_elements; ++idx) {
    data[idx] = init_value;
  }
  return true;
}

template <typename DataType>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      const TfLiteTensor* input,
                      const TfLiteTensor* init_value_tensor,
                      const int64_t* dimension, const int64_t num_axis,
                      TfLiteTensor* output) {
  const TfLiteStablehloReduceParams* data =
      reinterpret_cast<TfLiteStablehloReduceParams*>(node->builtin_data);

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph& body_subgraph = *(*subgraphs)[data->body_subgraph_index];

  const DataType* input_data = GetTensorData<DataType>(input);
  const DataType* init_value = GetTensorData<DataType>(init_value_tensor);
  DataType* output_data = GetTensorData<DataType>(output);

  const int input_num_dims = input->dims->size;
  const int* input_dims = input->dims->data;
  const int output_num_dims = output->dims->size;
  const int* output_dims = output->dims->data;
  int64_t num_resolved_axis = 0;
  int resolved_axis[kMaxStablehloReduceRank] = {0};
  int input_iter[kMaxStablehloReduceRank] = {0};

  TF_LITE_ENSURE_MSG(context,
                     StablehloResolveAxis(input_num_dims, dimension, num_axis,
                                          resolved_axis, &num_resolved_axis),
                     "Invalid resolve axis");

  // Reset output data.
  if (!InitTensorDataForReduce<DataType>(output_dims, output_num_dims,
                                         init_value[0], output_data)) {
    return TfLiteStatus::kTfLiteError;
  }

  // Resolve axis.
  if (!StablehloResolveAxis(input_num_dims, dimension, num_axis, resolved_axis,
                            &num_resolved_axis)) {
    return TfLiteStatus::kTfLiteError;
  }

  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx) {
    input_iter[idx] = 0;
  }

  // Iterate through input_data and send inputs to the respective subgraph
  // reduce function.
  do {
    size_t input_offset = StablehloReducedOutputOffset(
        input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset = StablehloReducedOutputOffset(
        input_num_dims, input_dims, input_iter, num_resolved_axis, dimension);

    // Copy input_data and output data with respective offset to the subraph's
    // inputs.
    for (int idx = 0; idx < 2; ++idx) {
      TfLiteTensor* subgraph_input =
          body_subgraph.tensor(body_subgraph.inputs()[idx]);
      std::memcpy(subgraph_input->data.raw,
                  (idx % 2 == 0 ? &output_data[output_offset]
                                : &input_data[input_offset]),
                  sizeof(DataType));
    }
    TF_LITE_ENSURE_OK(context, body_subgraph.Invoke());

    TfLiteTensor* subgraph_output =
        body_subgraph.tensor(body_subgraph.outputs()[0]);

    output_data[output_offset] = GetTensorData<DataType>(subgraph_output)[0];
  } while (NextIndex(input_num_dims, input_dims, input_iter));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  const TfLiteTensor* init_value;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInitValue, &init_value));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TfLiteType data_type = input->type;
  const TfLiteStablehloReduceParams* data =
      reinterpret_cast<TfLiteStablehloReduceParams*>(node->builtin_data);
  const int64_t* dimension = data->dimensions;
  const int64_t num_dimension = data->num_dimensions;

  if (data_type == kTfLiteFloat32) {
    return EvalImpl<float>(context, node, input, init_value, dimension,
                           num_dimension, output);
  } else if (data_type == kTfLiteFloat64) {
    return EvalImpl<double>(context, node, input, init_value, dimension,
                            num_dimension, output);
  } else if (data_type == kTfLiteFloat16) {
    return EvalImpl<Eigen::half>(context, node, input, init_value, dimension,
                                 num_dimension, output);
  } else if (data_type == kTfLiteBFloat16) {
    return EvalImpl<Eigen::bfloat16>(context, node, input, init_value,
                                     dimension, num_dimension, output);
  } else if (data_type == kTfLiteInt8) {
    return EvalImpl<int8_t>(context, node, input, init_value, dimension,
                            num_dimension, output);
  } else if (data_type == kTfLiteInt16) {
    return EvalImpl<int16_t>(context, node, input, init_value, dimension,
                             num_dimension, output);
  } else if (data_type == kTfLiteInt32) {
    return EvalImpl<int32_t>(context, node, input, init_value, dimension,
                             num_dimension, output);
  } else if (data_type == kTfLiteInt64) {
    return EvalImpl<int64_t>(context, node, input, init_value, dimension,
                             num_dimension, output);
  } else if (data_type == kTfLiteBool) {
    return EvalImpl<bool>(context, node, input, init_value, dimension,
                          num_dimension, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return TfLiteStatus::kTfLiteError;
  }
}

// Returns the output shape.
TfLiteStatus GetOutputShape(TfLiteContext* context,
                            const TfLiteIntArray* input_dims,
                            const int input_num_dims, const int64_t* axis,
                            const int64_t num_axis,
                            TfLiteIntArray** output_shape) {
  if (input_num_dims == 0) {
    *output_shape = TfLiteIntArrayCreate(0);
    return kTfLiteOk;
  }

  // Calculates size of reducing axis.
  int num_reduce_axis = num_axis;
  for (int i = 0; i < num_axis; ++i) {
    int current = axis[i];
    if (current < 0) {
      current += input_num_dims;
    }
    TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
    for (int j = 0; j < i; ++j) {
      int previous = axis[j];
      if (previous < 0) {
        previous += input_num_dims;
      }
      if (current == previous) {
        --num_reduce_axis;
        break;
      }
    }

    // Determines output dimensions.
    TfLiteIntArray* output_dims =
        TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    *output_shape = output_dims;
    return kTfLiteOk;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  const TfLiteTensor* init_value;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInitValue, &init_value));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));  
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  int input_rank = input->dims->size;
  int init_value_rank = init_value->dims->size;

  TF_LITE_ENSURE_MSG(context, node->inputs->size > 0,
                     "'stablehlo.reduce' Input should not be empty.");
  TF_LITE_ENSURE_MSG(context,
                     input_rank > 0 && input_rank <= kMaxStablehloReduceRank,
                     "'stablehlo.reduce' Input rank out of range.");
  TF_LITE_ENSURE_MSG(
      context,
      init_value_rank >= 0 && init_value_rank < kMaxStablehloReduceRank,
      "'stablehlo.reduce' Init Value rank out of range.");

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();

  const TfLiteStablehloReduceParams* data =
      reinterpret_cast<TfLiteStablehloReduceParams*>(node->builtin_data);
  if (data->body_subgraph_index >= subgraphs->size()) {
    TF_LITE_KERNEL_LOG(context,
                       "body subgraph not found for stablehlo.reduce.");
    return TfLiteStatus::kTfLiteError;
  }

  Subgraph* body_subgraph = (*subgraphs)[data->body_subgraph_index].get();
  TF_LITE_ENSURE_EQ(context, body_subgraph->outputs().size(), 1);

  for (int i = 0; i < node->inputs->size; ++i) {
    int input_idx = body_subgraph->inputs()[i];

    TfLiteTensor* body_subgraph_input = body_subgraph->tensor(input_idx);
    body_subgraph_input->params = input->params;

    if (input->type == kTfLiteInt16 &&
        input->quantization.type != kTfLiteNoQuantization) {
      TfLiteQuantizationFree(&body_subgraph_input->quantization);
      body_subgraph_input->quantization.type = kTfLiteAffineQuantization;
      auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
          malloc(sizeof(TfLiteAffineQuantization)));
      affine_quantization->quantized_dimension = 0;
      affine_quantization->scale = TfLiteFloatArrayCreate(1);
      affine_quantization->zero_point = TfLiteIntArrayCreate(1);
      affine_quantization->scale->data[0] = input->params.scale;
      affine_quantization->zero_point->data[0] = input->params.zero_point;
      body_subgraph_input->quantization.params = affine_quantization;
    }
  }
  TfLiteTensor* body_subgraph_output =
      body_subgraph->tensor(body_subgraph->outputs()[0]);
  body_subgraph_output->params = output->params;

  if (input->type == kTfLiteInt16 &&
      input->quantization.type != kTfLiteNoQuantization) {
    TfLiteQuantizationFree(&body_subgraph_output->quantization);
    body_subgraph_output->quantization.type = kTfLiteAffineQuantization;
    auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    affine_quantization->quantized_dimension = 0;
    affine_quantization->scale = TfLiteFloatArrayCreate(1);
    affine_quantization->zero_point = TfLiteIntArrayCreate(1);
    affine_quantization->scale->data[0] = input->params.scale;
    affine_quantization->zero_point->data[0] = input->params.zero_point;
    body_subgraph_output->quantization.params = affine_quantization;
  }

  TF_LITE_ENSURE_OK(context, body_subgraph->AllocateTensors());

  for (int idx = 0; idx < data->num_dimensions; ++idx) {
    TF_LITE_ENSURE_MSG(
        context,
        data->dimensions[idx] >= 0 && data->dimensions[idx] < input->dims->size,
        "'stablehlo.reduce' Dimension out of range.");
  }

  TfLiteIntArray* output_dims;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputShape(context, input->dims, input->dims->size, data->dimensions,
                     data->num_dimensions, &output_dims));
  context->ResizeTensor(context, output, output_dims);

  return TfLiteStatus::kTfLiteOk;
}

}  // namespace stablehlo_reduce

TfLiteRegistration* Register_STABLEHLO_REDUCE() {
  static TfLiteRegistration r = {nullptr, nullptr, stablehlo_reduce::Prepare,
                                 stablehlo_reduce::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
