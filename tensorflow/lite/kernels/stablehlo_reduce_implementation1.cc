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
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include <iostream>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reduce_op {

constexpr int kInputTensor = 0;
constexpr int kInitValue = 1;
constexpr int kOutputTensor = 0;

// Holds the data needed throughout the node lifetime.
struct NodeData {
  TfLiteReduceFunction body;
};

// This file has reference implementation of reduce_* operators.
enum KernelType {
  kReference,
  kGenericOptimized,
};

// enum for various reduce operations
enum ReduceType {
  kSum,
  kProd,
  kMax,
  kMin,
  kAny,
  kAll,
};

struct OpData {
  OpData(TfLiteContext* context, TfLiteNode* node)
      : context(context), node(node) {}

  TfLiteContext* context;
  TfLiteNode* node;

  TfLiteType type;
  int rank;
  int64_t element_size;
  int64_t input_dims[TFLITE_STABLEHLO_REDUCE_PARAMS_MAX_DIMENSION_COUNT];
  const TfLiteTensor* input;
  const TfLiteTensor* init_value;
  const int64_t* dimensions;
  TfLiteTensor* output;

  // TfLiteTensor* GetTemporary(int id) {
  //   return tflite::GetTemporary(context, node, id);
  // }

  // Helper to resize a tensor.
  TfLiteStatus ResizeTensor(TfLiteTensor* const tensor,
                            const int64_t* const shape) {
    auto dims = BuildTfLiteArray<int32_t>(rank, shape);
    return context->ResizeTensor(context, tensor, dims.release());
  }

  // Sets the operation data type and the associated byte size.
  TfLiteStatus SetElementType(TfLiteType t) {
    type = t;
    size_t unsigned_element_size;
    TF_LITE_ENSURE_OK(context,
                      GetSizeOfType(context, type, &unsigned_element_size));
    TF_LITE_ENSURE_MSG(
        context,
        // Directly comparing the unsigned_element_size to the max value of
        // int64_t fails the -Wtautological-constant-out-of-range-compare
        // warning when building on 32 bit targets.
        sizeof(unsigned_element_size) < sizeof(int64_t) ||
            unsigned_element_size <= std::numeric_limits<int64_t>::max(),
        "The element size cannot be contained in an int64_t value.");
    element_size = unsigned_element_size;
    return kTfLiteOk;
  }

  // Factors the initialization that are common across semantics.
  //
  // Semantic is one of StablehloData or TFLiteData.
  template <class Semantic>
  TfLiteStatus InitializeBase() {
    // init_value = reinterpret_cast<const char*>(
    //     GetInput(context, node,kInitValue)->data.data);
    init_value = GetInput(context,node,kInitValue);
    input = GetInput(context,node,kInputTensor);
    output = GetOutput(context,node,kOutputTensor);

    // const TfLiteTensor* const input_tensor =
    //     GetInput(context, node, kInputTensor);
    SetElementType(input->type);
    rank = input->dims->size;
    std::copy_n(input->dims->data, rank, input_dims);
    // input = reinterpret_cast<const char*>(input_tensor->data.data);

    // TfLiteTensor* const output_tensor =
    //     GetOutput(context, node, kOutputTensor);
    // output = reinterpret_cast<char*>(output_tensor->data.data);
    return kTfLiteOk;
  }
};

// Speciliazes OpData for the STABLEHLO_REDUCE operation.
struct StablehloData : public OpData {
  enum InputTensorId { kInput, kInitValue, kNumInputTensors };
  enum OutputTensorId { kOutput, kNumOutputTensors };

  using OpData::OpData;

  // TfLiteTensor* GetTemporary(int id) {
  //   return tflite::GetTemporary(context, node, id);
  // }

  TfLiteStatus Check() const {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), kNumInputTensors);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), kNumOutputTensors);
    const TfLiteTensor* const input_tensor = GetInput(context, node, kInput);
    const TfLiteTensor* const output_tensor = GetOutput(context, node, kOutput);
    const TfLiteTensor* const init_value_tensor =
        GetInput(context, node, kInitValue);
    TF_LITE_ENSURE_EQ(context, input_tensor->type, output_tensor->type);
    TF_LITE_ENSURE_EQ(context, input_tensor->type, init_value_tensor->type);
    TF_LITE_ENSURE(context, input_tensor->dims != nullptr);
    TF_LITE_ENSURE(context, input_tensor->dims->size > 0);
    TF_LITE_ENSURE(context, input_tensor->dims->size <= TFLITE_STABLEHLO_REDUCE_PARAMS_MAX_DIMENSION_COUNT);
    return kTfLiteOk;
  }

  TfLiteStatus Initialize() {
    TF_LITE_ENSURE_OK(context, InitializeBase<StablehloData>());
    const auto& params = *reinterpret_cast<TfLiteStablehloReduceParams*>(
        node->builtin_data);
    dimensions = params.dimensions;
    auto AllGtThanZero = [&](const int64_t* const attr) {
      return std::all_of(attr, attr + rank, [](int64_t d) { return d >= 0; });
    };
    TF_LITE_ENSURE(context, AllGtThanZero(dimensions));

    return kTfLiteOk;
  }

  // Sets up the temporary and output tensors and the sub-ops to dilate, pad,
  // crop and reduce.
  //
  // This should be called during Prepare.
  TfLiteStatus Setup() {
    NodeData& node_data = *reinterpret_cast<NodeData*>(node->user_data);

    node_data.body = GetBodyFunction();
    std::cout << "Came here\n\n";
    TfLiteTensor* const output_tensor = GetOutput(context, node, kOutput);
    return kTfLiteOk;
  }

  // Inspects the subgraph associated to the STABLEHLO_REDUCE node to
  // find out the reduction body.
  TfLiteReduceFunction GetBodyFunction() {
    const TfLiteStablehloReduceParams& params =
        *reinterpret_cast<TfLiteStablehloReduceParams*>(
            node->builtin_data);
    const int body_subgraph_index = params.body_subgraph_index;
    const Subgraph& parent_subgraph =
        *reinterpret_cast<Subgraph*>(context->impl_);
    const std::vector<std::unique_ptr<Subgraph>>& subgraphs =
        *parent_subgraph.GetSubgraphs();
    if (body_subgraph_index >= subgraphs.size()) {
      TF_LITE_KERNEL_LOG(
          context, "Body subgraph not found for stablehlo.reduce_window: %d.",
          body_subgraph_index);
      return TfLiteReduceFunctionUnsupported;
    }
    const Subgraph& body_subgraph = *subgraphs[body_subgraph_index];
    const std::vector<int>& execution_plan =
        body_subgraph.pre_delegation_execution_plan().empty()
            ? body_subgraph.execution_plan()
            : body_subgraph.pre_delegation_execution_plan();

    if (execution_plan.size() != 1) {
      TF_LITE_KERNEL_LOG(context,
                         "Only one kernel is allowed within "
                         "stablehlo.reduce_window body. (%zu) kernels found.\n",
                         execution_plan.size());
      return TfLiteReduceFunctionUnsupported;
    }
    const int body_kernel_index = execution_plan[0];
    const TfLiteRegistration& body_kernel_registration =
        body_subgraph.node_and_registration(body_kernel_index)->second;
    switch (body_kernel_registration.builtin_code) {
      case kTfLiteBuiltinAdd:
      case kTfLiteBuiltinStablehloAdd:
        return TfLiteReduceFunctionAdd;
      case kTfLiteBuiltinMul:
      case kTfLiteBuiltinStablehloMultiply:
        return TfLiteReduceFunctionMul;
      case kTfLiteBuiltinMaximum:
      case kTfLiteBuiltinStablehloMaximum:
        return TfLiteReduceFunctionMax;
      case kTfLiteBuiltinMinimum:
      case kTfLiteBuiltinStablehloMinimum:
        return TfLiteReduceFunctionMin;
      case kTfLiteBuiltinLogicalAnd:
      case kTfLiteBuiltinStablehloAnd:
        return TfLiteReduceFunctionAll;
      case kTfLiteBuiltinLogicalOr:
      case kTfLiteBuiltinStablehloOr:
        return TfLiteReduceFunctionAny;
      default:
        TF_LITE_KERNEL_LOG(
            context, "%s:%d unsupported reduction body builtin code: %d.\n",
            __FILE__, __LINE__, body_kernel_registration.builtin_code);
    }
  }
};

// Initializes the node's user data when the STABLEHLO_REDUCE_WINDOW sematic is
// used.
void* StablehloInit(TfLiteContext* context, const char* options,
                    size_t options_len) {
  NodeData* node_data = new NodeData();
  return node_data;
}

// Frees the node's user data when the STABLEHLO_REDUCE_WINDOW sematic is used.
void Free(TfLiteContext* context, void* node_data) {
  delete static_cast<NodeData*>(node_data);
}

template <class Semantic>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  Semantic ctx(context, node);
  TF_LITE_ENSURE_OK(context, ctx.Check());
  TF_LITE_ENSURE_OK(context, ctx.Initialize());
  return ctx.Setup();
}

struct Max {
  template <class T>
  constexpr T operator()(const T& a, const T& b) const {
    return a >= b ? a : b;
  }
};

struct Min {
  template <class T>
  constexpr T operator()(const T& a, const T& b) const {
    return a <= b ? a : b;
  }
};

template <typename T,ReduceType reduce_type>
TfLiteStatus EvalType(const OpData& op_context) {
  // int64_t num_axis = NumElements(op_context.dimensions);
  int64_t num_axis = TFLITE_STABLEHLO_REDUCE_PARAMS_MAX_DIMENSION_COUNT;
  // TfLiteTensor* temp_index;
  // TF_LITE_ENSURE_OK(op_context.context,
  //                   GetTemporarySafe(op_context.context, op_context.node, /*index=*/0, &temp_index));
   TfLiteTensor* temp_index =
          tflite::GetTemporary(op_context.context, op_context.node, 0);
  TfLiteTensor* resolved_axis =
          tflite::GetTemporary(op_context.context, op_context.node, 1);
  std::cout << "Inside Eval Type\n\n";
  // TfLiteTensor* resolved_axis;
  // TF_LITE_ENSURE_OK(
  //     op_context.context, GetTemporarySafe(op_context.context, op_context.node, /*index=*/1, &resolved_axis));
  const TfLiteTensor* input = op_context.input;
  // if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
  //     input->type == kTfLiteInt16) {
  //   TF_LITE_ENSURE_EQ(op_context.context, input->params.scale,
  //                     op_context.output->params.scale);
  //   TF_LITE_ENSURE_EQ(op_context.context, input->params.zero_point,
  //                     op_context.output->params.zero_point);
  // }
  std::cout << "Before init_value\n\n";
  const TfLiteTensor* init_value_tensor = op_context.init_value;
  // T init_value_data = GetTensorData<T>(init_value_tensor);
  T init_value = 0;
  T (*reducer)(const T current, const T in);
  switch (reduce_type) {
    case kSum:
      reducer = [](const T current, const T in) -> T { return in + current; };
      init_value = T(0);
      break;
    case kProd:
      init_value = static_cast<T>(1);
      reducer = [](const T current, const T in) -> T { return in * current; };
      break;
    case kMax:
      init_value = std::numeric_limits<T>::lowest();
      reducer = [](const T current, const T in) -> T {
        return (in > current) ? in : current;
      };
      break;
    case kMin:
      init_value = std::numeric_limits<T>::max();
      reducer = [](const T current, const T in) -> T {
        return (in < current) ? in : current;
      };
      break;
    case kAny:
      init_value = false;
      reducer = [](const T current, const T in) -> T {
        return in || current;
      };
      break;
    case kAll:
      init_value = true;
      reducer = [](const T current, const T in) -> T {
        return in && current;
      };
      break;
    default:
      TF_LITE_KERNEL_LOG(op_context.context, "Unsupported ReduceType: %d", reduce_type);
      return kTfLiteError;
  }
  int dimension_axis[8];
  for (int i=0;i<8;++i) {
    dimension_axis[i] = static_cast<int>(op_context.dimensions[i]); 
  }  
  int num_resolved_axis = 0;
  // TF_LITE_ENSURE_MSG(
  //     op_context.context,
  //     tflite::reference_ops::ResolveAxis(
  //         input->dims->size, dimension_axis, num_axis,
  //         GetTensorData<int>(resolved_axis), &num_resolved_axis),
  //     "Invalid axis index.");

  // if (IsReduceAllDims(resolved_axis, num_resolved_axis, input->dims->size)) {
  //   ReduceAllDims(GetTensorData<T>(input), input->dims->data,
  //                 input->dims->size, GetTensorData<T>(op_context.output),
  //                 init_value_data, reducer, op_context.context);
  //   return kTfLiteOk;
  // }
  TF_LITE_ENSURE(
      op_context.context,
      reference_ops::ReduceGeneric<T>(
          GetTensorData<T>(input), input->dims->data, input->dims->size,
          GetTensorData<T>(op_context.output),
          op_context.output->dims->data, op_context.output->dims->size,
          dimension_axis, num_axis,
          false, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), init_value, reducer));
  std::cout << op_context.output->dims->data << std::endl << op_context.output->dims->size;
  return kTfLiteOk;
}

template<ReduceType reduce_type>
TfLiteStatus EvalGeneric(const OpData& op_ctx) {
  switch (op_ctx.type) {
    case kTfLiteFloat32:
      return EvalType<float, reduce_type>(op_ctx);
      break;
    case kTfLiteInt32:
      return EvalType<int, reduce_type>(op_ctx);
      break;
    case kTfLiteInt64:
      return EvalType<int64_t, reduce_type>(op_ctx);
      break;
    case kTfLiteUInt8:
      return EvalType<uint8_t, reduce_type>(op_ctx);
      break;
    case kTfLiteInt8:
      return EvalType<int8_t, reduce_type>(op_ctx);
      break;
    case kTfLiteInt16:
      return EvalType<int16_t, reduce_type>(op_ctx);
      break;
    case kTfLiteBool:
      return EvalType<bool, reduce_type>(op_ctx);
      break;
    default:
      return kTfLiteError;
  }
}

// // Dispatches to the template implementation according to the tensor type.
template <ReduceType reduce_type>
TfLiteStatus DispatchReduceType(OpData& ctx) {
#define REDUCE_TYPE_CASE(CPP_TYPE, TENSOR_TYPE) \
  case TENSOR_TYPE:                                    \
    EvalGeneric<reduce_type>(ctx);            \
    break;
  switch (ctx.type) {
    REDUCE_TYPE_CASE(bool, kTfLiteBool);
    REDUCE_TYPE_CASE(int8_t, kTfLiteInt8);
    REDUCE_TYPE_CASE(int16_t, kTfLiteInt16);
    REDUCE_TYPE_CASE(int32_t, kTfLiteInt32);
    REDUCE_TYPE_CASE(int64_t, kTfLiteInt64);
    REDUCE_TYPE_CASE(uint8_t, kTfLiteUInt8);
    REDUCE_TYPE_CASE(float, kTfLiteFloat32);
    REDUCE_TYPE_CASE(double, kTfLiteFloat64);
    default:
      TF_LITE_KERNEL_LOG(
          ctx.context,
          "%s:%d unsupported kernel data type (TfliteType: %d a.k.a %s).",
          __FILE__, __LINE__, ctx.type, TfLiteTypeGetName(ctx.type));
      return kTfLiteError;
  }
#undef REDUCE_WINDOW_TYPE_CASE
  return kTfLiteOk;
}

// Dispatches to the template instanciation according to the reduction body.
TfLiteStatus DispatchReduceWindowBody(OpData& ctx) {
  const NodeData& node_data = *static_cast<NodeData*>(ctx.node->user_data);
  switch (node_data.body) {
    case TfLiteReduceFunctionUnsupported:
      TF_LITE_KERNEL_LOG(ctx.context, "%s:%d unsupported reduction body.\n",
                         __FILE__, __LINE__);
      return kTfLiteError;
    case TfLiteReduceFunctionAdd:
      return DispatchReduceType<ReduceType::kSum>(ctx);
    case TfLiteReduceFunctionMul:
      return DispatchReduceType<ReduceType::kProd>(ctx);
    case TfLiteReduceFunctionAll:
      return DispatchReduceType<ReduceType::kAll>(ctx);
    case TfLiteReduceFunctionAny:
      return DispatchReduceType<ReduceType::kAny>(ctx);
    case TfLiteReduceFunctionMin:
      return DispatchReduceType<ReduceType::kMin>(ctx);
    case TfLiteReduceFunctionMax:
      return DispatchReduceType<ReduceType::kMax>(ctx);
  }
  TF_LITE_KERNEL_LOG(ctx.context, "%s:%d unhandled reduction body case.\n",
                     __FILE__, __LINE__);
  return kTfLiteError;
}

template <class Semantic>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  Semantic ctx(context, node);
  TF_LITE_ENSURE_OK(context, ctx.Initialize());
  // Too much cropping can lead to a negative dimension.
  //
  // This never happens with the REDUCE_WINDOW (TFLiteData) semantic but since
  // that op is deprecated we don't care about the extra check.
  NodeData& node_data = *reinterpret_cast<NodeData*>(node->user_data);

  return DispatchReduceWindowBody(ctx);
}

}  // namespace reduce_op

TfLiteRegistration* Register_STABLEHLO_REDUCE2() {
  static TfLiteRegistration r = {
      /*.init=*/reduce_op::StablehloInit,
      /*.free=*/reduce_op::Free,
      /*.prepare=*/reduce_op::Prepare<reduce_op::StablehloData>,
      /*.invoke=*/reduce_op::Eval<reduce_op::StablehloData>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite