/* * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <memory>
#include <string>

#include "trt_builder.h"
#include "trt_ops.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {

const std::unordered_map<std::string, std::shared_ptr<TrtOpConverter>>*
GetOpConverters() {
  static auto* const map =
      new std::unordered_map<std::string, std::shared_ptr<TrtOpConverter>>({
        {"nn.relu", std::make_shared<ActivationOpConverter>()},
            {"sigmoid", std::make_shared<ActivationOpConverter>()},
            {"tanh", std::make_shared<ActivationOpConverter>()},
            {"nn.batch_norm", std::make_shared<BatchNormOpConverter>()},
            {"nn.softmax", std::make_shared<SoftmaxOpConverter>()},
            {"nn.conv2d", std::make_shared<Conv2DOpConverter>()},
            {"nn.dense", std::make_shared<DenseOpConverter>()},
            {"nn.bias_add", std::make_shared<BiasAddOpConverter>()},
            {"add", std::make_shared<ElementWiseBinaryOpConverter>()},
            {"subtract", std::make_shared<ElementWiseBinaryOpConverter>()},
            {"multiply", std::make_shared<ElementWiseBinaryOpConverter>()},
            {"divide", std::make_shared<ElementWiseBinaryOpConverter>()},
            {"power", std::make_shared<ElementWiseBinaryOpConverter>()},
            {"nn.max_pool2d", std::make_shared<PoolingOpConverter>()},
            {"nn.avg_pool2d", std::make_shared<PoolingOpConverter>()},
            {"nn.global_max_pool2d",
             std::make_shared<GlobalPoolingOpConverter>()},
            {"nn.global_avg_pool2d",
             std::make_shared<GlobalPoolingOpConverter>()},
            {"exp", std::make_shared<UnaryOpConverter>()},
            {"log", std::make_shared<UnaryOpConverter>()},
            {"sqrt", std::make_shared<UnaryOpConverter>()},
            {"abs", std::make_shared<UnaryOpConverter>()},
            {"negative", std::make_shared<UnaryOpConverter>()},
            {"nn.batch_flatten", std::make_shared<BatchFlattenOpConverter>()},
            {"expand_dims", std::make_shared<ExpandDimsOpConverter>()},
            {"squeeze", std::make_shared<SqueezeOpConverter>()},
            {"concatenate", std::make_shared<ConcatOpConverter>()},
            {"nn.conv2d_transpose",
             std::make_shared<Conv2DTransposeOpConverter>()},
            {"transpose", std::make_shared<TransposeOpConverter>()},
            {"reshape", std::make_shared<ReshapeOpConverter>()},
            {"nn.pad", std::make_shared<PadOpConverter>()},
            {"sum", std::make_shared<ReduceOpConverter>()},
            {"prod", std::make_shared<ReduceOpConverter>()},
            {"max", std::make_shared<ReduceOpConverter>()},
            {"min", std::make_shared<ReduceOpConverter>()},
            {"mean", std::make_shared<ReduceOpConverter>()},
            {"contrib.adaptive_max_pool2d",
             std::make_shared<AdaptivePoolingOpConverter>()},
            {"contrib.adaptive_avg_pool2d",
             std::make_shared<AdaptivePoolingOpConverter>()},
#if TRT_VERSION_GE(5, 1, 5)
            {"clip", std::make_shared<ActivationOpConverter>()},
            {"nn.leaky_relu", std::make_shared<ActivationOpConverter>()},
            {"sin", std::make_shared<UnaryOpConverter>()},
            {"cos", std::make_shared<UnaryOpConverter>()},
            {"atan", std::make_shared<UnaryOpConverter>()},
            {"ceil", std::make_shared<UnaryOpConverter>()},
            {"floor", std::make_shared<UnaryOpConverter>()},
            {"strided_slice", std::make_shared<StridedSliceOpConverter>()},
#endif
#if TRT_VERSION_GE(6, 0, 1)
            {"image.resize", std::make_shared<ResizeOpConverter>()},
#endif
      });
  return map;
}

TrtBuilder::TrtBuilder(const std::vector<DLTensor*>& args)
    : execution_args_(args) {
  // Create TRT builder and network.
  builder_ = nvinfer1::createInferBuilder(logger_);
  batch_size_ = args[0]->shape[0];
  builder_->setMaxBatchSize(batch_size_);
  const size_t workspace_size =
      dmlc::GetEnv("TVM_TENSORRT_MAX_WORKSPACE_SIZE", size_t(1) << 31);
  builder_->setMaxWorkspaceSize(workspace_size);
  const bool use_fp16 = dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false);
  builder_->setFp16Mode(use_fp16);
  network_ = builder_->createNetwork();
}

TrtEngineAndContext TrtBuilder::BuildEngine(const Expr& expr) {
  // Process graph and create INetworkDefinition.
  VisitExpr(expr);
  // Mark outputs.
  auto it = node_output_map_.find(expr.operator->());
  CHECK(it != node_output_map_.end()) << "Output was not found.";
  auto network_outputs = it->second;
  std::vector<std::string> network_output_names;
  for (int i = 0; i < network_outputs.size(); ++i) {
    CHECK(network_outputs[i].type == kTensor);
    auto out_tensor = network_outputs[i].tensor;
    std::string output_name = "tensorrt_output" + std::to_string(i);
    out_tensor->setName(output_name.c_str());
    network_output_names.push_back(output_name);
    network_->markOutput(*out_tensor);
    DLOG(INFO) << "Added TRT network output: " << out_tensor->getName()
               << " -> " << output_name;
  }
  nvinfer1::ICudaEngine* engine = builder_->buildCudaEngine(*network_);
  CHECK_EQ(engine->getNbBindings(),
           network_input_map_.size() + network_outputs.size());
  CleanUp();
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  return {engine, context, network_input_map_, network_output_names};
}

nvinfer1::Weights TrtBuilder::GetDLTensorAsWeights(DLTensor* dptr,
                                                   DLDeviceType src_device) {
  CHECK_EQ(dptr->ctx.device_type, src_device);
  CHECK_EQ(static_cast<int>(dptr->dtype.code), kDLFloat);
  const size_t weight_bytes = runtime::GetDataSize(*dptr);
  nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, 0};
  size_t count = 1;
  for (tvm_index_t i = 0; i < dptr->ndim; ++i) {
    count *= dptr->shape[i];
  }
  CHECK_EQ(count * 4, weight_bytes);
  weight.count = count;
  weight.values = new float[count];
  CHECK_EQ(
      TVMArrayCopyToBytes(dptr, const_cast<void*>(weight.values), weight_bytes),
      0)
      << TVMGetLastError();
  trt_weights_.push_back(weight);
  return weight;
}

nvinfer1::Weights TrtBuilder::GetNdArrayAsWeights(const runtime::NDArray& array,
                                                  DLDeviceType src_device) {
  DLTensor* dptr = const_cast<DLTensor*>(array.operator->());
  return GetDLTensorAsWeights(dptr, src_device);
}

void TrtBuilder::GetInputAsWeights(const VarNode* node) {
  const int var_node_idx = TrackVarNode(node);
  nvinfer1::Weights weight =
      GetDLTensorAsWeights(execution_args_[var_node_idx], kDLGPU);
  node_output_map_[node] = {TrtOpInput(weight, GetShape(node->checked_type()))};
}

void TrtBuilder::GetConstantAsWeights(const ConstantNode* node) {
  auto weight = GetNdArrayAsWeights(node->data, kDLCPU);
  auto shape_long = node->data.Shape();
  std::vector<int> shape(shape_long.begin(), shape_long.end());
  node_output_map_[node] = {TrtOpInput(weight, shape)};
}

void TrtBuilder::GetInputAsTransposedWeights(const CallNode* transpose,
                                             const VarNode* node) {
  GetInputAsWeights(node);
  CHECK_EQ(node_output_map_[node].size(), 1);
  const nvinfer1::Weights& original_weight = node_output_map_[node][0].weight;
  const auto& original_shape = node_output_map_[node][0].weight_shape;
  float* values = new float[original_weight.count];
  // Get order and new shape.
  const auto* attrs = transpose->attrs.as<TransposeAttrs>();
  std::vector<int> order;
  std::vector<int> new_shape;
  for (size_t i = 0; i < attrs->axes.size(); i++) {
    const int axis = attrs->axes[i].as<IntImm>()->value;
    order.push_back(axis);
    new_shape.push_back(original_shape[axis]);
  }
  // Perform transpose.
  if (order.size() == 4 && order[0] == 3 && order[1] == 2 && order[2] == 0 &&
      order[3] == 1) {
    TransposeRSCKtoKCRS(original_shape,
                        static_cast<const float*>(original_weight.values),
                        values);
  } else if (order.size() == 4 && order[0] == 2 && order[1] == 3 &&
             order[2] == 0 && order[3] == 1) {
    TransposeRSCKtoCKRS(original_shape,
                        static_cast<const float*>(original_weight.values),
                        values);
  } else if (order.size() == 2 && order[0] == 1 && order[1] == 0) {
    TransposeCKtoKC(original_shape,
                    static_cast<const float*>(original_weight.values), values);
  } else {
    LOG(FATAL) << "Constant transpose " << DebugString(order)
               << " is not supported.";
  }
  // Map as output of transpose op.
  nvinfer1::Weights transposed_weight{nvinfer1::DataType::kFLOAT, values,
                                      original_weight.count};
  trt_weights_.push_back(transposed_weight);
  node_output_map_[transpose] = {TrtOpInput(transposed_weight, new_shape)};
}

void TrtBuilder::VisitExpr_(const TupleGetItemNode* op) {
  if (const auto* tuple = op->tuple.as<TupleNode>()) {
    Expr item = tuple->fields[op->index];
    VisitExpr(item);
    node_output_map_[op] = node_output_map_[item.operator->()];
  } else {
    VisitExpr(op->tuple);
    // Index into tensor outputs from expr.
    node_output_map_[op] = {
        node_output_map_[op->tuple.operator->()][op->index]};
  }
}

void TrtBuilder::VisitExpr_(const TupleNode* op) {
  std::vector<TrtOpInput> outputs;
  for (auto item : op->fields) {
    VisitExpr(item);
    auto item_outputs = node_output_map_[item.operator->()];
    outputs.reserve(outputs.size() + item_outputs.size());
    outputs.insert(outputs.end(), item_outputs.begin(), item_outputs.end());
  }
  node_output_map_[op] = outputs;
}

void TrtBuilder::VisitExpr_(const VarNode* node) {
  const int id = TrackVarNode(node);

  const std::string& tensor_name = node->name_hint();
  auto shape = GetShape(node->checked_type(), /*remove_batch_dim=*/true);
  DLOG(INFO) << "Added TRT network input: " << node->name_hint() << " "
             << DebugString(shape);
  nvinfer1::Dims dims = VectorToTrtDims(shape);
  auto type = GetType(node->checked_type());
  CHECK(type.is_float()) << "Only FP32 inputs are supported.";
  auto input =
      network_->addInput(tensor_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
  network_input_map_[id] = tensor_name;
  node_output_map_[node] = {TrtOpInput(input)};
}

void TrtBuilder::VisitExpr_(const ConstantNode* node) {
  nvinfer1::Weights weight = GetNdArrayAsWeights(node->data, kDLCPU);
  nvinfer1::Dims dims = VectorToTrtDims(node->data.Shape());
  auto const_layer = network_->addConstant(dims, weight);
  CHECK(const_layer != nullptr);
  node_output_map_[node] = {TrtOpInput(const_layer->getOutput(0))};
}

void TrtBuilder::VisitExpr_(const CallNode* call) {
  AddTrtLayerParams params(network_, call);
  // Look up converter.
  auto it = GetOpConverters()->find(params.op_name);
  CHECK(it != GetOpConverters()->end())
      << "Unsupported operator conversion to TRT, op name: " << params.op_name;
  const auto converter = it->second;

  // Ensure that nodes are processed in topological order by visiting their
  // inputs first.
  for (int i = 0; i < call->args.size(); ++i) {
    // Handle special case where input must be constant array on CPU.
    if (!converter->variable_input_count &&
        converter->input_types[i] == kWeight) {
      // Input must be a constant weight
      if (auto* var = call->args[i].as<VarNode>()) {
        GetInputAsWeights(var);
      } else if (auto* node = call->args[i].as<ConstantNode>()) {
        GetConstantAsWeights(node);
      } else {
        // Temporary workaround for transposed weights. Once partitioning is
        // available, the transpose will be computed by tvm and the result will
        // be a var input.
        if (auto* transpose = call->args[i].as<CallNode>()) {
          if (transpose->op.as<OpNode>()->name == "transpose") {
            if (auto* weights = transpose->args[0].as<VarNode>()) {
              GetInputAsTransposedWeights(transpose, weights);
            } else {
              LOG(FATAL) << "TRT requires a constant input here.";
            }
          } else {
            LOG(FATAL) << "TRT requires a constant input here.";
          }
        } else {
          LOG(FATAL) << "TRT requires a constant input here.";
        }
      }
    } else {
      VisitExpr(call->args[i]);
    }
  }

  // Get inputs.
  for (int i = 0; i < call->args.size(); ++i) {
    auto it = node_output_map_.find(call->args[i].operator->());
    CHECK(it != node_output_map_.end()) << "Input was not found.";
    for (auto out : it->second) {
      params.inputs.push_back(out);
    }
  }
  if (!converter->variable_input_count) {
    CHECK_EQ(converter->input_types.size(), params.inputs.size())
        << "Op expected a different number of inputs.";
  }

  // Convert op to TRT.
  converter->Convert(&params);

  // Get outputs.
  node_output_map_[call] = {};
  std::vector<TrtOpInput> outputs;
  for (auto out : params.outputs) {
    node_output_map_[call].push_back(TrtOpInput(out));
  }
}

int TrtBuilder::TrackVarNode(const VarNode* node) {
  // TODO(trevmorr): make more robust
  const int trim_length = std::string("tensorrt_input").length();
  int var_node_idx =
      std::stoi(node->name_hint().substr(trim_length, std::string::npos));
  return var_node_idx;
}

void TrtBuilder::CleanUp() {
  network_->destroy();
  for (auto weight : trt_weights_) {
    if (weight.type == nvinfer1::DataType::kFLOAT) {
      delete[] static_cast<const float*>(weight.values);
    } else {
      delete[] static_cast<const uint16_t*>(weight.values);
    }
  }
}

void TransposeRSCKtoKCRS(const std::vector<int>& original_shape,
                         const float* input_values, float* output_values) {
  const int r = original_shape[0];
  const int s = original_shape[1];
  const int c = original_shape[2];
  const int k = original_shape[3];
  for (int x = 0; x < k; x++) {
    for (int y = 0; y < c; y++) {
      for (int z = 0; z < r; z++) {
        for (int w = 0; w < s; w++) {
          const int input_index = (x) + (y * k) + (z * s * c * k) + (w * c * k);
          const int output_index =
              (x * c * r * s) + (y * r * s) + (z * s) + (w);
          output_values[output_index] = input_values[input_index];
        }
      }
    }
  }
}

void TransposeRSCKtoCKRS(const std::vector<int>& original_shape,
                         const float* input_values, float* output_values) {
  const int r = original_shape[0];
  const int s = original_shape[1];
  const int c = original_shape[2];
  const int k = original_shape[3];
  for (int x = 0; x < k; x++) {
    for (int y = 0; y < c; y++) {
      for (int z = 0; z < r; z++) {
        for (int w = 0; w < s; w++) {
          const int input_index = (x) + (y * k) + (z * s * c * k) + (w * c * k);
          const int output_index =
              (y * k * r * s) + (x * r * s) + (z * s) + (w);
          output_values[output_index] = input_values[input_index];
        }
      }
    }
  }
}

void TransposeCKtoKC(const std::vector<int>& original_shape,
                     const float* input_values, float* output_values) {
  const int c = original_shape[0];
  const int k = original_shape[1];
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < k; j++) {
      const int input_index = i * k + j;
      const int output_index = j * c + i;
      output_values[output_index] = input_values[input_index];
    }
  }
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
