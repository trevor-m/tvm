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

#include "trt_builder.h"
#include "trt_logger.h"
#include "trt_ops.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {

// TODO(trevmorr): Make a function to return this
static const std::unordered_map<std::string, TrtOpConverter*>
    trt_op_converters = {
        // Activation ops
        {"nn.relu", new ActivationOpConverter()},
        {"sigmoid", new ActivationOpConverter()},
        {"tanh", new ActivationOpConverter()},
        {"clip", new ActivationOpConverter()},
        {"nn.leaky_relu", new ActivationOpConverter()},
        {"nn.batch_norm", new BatchNormOpConverter()},
        {"nn.softmax", new SoftmaxOpConverter()},
        {"nn.conv2d", new Conv2DOpConverter()},
        // {"conv2d_transpose", AddDeconvolution},
        {"nn.dense", new DenseOpConverter()},
        {"nn.bias_add", new BiasAddOpConverter()},
        {"add", new ElementWiseBinaryOpConverter()},
        {"subtract", new ElementWiseBinaryOpConverter()},
        {"multiply", new ElementWiseBinaryOpConverter()},
        {"divide", new ElementWiseBinaryOpConverter()},
        {"power", new ElementWiseBinaryOpConverter()},
        {"nn.max_pool2d", new PoolingOpConverter()},
        {"nn.avg_pool2d", new PoolingOpConverter()},
        {"nn.global_max_pool2d", new GlobalPoolingOpConverter()},
        {"nn.global_avg_pool2d", new GlobalPoolingOpConverter()},
        {"exp", new UnaryOpConverter()},
        {"log", new UnaryOpConverter()},
        {"sqrt", new UnaryOpConverter()},
        {"abs", new UnaryOpConverter()},
        {"negative", new UnaryOpConverter()},
        {"sin", new UnaryOpConverter()},
        {"cos", new UnaryOpConverter()},
        {"atan", new UnaryOpConverter()},
        {"ceil", new UnaryOpConverter()},
        {"floor", new UnaryOpConverter()},
        {"nn.batch_flatten", new BatchFlattenOpConverter()},
        {"expand_dims", new ExpandDimsOpConverter()},
        {"concatenate", new ConcatOpConverter()},
        // {"slice_like", AddSliceLike},
};

TrtBuilder::TrtBuilder(const std::vector<DLTensor*>& args)
    : execution_args_(args), var_node_counter_(0) {
  // Create TRT builder and network.
  static TensorRTLogger trt_logger;
  builder_ = nvinfer1::createInferBuilder(trt_logger);
  const int batch_size = args[0]->shape[0];
  builder_->setMaxBatchSize(batch_size);
  const size_t workspace_size = size_t(1) << 31;
  builder_->setMaxWorkspaceSize(workspace_size);
  const bool use_fp16 = dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false);
  builder_->setFp16Mode(use_fp16);
  network_ = builder_->createNetwork();
}

TrtEngineAndContext TrtBuilder::BuildEngine(const Expr& expr) {
  // Process graph and create INetworkDefinition.
  VisitExpr(expr);
  // Mark outputs.
  auto network_outputs = node_output_map_[expr.operator->()];
  for (int i = 0; i < network_outputs.size(); ++i) {
    CHECK(network_outputs[i].type == kTensor);
    auto out_tensor = network_outputs[i].tensor;
    std::string output_name = "tensorrt_output" + std::to_string(i);
    out_tensor->setName(output_name.c_str());
    network_->markOutput(*out_tensor);
    DLOG(INFO) << "Added TRT network output: " << out_tensor->getName()
               << " -> " << output_name;
  }
  nvinfer1::ICudaEngine* engine = builder_->buildCudaEngine(*network_);
  CHECK_EQ(engine->getNbBindings(),
           network_input_map_.size() + network_outputs.size());
  CleanUp();
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  return {engine, context, network_input_map_};
}

nvinfer1::Weights TrtBuilder::GetDLTensorAsWeights(DLTensor* dptr,
                                                   DLDeviceType src_device) {
  CHECK_EQ(dptr->ctx.device_type, src_device);
  CHECK_EQ(static_cast<int>(dptr->dtype.code), kDLFloat);
  const size_t weight_bytes = runtime::GetDataSize(*dptr);
  nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, 0};
  weight.values = new uint32_t[weight_bytes];
  size_t count = 1;
  for (tvm_index_t i = 0; i < dptr->ndim; ++i) {
    count *= dptr->shape[i];
  }
  weight.count = count;
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

nvinfer1::Weights TrtBuilder::GetInputAsWeights(const VarNode* node) {
  const int var_node_idx = TrackVarNode(node);
  nvinfer1::Weights weight =
      GetDLTensorAsWeights(execution_args_[var_node_idx], kDLGPU);
  node_output_map_[node] = {TrtOpInput(weight, GetShape(node->checked_type()))};
}

void TrtBuilder::VisitExpr_(const TupleGetItemNode* op) {
  if (const auto* tuple = op->tuple.as<TupleNode>()) {
    Expr item = tuple->fields[op->index];
    VisitExpr(item);
    // TODO(trevmorr): Index into outputs?
    node_output_map_[op] = node_output_map_[item.operator->()];
  } else {
    VisitExpr(op->tuple);
    // TODO(trevmorr): Index into outputs?
    node_output_map_[op] = node_output_map_[op->tuple.operator->()];
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
  auto it = trt_inputs_.find(tensor_name);
  if (it == trt_inputs_.end()) {
    auto shape = GetShape(node->checked_type(), /*remove_batch_dim=*/true);
    DLOG(INFO) << "Added TRT network input: " << node->name_hint() << " "
               << DebugString(shape);
    nvinfer1::Dims dims = VectorToTrtDims(shape);
    auto type = GetType(node->checked_type());
    CHECK(type.is_float()) << "Only FP32 inputs are supported.";
    trt_inputs_[tensor_name] = network_->addInput(
        tensor_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
    network_input_map_[id] = tensor_name;
  } else {
    LOG(WARNING) << "Found same input twice: " << tensor_name;
  }

  node_output_map_[node] = {TrtOpInput(trt_inputs_[tensor_name])};
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
  auto it = trt_op_converters.find(params.op_name);
  CHECK(it != trt_op_converters.end())
      << "Unsupported operator conversion to TRT, op name: " << params.op_name;
  const TrtOpConverter* converter = it->second;

  // Ensure that nodes are processed in topological order by visiting their
  // inputs first.
  for (int i = 0; i < call->args.size(); ++i) {
    // Handle special case where input must be constant array on CPU.
    if (!converter->variable_input_count &&
        converter->input_types[i] == kWeight) {
      // Input must be a constant weight
      if (auto* var = call->args[i].as<VarNode>()) {
        GetInputAsWeights(var);
      } else if (call->args[i].as<ConstantNode>()) {
        LOG(FATAL) << "Not implemented.";
      } else {
        LOG(FATAL) << "TRT requires a constant input here.";
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
  converter->Convert(params);

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
  // int var_node_idx = var_node_counter_++;
  var_node_input_map_[node] = var_node_idx;
  return var_node_idx;
}

void TrtBuilder::CleanUp() {
  network_->destroy();
  for (auto weight : trt_weights_) {
    if (weight.type == nvinfer1::DataType::kFLOAT) {
      delete[] static_cast<const uint32_t*>(weight.values);
    } else {
      delete[] static_cast<const uint16_t*>(weight.values);
    }
  }
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
