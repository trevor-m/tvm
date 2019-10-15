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
#include "utils.h"
#include "trt_ops.h"

namespace tvm {
namespace relay {
namespace contrib {

static size_t GetTensorSize(const DLTensor* arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  return size;
}

static size_t GetTensorBytes(const DLTensor* arr) {
  size_t size = GetTensorSize(arr);
  size *= (arr->dtype.bits * arr->dtype.lanes + 7) / 8;
  return size;
}

// TODO(trevmorr): Make a function to return this
static const std::unordered_map<std::string, TrtOpConverter*> trt_op_converters =
    {
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
        {"nn.bias_add", new ElementWiseBinaryOpConverter()}, // {"nn.bias_add", new BiasAddOpConverter()},
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
        {"exp", new UnaryOpConverter()},
        {"abs", new UnaryOpConverter()},
        {"negative", new UnaryOpConverter()},
        {"sin", new UnaryOpConverter()},
        {"cos", new UnaryOpConverter()},
        {"atan", new UnaryOpConverter()},
        {"ceil", new UnaryOpConverter()},
        {"floor", new UnaryOpConverter()},
        {"nn.batch_flatten", new BatchFlattenOpConverter()},
        {"expand_dims", new ExpandDimsOpConverter()},
        // {"concatenate", AddConcatenate},
        // {"slice_like", AddSliceLike},
};

TrtBuilder::TrtBuilder(tvm::TVMArgs args) : execution_args_(args), var_node_counter_(0) {
  // Create TRT builder and network.
  static TensorRTLogger trt_logger;
  builder_ = nvinfer1::createInferBuilder(trt_logger);
  builder_->setMaxBatchSize(1);
  builder_->setMaxWorkspaceSize(1 << 29);
  // builder_->setFp16Mode(use_fp16_);
  network_ = builder_->createNetwork();
  // network_ = builder_->createNetworkV2(1U <<
  //       static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
}

TrtEngineAndContext TrtBuilder::BuildEngine(const Expr& expr) {
  // Process graph and create INetworkDefinition.
  VisitExpr(expr);
  // Mark outputs.
  for (int i = 0; i < out_tensors_.size(); ++i) {
    std::string output_name = "tensorrt_output" + std::to_string(i);
    LOG(INFO) << "Added TRT network output: " << out_tensors_[i]->getName() << " -> " << output_name;
    out_tensors_[i]->setName(output_name.c_str());
    network_->markOutput(*out_tensors_[i]);
  }
  nvinfer1::ICudaEngine* engine = builder_->buildCudaEngine(*network_);
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  // TODO(trevmorr): add more validations
  CHECK_EQ(engine->getNbBindings(), network_input_map_.size() + out_tensors_.size());
  return {engine, context, network_input_map_};
}

nvinfer1::Weights TrtBuilder::GetInputAsWeights(const VarNode* node) {
  const int var_node_idx = var_node_input_map_[node];
  runtime::NDArray arg = execution_args_[var_node_idx];
  DLTensor* dptr = const_cast<DLTensor*>(arg.operator->());
  CHECK_EQ(dptr->ctx.device_type, kDLGPU);
  CHECK_EQ(static_cast<int>(dptr->dtype.code), kDLFloat);
  const size_t weight_bytes = GetTensorBytes(dptr);
  nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
  // TODO(trevmorr): manage memory properly
  wt.values = malloc(weight_bytes);
  wt.count = GetTensorSize(dptr);
  CHECK_EQ(TVMArrayCopyToBytes(dptr, const_cast<void*>(wt.values), weight_bytes), 0)
      << TVMGetLastError();
  return wt;
}

void TrtBuilder::VisitExpr_(const VarNode* node) {
  const int id = TrackVarNode(node);

  const std::string& tensor_name = node->name_hint();
  auto it = trt_inputs_.find(tensor_name);
  if (it == trt_inputs_.end()) {
    auto shape = GetShape(node->checked_type(), /*remove_batch_dim=*/true);
    LOG(INFO) << "Added TRT network input " << node->name_hint() << " " << DebugString(shape);
    nvinfer1::Dims dims = VectorToTrtDims(shape);
    auto type = GetType(node->checked_type());
    CHECK(type.is_float()) << "Only FP32 inputs are supported.";
    trt_inputs_[tensor_name] =
        network_->addInput(tensor_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
    network_input_map_[id] = tensor_name;
  }

  out_tensors_.clear();
  out_tensors_.push_back(trt_inputs_[tensor_name]);
}

void TrtBuilder::VisitExpr_(const ConstantNode* node) {
  // TODO(trevmorr)
  runtime::NDArray arg = node->data;
  DLTensor* dptr = const_cast<DLTensor*>(arg.operator->());
  // CHECK_EQ(dptr->ctx.device_type, kDLGPU);
  CHECK_EQ(static_cast<int>(dptr->dtype.code), kDLFloat);
  const size_t weight_bytes = GetTensorBytes(dptr);
  nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
  // TODO(trevmorr): manage memory properly
  wt.values = malloc(weight_bytes);
  wt.count = GetTensorSize(dptr);
  CHECK_EQ(TVMArrayCopyToBytes(dptr, const_cast<void*>(wt.values), weight_bytes), 0)
      << TVMGetLastError();

  nvinfer1::Dims dims;
  dims.d[0] = 1; // in case of scalar
  dims.nbDims = dptr->ndim;
  for (int i = 0; i < dims.nbDims; ++i) dims.d[i] = dptr->shape[i];
  LOG(INFO) << "Adding constant with shape " << DebugString(TrtDimsToVector(dims));
  auto const_layer = network_->addConstant(dims, wt);

  out_tensors_.clear();
  out_tensors_.push_back(const_layer->getOutput(0));
  // LOG(FATAL) << "not implemented";
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
  CHECK(converter->input_types.size() == call->args.size())
      << "Op expected a different number of inputs.";
  for (int i = 0; i < call->args.size(); ++i) {
    // Handle special case where input must be constant array on CPU.
    if (converter->input_types[i] == kWeight) {
      // Input must be a constant weight
      if (auto* var = call->args[i].as<VarNode>()) {
        LOG(WARNING) << "TRT requires a constant input for input " << i << " of op " << params.op_name << ", but relay.Var was used instead. The value supplied to the Var during this execution will be used for all future executions.";
        // Hack: get from input
        TrackVarNode(var);
        nvinfer1::Weights weights = GetInputAsWeights(var);
        // Include shape
        params.inputs.push_back(TrtOpInput(weights, GetShape(var->checked_type())));
      } else if (call->args[i].as<ConstantNode>()) {
        // TODO
        LOG(FATAL) << "not implemented";
      } else {
        LOG(FATAL) << "TRT requires a constant input here.";
      }
    } else {
      VisitExpr(call->args[i]);
      // Add outputs from args as inputs to this op.
      for (auto out : out_tensors_) {
        params.inputs.push_back(out);
      }
    }
  }

  // Convert op to TRT.
  converter->Convert(params);

  // Get outputs.
  out_tensors_.clear();
  for (auto out : params.outputs) {
    out_tensors_.push_back(out);
  }
}

int TrtBuilder::TrackVarNode(const VarNode* node) {
  int var_node_idx = var_node_counter_++;
  var_node_input_map_[node] = var_node_idx;
  return var_node_idx;
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
