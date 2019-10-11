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
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {

// Logger for TensorRT info/warning/errors
class TensorRTLogger : public nvinfer1::ILogger {
 public:
  TensorRTLogger() : TensorRTLogger(Severity::kWARNING) {}
  explicit TensorRTLogger(Severity severity) : reportable_severity(severity) {}
  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    // if (severity > reportable_severity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR: LOG(ERROR) << "INTERNAL_ERROR: " << msg; break;
      case Severity::kERROR: LOG(ERROR) << "ERROR: " << msg; break;
      case Severity::kWARNING: LOG(WARNING) << "WARNING: " << msg; break;
      case Severity::kINFO: LOG(INFO) << "INFO: " << msg; break;
      case Severity::kVERBOSE: LOG(INFO) << "VERBOSE: " << msg; break;
      default: LOG(INFO) << "UNKNOWN: " << msg; break;
    }
  }

 private:
  Severity reportable_severity{Severity::kWARNING};
};

// Parameters to convert an Op from relay to TensorRT
struct AddTrtLayerParams {
  const CallNode* call;
  nvinfer1::INetworkDefinition* network;
  std::string op_name;
  std::vector<nvinfer1::ITensor*> inputs;
  std::vector<nvinfer1::ITensor*> outputs;

  AddTrtLayerParams(nvinfer1::INetworkDefinition* network) : network(network) {}
};

void AddActivation(AddTrtLayerParams& params) {
  CHECK(params.inputs.size() == 1) << "Activation op expects 1 input.";
  static const std::unordered_map<std::string, nvinfer1::ActivationType>
      op_map = {{"nn.relu", nvinfer1::ActivationType::kRELU},
                {"sigmoid", nvinfer1::ActivationType::kSIGMOID},
                {"tanh", nvinfer1::ActivationType::kTANH}};
  auto it = op_map.find(params.op_name);
  CHECK(it != op_map.end()) << "Unsupported activation type " << params.op_name;
  nvinfer1::IActivationLayer* act_layer = params.network->addActivation(*params.inputs.at(0), nvinfer1::ActivationType::kRELU);
  CHECK(act_layer != nullptr);
  params.outputs.push_back(act_layer->getOutput(0));
}

void AddElementWiseBinaryOp(AddTrtLayerParams& params) {
  CHECK(params.inputs.size() == 2) << "Binary op expects 2 inputs.";
  static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> op_map =
      {{"add", nvinfer1::ElementWiseOperation::kSUM},
       {"subtract", nvinfer1::ElementWiseOperation::kSUB},
       {"multiply", nvinfer1::ElementWiseOperation::kPROD},
       {"divide", nvinfer1::ElementWiseOperation::kDIV},
       {"power", nvinfer1::ElementWiseOperation::kPOW}};
  auto it = op_map.find(params.op_name);
  CHECK(it != op_map.end()) << "Unsupported elementwise type " << params.op_name;
  nvinfer1::IElementWiseLayer* elemwise_layer =
      params.network->addElementWise(*params.inputs.at(0), *params.inputs.at(1), it->second);
  CHECK(elemwise_layer != nullptr);
  params.outputs.push_back(elemwise_layer->getOutput(0));
}

using AddTrtLayer = std::function<void(AddTrtLayerParams& params)>;

static const std::unordered_map<std::string, AddTrtLayer> add_trt_layer_funcs =
    {
        // {"nn.conv2d", AddConvolution},
        {"nn.relu", AddActivation},
        {"sigmoid", AddActivation},
        {"tanh", AddActivation},
        // {"nn.batch_norm", AddBatchNorm},
        {"add", AddElementWiseBinaryOp},
        {"subtract", AddElementWiseBinaryOp},
        {"multiply", AddElementWiseBinaryOp},
        {"divide", AddElementWiseBinaryOp},
        {"power", AddElementWiseBinaryOp},
        // {"clip", AddActivation},
        // {"max_pool2d", AddPooling},
        // {"avg_pool2d", AddPooling},
        // {"global_max_pool2d", AddPooling},
        // {"global_avg_pool2d", AddPooling},
        // {"dense", AddFullyConnected},
        // {"softmax", AddSoftmax},
        // {"concatenate", AddConcatenate},
        // {"conv2d_transpose", AddDeconvolution},
        // {"slice_like", AddSliceLike},
};

TrtBuilder::TrtBuilder() {
  // Create TRT builder and network.
  static TensorRTLogger trt_logger;
  builder_ = nvinfer1::createInferBuilder(trt_logger);
  builder_->setMaxBatchSize(1);
  builder_->setMaxWorkspaceSize(1 << 29);
  // builder_->setFp16Mode(use_fp16_);
  network_ = builder_->createNetwork();
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
  return {engine, context};
}

void TrtBuilder::VisitExpr_(const VarNode* node) {
  const std::string& tensor_name = node->name_hint();
  auto it = trt_inputs_.find(tensor_name);
  if (it == trt_inputs_.end()) {
    LOG(INFO) << "Added TRT network input " << node->name_hint();
    auto shape = GetShape(node->checked_type());
    nvinfer1::Dims dims = VectorToTrtDims(shape);
    auto type = GetType(node->checked_type());
    CHECK(type.is_float()) << "Only FP32 inputs are supported.";
    trt_inputs_[tensor_name] =
        network_->addInput(tensor_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
  }

  out_tensors_.clear();
  out_tensors_.push_back(trt_inputs_[tensor_name]);
}

void TrtBuilder::VisitExpr_(const ConstantNode* node) {
  // TODO(trevmorr)
}

void TrtBuilder::VisitExpr_(const CallNode* call) {
  AddTrtLayerParams params(network_);

  // Ensure that nodes are processed in topological order by visiting their
  // inputs first.
  for (int i = 0; i < call->args.size(); ++i) {
    VisitExpr(call->args[i]);
    // Add outputs from args as inputs to this op.
    for (auto out : out_tensors_) {
      params.inputs.push_back(out);
    }
  }

  // Get op name and look up conversion function.
  params.op_name = (call->op.as<OpNode>())->name;
  LOG(INFO) << "Processing op " << params.op_name;
  auto it = add_trt_layer_funcs.find(params.op_name);
  CHECK(it != add_trt_layer_funcs.end())
      << "Unsupported operator conversion to TRT, op name: " << params.op_name;

  // Convert op to TRT.
  it->second(params);

  // Get outputs.
  out_tensors_.clear();
  for (auto out : params.outputs) {
    out_tensors_.push_back(out);
  }
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
