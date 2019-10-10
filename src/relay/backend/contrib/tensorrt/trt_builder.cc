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
  TensorRTLogger(): TensorRTLogger(Severity::kWARNING) {}
  explicit TensorRTLogger(Severity severity): reportable_severity(severity) {}
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
};

void AddActivation(AddTrtLayerParams& params) {
  CHECK(params.inputs.size() == 1) << "Activation expects 1 input.";
  static const std::unordered_map<std::string, nvinfer1::ActivationType> op_map = {
      {"nn.relu", nvinfer1::ActivationType::kRELU},
      {"sigmoid", nvinfer1::ActivationType::kSIGMOID},
      {"tanh", nvinfer1::ActivationType::kTANH}};
  auto it = op_map.find(params.op_name);
  CHECK(it != op_map.end()) << "Unsupported activation type "
                            << params.op_name << " in TensorRT";
  nvinfer1::IActivationLayer* act_layer = params.network->addActivation(*params.inputs.at(0), nvinfer1::ActivationType::kRELU);
  CHECK(act_layer != nullptr);
  // act_layer->setName(nodes[nid].node_name.c_str());
  params.outputs.push_back(act_layer->getOutput(0));
}

using AddTrtLayer = std::function<void(AddTrtLayerParams& params)>;

static const std::unordered_map<std::string, AddTrtLayer> add_trt_layer_funcs = {
    // {{"conv2d", AddConvolution},
    //  {"batch_norm", AddBatchNorm},
      {"nn.relu", AddActivation},
      {"sigmoid", AddActivation},
      {"tanh", AddActivation},
    //  {"clip", AddActivation},
    //  {"add", AddElementWiseBinaryOp},
    //  {"elemwise_sub", AddElementWiseBinaryOp},
    //  {"elemwise_mul", AddElementWiseBinaryOp},
    //  {"elemwise_div", AddElementWiseBinaryOp},
    //  {"elemwise_pow", AddElementWiseBinaryOp},
    //  {"max_pool2d", AddPooling},
    //  {"avg_pool2d", AddPooling},
    //  {"global_max_pool2d", AddPooling},
    //  {"global_avg_pool2d", AddPooling},
    //  {"dense", AddFullyConnected},
    //  {"softmax", AddSoftmax},
    //  {"concatenate", AddConcatenate},
    //  {"conv2d_transpose", AddDeconvolution},
    //  {"slice_like", AddSliceLike},
    };

TrtBuilder::TrtBuilder(std::string id) {
  // this->subgraph_id = id;

  // Init TRT stuff
  static TensorRTLogger trt_logger;
  builder_ = nvinfer1::createInferBuilder(trt_logger);
  builder_->setMaxBatchSize(1);
  builder_->setMaxWorkspaceSize(1 << 29);
  //builder_->setFp16Mode(use_fp16_);
  network_ = builder_->createNetwork();
}

TrtEngineAndContext TrtBuilder::BuildEngine(const Expr& expr) {
  // Process graph and create INetworkDefinition.
  VisitExpr(expr);
  // out_tensors_ will hold the output of the network
  for (int i = 0; i < out_tensors_.size(); ++i) {
    // std::string output_name = "output_" + std::to_string(i);
    // out_tensors_->setName(output_name.c_str());
    network_->markOutput(*out_tensors_[i]);
    LOG(INFO) << "Added network output: " << out_tensors_[i]->getName();
  }
  nvinfer1::ICudaEngine* engine = builder_->buildCudaEngine(*network_);
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  return {engine, context};
}

void TrtBuilder::VisitExpr_(const VarNode* node) {
  LOG(INFO) << "Adding input " << node->name_hint();

  const std::string& tensor_name = node->name_hint();
  auto shape = GetShape(node->checked_type());
  nvinfer1::Dims dims = VectorToTrtDims(shape);
  auto type = GetType(node->checked_type());
  CHECK(type.is_float()) << "Only FP32 inputs are supported.";
  nvinfer1::ITensor* input_tensor = network_->addInput(tensor_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
  // _trt_tensors[tensor_name] = input_tensor;

  out_tensors_.clear();
  out_tensors_.push_back(input_tensor);
}

void TrtBuilder::VisitExpr_(const CallNode* call) {
  AddTrtLayerParams params;
  params.network = network_;
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
  auto it = add_trt_layer_funcs.find(params.op_name);
  CHECK(it != add_trt_layer_funcs.end()) << "Unsupported operator conversion to TRT, op name: "
      << params.op_name;

  LOG(INFO) << "Processing op " << params.op_name;
  it->second(params);

  out_tensors_.clear();
  for (auto out : params.outputs) {
    out_tensors_.push_back(out);
  }
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
