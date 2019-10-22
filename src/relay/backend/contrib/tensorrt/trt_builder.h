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

#ifndef TVM_RELAY_BACKEND_TRT_BUILDER_H_
#define TVM_RELAY_BACKEND_TRT_BUILDER_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <unordered_map>
#include <vector>
#include "NvInfer.h"

namespace tvm {
namespace relay {
namespace contrib {

struct TrtEngineAndContext {
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;
  std::unordered_map<int, std::string> network_input_map;
};

enum TrtInputType {
  kTensor,
  kWeight,
};

struct TrtOpInput {
  TrtInputType type;
  nvinfer1::ITensor* tensor;
  nvinfer1::Weights weight;
  std::vector<int> weight_shape;

  TrtOpInput(nvinfer1::ITensor* tensor) : tensor(tensor), type(kTensor) {}
  TrtOpInput(nvinfer1::Weights weight, const std::vector<int>& shape)
      : weight(weight), type(kWeight), weight_shape(shape) {}
};

class TrtBuilder : public ExprVisitor {
 public:
  TrtBuilder(const std::vector<DLTensor*>& args);

  void VisitExpr_(const VarNode* node) final;

  void VisitExpr_(const ConstantNode* node) final;

  void VisitExpr_(const TupleGetItemNode* op) final;

  void VisitExpr_(const TupleNode* op) final;

  void VisitExpr_(const CallNode* call) final;

  TrtEngineAndContext BuildEngine(const Expr& expr);

 private:
  nvinfer1::Weights GetNdArrayAsWeights(const runtime::NDArray& array,
                                        DLDeviceType src_device);
  nvinfer1::Weights GetDLTensorAsWeights(DLTensor* dptr,
                                         DLDeviceType src_device);
  nvinfer1::Weights GetInputAsWeights(const VarNode* node);
  void CleanUp();

  int TrackVarNode(const VarNode* node);

  // Maps a node to its outputs.
  std::unordered_map<const ExprNode*, std::vector<TrtOpInput>> node_output_map_;

  // For TRT conversion
  nvinfer1::IBuilder* builder_;
  nvinfer1::INetworkDefinition* network_;

  // VarNode name_hint -> input tensor
  std::unordered_map<std::string, nvinfer1::ITensor*> trt_inputs_;
  // TODO(trevmorr): cache weights into here
  std::vector<nvinfer1::Weights> trt_weights_;

  // Maps execution_args_ input index -> TRT input tensor name / VarNode
  // name_hint
  std::unordered_map<int, std::string> network_input_map_;

  // Execution inputs from this invocation.
  const std::vector<DLTensor*>& execution_args_;

  // Maps VarNodes to index in execution_args_.
  int var_node_counter_;
  std::unordered_map<const VarNode*, int> var_node_input_map_;
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TRT_BUILDER_H_
