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

#include <stdlib.h>
// #include <tvm/relay/contrib_codegen.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
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

class TrtBuilder : public ExprVisitor {
 public:
  TrtBuilder(tvm::TVMArgs args);

  void VisitExpr_(const VarNode* node) final;

  void VisitExpr_(const ConstantNode* node) final;

  // TODO(trevmorr)
  // void VisitExpr_(const TupleGetItemNode* op) final { 
  //   ;
  // } 

  void VisitExpr_(const CallNode* call) final;

  TrtEngineAndContext BuildEngine(const Expr& expr);

 private:
  nvinfer1::Weights GetInputAsWeights(const VarNode* node);

  int TrackVarNode(const VarNode* node);

  // Tracks outputs of operators as they are processed.
  std::vector<nvinfer1::ITensor*> out_tensors_;

  // For TRT conversion
  nvinfer1::IBuilder* builder_;
  nvinfer1::INetworkDefinition* network_;

  std::unordered_map<std::string, nvinfer1::ITensor*> trt_inputs_;
  // TODO(trevmorr): cache weights into here
  // std::unordered_map<std::string, nvinfer1::Weights> trt_weights_;
  // TODO(trevmorr): populate these and use for execution

  // index -> name
  std::unordered_map<int, std::string> network_input_map_;
  // std::vector<std::string> network_output_names_;

  // Execution inputs.
  tvm::TVMArgs execution_args_;

  // Map VarNodes to index in execution_args_.
  int var_node_counter_;
  std::unordered_map<const VarNode*, int> var_node_input_map_;
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif // TVM_RELAY_BACKEND_TRT_BUILDER_H_
