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
};

class TrtBuilder : public ExprVisitor {
 public:
  TrtBuilder(std::string id);

  void VisitExpr_(const VarNode* node) final;

  // TODO(trevmorr): should this do nothing??
  void VisitExpr_(const TupleGetItemNode* op) final { } 

  void VisitExpr_(const CallNode* call) final;

  TrtEngineAndContext BuildEngine(const Expr& expr);

 private:
  // Tracks outputs of operators as they are processed.
  std::vector<nvinfer1::ITensor*> out_tensors_;

  // For TRT conversion
  nvinfer1::IBuilder* builder_;
  nvinfer1::INetworkDefinition* network_;
  // std::unordered_map<std::string, nvinfer1::ITensor*> trt_tensors_;
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif // TVM_RELAY_BACKEND_TRT_BUILDER_H_
