/*
 * Licensed to the Apache Software Foundation (ASF) under one
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

/*! Copyright (c) 2019 by Contributorsr
 * \file src/relay/pass/partition_graph.cc
 *
 * \brief  Partition an input function into multiple Functions according based
 * on the inserted annotation nodes (i.e. begin and end). These nodes are used
 * as boundaries to partition the Relay function into multiple regions that can
 * be offloaded to different accelerators.
 *
 * Each of these paritioned functions, a.k.a subgraphs, will be viewed as
 * external functions, and they will use external tools for codegen.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace graph_partitioning {


class TrtEnabler : public ExprMutator {
 public:
  Expr VisitExpr_(const VarNode* op) {
    // Rename inputs
    auto var = VarNode::make("tensorrt_input" + std::to_string(var_id_++),
                             op->checked_type_);
    
    original_inputs.push_back({var, GetRef<Expr>(op)});
    return std::move(var);
  }

  Expr Enable(const Expr& expr) {
    // Get inputs to func.
    auto func = expr.as<FunctionNode>();
    CHECK(func != nullptr);
    Array<Var> func_params;
    for (auto param : func->params) {
      func_params.push_back(param);
    }

    // Process body
    auto body = VisitExpr(func->body);
    Array<Var> params;
    Array<Expr> args;
    for (auto pair : original_inputs) {
      params.push_back(pair.first);
      args.push_back(pair.second);
    }
    auto subgraph_func =
        FunctionNode::make(params, body, body->checked_type_, {}, Attrs());
    std::string name = "subgraph_0";
    subgraph_func =
        FunctionSetAttr(subgraph_func, "func_name", tvm::ir::StringImm::make(name));
    subgraph_func = FunctionSetAttr(subgraph_func, "Primitive", tvm::Integer(1));
    subgraph_func = FunctionSetAttr(subgraph_func, "External",
                                    tvm::ir::StringImm::make("tensorrt"));
    auto call = CallNode::make(subgraph_func, args);

    // Build outer func
    return FunctionNode::make(func_params, call, subgraph_func->ret_type, subgraph_func->type_params, subgraph_func->attrs);
  }

 private:
  int var_id_{0};
  std::vector<std::pair<Var, Expr>> original_inputs;
};

Expr EnableTrt(const Expr& expr) {
  TrtEnabler trt_enabler;
  return trt_enabler.Enable(expr);
}

}  // namespace graph_partitioning

namespace transform {

Pass EnableTrt() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> part_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(graph_partitioning::EnableTrt(f));
      };
  auto partitioned = CreateFunctionPass(part_func, 1, "EnableTrt", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_API("relay._transform.EnableTrt")
.set_body_typed(transform::EnableTrt);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
