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
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/attrs/nn.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace graph_partitioning {

static const std::unordered_set<std::string> trt_compatible_ops = {
    {"nn.relu"},
    {"sigmoid"},
    {"tanh"},
    {"clip"},
    {"nn.leaky_relu"},
    {"nn.batch_norm"},
    {"nn.softmax"},
    {"nn.conv2d"},
    {"nn.dense"},
    {"nn.bias_add"},
    {"add"},
    {"subtract"},
    {"multiply"},
    {"divide"},
    {"power"},
    {"nn.max_pool2d"},
    {"nn.avg_pool2d"},
    {"nn.global_max_pool2d"},
    {"nn.global_avg_pool2d"},
    {"exp"},
    {"log"},
    {"sqrt"},
    {"abs"},
    {"negative"},
    {"sin"},
    {"cos"},
    {"atan"},
    {"ceil"},
    {"floor"},
    {"nn.batch_flatten"},
    {"expand_dims"},
    {"concatenate"},
    {"nn.conv2d_transpose"}
};

std::vector<int> GetShape(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK(ttype);
  std::vector<int> _shape;
  for (int i = 0; i < ttype->shape.size(); ++i) {
    auto* val = ttype->shape[i].as<IntImm>();
    CHECK(val);
    _shape.push_back(val->value);
  }
  return _shape;
}

class TrtChecker : public ExprVisitor {
 public:
  void VisitExpr_(const VarNode* op) {
    const auto* ttype = op->checked_type().as<TensorTypeNode>();
    CHECK(ttype);
    if (!ttype->dtype.is_float()) {
      compatible_ = false;
    }
  }

  void VisitExpr_(const CallNode* call) final {
    const std::string op_name = (call->op.as<OpNode>())->name;
    if (trt_compatible_ops.count(op_name) == 0) {
      compatible_ = false;
    }
    if (op_name == "nn.conv2d") {
      if (const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>()) {
        if (conv2d_attr->data_layout != "NCHW" ||
            conv2d_attr->kernel_layout != "OIHW") {
          compatible_ = false;
        }
        if (conv2d_attr->out_layout != "" && 
            conv2d_attr->out_layout != "NCHW") {
          compatible_ = false;
        }
      }
    }
    if (op_name == "nn.dense") {
      auto shape0 = GetShape(call->type_args[0]);
      auto shape1 = GetShape(call->type_args[0]);
      if (shape0.size() < 2 || shape0.size() > 4) {
        compatible_ = false;
      }
      if (shape1.size() != 2) {
        compatible_ = false;
      }
    }
    if (op_name == "nn.batch_norm") {
      if (const auto* bn_attr = call->attrs.as<BatchNormAttrs>()) {
        if (bn_attr->axis != 1) {
          compatible_ = false;
        }
      }
    }
    if (op_name == "nn.max_pool2d") {
      if (const auto* pool_attr = call->attrs.as<MaxPool2DAttrs>()) {
        if (pool_attr->layout != "NCHW") {
          compatible_ = false;
        }
      }
    }
    if (op_name == "nn.avg_pool2d") {
      if (const auto* pool_attr = call->attrs.as<AvgPool2DAttrs>()) {
        if (pool_attr->layout != "NCHW") {
          compatible_ = false;
        }
      }
    }
    if (op_name == "nn.global_max_pool2d" || op_name == "nn.global_avg_pool2d") {
      if (const auto* pool_attr = call->attrs.as<GlobalPool2DAttrs>()) {
        if (pool_attr->layout != "NCHW") {
          compatible_ = false;
        }
      }
    }
    if (op_name == "expand_dims") {
      if (const auto* expand_dims_attr = call->attrs.as<ExpandDimsAttrs>()) {
        if (expand_dims_attr->axis <= 0) {
          compatible_ = false;
        }
      }
    }
    if (op_name == "concatenate") {
      if (const auto* concat_attr = call->attrs.as<ConcatenateAttrs>()) {
        if (concat_attr->axis <= 0) {
          compatible_ = false;
        }
      }
    }
    if (op_name == "nn.bias_add") {
      auto shape0 = GetShape(call->type_args[0]);
      if (shape0.size() < 2 || shape0.size() > 4) {
        compatible_ = false;
      }
    }
    if (op_name == "nn.conv2d_transpose") {
      if (const auto* conv2d_attr = call->attrs.as<Conv2DTransposeAttrs>()) {
        if (conv2d_attr->data_layout != "NCHW" ||
            conv2d_attr->kernel_layout != "OIHW") {
          compatible_ = false;
        }
        if (conv2d_attr->out_layout != "" && 
            conv2d_attr->out_layout != "NCHW") {
          compatible_ = false;
        }
        if (conv2d_attr->dilation[0].as<IntImm>()->value != 1 ||
            conv2d_attr->dilation[1].as<IntImm>()->value != 1) {
          compatible_ = false;
        }
      }
    }
  }

  bool Check(const Expr& expr) {
    compatible_ = true;
    VisitExpr(expr);
    return compatible_;
  }

 private:
  bool compatible_;
};

class TrtEnabler : public ExprMutator {
 public:
  Expr VisitExpr_(const TupleGetItemNode* n) final {
    // Remove nn.dropout
    static const Op& dropout = Op::Get("nn.dropout");
    Expr new_e = ExprMutator::VisitExpr_(n);
    const auto* new_n = new_e.as<TupleGetItemNode>();
    if (new_n->index != 0) {
      return new_e;
    }
    if (const auto* call = new_n->tuple.as<CallNode>()) {
      if (call->op.same_as(dropout)) {
        return call->args[0];
      }
    }
    return new_e;
  }

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
  TrtChecker trt_checker;
  bool can_use_trt = trt_checker.Check(expr);
  if (can_use_trt) {
    TrtEnabler trt_enabler;
    return trt_enabler.Enable(expr);
  }
  LOG(WARNING) << "Model is not TRT compatible. Falling back to Relay/CUDA";
  return expr;
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
