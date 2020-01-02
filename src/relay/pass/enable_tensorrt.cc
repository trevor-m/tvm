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

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/node/container.h>

#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#if TVM_COMPILER_TENSORRT
#include "NvInfer.h"
#endif

namespace tvm {
namespace relay {

// Pytorch's "addmm" operator will add redundant transpose and scale on weights
// for dense operator. Look for pattern of Transpose([1, 0]) -> Scale(1.0f) ->
// Transpose([1, 0]) and remove.
class PyTorchAddmmFixer : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* n) {
    static const Op& multiply_op = Op::Get("multiply");
    static const Op& transpose_op = Op::Get("transpose");
    if (n->op.same_as(transpose_op)) {
      const auto* attrs = n->attrs.as<TransposeAttrs>();
      if (attrs->axes.defined() && attrs->axes.size() == 2 &&
          attrs->axes[0].as<IntImm>()->value == 1 &&
          attrs->axes[1].as<IntImm>()->value == 0) {
        auto* multiply = n->args[0].as<CallNode>();
        if (multiply && multiply->op.same_as(multiply_op)) {
          const ConstantNode* const_1 = multiply->args[1].as<ConstantNode>();
          if (const_1 && const_1->is_scalar() &&
              *static_cast<float*>(const_1->data->data) == 1.0f) {
            auto* transpose = multiply->args[0].as<CallNode>();
            if (transpose && transpose->op.same_as(transpose_op)) {
              const auto* attrs = n->attrs.as<TransposeAttrs>();
              if (attrs->axes.defined() && attrs->axes.size() == 2 &&
                  attrs->axes[0].as<IntImm>()->value == 1 &&
                  attrs->axes[1].as<IntImm>()->value == 0) {
                return transpose->args[0];
              }
            }
          }
        }
      }
    }
    return ExprMutator::VisitExpr_(n);
  }
};

// Base TRT ops.
static const std::unordered_set<std::string> trt_base_compatible_ops = {
    {"nn.dropout"},  // Will be removed by EnableTrt pass.
    {"nn.relu"},
    {"sigmoid"},
    {"tanh"},
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
    {"nn.batch_flatten"},
    {"expand_dims"},
    {"squeeze"},
    {"concatenate"},
    {"nn.conv2d_transpose"},
    {"transpose"},
    {"reshape"},
    {"nn.pad"},
    {"sum"},
    {"prod"},
    {"max"},
    {"min"},
    {"mean"},
    {"contrib.adaptive_max_pool2d"},
    {"contrib.adaptive_avg_pool2d"}};

// Ops which require TRT 5.1.5+
static const std::unordered_set<std::string> trt_5_1_5_compatible_ops = {
    {"clip"}, {"nn.leaky_relu"}, {"sin"},   {"cos"},
    {"atan"}, {"ceil"},          {"floor"}, {"strided_slice"}};

// Ops which require TRT 6.0.1+
static const std::unordered_set<std::string> trt_6_0_1_compatible_ops = {
    {"image.resize"}};

bool TrtVersionGe(const std::tuple<int, int, int>& curr_version, int major,
                  int minor, int patch) {
  if (std::get<0>(curr_version) > major) return true;
  if (std::get<0>(curr_version) == major && std::get<1>(curr_version) > minor)
    return true;
  if (std::get<0>(curr_version) == major &&
      std::get<1>(curr_version) == minor && std::get<2>(curr_version) >= patch)
    return true;
  return false;
}

std::vector<int> GetShape(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK(ttype);
  std::vector<int> _shape;
  for (size_t i = 0; i < ttype->shape.size(); ++i) {
    auto* val = ttype->shape[i].as<IntImm>();
    CHECK(val);
    _shape.push_back(val->value);
  }
  return _shape;
}

class TrtChecker : public ExprVisitor {
 public:
  explicit TrtChecker(const std::tuple<int, int, int>& trt_version)
      : trt_version_(trt_version) {
    // Create set of compatible ops for this version.
    trt_compatible_ops.insert(trt_base_compatible_ops.begin(),
                              trt_base_compatible_ops.end());
    if (TrtVersionGe(trt_version_, 5, 1, 5)) {
      // Add TRT 5.1.5 ops to whitelist.
      trt_compatible_ops.insert(trt_5_1_5_compatible_ops.begin(),
                                trt_5_1_5_compatible_ops.end());
    }
    if (TrtVersionGe(trt_version_, 6, 0, 1)) {
      // Add TRT 6.0.1 ops to whitelist.
      trt_compatible_ops.insert(trt_6_0_1_compatible_ops.begin(),
                                trt_6_0_1_compatible_ops.end());
    }
  }

  void VisitExpr_(const VarNode* op) {
    const auto* ttype = op->checked_type().as<TensorTypeNode>();
    CHECK(ttype);
    if (!ttype->dtype.is_float()) {
      compatible_ = false;
    }
  }

  void VisitExpr_(const CallNode* call) final {
    const std::string op_name = (call->op.as<OpNode>())->name;
    for (size_t i = 0; i < call->args.size(); ++i) {
      // Workaround for check for transpose of weight.
      if ((op_name == "nn.conv2d" || op_name == "nn.dense") && i == 1) {
        if (auto* transpose = call->args[i].as<CallNode>()) {
          if (transpose->op.as<OpNode>()->name == "transpose") {
            if (!transpose->args[0].as<VarNode>()) {
              compatible_ = false;
              LOG(INFO) << op_name
                        << " not supported: most have constant weight.";
            }
          } else {
            compatible_ = false;
            LOG(INFO) << op_name
                      << " not supported: most have constant weight.";
          }
        } else {
          VisitExpr(call->args[i]);
        }
      } else {
        VisitExpr(call->args[i]);
      }
    }
    if (trt_compatible_ops.count(op_name) == 0) {
      LOG(INFO) << op_name << " not supported.";
      compatible_ = false;
    }
    if (op_name == "nn.conv2d") {
      if (const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>()) {
        if (conv2d_attr->data_layout != "NCHW" ||
            conv2d_attr->kernel_layout != "OIHW") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be NCHW.";
        }
        if (conv2d_attr->out_layout != "" &&
            conv2d_attr->out_layout != "NCHW") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be NCHW.";
        }
      }
    }
    if (op_name == "nn.dense") {
      auto shape0 = GetShape(call->type_args[0]);
      auto shape1 = GetShape(call->type_args[1]);
      if (shape0.size() < 2 || shape0.size() > 4) {
        compatible_ = false;
        LOG(INFO) << op_name
                  << " not supported: input must be rank 2, 3, or 4.";
      }
      if (shape1.size() != 2) {
        compatible_ = false;
        LOG(INFO) << op_name << " not supported: weight must be rank 2.";
      }
    }
    if (op_name == "nn.batch_norm") {
      if (const auto* bn_attr = call->attrs.as<BatchNormAttrs>()) {
        if (bn_attr->axis != 1 && bn_attr->axis != 3) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be on axis 1 or 3."
                    << bn_attr->axis;
        }
      }
    }
    if (op_name == "nn.softmax") {
      if (const auto* attrs = call->attrs.as<SoftmaxAttrs>()) {
        if (attrs->axis == 0) {
          compatible_ = false;
          LOG(INFO) << op_name
                    << " not supported: can't modify batch dimension.";
        }
      }
    }
    if (op_name == "nn.max_pool2d") {
      if (const auto* pool_attr = call->attrs.as<MaxPool2DAttrs>()) {
        if (pool_attr->layout != "NCHW") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be NCHW.";
        }
      }
    }
    if (op_name == "nn.avg_pool2d") {
      if (const auto* attrs = call->attrs.as<AvgPool2DAttrs>()) {
        if (attrs->layout != "NCHW") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be NCHW.";
        }
        if (attrs->count_include_pad) {
          if (attrs->padding.size() == 4 ||
              (attrs->strides.size() == 2 &&
               (attrs->strides[0].as<IntImm>()->value != 1 ||
                attrs->strides[1].as<IntImm>()->value != 1))) {
            compatible_ = false;
            LOG(INFO) << op_name
                      << " not supported: inclusive-counted blended or average "
                         "pooling is not supported in combination with "
                         "asymmetric padding or with strides.";
          }
        }
        if (attrs->ceil_mode && !TrtVersionGe(trt_version_, 5, 1, 5)) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: ceil_mode=True requires "
                                  "TensorRT 5.1.5 or greater.";
        }
      }
    }
    if (op_name == "nn.global_max_pool2d" ||
        op_name == "nn.global_avg_pool2d") {
      if (const auto* pool_attr = call->attrs.as<GlobalPool2DAttrs>()) {
        if (pool_attr->layout != "NCHW") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be NCHW.";
        }
      }
    }
    if (op_name == "expand_dims") {
      if (const auto* attrs = call->attrs.as<ExpandDimsAttrs>()) {
        if (attrs->axis == 0) {
          compatible_ = false;
          LOG(INFO) << op_name
                    << " not supported: cannot modify batch dimension.";
        }
      }
    }
    if (op_name == "squeeze") {
      if (const auto* attrs = call->attrs.as<SqueezeAttrs>()) {
        if (!attrs->axis.defined()) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must explicitly set axis.";
        } else {
          for (size_t i = 0; i < attrs->axis.size(); ++i) {
            if (attrs->axis[i].as<IntImm>()->value == 0) {
              compatible_ = false;
              LOG(INFO) << op_name
                        << " not supported: cannot modify batch dimension.";
              break;
            }
          }
        }
      }
    }
    if (op_name == "concatenate") {
      if (const auto* concat_attr = call->attrs.as<ConcatenateAttrs>()) {
        if (concat_attr->axis == 0) {
          compatible_ = false;
          LOG(INFO) << op_name
                    << " not supported: cannot modify batch dimension.";
        }
      }
    }
    if (op_name == "nn.bias_add") {
      auto shape0 = GetShape(call->type_args[0]);
      if (shape0.size() < 2 || shape0.size() > 4) {
        compatible_ = false;
        LOG(INFO) << op_name
                  << " not supported: input must be rank 2, 3, or 4.";
      }
    }
    if (op_name == "nn.conv2d_transpose") {
      if (const auto* conv2d_attr = call->attrs.as<Conv2DTransposeAttrs>()) {
        if (conv2d_attr->data_layout != "NCHW" ||
            conv2d_attr->kernel_layout != "OIHW") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be NCHW.";
        }
        if (conv2d_attr->out_layout != "" &&
            conv2d_attr->out_layout != "NCHW") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: must be NCHW.";
        }
        if (conv2d_attr->dilation[0].as<IntImm>()->value != 1 ||
            conv2d_attr->dilation[1].as<IntImm>()->value != 1) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: dilation rate must be 1.";
        }
      }
    }
    if (op_name == "transpose") {
      if (const auto* attrs = call->attrs.as<TransposeAttrs>()) {
        if (attrs->axes[0].as<IntImm>()->value != 0) {
          compatible_ = false;
          LOG(INFO) << op_name
                    << " not supported: can't modify batch dimension.";
        }
      }
    }
    if (op_name == "reshape") {
      if (const auto* attrs = call->attrs.as<ReshapeAttrs>()) {
        // TODO(trevmorr): check for modified batch dim.
        for (size_t i = 0; i < attrs->newshape.size(); i++) {
          if (attrs->newshape[i].as<IntImm>()->value < -1) {
            compatible_ = false;
            LOG(INFO) << op_name
                      << " not supported: reshape dims must be explicit.";
          }
        }
      }
    }
    if (op_name == "nn.pad") {
      if (const auto* attrs = call->attrs.as<PadAttrs>()) {
        if (attrs->pad_mode != "constant") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: pad mode must be constant.";
        } else if (attrs->pad_value != 0.0) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: pad value must be zero.";
        }
      }
    }
    if (op_name == "sum" || op_name == "prod" || op_name == "max" ||
        op_name == "min" || op_name == "mean") {
      if (const auto* attrs = call->attrs.as<ReduceAttrs>()) {
        if (!attrs->axis.defined() || attrs->axis.size() == 0) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: cannot reduce to scalar.";
        } else {
          for (size_t i = 0; i < attrs->axis.size(); i++) {
            if (attrs->axis[i].as<IntImm>()->value == 0) {
              compatible_ = false;
              LOG(INFO) << op_name
                        << " not supported: can't modify batch dimension.";
              break;
            }
          }
        }
        if (attrs->exclude) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: exclude not supported.";
        }
      }
    }
    if (op_name == "strided_slice") {
      auto shape = GetShape(call->type_args[0]);
      if (const auto* attrs = call->attrs.as<StridedSliceAttrs>()) {
        if (attrs->begin[0].as<IntImm>()->value != 0 ||
            (attrs->end[0].as<IntImm>()->value != -1 &&
             attrs->end[0].as<IntImm>()->value != shape[0])) {
          compatible_ = false;
          LOG(INFO) << op_name
                    << " not supported: can't modify batch dimension.";
        }
        for (size_t i = 0; i < attrs->begin.size(); i++) {
          if (attrs->begin[i].as<IntImm>()->value < 0 ||
              attrs->end[i].as<IntImm>()->value < 0) {
            compatible_ = false;
            LOG(INFO) << op_name
                      << " not supported: start/end values must be positive.";
          }
        }
      }
    }
    if (op_name == "adaptive_max_pool2d" || op_name == "adaptive_avg_pool2d") {
      if (const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>()) {
        if ((attrs->output_size.size() == 1 &&
             attrs->output_size[0].as<IntImm>()->value != 1) ||
            (attrs->output_size.size() == 2 &&
             (attrs->output_size[0].as<IntImm>()->value != 1 ||
              attrs->output_size[1].as<IntImm>()->value != 1))) {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: output size must be (1, 1).";
        }
      }
    }
    if (op_name == "resize") {
      if (const auto* attrs = call->attrs.as<ResizeAttrs>()) {
        if (attrs->method != "nearest_neighbor" && attrs->method != "bilinear") {
          compatible_ = false;
          LOG(INFO) << op_name << " not supported: method must be nearest_neighor or bilinear";
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
  std::unordered_set<std::string> trt_compatible_ops;
  std::tuple<int, int, int> trt_version_;
};

class TrtEnabler : public ExprMutator {
 public:
  explicit TrtEnabler(const std::tuple<int, int, int>& trt_version)
      : trt_version_(trt_version) {}

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

    original_inputs_.push_back({var, GetRef<Expr>(op)});
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
    for (auto pair : original_inputs_) {
      params.push_back(pair.first);
      args.push_back(pair.second);
    }
    auto subgraph_func =
        FunctionNode::make(params, body, body->checked_type_, {}, Attrs());
    std::string name = "subgraph_0";
    subgraph_func = FunctionSetAttr(subgraph_func, "func_name",
                                    tvm::ir::StringImm::make(name));
    subgraph_func =
        FunctionSetAttr(subgraph_func, "Primitive", tvm::Integer(1));
    subgraph_func = FunctionSetAttr(subgraph_func, "External",
                                    tvm::ir::StringImm::make("tensorrt"));
    auto call = CallNode::make(subgraph_func, args);

    // Build outer func
    return FunctionNode::make(func_params, call, subgraph_func->ret_type,
                              subgraph_func->type_params, subgraph_func->attrs);
  }

 private:
  int var_id_{0};
  std::vector<std::pair<Var, Expr>> original_inputs_;
  std::tuple<int, int, int> trt_version_;
};

bool IsTrtCompatible(const Expr& expr,
                     const std::tuple<int, int, int>& trt_version) {
  return TrtChecker(trt_version).Check(expr);
}

Expr EnableTrt(const Expr& expr, const std::tuple<int, int, int>& trt_version) {
  if (IsTrtCompatible(expr, trt_version)) {
    return TrtEnabler(trt_version).Enable(expr);
  }
  LOG(WARNING) << "Model is not TRT compatible. Falling back to Relay/CUDA.";
  return expr;
}

Expr FixPyTorchAddmm(const Expr& e) { return PyTorchAddmmFixer().Mutate(e); }

namespace transform {

Array<Integer> GetTrtVersion() {
#if TVM_COMPILER_TENSORRT
  return {Integer(NV_TENSORRT_MAJOR), Integer(NV_TENSORRT_MINOR), Integer(NV_TENSORRT_PATCH)};
#else
  return {};
#endif
}

Pass FixPyTorchAddmm() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(FixPyTorchAddmm(f));
      };
  return CreateFunctionPass(pass_func, 0, "FixPyTorchAddmm",
                            {ir::StringImm::make("InferType")});
}

Pass EnableTrt(int trt_ver_major, int trt_ver_minor, int trt_ver_patch) {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> part_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(EnableTrt(
            f, std::make_tuple(trt_ver_major, trt_ver_minor, trt_ver_patch)));
      };
  auto enable_trt = CreateFunctionPass(part_func, 1, "EnableTrt", {});
  const auto* remove_unused =
      tvm::runtime::Registry::Get("relay._transform.RemoveUnusedFunctions");
  Array<tvm::Expr> entry_functions{tvm::Expr{"main"}};
  // auto pass = "relay._transform.RemoveUnusedFunctions"
  return Sequential({(*remove_unused)(entry_functions), FixPyTorchAddmm(),
                     enable_trt, InferType()});
}

TVM_REGISTER_API("relay._transform.EnableTrt").set_body_typed(EnableTrt);
TVM_REGISTER_API("relay._transform.GetTrtVersion").set_body_typed(GetTrtVersion);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
