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

/*!
 * \file src/relay/transforms/annotate_target.cc
 * \brief Wraps a call with compiler_begin and compiler_end to indicate that
 * the op of this call node will use external compiler.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/container.h>

namespace tvm {
namespace relay {
namespace annotate_target {

static const Op& compiler_begin_op = Op::Get("annotation.compiler_begin");

// A helper class to insert annotation boundaries for a program region that will
// be handled by a specific compiler.
class AnnotateTargetWrapper : public ExprMutator {
 public:
  AnnotateTargetWrapper(const Array<runtime::String> targets) {
    for (auto target : targets) {
      targets_.push_back(target.data());
    }
  }

  Expr Annotate(const Expr& expr) {
    auto new_expr = Mutate(expr);
    //std::cerr << AsText(new_expr);
    return new_expr;
  }

  /*! \brief This function 1) annotates a compiler end and a compiler begin to all arguments.
   * The compiler end is based on the arg target while the compiler begin is based on the given
   * target. If target is not given and all arguments are going to the same target, then we will
   * use that target; otherwise we use default for this op. Note that all arg exprs must be
   * available in op_expr_to_target before calling this function.
   *
   * \param args An array of arguments of the given node.
   * \param target The target of the current node.
   * \return A pair of target and annotated argument expressions.
   */
  std::pair<std::string, Array<Expr>> AnnotateArgs(const Array<Expr> args,
                                                   const std::string target = "") {
    std::string ref_target = "";
    Array<Expr> compiler_ends;
    for (auto arg : args) {
      if (op_expr_to_target_.find(arg) != op_expr_to_target_.end()) {
        std::string arg_target = op_expr_to_target_[arg];
        compiler_ends.push_back(InsertAnnotation(arg, arg_target, end_op));
        if (ref_target == "") {
          ref_target = arg_target;
        } else if (ref_target != arg_target) {
          ref_target = "__inconsist__";
        }
      } else {
        // Input vars.
        compiler_ends.push_back(arg);
      }
    }
    ref_target = (ref_target == "__inconsist__") ? "default" : ref_target;

    // Determine compiler begin target.
    std::string op_target = (target == "")? ref_target: target;

    Array<Expr> compiler_begins;
    for (auto end : compiler_ends) {
      compiler_begins.push_back(InsertAnnotation(end, op_target, begin_op));
    }

    return {op_target, compiler_begins};
  }

  Expr InsertAnnotation(const Expr& expr, const std::string target, const PackedFunc* ann_op) {
    Expr new_op = (*ann_op)(expr, target);
    new_op->checked_type_ = expr->checked_type_;
    return new_op;
  }

  Expr VisitExpr_(const CallNode* cn) {
    // TODO(@zhiics, @comaniac) Handle composite functions.

    Op op = Downcast<Op>(cn->op);
    CHECK(op.defined());

    // Supported targets for this node. The order implies the priority.
    std::vector<std::string> supported_targets;

    // Check which targets this op can be offloaded.
    for (auto target : this->targets_) {
      auto fannotate = Op::GetAttr<FTVMAnnotateTarget>("target." + target);
      if (fannotate.count(op) && fannotate[op](cn->attrs, cn->args)) {
        supported_targets.push_back(target);
      }
    }
    supported_targets.push_back("default");  // Make default as the last option.

    // TODO(@comaniac, @zhiics): Now we simply assign this node to the target with
    // the highest priority, but we should preserve all supported targets so that
    // we can make a better decision.
    std::string target = supported_targets[0];

    // Visit and mutate arguments after the target of this op has been determined.
    auto new_e = ExprMutator::VisitExpr_(cn);
    Call call = Downcast<Call>(new_e);

    // Add annotations to each arg.
    auto target_n_args = AnnotateArgs(call->args, target);
    Array<Expr> compiler_begins = std::get<1>(target_n_args);
    // for (auto b : compiler_begins) {
    //   std::cerr << AsText(b);
    //   std::cerr << "===============\n";
    // }
    //std::cerr << "*********************************************\n";
    call = Call(call->op, compiler_begins, call->attrs);
    call->checked_type_ = cn->checked_type_;

    // Update the target map.
    op_expr_to_target_[call] = target;

    return std::move(call);
  }

  Expr VisitExpr_(const TupleNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto expr = Downcast<Tuple>(new_e);

    auto target_n_args = AnnotateArgs(expr->fields);
    auto new_expr = Tuple(std::get<1>(target_n_args));
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return new_expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto expr = Downcast<TupleGetItem>(new_e);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->tuple}));

    std::string target = std::get<0>(target_n_args);
    auto new_expr = TupleGetItem(std::get<1>(target_n_args)[0], expr->index);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return new_expr;
  }

  Expr VisitExpr_(const FunctionNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto func = Downcast<Function>(new_e);
    auto new_body = func->body;
    if (op_expr_to_target_.find(func->body) != op_expr_to_target_.end()) {
      new_body = InsertAnnotation(func->body, op_expr_to_target_[func->body], end_op);
      op_expr_to_target_[new_body] = op_expr_to_target_[func->body];
    }

    return Function(func->params, new_body, func->ret_type, func->type_params, func->attrs);
  }

  Expr VisitExpr_(const LetNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto expr = Downcast<Let>(new_e);

    std::vector<Expr> args = {expr->value, expr->body};
    auto target_n_args = AnnotateArgs(Array<Expr>(args));

    std::string target = std::get<0>(target_n_args);
    auto new_expr = Let(expr->var, std::get<1>(target_n_args)[0], std::get<1>(target_n_args)[1]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return new_expr;
  }

  Expr VisitExpr_(const IfNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto expr = Downcast<If>(new_e);

    std::vector<Expr> args = {expr->cond, expr->true_branch, expr->false_branch};
    auto target_n_args = AnnotateArgs(Array<Expr>(args));

    std::string target = std::get<0>(target_n_args);
    auto new_expr = If(std::get<1>(target_n_args)[0], std::get<1>(target_n_args)[1],
                       std::get<1>(target_n_args)[2]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return new_expr;
  }

  Expr VisitExpr_(const RefCreateNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto expr = Downcast<RefCreate>(new_e);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->value}));
    auto new_expr = RefCreate(std::get<1>(target_n_args)[0]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return new_expr;
  }

  Expr VisitExpr_(const RefReadNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto expr = Downcast<RefRead>(new_e);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->ref}));

    auto new_expr = RefRead(std::get<1>(target_n_args)[0]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return new_expr;
  }

  Expr VisitExpr_(const RefWriteNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto expr = Downcast<RefWrite>(new_e);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->ref, expr->value}));

    std::string target = std::get<0>(target_n_args);
    auto new_expr = RefWrite(std::get<1>(target_n_args)[0], std::get<1>(target_n_args)[1]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return new_expr;
  }

 private:
  /*! \brief The target backends for annotation. */
  std::vector<std::string> targets_;
  /*! \brief Maintain the decision of the target for each op expr. */
  std::unordered_map<Expr, std::string, ObjectHash, ObjectEqual> op_expr_to_target_;
  const PackedFunc* begin_op = runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
  const PackedFunc* end_op = runtime::Registry::Get("relay.op.annotation._make.compiler_end");
};

Expr AnnotateTarget(const Expr& expr, const Array<runtime::String> targets) {
  return AnnotateTargetWrapper(targets).Annotate(expr);
}

}  // namespace annotate_target

namespace transform {

Pass AnnotateTarget(const Array<runtime::String> targets) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::annotate_target::AnnotateTarget(f, targets));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "AnnotateTargetFunc",
                                      {tir::StringImmNode::make("InferType")});
  return transform::Sequential({func_pass, InferType()}, "AnnotateTarget");
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotateTarget")
.set_body_typed(AnnotateTarget);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
