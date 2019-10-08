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
// #include <dlfcn.h>
#include <stdlib.h>
#include <tvm/relay/contrib_codegen.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include "trt_executor.h"

namespace tvm {
namespace relay {
namespace contrib {

// TODO(trevmorr): Disable SimplifyInference path when using TRT.
class TrtModuleNode : public ExternModuleNodeBase {
 public:
  void CompileExternLib() override {}

  TVM_DLL std::string GetSource(const std::string& format = "") override {
    return "";
  }

  const char* type_key() const override { return "TrtModule"; }

  runtime::PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    // Generate an external packed function
    std::string id = GetSubgraphID(name);
    return trt_exec_.GetFunction(id, this->serialized_json_);
  }

  void Build(const NodeRef& ref) override {
    if (ref->derived_from<FunctionNode>()) {
      Function func = Downcast<Function>(ref);
      serialized_json_ = SaveJSON(func->body);
    } else if (ref->derived_from<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      for (const auto& it : mod->functions) {
        // TODO(trevmorr): handle this loop properly
        Function func = Downcast<Function>(it.second);
        serialized_json_ = SaveJSON(func->body);
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module";
    }
  }

 private:
  std::string serialized_json_;
  TrtExecutor trt_exec_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 */
runtime::Module TrtCompiler(const NodeRef& ref) {
  std::shared_ptr<TrtModuleNode> n = std::make_shared<TrtModuleNode>();
  n->Build(ref);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.tensorrt").set_body_typed(TrtCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
