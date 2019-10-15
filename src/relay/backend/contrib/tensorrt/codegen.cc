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

#include "trt_builder.h"
#include <unordered_map>
#include <vector>
#include "NvInfer.h"

namespace tvm {
namespace relay {
namespace contrib {


void ExecuteEngine(const TrtEngineAndContext& engine_and_context, tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  auto engine = engine_and_context.engine;
  auto context = engine_and_context.context;
  const DLTensor* dptr = ((runtime::NDArray)args[0]).operator->();

  // const int num_outputs = 1;
  const int num_bindings = engine->getNbBindings();
  // CHECK(args.size() == num_bindings);
  void* bindings[num_bindings];
  if (!runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
    LOG(FATAL) << "Only support float32 type.";
  }
  // Set inputs.
  // TODO(trevmorr): Look up bindings by name.
  for (int i = 0; i < args.size() - 1; ++i) {
    auto it = engine_and_context.network_input_map.find(i);
    if (it != engine_and_context.network_input_map.end()) {
      runtime::NDArray arg = args[i];
      int binding_index = engine->getBindingIndex(it->second.c_str());
      CHECK(binding_index != -1);
      bindings[binding_index] = reinterpret_cast<float*>(arg->data);
    } else {
      // LOG(WARNING) << "input " << i << " was baked into TRT engine already.";
    }
  }
  // Set outputs.
  // TODO(trevmorr): Allow multiple outputs.
  runtime::NDArray out_arg = args[args.size() - 1];
  auto out = reinterpret_cast<float*>(out_arg->data);
  bindings[num_bindings-1] = out;
  const int batch_size = dptr->shape[0];
  CHECK(context->execute(batch_size, bindings)) << "Running TensorRT failed.";

  // TODO(trevmorr): Look up bindings by name.
  // TODO(trevmorr): Allow multiple outputs.
  *rv = bindings[num_bindings-1];
}

class TrtModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(const std::string& id = "") const override {
    // TensorRT doesn't create external libraries.
    return {};
  }

  void CompileExternLib() override { }

  const std::string GetPrefix() const override {
    return "tensorrt_";
  }

  TVM_DLL std::string GetSource(const std::string& format = "") override {
    return "";
  }

  const char* type_key() const override {
    return "TrtModule";
  }

  runtime::PackedFunc GetFunction(const std::string& name,
                                  const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    // LOG(INFO) << "getfn " << name;
    // Generate an external packed function
    return PackedFunc([sptr_to_self, this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      curr_id_ = GetSubgraphID(name);
      auto it = trt_engine_cache_.find(curr_id_);
      if (it == trt_engine_cache_.end()) {
        // Build new trt engine and place in cache.
        LOG(INFO) << "Building new TensorRT engine for subgraph " << curr_id_;
        // for(int i = 0; i < args.size(); i++) {
        //   runtime::NDArray arg = args[i];
        //   DLTensor* dptr = const_cast<DLTensor*>(arg.operator->());
        //   size_t size = 1;
        //   for (tvm_index_t i = 0; i < dptr->ndim; ++i) {
        //     size *= dptr->shape[i];
        //   }
        //   LOG(INFO) << "input " << i << ": " << size;
        // }
        Expr expr = LoadJSON<Expr>(this->serialized_json_);
        auto builder = TrtBuilder(args);
        auto engine_and_context = builder.BuildEngine(expr);
        trt_engine_cache_[curr_id_] = engine_and_context;
      }

      // LOG(INFO) << "Executing TensorRT engine for subgraph " << curr_id_;
      auto engine_and_context = trt_engine_cache_[curr_id_];
      ExecuteEngine(engine_and_context, args, rv);
    });
  }

  void Build(const NodeRef& ref) override {
    if (ref->derived_from<FunctionNode>()) {
      Function func = Downcast<Function>(ref);
      serialized_json_ = SaveJSON(func->body);
      // LOG(INFO) << AsText(func->body);
    } else if (ref->derived_from<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      for (const auto& it : mod->functions) {
        // TODO(trevmorr): handle this loop properly
        Function func = Downcast<Function>(it.second);
        serialized_json_ = SaveJSON(func->body);
        // LOG(INFO) << AsText(func->body);
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module";
    }
  }

 private:
  std::string curr_id_;
  std::string serialized_json_;
  std::unordered_map<std::string, TrtEngineAndContext> trt_engine_cache_;
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

TVM_REGISTER_API("relay.ext.tensorrt")
.set_body_typed(TrtCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
