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

#ifndef TVM_RELAY_BACKEND_TRT_EXECUTOR_H_
#define TVM_RELAY_BACKEND_TRT_EXECUTOR_H_

#include <stdlib.h>
#include <tvm/relay/contrib_codegen.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include <unordered_map>
#include <vector>
#include "trt_builder.h"

#include "NvInfer.h"

namespace tvm {
namespace relay {
namespace contrib {

// Logger for TensorRT info/warning/errors
class TrtExecutor {
 public:
  runtime::PackedFunc GetFunction(const std::string& id, const std::string& serialized_subgraph) {
    // Generate an external packed function
    return PackedFunc([this, id, serialized_subgraph](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      auto it = trt_engine_cache_.find(id);
      if (it == trt_engine_cache_.end()) {
        // Build new trt engine and place in cache.
        LOG(INFO) << "Building new TensorRT engine for subgraph " << id;
        Expr expr = LoadJSON<Expr>(serialized_subgraph);

        auto inputs = ConvertInputs(args);
        auto builder = TrtBuilder(inputs);
        auto engine_and_context = builder.BuildEngine(expr);
        LOG(INFO) << "Finished building engine";
        this->trt_engine_cache_[id] = engine_and_context;
      }

      auto engine_and_context = this->trt_engine_cache_[id];
      this->ExecuteEngine(engine_and_context, args, rv);
    });
  }
  
 private:
  std::unordered_map<std::string, TrtEngineAndContext> trt_engine_cache_;

  // Convert TVMArgs to make compatible with VM or graph runtime.
  std::vector<DLTensor*> ConvertInputs(tvm::TVMArgs args) {
    std::vector<DLTensor*> inputs(args.size(), nullptr);
    for (int i = 0; i < args.size(); i++) {
      if (args[i].type_code() == kNDArrayContainer) {
        // Relay Debug/VM uses NDArray
        runtime::NDArray array = args[i];
        inputs[i] = const_cast<DLTensor*>(array.operator->());
      } else if (args[i].type_code() == kArrayHandle) {
        // Graph runtime uses DLTensors
        inputs[i] = args[i];
      } else {
        LOG(FATAL) << "Invalid TVMArgs type.";
      }
    }
    return inputs;
  }

  void ExecuteEngine(const TrtEngineAndContext& engine_and_context,
                     tvm::TVMArgs args, tvm::TVMRetValue* rv) {
    auto engine = engine_and_context.engine;
    auto context = engine_and_context.context;
    const int num_bindings = engine->getNbBindings();
    std::vector<void*> bindings(num_bindings, nullptr);
    // Set inputs.
    auto inputs = ConvertInputs(args);
    // TODO(trevmorr): Assumes output is at the end - is this true?
    for (int i = 0; i < inputs.size() - 1; ++i) {
      auto it = engine_and_context.network_input_map.find(i);
      if (it != engine_and_context.network_input_map.end()) {
        DLTensor* arg = inputs[i];
        int binding_index = engine->getBindingIndex(it->second.c_str());
        CHECK(binding_index != -1);
        if (!runtime::TypeMatch(arg->dtype, kDLFloat, 32)) {
          LOG(FATAL) << "Only float32 inputs are supported.";
        }
        bindings[binding_index] = reinterpret_cast<float*>(arg->data);
      }
    }
    // Set outputs.
    // TODO(trevmorr): Allow multiple outputs.
    DLTensor* out_arg = inputs[inputs.size() - 1];
    bindings[num_bindings - 1] = reinterpret_cast<float*>(out_arg->data);
    // Use batch size from first input.
    const int batch_size = inputs[0]->shape[0];
    CHECK(context->execute(batch_size, bindings.data()))
        << "Running TensorRT failed.";

    // TODO(trevmorr): Look up bindings by name.
    // TODO(trevmorr): Allow multiple outputs.
    *rv = bindings[num_bindings - 1];
  }
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TRT_EXECUTOR_H_
