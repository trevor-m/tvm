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
 * \file src/relay/backend/contrib/dnnl/codegen.cc
 * \brief Implementation of DNNL codegen APIs.
 */

#include <tvm/node/serialization.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>

#include "../../../../runtime/contrib/tensorrt/tensorrt_module.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

class TrtModuleCodegen : public CSourceModuleCodegenBase {
 public:
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    std::string serialized_subgraph;
    if (ref->IsInstance<FunctionNode>()) {
      serialized_subgraph = SaveJSON(Downcast<Function>(ref)->body);
    } else if (ref->IsInstance<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      // TODO: support multiple functions. It is currently not possible for
      // there to be more than one TRT func, so not a problem yet.
      for (const auto& it : mod->functions) {
        serialized_subgraph = SaveJSON(Downcast<Function>(it.second)->body);
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }
    return runtime::TensorRTModuleCreate(serialized_subgraph);
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module
 * and compiles it into a runtime module.
 */
runtime::Module TrtCompiler(const ObjectRef& ref) {
  TrtModuleCodegen tensorrt;
  return tensorrt.CreateCSourceModule(ref);
}

TVM_REGISTER_API("relay.ext.tensorrt").set_body_typed(TrtCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
