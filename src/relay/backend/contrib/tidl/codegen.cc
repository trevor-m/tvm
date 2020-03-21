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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief TIDL codegen
 */
class TIDLModuleCodeGen : public CSourceModuleCodegenBase {
 public:

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    int total_subgraphs = 0;
    int subgraph_id = 0;

    if (ref->IsInstance<FunctionNode>()) {
      std::cout << "FunctionNode: ";
      std::string subgraph_name = GetExtSymbol(Downcast<Function>(ref));
      std::cout << "subgraph_name is: " << subgraph_name <<std::endl;
      total_subgraphs = 1;
    } else if (ref->IsInstance<IRModuleNode>()) {
      // TODO: support multiple functions.
      IRModule mod = Downcast<IRModule>(ref);
      total_subgraphs = mod->functions.size();
      std::cout << "IRModuleNode, total subgraphs: " << total_subgraphs << std::endl;
      for (const auto& it : mod->functions) {
        //std::string subgraph_name = GetExtSymbol(Downcast<Function>(it.second));
        //std::cout << "subgraph id: " << subgraph_id << ", ";
        //std::cout << "subgraph_name: " << subgraph_name <<std::endl;

        auto func = Downcast<Function>(it.second);
        auto subgraph_name = GetExtSymbol(func);
        const int num_inputs = func->params.size();
        for (int i = 0; i < num_inputs; i++) {
          std::cout << subgraph_name << " input " << i << " is named " << func->params[i]->name_hint() << std::endl;
        }

        const int num_outputs = func->ret_type.as<TensorTypeNode>() ? 1 : func->ret_type.as<TupleTypeNode>()->fields.size();
        std::cout << subgraph_name << " has " << num_inputs << " inputs." << std::endl;
        std::cout << subgraph_name << " has " << num_outputs << " outputs." << std::endl;
        //LOG(INFO) << subgraph_name << " has " << num_outputs << " outputs.";

        subgraph_id++;
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
    }

    // TODO: support multiple functions - how to do it?
    const PackedFunc* pf =
          runtime::Registry::Get("tvm.contrib.tidl.create");
    CHECK(pf != nullptr)
          << "tvm.contrib.tidl.create was not found in the registry.";
    // need to pass what TIDL runtime needs:
    //    - 
    return (*pf)(total_subgraphs, subgraph_id);
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 *
 */
runtime::Module TIDLCompiler(const ObjectRef& ref) {
  std::cout << "TIDL compiler invoked" << std::endl;
  TIDLModuleCodeGen tidl;
  return tidl.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.tidl").set_body_typed(TIDLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
