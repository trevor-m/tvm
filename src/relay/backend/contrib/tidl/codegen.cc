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
#include <tvm/ir/module.h>

#include <fstream>
#include <sstream>
#include <unordered_map>

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace runtime {
Module TIDLModuleCreate(int total_subgraphs, 
                      const std::unordered_map<std::string, int>& num_inputs,
                      const std::unordered_map<std::string, int>& num_outputs);
}
}

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
    //std::string subgraph_names[MAX_NUM_TIDL_SUBRAPHS];  // can be used for error checking

    if (ref->IsInstance<FunctionNode>()) {
      //std::cout << "FunctionNode: ";
      //subgraph_names[0] = GetExtSymbol(Downcast<Function>(ref));
      //std::cout << "subgraph_name is: " << subgraph_names[0] <<std::endl;
      total_subgraphs = 1;
      
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      total_subgraphs = mod->functions.size();
      std::cout << "IRModuleNode, total subgraphs: " << total_subgraphs << std::endl;

      for (const auto& it : mod->functions) {
        auto func = Downcast<Function>(it.second);
        auto subgraph_name = GetExtSymbol(func);
        // get subgraph_id from subgraph_name
        //subgraph_names[subgraph_id] = subgraph_name;

        const int num_inputs = func->params.size();
        subgraph_num_inputs[subgraph_name] = num_inputs;
        for (int i = 0; i < num_inputs; i++) {
          std::cout << subgraph_name << " input " << i << " is named " << func->params[i]->name_hint() << std::endl;
        }

        const int num_outputs = func->ret_type.as<TensorTypeNode>() ? 1 : func->ret_type.as<TupleTypeNode>()->fields.size();
        subgraph_num_outputs[subgraph_name] = num_outputs;
        std::cout << subgraph_name << " has " << num_inputs << " inputs." << std::endl;
        std::cout << subgraph_name << " has " << num_outputs << " outputs." << std::endl;
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
    }

    //const PackedFunc* pf = runtime::Registry::Get("tvm.contrib.tidl.create");
    //CHECK(pf != nullptr)
    //      << "tvm.contrib.tidl.create was not found in the registry.";
    //// subgraph_names can be used for error checking at runtime
    ////return (*pf)(total_subgraphs, subgraph_names);
    //return (*pf)(total_subgraphs, subgraph_num_inputs, subgraph_num_outputs);
    
    return runtime::TIDLModuleCreate(total_subgraphs, subgraph_num_inputs, subgraph_num_outputs);
  }

 private:
  /*! \brief  */
  std::unordered_map<std::string, int> subgraph_num_inputs;
  std::unordered_map<std::string, int> subgraph_num_outputs;
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
