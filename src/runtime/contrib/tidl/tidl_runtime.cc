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
 * \file external_runtime_test.cc
 * \brief Test an example runtime module to interpreting a json string.
 *
 * This is an exmaple runtime employed to show how we can interprete and execute
 * a json string that represents a simple computational (sub)graph. Users will
 * mainly need to implement four functions as follows:
 *  - GetFunction. It is used to get the packed function from the json runtime
 * module using a provided function name. This function returns a PackedFunc
 * that can be directly invoked by feeding it with parameters.
 *  - SaveToBinary. This function is used to achieve the serialization purpose.
 * The emitted binary stream can be directly saved to disk so that users can
 * load then back when needed.
 *  - LoadFromBinary. This function uses binary stream to load the json that
 * saved by SaveToBinary which essentially performs deserialization.
 */
#include <stdlib.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../file_util.h"

extern "C" {
extern void TidlRunSubgraph(
    int total_subgraphs,   // passed from TIDL codegen
    int subgraph_id,       // available from TIDL codegen,
                           // also available from 1st argument of GetFunction(): const std::string& name
    int batch_size,        // batch value of input data tensor dimension (N,H,W,C) or (N,C,H,W)
    int num_inputs,        // number of data inputs to the subgraph (not including weights)
    int num_outputs,       // number of data outputs
    float **inputTensors,  // array of input tensors
    float **outputTensors  // array of output tensors
    );
}


namespace tvm {
namespace runtime {

Module TIDLModuleCreate(int total_subgraphs, int subgraph_id);

/*! \brief A module for TIDL runtime. */
class TIDLModule : public runtime::ModuleNode {
 public:
  // TIDL representation for a subgraph is defined in two .bin files:
  //    - tidl_subgraph<subgraph_id>_net.bin
  //    - tidl_subgraph<subgraph_id>_params.bin
  // These two .bin files will be loaded by TIDL shared lib.
  explicit TIDLModule(int total_subgraphs, int subgraph_id) {
    std::cout << "TIDL runtime module: total subgraphs is " << total_subgraphs << ", ";
    std::cout << "subgraph id is " << subgraph_id << std::endl;
    this->total_subgraphs = total_subgraphs;
    this->subgraph_id = subgraph_id;
    //sprintf(this->subgraph_name, "tidl_subgraph_%d_%d", subgraph_id, total_subgraphs);
    subgraph_name = std::to_string(total_subgraphs) + "_" + std::to_string(subgraph_id);
  }

  /* When TVM runtime wants to execute a subgraph with your compiler tag, 
   * TVM runtime invokes this function from your customized runtime module. 
   * It provides the function name as well as runtime arguments, and GetFunction
   * should return a packed function implementation for TVM runtime to execute.
  */
  PackedFunc GetFunction(const std::string& name, // name is tidl_0, tidl_1, etc.
                         const ObjectPtr<Object>& sptr_to_self) final {
//#if TVM_GRAPH_RUNTIME_TIDL
#if 1
    // need to invoke tidl_inference:
    //   - input and output can be found from TVMArgs args??
    //   - total_subgraphs and subgraph_id are passed from args_to_tidl_runtime
    //tidl_inference(input, output, total_subgraphs, subgraph_id);
    std::cout << "subgraph name is " << name << std::endl;
#else
    LOG(FATAL) << "TVM was not built with TIDL runtime enabled. Build "
                << "with USE_TIDL=ON.";
    return PackedFunc();
#endif
  }

  const char* type_key() const { return "tidl"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    std::cout << "SaveToFile: file name is " << file_name << ", subgraph_name is " << subgraph_name << std::endl;
    SaveBinaryToFile(file_name, subgraph_name);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    std::cout << "SaveToBinary: subgraph_name is " << subgraph_name << std::endl;
    stream->Write(subgraph_name);
  }

  static Module LoadFromFile(const std::string& path) {
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string subgraph_name(size, ' ');
    filep.seekg(0);
    filep.read(&subgraph_name[0], size);
    return TIDLModuleCreate(1,0);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string subgraph_name;
    stream->Read(&subgraph_name);
    //get total_subgraphs and subgraph_id from subgraph_name
    return TIDLModuleCreate(1, 0);
  }

 private:
  int total_subgraphs;
  int subgraph_id;
  std::string subgraph_name;
  
};


Module TIDLModuleCreate(int total_subgraphs, int subgraph_id) {
  auto n = make_object<TIDLModule>(total_subgraphs, subgraph_id);
  return Module(n);
}

TVM_REGISTER_GLOBAL("tvm.contrib.tidl.create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TIDLModuleCreate(args[0],args[1]);
});

}  // namespace runtime
}  // namespace tvm
