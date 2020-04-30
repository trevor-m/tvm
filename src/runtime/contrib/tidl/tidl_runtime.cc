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

#include <dmlc/logging.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../file_util.h"

//#define MAX_INPUT_TENSORS  4
//#define MAX_OUTPUT_TENSORS 4
//#define MAX_BATCH_SIZE     256

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

inline std::string ToJSON(int total_subgraphs,
                          const std::unordered_map<std::string, int>& num_inputs_,
                          const std::unordered_map<std::string, int>& num_outputs_)
{
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  writer.BeginObject();
  writer.WriteObjectKeyValue("total subgraphs",  total_subgraphs);
  writer.WriteObjectKeyValue("subgraph inputs",  num_inputs_);
  writer.WriteObjectKeyValue("subgraph outputs", num_outputs_);
  writer.EndObject();
  return os.str();
}

inline void FromJSON(const std::string& str,
                     int &total_subgraphs,
                     std::unordered_map<std::string, int>& num_inputs,
                     std::unordered_map<std::string, int>& num_outputs)
{
  std::istringstream is(str);
  dmlc::JSONReader reader(&is);
  dmlc::JSONObjectReadHelper helper;
  // Read total subgraphs
  helper.DeclareField("total subgraphs", &total_subgraphs);
  // Read num_inputs
  helper.DeclareField("subgraph inputs", &num_inputs);
  // Read num_outputs
  helper.DeclareField("subgraph outputs",&num_outputs);
  helper.ReadAllFields(&reader);
}

namespace tvm {
namespace runtime {

typedef void (*tidl_subgraph_t)(int, int, int, int, int, float **, float **);

Module TIDLModuleCreate(int total_subgraphs,
                        const std::unordered_map<std::string, int>& num_inputs,
                        const std::unordered_map<std::string, int>& num_outputs);

/*! \brief A module for TIDL runtime. */
class TIDLModule : public runtime::ModuleNode {
 public:
  explicit TIDLModule(int total_subgraphs, 
                      const std::unordered_map<std::string, int>& num_inputs,
                      const std::unordered_map<std::string, int>& num_outputs) {
    std::cout << "TIDL runtime module: total subgraphs is " << total_subgraphs << ", " << std::endl;
    for (auto it : num_inputs) 
      std::cout << "Subgraph " << it.first << ": numinputs is " << it.second << std::endl;
    for (auto it : num_outputs) 
      std::cout << "Subgraph " << it.first << ": numoutputs is " << it.second << std::endl;

    this->total_subgraphs_ = total_subgraphs;
    this->num_inputs_  = num_inputs;
    this->num_outputs_ = num_outputs;
    this->tidl_handle  = NULL;
  }

  void TidlInit() {
    // Load TIDL shared library and get graph execute function
    if(!tidl_handle) {
      // reset errors
      dlerror();
      tidl_handle = dlopen("libtidl_api.so", RTLD_NOW | RTLD_GLOBAL );
      const char *dlsym_error1 = dlerror();
      if (dlsym_error1) {
        LOG(FATAL) << "Cannot open libtidl_api.so! " << dlsym_error1 << '\n';
      }
      dlerror();
      tidl_subgraph = (tidl_subgraph_t) dlsym(tidl_handle, "TidlRunSubgraph");
      const char *dlsym_error2 = dlerror();
      if (dlsym_error2) {
         LOG(FATAL) << "Cannot load symbol 'TidlRunSubgraph': " << dlsym_error2 << '\n';
         dlclose(tidl_handle);
         //return;
      }
    }
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
    std::cout << "TVM runtime GetFunc: subgraph " << name << std::endl;
    if (name.find("tidl_") == std::string::npos) {
      std::cout << "Subgraph name doesn't contain \"tidl_\"!" << std::endl;
      return PackedFunc(nullptr);
    }
    std::cout << "Initializing TIDL ..." << std::endl;
    TidlInit();  // Question - should TidlInit() be called here??
    std::cout << "TIDL Initialized. Now running inference... " << std::endl;

    // get subgraph id from name
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      std::string subgraph_name = (std::string)name;
      // Get subgraph id which is after "tidl_" (5 characters)
      int subgraph_id = std::stoi(subgraph_name.erase(0,5));
      std::cout << "TIDL subgraph name: " << name << " , subgraph id: " << subgraph_id << std::endl;
      DLTensor *arg0 = (DLTensor *)args[0];
      const int batch_size = arg0->shape[0];
      int num_inputs  = num_inputs_[name];
      int num_outputs = num_outputs_[name];
      //float** inputs  = new float*[num_inputs];
      //float** outputs = new float*[num_outputs];
      std::vector<float *> inputs;
      std::vector<float *> outputs;
      //float *inputs[MAX_INPUT_TENSORS*MAX_BATCH_SIZE];
      //float *outputs[MAX_OUTPUT_TENSORS*MAX_BATCH_SIZE];
      for (int i = 0; i < num_inputs; i++) {
        DLTensor *arg = (DLTensor *)args[i];
        //inputs[i] = reinterpret_cast<float*>(arg->data);
        inputs.push_back(reinterpret_cast<float*>(arg->data));
      }
      for (int i = 0; i < num_outputs; i++) {
        const int index_in_args = num_inputs + i;
        DLTensor *arg = (DLTensor *)args[index_in_args];
        //outputs[i] = reinterpret_cast<float*>(arg->data);
        outputs.push_back(reinterpret_cast<float*>(arg->data));
      }

      // ... Execute TidlRunSubgraph ...
      std::cout << "Invoking TIDL with args: " << total_subgraphs_ << subgraph_id;
      std::cout << batch_size << num_inputs << num_outputs << std::endl;
      tidl_subgraph(total_subgraphs_, subgraph_id, batch_size, 
                    num_inputs, num_outputs, &inputs[0], &outputs[0]);
      std::cout << "TIDL execution finished." << std::endl;
    });
#else
    LOG(FATAL) << "TVM was not built with TIDL runtime enabled. Build "
                << "with USE_TIDL=ON.";
    return PackedFunc();
#endif
  }

  const char* type_key() const { return "tidl"; }

  // used for lib.save in case of whole graph offload compilation
  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    std::cout << "TIDL SaveToFile: file name is " << file_name << std::endl;
    SaveBinaryToFile(file_name, ToJSON(total_subgraphs_, num_inputs_, num_outputs_));
  }

  // used for export in case of heterogeneous execute
  void SaveToBinary(dmlc::Stream* stream) final {
    std::cout << "TIDL SaveToBinary: " << std::endl;
    stream->Write(ToJSON(total_subgraphs_, num_inputs_, num_outputs_));
  }

  static Module LoadFromFile(const std::string& path) {
    int total_subgraphs;
    std::unordered_map<std::string, int> num_inputs;
    std::unordered_map<std::string, int> num_outputs;
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string graph_info(size, ' ');
    filep.seekg(0);
    filep.read(&graph_info[0], size);
    FromJSON(graph_info, total_subgraphs, num_inputs, num_outputs);
    std::cout << "TIDL load from file..." << std::endl;

    return TIDLModuleCreate(total_subgraphs, num_inputs, num_outputs);
  }

  static Module LoadFromBinary(void* strm) {
    int total_subgraphs;
    std::unordered_map<std::string, int> num_inputs;
    std::unordered_map<std::string, int> num_outputs;
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string graph_info;
    stream->Read(&graph_info);
    FromJSON(graph_info, total_subgraphs, num_inputs, num_outputs);
    return TIDLModuleCreate(total_subgraphs, num_inputs, num_outputs);
  }

 private:
  int total_subgraphs_;
  //std::string subgraph_name;
  std::unordered_map<std::string, int> num_inputs_;
  std::unordered_map<std::string, int> num_outputs_;
  void *tidl_handle;
  tidl_subgraph_t tidl_subgraph;
};

Module TIDLModuleCreate(int total_subgraphs, 
                        const std::unordered_map<std::string, int>& num_inputs,
                        const std::unordered_map<std::string, int>& num_outputs) {
  auto n = make_object<TIDLModule>(total_subgraphs, num_inputs, num_outputs);
  return Module(n);
}

/*
TVM_REGISTER_GLOBAL("tvm.contrib.tidl.create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TIDLModuleCreate(args[0],args[1],args[2]);
});
*/

TVM_REGISTER_GLOBAL("runtime.module.loadfile_tidl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TIDLModule::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tidl")
.set_body_typed(TIDLModule::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
