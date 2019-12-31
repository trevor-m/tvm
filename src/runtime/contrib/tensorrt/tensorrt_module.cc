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
#include "tensorrt_builder.h"
#include "tensorrt_module.h"

#include "NvInfer.h"

namespace tvm {
namespace runtime {

class TensorRTModule : public runtime::ModuleNode {
 public:
  explicit TensorRTModule(const std::string& serialized_subgraph)
      : serialized_subgraph_(serialized_subgraph) {}

  ~TensorRTModule() {
    for (auto& it : trt_engine_cache_) {
      it.second.context->destroy();
      it.second.engine->destroy();
    }
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    // Generate an external packed function
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      auto it = trt_engine_cache_.find(name);
      if (it == trt_engine_cache_.end()) {
        // Build new trt engine and place in cache.
        LOG(INFO) << "Building new TensorRT engine for subgraph " << name;
        auto expr = Downcast<relay::Expr>(LoadJSON(this->serialized_subgraph_));

        auto inputs = ConvertInputs(args);
        auto builder = relay::contrib::TensorRTBuilder(inputs);
        auto engine_and_context = builder.BuildEngine(expr);
        LOG(INFO) << "Finished building engine";
        this->trt_engine_cache_[name] = engine_and_context;
      }

      auto engine_and_context = this->trt_engine_cache_[name];
      this->ExecuteEngine(engine_and_context, args, rv);
    });
  }

  const char* type_key() const { return "tensorrt"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    SaveBinaryToFile(file_name, serialized_subgraph_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(serialized_subgraph_);
  }

  static Module LoadFromFile(const std::string& path) {
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string serialized_subgraph(size, ' ');
    filep.seekg(0);
    filep.read(&serialized_subgraph[0], size);
    return TensorRTModuleCreate(serialized_subgraph);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string serialized_subgraph;
    stream->Read(&serialized_subgraph);
    return TensorRTModuleCreate(serialized_subgraph);
  }

 private:
  std::string serialized_subgraph_;
  std::unordered_map<std::string, TrtEngineAndContext> trt_engine_cache_;

  // Convert TVMArgs to make compatible with VM or graph runtime.
  std::vector<DLTensor*> ConvertInputs(tvm::TVMArgs args) {
    std::vector<DLTensor*> inputs(args.size(), nullptr);
    for (size_t i = 0; i < args.size(); i++) {
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
    const size_t num_outputs = engine_and_context.network_outputs.size();
    CHECK_GT(inputs.size(), num_outputs);
    // TODO(trevmorr): Assumes output is at the end - is this true?
    for (size_t i = 0; i < inputs.size() - num_outputs; ++i) {
      auto it = engine_and_context.network_input_map.find(i);
      if (it != engine_and_context.network_input_map.end()) {
        DLTensor* arg = inputs[i];
        int binding_index = engine->getBindingIndex(it->second.c_str());
        CHECK_NE(binding_index, -1);
        if (!runtime::TypeMatch(arg->dtype, kDLFloat, 32)) {
          LOG(FATAL) << "Only float32 inputs are supported.";
        }
        bindings[binding_index] = reinterpret_cast<float*>(arg->data);
      }
    }
    // Set outputs.
    // TODO(trevmorr): Allow multiple outputs.
    for (size_t i = 0; i < num_outputs; ++i) {
      const int index_in_inputs = inputs.size() - num_outputs + i;
      DLTensor* out_arg = inputs[index_in_inputs];
      int binding_index = engine->getBindingIndex(
          engine_and_context.network_outputs[i].c_str());
      CHECK_NE(binding_index, -1);
      bindings[binding_index] = reinterpret_cast<float*>(out_arg->data);
    }
    // Use batch size from first input.
    const int batch_size = inputs[0]->shape[0];
    CHECK(context->execute(batch_size, bindings.data()))
        << "Running TensorRT failed.";

    // TODO(trevmorr): Look up bindings by name.
    // TODO(trevmorr): Allow multiple outputs.
    *rv = bindings[num_bindings - num_outputs];
  }
};

Module TensorRTModuleCreate(const std::string& serialized_subgraph) {
  auto n = make_object<TensorRTModule>(serialized_subgraph);
  return Module(n);
}

TVM_REGISTER_GLOBAL("module.loadfile_tensorrt")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TensorRTModule::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("module.loadbinary_tensorrt")
.set_body_typed(TensorRTModule::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
