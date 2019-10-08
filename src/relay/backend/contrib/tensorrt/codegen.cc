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
#include <dlfcn.h>
#include <stdlib.h>
#include <tvm/relay/contrib_codegen.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include "utils.h"
#include <unordered_map>
// #include "libs.h"
#include "NvInfer.h"

namespace tvm {
namespace relay {
namespace contrib {

  // Logger for TensorRT info/warning/errors
class TensorRTLogger : public nvinfer1::ILogger {
 public:
  TensorRTLogger(): TensorRTLogger(Severity::kWARNING) {}
  explicit TensorRTLogger(Severity severity): reportable_severity(severity) {}
  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    // if (severity > reportable_severity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR: LOG(ERROR) << "INTERNAL_ERROR: " << msg; break;
      case Severity::kERROR: LOG(ERROR) << "ERROR: " << msg; break;
      case Severity::kWARNING: LOG(WARNING) << "WARNING: " << msg; break;
      case Severity::kINFO: LOG(INFO) << "INFO: " << msg; break;
      default: LOG(INFO) << "UNKNOWN: " << msg; break;
    }
  }
 private:
  Severity reportable_severity{Severity::kWARNING};
};

// Parameters to convert an Op from relay to TensorRT
struct AddTrtLayerParams {
  const CallNode* call;
};

struct TrtEngineAndContext {
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;
};


void AddActivation(const AddTrtLayerParams& params) {
  // CHECK(call->args.size() == 1) << "Activation requires one input.";
  //params.call->args[0]
  //nvinfer1::IActivationLayer* act_layer = network->addActivation(*data, it->second);
}

using AddTrtLayer = std::function<void(
    const AddTrtLayerParams& params)>;

static const std::unordered_map<std::string, AddTrtLayer> add_trt_layer_funcs = {
    // {{"conv2d", AddConvolution},
    //  {"batch_norm", AddBatchNorm},
     {"nn.relu", AddActivation},
    //  {"sigmoid", AddActivation},
    //  {"tanh", AddActivation},
    //  {"clip", AddActivation},
    //  {"add", AddElementWiseBinaryOp},
    //  {"elemwise_sub", AddElementWiseBinaryOp},
    //  {"elemwise_mul", AddElementWiseBinaryOp},
    //  {"elemwise_div", AddElementWiseBinaryOp},
    //  {"elemwise_pow", AddElementWiseBinaryOp},
    //  {"max_pool2d", AddPooling},
    //  {"avg_pool2d", AddPooling},
    //  {"global_max_pool2d", AddPooling},
    //  {"global_avg_pool2d", AddPooling},
    //  {"dense", AddFullyConnected},
    //  {"softmax", AddSoftmax},
    //  {"concatenate", AddConcatenate},
    //  {"conv2d_transpose", AddDeconvolution},
    //  {"slice_like", AddSliceLike},
    };

// FIXME: This is an experimental implementation. We should implement all utilities
// and make a base class such as ExternBuilder for users to implement.
class TrtBuilder : public ExprVisitor {
 public:
  TrtBuilder(std::string id) {
    this->_subgraph_id = id;

    // Init TRT stuff
    static TensorRTLogger trt_logger;
    _builder = nvinfer1::createInferBuilder(trt_logger);
    _builder->setMaxBatchSize(1);
    _builder->setMaxWorkspaceSize(1 << 29);
    //_builder->setFp16Mode(use_fp16_);
    _network = _builder->createNetwork();
  }

  void VisitExpr_(const VarNode* node) final {
    LOG(INFO) << "Adding input " << node->name_hint();

    const std::string& tensor_name = node->name_hint();
    auto shape = GetShape(node->checked_type());
    nvinfer1::Dims dims = VectorToTrtDims(shape);
    auto type = GetType(node->checked_type());
    CHECK(type.is_float()) << "Only FP32 inputs are supported.";
    auto tensor = _network->addInput(tensor_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
    _trt_tensors[tensor_name] = tensor;

    _subgraph_args.push_back(node->name_hint());
    _out.clear();
    _out.push_back(tensor_name);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    ; // Do nothing
  }

  void VisitExpr_(const CallNode* call) final {
    // Ensure that nodes are processed in topological order by visiting their
    // inputs first.
    for (int i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
    }

    // auto type_node = call->checked_type().as<TensorTypeNode>();
    // CHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
    //     << "Only support single output tensor with float type";


    const std::string& op_name = (call->op.as<OpNode>())->name;
    // auto it = add_trt_layer_funcs.find(op_name);
    // CHECK(it != add_trt_layer_funcs.end()) << "Unsupported operator conversion to TRT, op name: "
    //     << op_name;

    LOG(INFO) << "Processing op " << op_name;
    // Prepare args
    //AddTrtLayerParams params;
    // Call convert
    //it->second(params);
    // TODO: Update output buffer
    _out.clear();
    _out.push_back({"hi"});
  }

  TrtEngineAndContext BuildEngine() {
    nvinfer1::ICudaEngine* engine = _builder->buildCudaEngine(*_network);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    return {engine, context};
  }

 private:
  // Unused??
  std::string _subgraph_id = "";
  int _func_idx = 0;
  int _buf_idx = 0;
  std::vector<std::string> _subgraph_args;
  std::vector<std::string> _subgraph_body;
  std::vector<std::string> _func_decl;
  std::vector<std::string> _buf_decl;
  std::vector<std::string> _out;

  // For TRT conversion
  nvinfer1::IBuilder* _builder;
  nvinfer1::INetworkDefinition* _network;
  std::unordered_map<std::string, nvinfer1::ITensor*> _trt_tensors;
};

class TrtModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(std::string id = "") const override {
    // TensorRT doesn't create external libraries.
    return {};
  }

  const std::string GetPrefix() const override {
    return "tensorrt_";
  }

  /*!
   * \brief Get the source code of the external module.
   *
   * \param format The format of the source code.
   *
   * \return The source code of the external library module in the text form.
   */
  TVM_DLL std::string GetSource(const std::string& format = "") override {
    return "";
  }

  const char* type_key() const override {
    return "TrtModule";
  }

  runtime::PackedFunc GetFunction(const std::string& name,
                                  const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    _curr_id = GetSubgraphID(name);
    // Open(this->GetExternLibPaths(_curr_id));
    // CHECK(handle_) << "The external module has not been built or failed to open.\n";
    LOG(INFO) << "get function";
    // Generate an external packed function
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      const DLTensor* dptr = ((runtime::NDArray)args[0]).operator->();
      runtime::NDArray out_arg = args[args.size() - 1];
      auto out = reinterpret_cast<float*>(out_arg->data);
      LOG(INFO) << "At TRT Runtime: " << this->_serialized_json;

      auto it = _trt_engine_cache.find(_curr_id);
      if (it == _trt_engine_cache.end()) {
        // Build new trt engine and place in cache.
        Expr e = LoadJSON<Expr>(this->_serialized_json);
        auto builder = TrtBuilder(GetPrefix() + _curr_id);
        builder.VisitExpr(e);
        auto engine_and_context = builder.BuildEngine();
        // std::string code = builder.build();
        LOG(INFO) << "built engine code";
        //_trt_engine_cache[_curr_id] = code;
      }
      
      // Get function from the library
      // std::string encoded_name = GetPrefix() + _curr_id;
      // auto func_s = reinterpret_cast<GccSubgraphFunc>(GetSymbol(encoded_name));

      // Reinterpret data and function to the right type and invoke
      // if (runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
      //   GccPackedArgs packed_args;
      //   packed_args.data = (float**)malloc(sizeof(float*) * args.size());
      //   for (int i = 0; i < args.size() - 1; ++i) {
      //     runtime::NDArray arg = args[i];
      //     packed_args.data[i] = reinterpret_cast<float*>(arg->data);
      //   }
      //   (*func_s)(packed_args, out);
      // } else {
      //   LOG(FATAL) << "Only support float32 type.";
      // }
      *rv = out;
    });
  }

  void Build(const Expr& expr) override {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    _serialized_json = SaveJSON(func->body);
    LOG(INFO) << "Serialized relay subgraph for TRT conversion.";
  }

 private:
  std::string _curr_id;
  std::string _serialized_json;
  std::unordered_map<std::string, TrtEngineAndContext> _trt_engine_cache;
  std::unordered_map<std::string, nvinfer1::ITensor*> _trt_tensors;
};


/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 */
runtime::Module TrtCompiler(const Expr& expr) {
  std::shared_ptr<TrtModuleNode> n = std::make_shared<TrtModuleNode>();
  n->Build(expr);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.tensorrt")
.set_body_typed(TrtCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
