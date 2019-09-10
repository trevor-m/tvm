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
#include <dlpack/dlpack.h>
#include <stdlib.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/contrib_codegen.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/util.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tvm {
namespace relay {
namespace contrib {

typedef void (*CblasFloat)(float* a, float* b, float* out, int M, int N, int K);

class CblasModuleNode : public ExternModuleNodeBase {
 public:
  const std::string GetExternLibPath() override {
    return "/tmp/relay_extern_cblas.so";
  }

  /*!
   * \brief Get a PackedFunc from module, which is a function ptr can be invoked
   * for execution given some parameters.
   *
   * \param name the name of the external function.
   * \param func_s The function symbol retrieved from the external library.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  runtime::PackedFunc InvokeExternFunc(const std::string& name, void* func_s,
                                       const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    if (name == "nn.dense") {
      func_s_ = reinterpret_cast<CblasFloat>(func_s);
      return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
        CHECK_EQ(args.size(), 3U);
        runtime::NDArray data = args[0];
        runtime::NDArray weight = args[1];
        runtime::NDArray out = args[2];

        const DLTensor* dptr = data.operator->();
        CHECK(runtime::TypeMatch(dptr->dtype, kDLFloat, 32));

        float* d_data = reinterpret_cast<float*>(data->data);
        float* weight_data = reinterpret_cast<float*>(weight->data);
        float* out_data = reinterpret_cast<float*>(out->data);

        int M = CountRow(data);
        int N = CountColumn(weight);
        int K = CountColumn(data);
        (*func_s_)(d_data, weight_data, out_data, M, N, K);
        *rv = out;
      });
    } else {
      LOG(INFO) << name << " is not Supported. Only nn.dense is supported so far";
      return PackedFunc();
    }
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
    return "CblasModule";
  }

  void Build(const Expr& expr) override {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "CBLAS expects a single convolution or dense op.";

    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "CBLAS expects a single convolution or dense op.";
    Op op = GetRef<Op>(op_node);
    if (op == Op::Get("nn.conv2d")) {
      // const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
      // TODO(@zhiics) Generate the template.
      ;
    } else if (op == Op::Get("nn.dense")) {
      // TODO(@zhiics) Generate the template.
      //const auto* dense_attr = call->attrs.as<DenseAttrs>();
      ;
    } else {
      LOG(FATAL) << "CBLAS expects a single convolution or dense op.";
    }

    if (!std::getenv("MKLROOT")) {
      LOG(FATAL) << "MKLROOT not found. Did you source mklvars.sh?";
    }
    int ret = std::system("g++ -O2 -Wall -std=c++11 -shared -fPIC "
                          "src/relay/backend/contrib/cblas/libs.cc "
                          "-o /tmp/relay_extern_cblas.so -ldl -lpthread -lm -lmkl_rt");
    if (ret != 0) {
      LOG(FATAL) << "Failed to compile CBLAS library. Error code: " << ret;
    }
  }

 private:
  // Get the number of row of a ndarray.
  int CountRow(const runtime::NDArray& data) {
    const DLTensor* tensor = data.operator->();
    return tensor->shape[0];
  }

  // Get the number of columns of a ndarray.
  int CountColumn(const runtime::NDArray& data) {
    const DLTensor* tensor = data.operator->();
    return tensor->shape[1];
  }

  CblasFloat func_s_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 */
runtime::Module CblasCompiler(const Expr& expr) {
  std::shared_ptr<CblasModuleNode> n = std::make_shared<CblasModuleNode>();
  n->Build(expr);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.cblas")
.set_body_typed(CblasCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
