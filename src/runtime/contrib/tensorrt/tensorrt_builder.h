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

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <string>
#include <unordered_map>
#include <vector>
#include "NvInfer.h"

#define TRT_VERSION_GE(major, minor, patch)                    \
  ((NV_TENSORRT_MAJOR > major) ||                              \
  (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR > minor) || \
  (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
  NV_TENSORRT_PATCH >= patch))

#include "tensorrt_logger.h"

namespace tvm {
namespace runtime {

struct TrtEngineAndContext {
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;
  std::unordered_map<int, std::string> network_input_map;
  std::vector<std::string> network_outputs;
};

}  // namespace runtime

namespace relay {
namespace contrib {

enum TrtInputType {
  kTensor,
  kWeight,
};

// An input to a TrtOpConverter. The type of the input is either kTensor or
// kWeight. For kTensor, "tensor" contains the input tensor. For kWeight,
// "weight" contains the input weight and "weight_shape" contains the shape.
struct TrtOpInput {
  TrtInputType type;
  nvinfer1::ITensor* tensor;
  nvinfer1::Weights weight;
  std::vector<int> weight_shape;

  explicit TrtOpInput(nvinfer1::ITensor* tensor)
      : tensor(tensor), type(kTensor) {}
  TrtOpInput(nvinfer1::Weights weight, const std::vector<int>& shape)
      : weight(weight), type(kWeight), weight_shape(shape) {}
};

// An ExprVisitor to convert a relay expression into a TensorRT engine and
// execution context.
class TensorRTBuilder : public ExprVisitor {
 public:
  explicit TensorRTBuilder(const std::vector<DLTensor*>& args);

  void VisitExpr_(const VarNode* node) final;

  void VisitExpr_(const ConstantNode* node) final;

  void VisitExpr_(const TupleGetItemNode* op) final;

  void VisitExpr_(const TupleNode* op) final;

  void VisitExpr_(const CallNode* call) final;

  // Convert Expr into TensorRT.
  runtime::TrtEngineAndContext BuildEngine(const Expr& expr);

 private:
  nvinfer1::Weights GetNdArrayAsWeights(const runtime::NDArray& array,
                                        DLDeviceType src_device);

  nvinfer1::Weights GetDLTensorAsWeights(DLTensor* dptr,
                                         DLDeviceType src_device);

  // Gets value from execution args and converts to constant weight stored in
  // node_output_map_ with node as the key.
  void GetInputAsWeights(const VarNode* node);

  // Gets value from ConstantNode data and converts to constant weight stored in
  // node_output_map_ with node as the key.
  void GetConstantAsWeights(const ConstantNode* node);

  // Temporary workaround for transposed weights.
  void GetInputAsTransposedWeights(const CallNode* transpose,
                                   const VarNode* node);

  // Deallocates weights and destroys network definition.
  void CleanUp();

  // Get corresponding index for VarNode in execution_args_.
  int TrackVarNode(const VarNode* node);

  // Maps a node to its outputs.
  std::unordered_map<const ExprNode*, std::vector<TrtOpInput>> node_output_map_;

  // TensorRT builder and network definition.
  runtime::TensorRTLogger logger_;
  nvinfer1::IBuilder* builder_;
  nvinfer1::INetworkDefinition* network_;

  // List of all weights held in memory.
  std::vector<nvinfer1::Weights> trt_weights_;

  // Execution inputs from this invocation.
  const std::vector<DLTensor*>& execution_args_;
  int batch_size_;

  // Maps execution_args_ input index -> TRT input tensor name / VarNode
  // name_hint.
  std::unordered_map<int, std::string> network_input_map_;
};

// Helper functions for GetInputAsTransposedWeights.
void TransposeRSCKtoKCRS(const std::vector<int>& original_shape,
                         const float* input_values, float* output_values);
void TransposeRSCKtoCKRS(const std::vector<int>& original_shape,
                         const float* input_values, float* output_values);
void TransposeCKtoKC(const std::vector<int>& original_shape,
                     const float* input_values, float* output_values);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_
