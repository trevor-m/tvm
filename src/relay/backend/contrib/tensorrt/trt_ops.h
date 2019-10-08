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

#ifndef TVM_RELAY_BACKEND_TRT_OPS_H_
#define TVM_RELAY_BACKEND_TRT_OPS_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <vector>
#include "NvInfer.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {

// Parameters to convert an Op from relay to TensorRT
struct AddTrtLayerParams {
  const CallNode* call;
  nvinfer1::INetworkDefinition* network;
  std::string op_name;
  std::vector<TrtOpInput> inputs;
  std::vector<nvinfer1::ITensor*> outputs;

  AddTrtLayerParams(nvinfer1::INetworkDefinition* network, const CallNode* call)
      : network(network), call(call) {
    op_name = (call->op.as<OpNode>())->name;
  }
};

class TrtOpConverter {
 public:
  // Used to specify whether each input is tensor or weight.
  const std::vector<TrtInputType> input_types;
  // If set to true, any number of tensor inputs can be used for the op.
  const bool variable_input_count;

  TrtOpConverter(const std::vector<TrtInputType>& input_types,
                 bool variable_input_count = false)
      : input_types(input_types), variable_input_count(variable_input_count) {}

  // Convert to TRT. Implementation should use inputs and attributes from the
  // CallNode to add the corresponding TRT layers to network. Outputs should be
  // pushed to outputs vector.
  virtual void Convert(AddTrtLayerParams& params) const = 0;

  // Helper functions.
  nvinfer1::ITensor* Reshape(AddTrtLayerParams& params,
                             nvinfer1::ITensor* input,
                             const std::vector<int>& new_shape) const {
    auto layer = params.network->addShuffle(*input);
    CHECK(layer != nullptr);
    layer->setReshapeDimensions(VectorToTrtDims(new_shape));
    return layer->getOutput(0);
  }
};

class ActivationOpConverter : public TrtOpConverter {
 public:
  ActivationOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    CHECK(params.inputs.size() == 1) << "Activation op expects 1 input.";
    static const std::unordered_map<std::string, nvinfer1::ActivationType>
        op_map = {
            {"nn.relu", nvinfer1::ActivationType::kRELU},
            {"sigmoid", nvinfer1::ActivationType::kSIGMOID},
            {"tanh", nvinfer1::ActivationType::kTANH},
            {"clip", nvinfer1::ActivationType::kCLIP},
            {"nn.leaky_relu", nvinfer1::ActivationType::kLEAKY_RELU},
        };
    auto it = op_map.find(params.op_name);
    CHECK(it != op_map.end()) << "Unsupported activation type "
                              << params.op_name;
    nvinfer1::IActivationLayer* act_layer = params.network->addActivation(
        *params.inputs.at(0).tensor, nvinfer1::ActivationType::kRELU);
    if (params.op_name == "clip") {
      const auto* clip_attr = params.call->attrs.as<ClipAttrs>();
      act_layer->setAlpha(clip_attr->a_min);
      act_layer->setBeta(clip_attr->a_max);
    } else if (params.op_name == "nn.leaky_relu") {
      const auto* leaky_relu_attr = params.call->attrs.as<LeakyReluAttrs>();
      act_layer->setAlpha(leaky_relu_attr->alpha);
    }
    CHECK(act_layer != nullptr);
    params.outputs.push_back(act_layer->getOutput(0));
  }
};

class ElementWiseBinaryOpConverter : public TrtOpConverter {
 public:
  ElementWiseBinaryOpConverter() : TrtOpConverter({kTensor, kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation>
        op_map = {{"add", nvinfer1::ElementWiseOperation::kSUM},
                  {"subtract", nvinfer1::ElementWiseOperation::kSUB},
                  {"multiply", nvinfer1::ElementWiseOperation::kPROD},
                  {"divide", nvinfer1::ElementWiseOperation::kDIV},
                  {"power", nvinfer1::ElementWiseOperation::kPOW}};
    auto it = op_map.find(params.op_name);
    CHECK(it != op_map.end()) << "Unsupported elementwise type "
                              << params.op_name;
    // Broadcast
    auto input0 = params.inputs.at(0).tensor;
    auto input0_dims = TrtDimsToVector(input0->getDimensions());
    auto input1 = params.inputs.at(1).tensor;
    auto input1_dims = TrtDimsToVector(input1->getDimensions());
    const bool need_broadcast = input0_dims.size() != input1_dims.size();
    if (need_broadcast) {
      if (input0_dims.size() < input1_dims.size()) {
        std::vector<int> new_shape(input0_dims);
        while (new_shape.size() < input1_dims.size())
          new_shape.insert(new_shape.begin(), 1);
        input0 = Reshape(params, input0, new_shape);
      } else if (input1_dims.size() < input0_dims.size()) {
        std::vector<int> new_shape(input1_dims);
        while (new_shape.size() < input0_dims.size())
          new_shape.insert(new_shape.begin(), 1);
        input1 = Reshape(params, input1, new_shape);
      }
    }

    nvinfer1::IElementWiseLayer* elemwise_layer =
        params.network->addElementWise(*input0, *input1, it->second);
    CHECK(elemwise_layer != nullptr);
    params.outputs.push_back(elemwise_layer->getOutput(0));
  }
};

class Conv2DOpConverter : public TrtOpConverter {
 public:
  Conv2DOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto input_tensor = params.inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params.inputs.at(1).weight_shape;
    const auto* conv2d_attr = params.call->attrs.as<Conv2DAttrs>();
    CHECK(conv2d_attr->data_layout == "NCHW");
    CHECK(conv2d_attr->out_layout == "" || conv2d_attr->out_layout == "NCHW");
    CHECK(conv2d_attr->kernel_layout == "OIHW");

    // Could use conv2d_attr->channels.as<IntImm>()->value
    const int num_outputs = weight_shape[0];
    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto conv_layer =
        params.network->addConvolution(*input_tensor, num_outputs, kernel_size,
                                       params.inputs.at(1).weight, bias);
    CHECK(conv_layer != nullptr);
    const auto padding =
        nvinfer1::DimsHW(conv2d_attr->padding[0].as<IntImm>()->value,
                         conv2d_attr->padding[1].as<IntImm>()->value);
    conv_layer->setPadding(padding);
    const auto strides =
        nvinfer1::DimsHW(conv2d_attr->strides[0].as<IntImm>()->value,
                         conv2d_attr->strides[1].as<IntImm>()->value);
    conv_layer->setStride(strides);
    const auto dilation =
        nvinfer1::DimsHW(conv2d_attr->dilation[0].as<IntImm>()->value,
                         conv2d_attr->dilation[1].as<IntImm>()->value);
    conv_layer->setDilation(dilation);
    conv_layer->setNbGroups(conv2d_attr->groups);
    params.outputs.push_back(conv_layer->getOutput(0));
  }
};

// Using FullyConnected
class DenseOpConverter : public TrtOpConverter {
 public:
  DenseOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto input_tensor = params.inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    CHECK(input_dims.size() > 0 && input_dims.size() <= 3);
    const bool need_reshape_on_input = input_dims.size() != 3;
    if (need_reshape_on_input) {
      // Add dims of size 1 until rank is 3.
      std::vector<int> new_shape(input_dims);
      while (new_shape.size() < 3) new_shape.insert(new_shape.end(), 1);
      input_tensor = Reshape(params, input_tensor, new_shape);
    }
    // Weights are in KC format.
    CHECK(params.inputs.at(1).weight_shape.size() == 2);
    const int num_units = params.inputs.at(1).weight_shape[0];
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IFullyConnectedLayer* fc_layer =
        params.network->addFullyConnected(*input_tensor, num_units,
                                          params.inputs.at(1).weight, bias);
    CHECK(fc_layer != nullptr);
    auto output_tensor = fc_layer->getOutput(0);
    if (need_reshape_on_input) {
      // Remove added dims.
      input_dims[input_dims.size() - 1] = num_units;
      output_tensor = Reshape(params, output_tensor, input_dims);
    }
    params.outputs.push_back(output_tensor);
  }
};

class BatchNormOpConverter : public TrtOpConverter {
 public:
  BatchNormOpConverter()
      : TrtOpConverter({kTensor, kWeight, kWeight, kWeight, kWeight}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto gamma = params.inputs.at(1).weight;
    auto beta = params.inputs.at(2).weight;
    auto mean = params.inputs.at(3).weight;
    auto var = params.inputs.at(4).weight;
    const auto* bn_attr = params.call->attrs.as<BatchNormAttrs>();
    CHECK_EQ(gamma.count, beta.count);
    CHECK_EQ(gamma.count, mean.count);
    CHECK_EQ(gamma.count, var.count);
    CHECK(bn_attr->axis == 1);

    // TODO(trevmorr): Track these weights in trt_weights_
    void* weight_scale_ptr = malloc(sizeof(float) * gamma.count);
    nvinfer1::Weights weight_scale{nvinfer1::DataType::kFLOAT, weight_scale_ptr,
                                   gamma.count};
    void* weight_shift_ptr = malloc(sizeof(float) * gamma.count);
    nvinfer1::Weights weight_shift{nvinfer1::DataType::kFLOAT, weight_shift_ptr,
                                   gamma.count};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    // fill in the content of weights for the Scale layer
    const float* gamma_ptr = reinterpret_cast<const float*>(gamma.values);
    const float* beta_ptr = reinterpret_cast<const float*>(beta.values);
    const float* mean_ptr = reinterpret_cast<const float*>(mean.values);
    const float* var_ptr = reinterpret_cast<const float*>(var.values);
    float* scale_ptr = reinterpret_cast<float*>(weight_scale_ptr);
    float* shift_ptr = reinterpret_cast<float*>(weight_shift_ptr);
    // TODO(trevmorr): consider parallelizing the following loop
    for (int i = 0; i < gamma.count; ++i) {
      scale_ptr[i] = 1.0 / std::sqrt(var_ptr[i] + bn_attr->epsilon);
      if (bn_attr->scale) {
        scale_ptr[i] *= gamma_ptr[i];
      }
      shift_ptr[i] = -mean_ptr[i] * scale_ptr[i];
      if (bn_attr->center) {
        shift_ptr[i] += beta_ptr[i];
      }
    }
    nvinfer1::IScaleLayer* scale_layer = params.network->addScale(
        *params.inputs.at(0).tensor, nvinfer1::ScaleMode::kCHANNEL,
        weight_shift, weight_scale, power);
    CHECK(scale_layer != nullptr);
    params.outputs.push_back(scale_layer->getOutput(0));
  }
};

class BatchFlattenOpConverter : public TrtOpConverter {
 public:
  BatchFlattenOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    params.outputs.push_back(Reshape(params, params.inputs.at(0).tensor, {-1}));
  }
};

class SoftmaxOpConverter : public TrtOpConverter {
 public:
  SoftmaxOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    const auto* softmax_attr = params.call->attrs.as<SoftmaxAttrs>();
    CHECK(softmax_attr->axis == -1);
    nvinfer1::ISoftMaxLayer* softmax_layer =
        params.network->addSoftMax(*params.inputs.at(0).tensor);
    CHECK(softmax_layer != nullptr);
    params.outputs.push_back(softmax_layer->getOutput(0));
  }
};

class PoolingOpConverter : public TrtOpConverter {
 public:
  PoolingOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto input_tensor = params.inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map =
        {{"nn.max_pool2d", nvinfer1::PoolingType::kMAX},
         {"nn.avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params.op_name);
    CHECK(it != op_map.end()) << "Unsupported pooling type " << params.op_name
                              << " in TensorRT";

    nvinfer1::DimsHW window_size, strides, padding;
    bool count_include_pad = false;
    bool ceil_mode = false;
    if (params.op_name == "nn.max_pool2d") {
      const auto* pool_attr = params.call->attrs.as<MaxPool2DAttrs>();
      CHECK(pool_attr->layout == "NCHW");
      window_size =
          nvinfer1::DimsHW(pool_attr->pool_size[0].as<IntImm>()->value,
                           pool_attr->pool_size[1].as<IntImm>()->value);
      strides = nvinfer1::DimsHW(pool_attr->strides[0].as<IntImm>()->value,
                                 pool_attr->strides[1].as<IntImm>()->value);
      padding = nvinfer1::DimsHW(pool_attr->padding[0].as<IntImm>()->value,
                                 pool_attr->padding[1].as<IntImm>()->value);
      ceil_mode = pool_attr->ceil_mode;
    } else if (params.op_name == "nn.avg_pool2d") {
      const auto* pool_attr = params.call->attrs.as<AvgPool2DAttrs>();
      CHECK(pool_attr->layout == "NCHW");
      window_size =
          nvinfer1::DimsHW(pool_attr->pool_size[0].as<IntImm>()->value,
                           pool_attr->pool_size[1].as<IntImm>()->value);
      strides = nvinfer1::DimsHW(pool_attr->strides[0].as<IntImm>()->value,
                                 pool_attr->strides[1].as<IntImm>()->value);
      padding = nvinfer1::DimsHW(pool_attr->padding[0].as<IntImm>()->value,
                                 pool_attr->padding[1].as<IntImm>()->value);
      ceil_mode = pool_attr->ceil_mode;
      count_include_pad = pool_attr->count_include_pad;
    }

    auto pool_layer =
        params.network->addPooling(*input_tensor, it->second, window_size);
    CHECK(pool_layer != nullptr);
    pool_layer->setStride(strides);
    pool_layer->setPadding(padding);
    pool_layer->setAverageCountExcludesPadding(!count_include_pad);
    if (ceil_mode) {
      pool_layer->setPaddingMode(nvinfer1::PaddingMode::kCAFFE_ROUND_UP);
    } else {
      pool_layer->setPaddingMode(nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN);
    }
    params.outputs.push_back(pool_layer->getOutput(0));
  }
};

class GlobalPoolingOpConverter : public TrtOpConverter {
 public:
  GlobalPoolingOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto input_tensor = params.inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map =
        {{"nn.global_max_pool2d", nvinfer1::PoolingType::kMAX},
         {"nn.global_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params.op_name);
    CHECK(it != op_map.end()) << "Unsupported pooling type " << params.op_name
                              << " in TensorRT";
    const auto* pool_attr = params.call->attrs.as<GlobalPool2DAttrs>();
    CHECK(pool_attr->layout == "NCHW");
    const auto window_size = nvinfer1::DimsHW(input_dims[1], input_dims[2]);
    auto pool_layer =
        params.network->addPooling(*input_tensor, it->second, window_size);
    CHECK(pool_layer != nullptr);
    params.outputs.push_back(pool_layer->getOutput(0));
  }
};

class ExpandDimsOpConverter : public TrtOpConverter {
 public:
  ExpandDimsOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto input_tensor = params.inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    const auto* expand_dims_attr = params.call->attrs.as<ExpandDimsAttrs>();
    CHECK(expand_dims_attr->axis > 0);
    // Subtract 1 for implicit batch dim.
    const int axis = expand_dims_attr->axis - 1;
    for (int i = 0; i < expand_dims_attr->num_newaxis; ++i) {
      input_dims.insert(input_dims.begin() + axis, 1);
    }
    params.outputs.push_back(
        Reshape(params, params.inputs.at(0).tensor, input_dims));
  }
};

class UnaryOpConverter : public TrtOpConverter {
 public:
  UnaryOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams& params) const {
    // The following ops are supported by TRT but don't exist in relay yet:
    // recip, tan, sinh, cosh, asin, acos, asinh, acosh, atanh
    static const std::unordered_map<std::string, nvinfer1::UnaryOperation>
        op_map = {
            {"exp", nvinfer1::UnaryOperation::kEXP},
            {"log", nvinfer1::UnaryOperation::kLOG},
            {"sqrt", nvinfer1::UnaryOperation::kSQRT},
            {"abs", nvinfer1::UnaryOperation::kABS},
            {"negative", nvinfer1::UnaryOperation::kNEG},
            {"sin", nvinfer1::UnaryOperation::kSIN},
            {"cos", nvinfer1::UnaryOperation::kCOS},
            {"atan", nvinfer1::UnaryOperation::kATAN},
            {"ceil", nvinfer1::UnaryOperation::kCEIL},
            {"floor", nvinfer1::UnaryOperation::kFLOOR},
        };
    auto it = op_map.find(params.op_name);
    CHECK(it != op_map.end()) << "Unsupported unary type " << params.op_name;
    nvinfer1::IUnaryLayer* unary_layer =
        params.network->addUnary(*params.inputs.at(0).tensor, it->second);
    CHECK(unary_layer != nullptr);
    params.outputs.push_back(unary_layer->getOutput(0));
  }
};

class ConcatOpConverter : public TrtOpConverter {
 public:
  ConcatOpConverter()
      : TrtOpConverter({}, /*variable_input_count=*/true) {}

  void Convert(AddTrtLayerParams& params) const {
    const int num_inputs = params.inputs.size();
    CHECK(num_inputs > 0);
    std::vector<nvinfer1::ITensor*> input_tensors;
    for (auto input : params.inputs) {
      CHECK(input.type == kTensor);
      input_tensors.push_back(input.tensor);
    }

    const auto* concat_attr = params.call->attrs.as<ConcatenateAttrs>();
    CHECK(concat_attr->axis >= 0) << "Negative axis not implemented.";
    CHECK(concat_attr->axis != 0) << "Can't concat on batch dimension.";
    // Subtract 1 for implicit batch dimension.
    const int trt_axis = concat_attr->axis - 1;

    nvinfer1::IConcatenationLayer* concat_layer =
        params.network->addConcatenation(input_tensors.data(), num_inputs);
    CHECK(concat_layer != nullptr);
    concat_layer->setAxis(trt_axis);
    params.outputs.push_back(concat_layer->getOutput(0));
  }
};

class BiasAddOpConverter : public TrtOpConverter {
 public:
  BiasAddOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto input_tensor = params.inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    CHECK(input_dims.size() > 0 && input_dims.size() <= 3);
    const bool need_reshape_on_input = input_dims.size() != 3;
    if (need_reshape_on_input) {
      // Add dims of size 1 until rank is 3.
      std::vector<int> new_shape(input_dims);
      while (new_shape.size() < 3) new_shape.insert(new_shape.end(), 1);
      input_tensor = Reshape(params, input_tensor, new_shape);
    }

    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IScaleLayer* scale_layer =
        params.network->addScale(*input_tensor, nvinfer1::ScaleMode::kCHANNEL,
                                 params.inputs.at(1).weight, shift, power);
    CHECK(scale_layer != nullptr);
    auto output_tensor = scale_layer->getOutput(0);
    if (need_reshape_on_input) {
      // Remove added dims.
      // input_dims[input_dims.size() - 1] = num_units;
      output_tensor = Reshape(params, output_tensor, input_dims);
    }
    params.outputs.push_back(output_tensor);
  }
};

class Conv2DTransposeOpConverter : public TrtOpConverter {
 public:
  Conv2DTransposeOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams& params) const {
    auto input_tensor = params.inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params.inputs.at(1).weight_shape;
    const auto* conv2d_attr = params.call->attrs.as<Conv2DTransposeAttrs>();
    CHECK(conv2d_attr->data_layout == "NCHW");
    CHECK(conv2d_attr->out_layout == "" || conv2d_attr->out_layout == "NCHW");
    CHECK(conv2d_attr->kernel_layout == "OIHW");
    CHECK(conv2d_attr->dilation[0].as<IntImm>()->value == 1 &&
          conv2d_attr->dilation[1].as<IntImm>()->value == 1);

    // Could use conv2d_attr->channels.as<IntImm>()->value
    const int num_outputs = weight_shape[0];
    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto deconv_layer =
        params.network->addDeconvolution(*input_tensor, num_outputs, kernel_size,
                                         params.inputs.at(1).weight, bias);
    CHECK(deconv_layer != nullptr);
    const auto padding =
        nvinfer1::DimsHW(conv2d_attr->padding[0].as<IntImm>()->value,
                         conv2d_attr->padding[1].as<IntImm>()->value);
    deconv_layer->setPadding(padding);
    const auto strides =
        nvinfer1::DimsHW(conv2d_attr->strides[0].as<IntImm>()->value,
                         conv2d_attr->strides[1].as<IntImm>()->value);
    deconv_layer->setStride(strides);
    deconv_layer->setNbGroups(conv2d_attr->groups);
    params.outputs.push_back(deconv_layer->getOutput(0));
  }
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TRT_OPS_H_
