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

#ifndef TVM_RELAY_BACKEND_TENSORRT_UTILS_H_
#define TVM_RELAY_BACKEND_TENSORRT_UTILS_H_

#include "NvInfer.h"

#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

template <typename T>
nvinfer1::Dims VectorToTrtDims(const std::vector<T>& vec) {
  nvinfer1::Dims dims;
  // Dims(nbDims=0, d[0]=1) is used to represent a scalar in TRT.
  dims.d[0] = 1;
  dims.nbDims = vec.size();
  for (int i = 0; i < vec.size(); ++i) {
    dims.d[i] = vec[i];
  }
  return dims;
}

std::vector<int> TrtDimsToVector(const nvinfer1::Dims& dims) {
  return std::vector<int>(dims.d, dims.d + dims.nbDims);
}

std::vector<int> GetShape(const Type& type, bool remove_batch_dim = false) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK(ttype);
  std::vector<int> _shape;
  const int start_index = remove_batch_dim ? 1 : 0;
  for (int i = start_index; i < ttype->shape.size(); ++i) {
    auto* val = ttype->shape[i].as<IntImm>();
    CHECK(val);
    _shape.push_back(val->value);
  }
  return _shape;
}

std::string DebugString(const std::vector<int>& vec) {
  std::ostringstream ss;
  ss << "(";
  for (int i = 0; i < vec.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << vec[i];
  }
  ss << ")";
  return ss.str();
}

DataType GetType(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK(ttype);
  return ttype->dtype;
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TENSORRT_UTILS_H_
