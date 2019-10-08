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
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/utils.h
 * \brief Utils function for backend
 */
#ifndef TVM_RELAY_BACKEND_TENSORRT_UTILS_H_
#define TVM_RELAY_BACKEND_TENSORRT_UTILS_H_
#include "NvInfer.h"

namespace tvm {
namespace relay {
namespace contrib {

nvinfer1::Dims VectorToTrtDims(const std::vector<int>& vec) {
  nvinfer1::Dims dims;
  dims.nbDims = vec.size();
  for (int i = 0; i < vec.size(); ++i) {
    dims.d[i] = vec[i];
  }
  return dims;
}

std::vector<int> GetShape(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK(ttype);
  std::vector<int> _shape;
  for (int i = 0; i < ttype->shape.size(); ++i) {
    auto* val = ttype->shape[i].as<IntImm>();
    CHECK(val);
    _shape.push_back(val->value);
  }
  return _shape;
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
