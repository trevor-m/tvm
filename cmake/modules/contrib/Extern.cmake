# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

message(STATUS "Build with relay.backend.contrib")

# Gcc (for demo purpose)
file(GLOB GCC_RELAY_CONTRIB_SRC src/relay/backend/contrib/gcc/codegen.cc)
list(APPEND COMPILER_SRCS ${GCC_RELAY_CONTRIB_SRC})

# DNNL (for demo purpose)
file(GLOB DNNL_RELAY_CONTRIB_SRC src/relay/backend/contrib/dnnl/codegen.cc)
list(APPEND COMPILER_SRCS ${DNNL_RELAY_CONTRIB_SRC})

# TensorRT (for demo purpose)
if(USE_TENSORRT)
    if(IS_DIRECTORY ${USE_TENSORRT})
        set(TENSORRT_ROOT_DIR ${USE_TENSORRT})
    endif()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES include)
    find_library(TENSORRT_LIB_DIR nvinfer HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES lib)
    find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIB_DIR)
    if(NOT TENSORRT_FOUND)
        message(ERROR "Could not find TensorRT.")
    endif()
    include_directories(${TENSORRT_INCLUDE_DIR})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${TENSORRT_LIB_DIR})
    # Build codegen source
    file(GLOB TENSORRT_RELAY_CONTRIB_SRC src/relay/backend/contrib/tensorrt/*.cc)
    list(APPEND COMPILER_SRCS ${TENSORRT_RELAY_CONTRIB_SRC})
endif()

