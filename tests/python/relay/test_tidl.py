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
"""Unit tests for TIDL compilation."""

import os
from tvm import relay
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data
from tvm.relay.backend.contrib import tidl

def get_compiler_path():
    arm_gcc_path = os.getenv("ARM_GCC_PATH")
    if arm_gcc_path is None:
        print("Environment variable ARM_GCC_PATH is not set! Model won't be compiled!")
        return None
    else:
        arm_gcc = os.path.join(arm_gcc_path, "arm-linux-gnueabihf-g++")
        if os.path.exists(arm_gcc):
            return arm_gcc
        else:
            print("ARM GCC arm-linux-gnueabihf-g++ does not exist! Model won't be compiled!")
            return None

def get_tidl_tools_path():
    tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
    if tidl_tools_path is None:
        print("Environment variable TIDL_TOOLS_PATH is not set! Model won't be compiled!")
        return None
    else:
        return tidl_tools_path

def model_compile(model_name, mod_orig, params, data_layout, input_node, input_data,
                  num_tidl_subgraphs=1):

    tidl_artifacts_folder = "./artifacts_" + model_name
    if os.path.isdir(tidl_artifacts_folder):
        filelist = [ f for f in os.listdir(tidl_artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(tidl_artifacts_folder, file))
    else:
        os.mkdir(tidl_artifacts_folder)

    tidl_compiler = tidl.TIDLCompiler("AM57", (6, 3),
                                      num_tidl_subgraphs=num_tidl_subgraphs,
                                      data_layout=data_layout,
                                      artifacts_folder=tidl_artifacts_folder,
                                      tidl_tools_path=get_tidl_tools_path())
    mod, status = tidl_compiler.enable(mod_orig, params, {input_node: input_data})
    assert status != -1   # For CI test

    arm_gcc = get_compiler_path()
    if arm_gcc is None:
        print("Skip build because ARM_GCC_PATH is not set")
        return

    if status == 1:
        print("Graph execution with TIDL")
    else:
        print("Graph execution without TIDL")

    target = "llvm -target=armv7l-linux-gnueabihf" # for AM57x or J6 devices
    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    path_lib = os.path.join(tidl_artifacts_folder, "deploy_lib.so")
    path_graph = os.path.join(tidl_artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(tidl_artifacts_folder, "deploy_param.params")
    lib.export_library(path_lib, cc=arm_gcc)
    with open(path_graph, "w") as fo:
        fo.write(graph)
    with open(path_params, "wb") as fo:
        fo.write(relay.save_param_dict(params))

    print("Artifacts can be found at " + tidl_artifacts_folder)
    return

def test_tidl_classification():
    classification_models = ['mobilenet1.0', 'mobilenetv2_1.0', 'resnet101_v1', 'densenet121']
    image_size = 224
    data_layout = "NCHW"
    input_name = "data"

    #======================== Load testing image ===========================
    im_fname = download_testdata('https://github.com/dmlc/mxnet.js/blob/master/' +
                                 'data/cat.png?raw=true', 'cat.png', module='data')
    img_norm, _ = data.transforms.presets.ssd.load_test(im_fname, short=image_size)
    input_data = img_norm.asnumpy()

    for model_name in classification_models:
        #======================== Load the model ========================
        model = model_zoo.get_model(model_name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape={input_name: input_data.shape})

        #======================== Compile the model ========================
        model_compile(model_name, mod, params, data_layout, input_name, input_data)

def test_tidl_object_detection():
    object_detection_models = ['ssd_512_mobilenet1.0_voc', 'yolo3_mobilenet1.0_coco']
    image_size = 512
    data_layout = "NCHW"
    input_name = "data"

    #======================== Load testing image ===========================
    im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')
    img_norm, _ = data.transforms.presets.ssd.load_test(im_fname, short=image_size)
    input_data = img_norm.asnumpy()

    for model_name in object_detection_models:
        #======================== Load the model ========================
        model = model_zoo.get_model(model_name, pretrained=True)
        ssd_mod, ssd_params = relay.frontend.from_mxnet(model, {input_name:input_data.shape})

        #======================== Compile the model ========================
        model_compile(model_name, ssd_mod, ssd_params, data_layout, input_name, input_data)

def test_tidl_segmentation():
    model_name = 'mask_rcnn_resnet18_v1b_coco'
    input_name = "data"
    data_layout = "NCHW"

    #======================== Load testing image =======================
    im_fname = download_testdata('https://raw.githubusercontent.com/dmlc/web-data/master/' +
                                 'gluoncv/segmentation/voc_examples/1.jpg', 'example.jpg',
                                 module='data')
    img_norm, _ = data.transforms.presets.rcnn.load_test(im_fname)
    input_data = img_norm.asnumpy()

    #======================== Load the model ========================
    model = model_zoo.get_model(model_name, pretrained=True)
    seg_mod, seg_params = relay.frontend.from_mxnet(model, {input_name:input_data.shape})

    #======================== Compile the model ========================
    model_compile(model_name, seg_mod, seg_params, data_layout, input_name, input_data)

if __name__ == '__main__':
    test_tidl_classification()
    test_tidl_object_detection()
    test_tidl_segmentation()
