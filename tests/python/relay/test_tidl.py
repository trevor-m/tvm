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
import sys

import argparse
parser = argparse.ArgumentParser(description="Testing TIDL code generation")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_tidl_subgraphs", type=int, default=1, help="number of TIDL subgraphs")
parser.add_argument("--force_arm_only", help="Force ARM-only execution", action='store_true', default=False)

try:
    args = parser.parse_args()
except SystemExit:
    quit()

batch_size = args.batch_size
num_tidl_subgraphs = args.num_tidl_subgraphs
force_arm_only = args.force_arm_only

if os.getenv("ARM_GCC_PATH") is None:
  sys.exit("Environment variable ARM_GCC_PATH is not set!")
else: 
  arm_gcc_path = os.getenv("ARM_GCC_PATH")
arm_gcc = os.path.join(arm_gcc_path, "arm-linux-gnueabihf-g++")
target  = "llvm -target=armv7l-linux-gnueabihf"

import numpy as np
import onnx
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.contrib import cc
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
import mxnet as mx
from mxnet import image
from matplotlib import pyplot as plt
import tensorflow as tf
from tvm.relay.testing import tf as tf_testing
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.presets.segmentation import test_transform
from tvm.relay.backend.contrib import tidl

def model_compile(model_name, mod_orig, params, 
                  data_layout, input_node, input_data):
    artifacts_folder = "./artifacts_" + model_name
    if os.path.isdir(artifacts_folder):
        filelist = [ f for f in os.listdir(artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(artifacts_folder, file))
    else:
        os.mkdir(artifacts_folder)

    if force_arm_only is True:
        mod = mod_orig
        tidlEnabled = False
    else:
        tidl_compiler = tidl.TIDLCompiler("AM57", (6,3), 
                                          num_tidl_subgraphs = num_tidl_subgraphs, 
                                          data_layout = data_layout, 
                                          artifacts_folder = artifacts_folder) 
        input = {input_node: input_data}
        mod = tidl_compiler.enable(mod_orig, params, input)
        tidlEnabled = True
        if mod == None:  # TIDL cannot be enabled - no offload to TIDL
            mod = mod_orig
            tidlEnabled = False

    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    path_lib    = os.path.join(artifacts_folder, "deploy_lib.so")
    path_graph  = os.path.join(artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(artifacts_folder, "deploy_param.params")
    lib.export_library(path_lib, cc=arm_gcc)
    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))

    return tidlEnabled

def load_gluoncv_model(model, x, input_name, input_shape, dtype):
    block = model_zoo.get_model(model, pretrained=True)

    if 'faster' in model:
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
    else:
        block.hybridize()
        block.forward(x)
        block.export('temp') # create file temp-symbol.json and temp-0000.params

        model_json = mx.symbol.load('temp-symbol.json')
        save_dict = mx.ndarray.load('temp-0000.params')
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            elif tp == 'aux':
                aux_params[name] = v
        mod, params = relay.frontend.from_mxnet(model_json, {input_name: input_shape}, arg_params=arg_params, aux_params=aux_params)
    return block, mod, params

def test_tidl_classification(model_name):
    dtype = 'float32'
    input_shape = (1, 3, 224, 224)
    data_layout = "NCHW"
    input_node  = "data"
    input_image = "./dog.npy"
    if os.path.exists(input_image):
        x = np.load(input_image)  # "NCHW"
        input_data = x/np.amax(np.abs(x))
    else:
        input_data = np.random.uniform(-1, 1, input_shape).astype(dtype)

    model = model_zoo.get_model(model_name, pretrained=True)
    relay_mod, relay_params = relay.frontend.from_mxnet(model, shape={input_node: input_shape}, dtype=dtype)

    #======================== Compile the model ========================
    assert model_compile(model_name, relay_mod, relay_params, data_layout, input_node, input_data)


def test_tidl_object_detection(model_name):
    im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')
    image_size = 512
    data_layout = "NCHW"
    dtype = "float32"
    input_name = "data"

    #======================== Load testing image and model ====================
    input_shape = (1, 3, image_size, image_size)
    x, img = data.transforms.presets.ssd.load_test(im_fname, short=image_size)
    block, ssd_mod, ssd_params = load_gluoncv_model(model_name, x, input_name, input_shape, dtype)

    #======================== Execute the full graph on TVM ====================
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(ssd_mod, "llvm", params=ssd_params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    input_data = x.asnumpy()
    np.save("ssd_input.npy",input_data) # to be used by inference testing on the target
    mod.set_input(input_name, input_data)
    mod.set_input(**params)
    mod.run()
    class_IDs, scores, bounding_boxs = mod.get_output(0), mod.get_output(1), mod.get_output(2)
    ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                             class_IDs.asnumpy()[0], class_names=block.classes)
    plt.savefig("gluoncv_"+model_name+"_tvm.png")
    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    print("Number of outputs: " + str(len(results)))

    #======================== Compile the model ========================
    assert model_compile(model_name, ssd_mod, ssd_params, data_layout, input_name, input_data)

def test_tidl_segmentation(model_name):
    input_name = "data"
    data_layout = "NCHW"
    dtype = "float32"

    #======================== Load testing image and model ====================
    img_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg'
    filename = 'example.jpg'
    utils.download(img_url, filename)
    img = image.imread(filename)
    img = test_transform(img, ctx = mx.cpu(0))
    input_shape = img.shape
    input_data = img.asnumpy()
    np.save("seg_input.npy",input_data) # to be used by inference testing on the target
    model = model_zoo.get_model(model_name, pretrained=True)
    model.hybridize()
    model.forward(img)
    model.export('gluoncv-temp')    # create file gluoncv-temp-symbol.json
    seg_mod, seg_params = relay.frontend.from_mxnet(model, {input_name:input_shape})

    #======================== Compile the model ========================
    assert model_compile(model_name, seg_mod, seg_params, data_layout, input_name, input_data)

if __name__ == '__main__':
    classification_models   = ['mobilenet1.0',
                               'mobilenetv2_1.0',
                              ]
    object_detection_models = ['ssd_512_mobilenet1.0_voc',
                               'yolo3_mobilenet1.0_coco',
                              ]
    segmentation_models     = ['mask_rcnn_fpn_resnet18_v1b_coco',
                              ]

    for model in classification_models:
        test_tidl_classification(model)
    for model in object_detection_models:
        test_tidl_object_detection(model)
    for model in segmentation_models:
        test_tidl_segmentation(model)
