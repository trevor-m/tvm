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
"""Unit tests for graph partitioning."""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

import onnx
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.contrib import cc
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from tvm.relay.backend.contrib import tidl
import tensorflow as tf
from tvm.relay.testing import tf as tf_testing
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.presets.segmentation import test_transform
import mxnet as mx
from mxnet import image


def create_tf_relay_graph(model, input_node, input_shape, layout):
    model_folder = "./tf_models/"
    if model == "MobileNetV1":
        model    = model_folder + "mobilenet_v1_1.0_224_frozen.pb"
        out_node = "MobilenetV1/Predictions/Softmax"
    elif model == "MobileNetV2":
        model    = model_folder + "mobileNet2/mobilenet_v2_1.0_224_frozen.pb"
        out_node = "MobilenetV2/Predictions/Softmax"
    elif model == "InceptionV1":
        model    = model_folder + "inception1/inception_v1_fbn.pb"
        out_node = "softmax/Softmax"
    elif model == "InceptionV3":
        model    = model_folder + "inception3/inception_v3_2016_08_28_frozen-with_shapes.pb"
        out_node = "InceptionV3/Predictions/Softmax"
    #if layout == "NCHW":
    #    input_shape = (input_shape[0],input_shape[2],input_shape[3],input_shape[1])
    with tf.gfile.GFile(model, 'rb') as f:
        # Import tensorflow graph definition to relay frontend.
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        # Add shapes to the graph.
        with tf.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)

        shape_dict = {input_node : input_shape}
        print("Inut node shape dict:" + str(shape_dict))
        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                     layout = layout,
                                                     shape  = shape_dict, 
                                                     outputs= None)
        print("Tensorflow model imported to Relay IR.")

    return mod, params


def model_compile(model_name, mod_orig, params, num_tidl_subgraphs, 
                  data_layout, input_node, input_data):
    artifacts_folder = "./artifacts_" + model_name
    if os.path.isdir(artifacts_folder):
        filelist = [ f for f in os.listdir(artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(artifacts_folder, file))
    else:
        os.mkdir(artifacts_folder)

    mod = tidl.EnableTIDL(mod_orig, params, num_tidl_subgraphs, 
                          data_layout, input_node, input_data, 
                          artifacts_folder, tidl_calib_tool)
    if mod == None:  # TIDL cannot be enabled - no offload to TIDL
        mod = mod_orig

    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    path_lib    = os.path.join(artifacts_folder, "deploy_lib.so")
    path_graph  = os.path.join(artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(artifacts_folder, "deploy_param.params")
    lib.export_library(path_lib, cc=arm_gcc)
    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))

def test_extern_tidl_tf(model_name, num_tidl_subgraphs):
    dtype = "float32"
    data_layout = "NHWC"
    input_shape = (1, 224, 224, 3)
    x = np.load(os.path.join(tidl_tools_path, 'dog.npy'))  # "NCHW"
    x = x.transpose(0,2,3,1)  # TF uses "NHWC" layout
    if x.shape != input_shape:
        sys.exit("Input data shape is not correct!")
    # Normalize input data to (-1,1)
    input_data = x/np.amax(np.abs(x))
    input_node = "input"

    #============= Create a Relay graph for MobileNet model ==============
    tf_mod, tf_params = create_tf_relay_graph(model = model_name,
                                              input_node  = input_node,
                                              input_shape = input_shape,
                                              layout = data_layout)
    print("---------- Original TF Graph ----------")
    print(tf_mod.astext(show_meta_data=False))

    #======================== TIDL code generation ====================
    model_compile(model_name, tf_mod, tf_params, num_tidl_subgraphs, data_layout, input_node, input_data)

def test_extern_tidl_onnx(model_name, num_tidl_subgraphs):
    model_folder = "./onnx_models/"
    if model_name == "resnet18v1":
        model = model_folder + "onnx_resNet18v1/resnet18v1.onnx"
    if model_name == "resnet18v2":
        model = model_folder + "onnx_resNet18v2/resnet18v2.onnx"
    if model_name == "resnet101v1":
        model = model_folder + "resnet101v1/resnet101-v1.onnx"
    if model_name == "squeezenet1.1":
        model = model_folder + "squeezeNet1.1/squeezenet1.1.onnx"
    onnx_model = onnx.load(model)

    image_shape = (1, 3, 224, 224)
    data_layout = "NCHW"
    x = np.load(os.path.join(tidl_tools_path, 'dog.npy'))  # "NCHW"
    input_data = x/np.amax(np.abs(x))

    input_name = "data"
    shape_dict = {input_name: image_shape }
    onnx_mod, onnx_params = relay.frontend.from_onnx(onnx_model, shape_dict)

    #======================== Compile the model ========================
    model_compile(model_name, onnx_mod, onnx_params, num_tidl_subgraphs, data_layout, input_name, input_data)

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

def test_extern_tidl_gluoncv_ssd(model_name, num_tidl_subgraphs):
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
    plt.savefig("gluoncv_ssd_tvm.png")
    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    print("Number of outputs: " + str(len(results)))
    #for i in range(len(results)):
    #    np.savetxt("graph_out_"+str(i)+".txt", results[i].flatten(), fmt='%10.5f')

    #======================== Compile the model ========================
    model_compile(model_name, ssd_mod, ssd_params, num_tidl_subgraphs, data_layout, input_name, input_data)

def test_extern_tidl_gluoncv_segmentation(model_name, num_tidl_subgraphs):
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
#    block, seg_mod, seg_params = load_gluoncv_model(model, img, input_name, input_shape, dtype)

    #======================== Compile the model ========================
    model_compile(model_name, seg_mod, seg_params, num_tidl_subgraphs, data_layout, input_name, input_data)

if __name__ == '__main__':
    if os.getenv("ARM_GCC_PATH") is None:
      sys.exit("Environment variable ARM_GCC_PATH is not set!")
    else: 
      arm_gcc_path = os.getenv("ARM_GCC_PATH")
    if os.getenv("TIDL_TOOLS_PATH") is None:
        sys.exit("Environment variable TIDL_TOOLS_PATH is not set!")
    else:
        tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
    tidl_calib_tool  = os.path.join(tidl_tools_path, "eve_test_dl_algo_ref.out")
    arm_gcc          = os.path.join(arm_gcc_path, "arm-linux-gnueabihf-g++")
    target           = "llvm -target=armv7l-linux-gnueabihf"

    tf_models  = [#'MobileNetV1',
#                  'MobileNetV2',
                  'InceptionV1',
                 ]
    onnx_models = [#'resnet18v1',
                   #'resnet18v2',
                   'squeezenet1.1'
                   ]
    ssd_models = ['ssd_512_mobilenet1.0_coco',
                  'ssd_512_mobilenet1.0_voc',
                  #'ssd_512_mobilenet2.0_coco',
                  #'ssd_512_mobilenet2.0_voc',
                 ]
    seg_models = ['mask_rcnn_resnet18_v1b_coco',
                 ]

    #test_extern_tidl()
#TODO: download classification models from web
#    for tf_model in tf_models:
#        test_extern_tidl_tf(tf_model, num_tidl_subgraphs=1)
#    for onnx_model in onnx_models:
#        test_extern_tidl_onnx(onnx_model, num_tidl_subgraphs=1)
    for ssd_model in ssd_models:
        test_extern_tidl_gluoncv_ssd(ssd_model, num_tidl_subgraphs=1)
    for seg_model in seg_models:
        test_extern_tidl_gluoncv_segmentation(seg_model, num_tidl_subgraphs=1)
