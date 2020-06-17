import os
import sys
import numpy as np
from matplotlib import pyplot as plt

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.contrib import cc
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from tvm.relay.backend.contrib import tidl
import mxnet as mx
from mxnet import image
from mxnet.gluon.model_zoo.vision import get_model
import gluoncv

def model_compile(model_name, mod_orig, params, input_data, num_tidl_subgraphs=4, 
                  data_layout="NCHW", input_node="data"):
    artifacts_folder = "./artifacts_" + model_name
    if os.path.isdir(artifacts_folder):
        filelist = [ f for f in os.listdir(artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(artifacts_folder, file))
    else:
        os.mkdir(artifacts_folder)

    mod = tidl.EnableTIDL(mod_orig, params, num_tidl_subgraphs, 
                          data_layout, input_node, input_data, 
                          artifacts_folder, os.path.join(os.getenv("TIDL_TOOLS_PATH"), "eve_test_dl_algo_ref.out"))
    # We expect somethign to be offloaded to TIDL.
    assert mod is not None

    target = "llvm -target=armv7l-linux-gnueabihf"
    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    path_lib = os.path.join(artifacts_folder, "deploy_lib.so")
    path_graph = os.path.join(artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(artifacts_folder, "deploy_param.params")
    cc_path = os.path.join(os.getenv("ARM_GCC_PATH"), "arm-linux-gnueabihf-g++")
    lib.export_library(path_lib, cc=cc_path)
    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))


def test_tidl_gluon():
    x = np.random.normal(0, 1, (1, 3, 224, 224)) #np.load(os.path.join(os.getenv("TIDL_TOOLS_PATH"), 'dog.npy'))
    input_data = x/np.amax(np.abs(x))

    def test_model(model, input_shape, dtype, use_tidl=True, num_iteration=1):
        block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        model_compile(model, mod, params, input_node='data', input_data=input_data)

    latency = {}
    models = [
        'alexnet',
        'resnet18_v1',
        'resnet34_v1',
        'resnet50_v1',
        'resnet101_v1',
        'resnet152_v1',
        'resnet18_v2',
        'resnet34_v2',
        'resnet50_v2',
        'resnet101_v2',
        'resnet152_v2',
        'squeezenet1.0',
        'mobilenet0.25',
        'mobilenet0.5',
        'mobilenet0.75',
        'mobilenet1.0',
        'mobilenetv2_0.25',
        'mobilenetv2_0.5',
        'mobilenetv2_0.75',
        'mobilenetv2_1.0',
        'vgg11',
        'vgg16',
        'densenet121',
        'densenet169',
        'densenet201',
    ]
    
    dtype = 'float32'
    input_shape = (1, 3, 224, 224)
    for model in models:
        print('testing model:', model),
        test_model(model, input_shape, dtype, use_tidl=True)

def test_tidl_gluoncv():
    def test_model(model, input_shape, dtype, use_tidl=True, num_iteration=1):
        block = gluoncv.model_zoo.get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

        input_data = np.random.normal(0, 1, input_shape)
        input_data = input_data/np.amax(np.abs(input_data))
        model_compile(model, mod, params, input_node='data', input_data=input_data)

    latency = {}
    models = [
        ('deeplab_resnet101_ade', (1, 3, 480, 480)),
        #('yolo3_mobilenet1.0_coco', (1, 3, 224, 224)),
    ]
    
    dtype = 'float32'
    for model, input_shape in models:
        print('testing model:', model)
        test_model(model, input_shape, dtype, use_tidl=True)

if __name__ == "__main__":
    # test_tidl_gluon()
    test_tidl_gluoncv()
