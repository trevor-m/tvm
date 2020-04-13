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

import numpy as np

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform
import tvm.relay.op.contrib.tidl

# For darknet tests
#Using darknet...
import sys
from tvm.contrib.download import download_testdata
download_testdata.__test__ = False
from tvm.relay.testing.darknet import LAYERTYPE
from tvm.relay.testing.darknet import __darknetffi__
from tvm.relay.frontend.darknet import ACTIVATION

REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'

DARKNET_LIB = 'libdarknet_mac2.0.so'
DARKNETLIB_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'

if sys.platform in ['linux', 'linux2']:
    DARKNET_LIB = 'libdarknet2.0.so'
    DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
elif sys.platform == 'darwin':
    DARKNET_LIB = 'libdarknet_mac2.0.so'
    DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

LIB = __darknetffi__.dlopen(download_testdata(DARKNETLIB_URL, DARKNET_LIB, module='darknet'))

DARKNET_TEST_IMAGE_NAME = 'dog.jpg'
DARKNET_TEST_IMAGE_URL = REPO_URL + 'data/' + DARKNET_TEST_IMAGE_NAME +'?raw=true'
DARKNET_TEST_IMAGE_PATH = download_testdata(DARKNET_TEST_IMAGE_URL, DARKNET_TEST_IMAGE_NAME, module='data')

def test_tidl_annotation():

    dtype = "float32"
    input_shape = (1, 3, 224, 224) # NCHW
    w1_shape    = (32, 3, 3, 3)    # OIHW

    data = relay.var('data', shape=(input_shape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1_shape), dtype=dtype)
    conv2d_1 = relay.nn.conv2d(data,
                               weight1,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               kernel_layout = 'OIHW')

    clip_1 = relay.clip(conv2d_1, 0, 6) # relu6

    squeeze_1 = relay.squeeze(clip_1)
    reshape_1 = relay.reshape(squeeze_1, (3, 3, 3, 32))

    out = reshape_1
    f1 = relay.Function([data, weight1], out)
    mod = tvm.IRModule.from_expr(f1)
    print('---------- Original graph ----------')
    print(mod.astext(show_meta_data=False))

    print("----------  Graph with composite fns ----------")
    #TODO: Uncomment after Cody refactor PR in
    mod = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod)
    print(mod.astext(show_meta_data=False))

    print("---------- Annotated graph ----------")
    mod = transform.AnnotateTarget("tidl")(mod)
    print(mod.astext(show_meta_data=False))

    print("---------- Annotated graph after merging ----------")
    mod = transform.MergeCompilerRegions()(mod)
    print(mod.astext(show_meta_data=False))

    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    print(mod.astext(show_meta_data=False))

def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

def test_tidl_mobilenet():
    import tensorflow as tf
    import tvm.relay.testing.tf as tf_testing

    with tf.Graph().as_default():

        graph_def = tf_testing.get_workload(
            "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
            "mobilenet_v2_1.4_224_frozen.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
        out_node = 'MobilenetV2/Predictions/Reshape_1'
        with tf.Session() as sess:
            # Add shapes to the graph.
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
            input_data = convert_to_list(data)
            input_node = convert_to_list('input')
            shape_dict = {e: i.shape for e, i in zip(input_node, input_data)}
            mod1, params = relay.frontend.from_tensorflow(graph_def,
                                                          shape=shape_dict)
            print('---------- Original Graph ----------')
            mod1 = relay.transform.RemoveUnusedFunctions()(mod1)
            print(mod1.astext(show_meta_data=False))
            print('---------- Merge Composite Functions ----------')
            mod3 = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod1) #Merge sequence of ops into composite functions/ops
            print(mod3.astext(show_meta_data=False))
            print("---------- Annotated Graph ----------")
            mod4 = transform.AnnotateTarget("tidl")(mod3) #Looks at annotated ops and marks them in the graph with compiler.begin and compiler.end
            print(mod4.astext(show_meta_data=False))
            print("---------- Merge Compiler Regions ----------")
            mod4 = transform.MergeCompilerRegions()(mod4) #Merge annotated regions together that use the same external target, combines marked regions for each target
            print(mod4.astext(show_meta_data=False))
            print("---------- Partioned Graph ----------")
            mod4 = transform.PartitionGraph()(mod4)
            print(mod4.astext(show_meta_data=False))

def _read_memory_buffer(shape, data, dtype='float32'):
    length = 1
    for x in shape:
        length *= x
    data_np = np.zeros(length, dtype=dtype)
    for i in range(length):
        data_np[i] = data[i]
    return data_np.reshape(shape)

def _load_net(cfg_url, cfg_name, weights_url, weights_name):
    cfg_path = download_testdata(cfg_url, cfg_name, module='darknet')
    weights_path = download_testdata(weights_url, weights_name, module='darknet')
    net = LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
    return net

def verify_darknet_frontend(net, build_dtype='float32'):
    '''Test network with given input image on both darknet and tvm'''
    def get_darknet_output(net, img):
        LIB.network_predict_image(net, img)
        out = []
        for i in range(net.n):
            layer = net.layers[i]
            if layer.type == LAYERTYPE.REGION:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.coords, layer.background],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.n*2, ), layer.biases))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif layer.type == LAYERTYPE.YOLO:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.total],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.total*2, ), layer.biases))
                out.insert(0, _read_memory_buffer((layer.n, ), layer.mask, dtype='int32'))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif i == net.n-1:
                if layer.type == LAYERTYPE.CONNECTED:
                    darknet_outshape = (layer.batch, layer.out_c)
                elif layer.type in [LAYERTYPE.SOFTMAX]:
                    darknet_outshape = (layer.batch, layer.outputs)
                else:
                    darknet_outshape = (layer.batch, layer.out_c,
                                        layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(darknet_outshape, layer.output))
        return out

    dtype = 'float32'

    img = LIB.letterbox_image(LIB.load_image_color(DARKNET_TEST_IMAGE_PATH.encode('utf-8'), 0, 0), net.w, net.h)
    darknet_output = get_darknet_output(net, img)
    batch_size = 1
    data = np.empty([batch_size, img.c, img.h, img.w], dtype)
    i = 0
    for c in range(img.c):
        for h in range(img.h):
            for k in range(img.w):
                data[0][c][h][k] = img.data[i]
                i = i + 1

    (mod, params) = _get_tvm_output(net, data, build_dtype)
    return (mod, params)

def _get_tvm_output(net, data, build_dtype='float32', states=None):
    '''Compute TVM output'''
    dtype = 'float32'
    mod, params = relay.frontend.from_darknet(net, data.shape, dtype)
    return (mod, params)

def test_tidl_yolo():
    model_name = 'yolov3'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    build_dtype = {}
    front_out = verify_darknet_frontend(net, build_dtype)

    mod = front_out[0]
    LIB.free_network(net)

    print('---------- Original Graph ----------')
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    print(mod.astext(show_meta_data=False))
    print('---------- Merge Composite Functions ----------')
    mod = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod) #Merge sequence of ops into composite functions/ops
    print(mod.astext(show_meta_data=False))
    print("---------- Annotated Graph ----------")
    mod = transform.AnnotateTarget("tidl")(mod) #Looks at annotated ops and marks them in the graph with compiler.begin and compiler.end
    print(mod.astext(show_meta_data=False))
    print("---------- Merge Compiler Regions ----------")
    mod = transform.MergeCompilerRegions()(mod) #Merge annotated regions together that use the same external target, combines marked regions for each target
    print(mod.astext(show_meta_data=False))
    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    print(mod.astext(show_meta_data=False))


def test_tidl_mobilenet_no_composite():

    import tensorflow as tf
    import tvm.relay.testing.tf as tf_testing

    with tf.Graph().as_default():

        graph_def = tf_testing.get_workload(
            "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
            "mobilenet_v2_1.4_224_frozen.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
        out_node = 'MobilenetV2/Predictions/Reshape_1'
        with tf.Session() as sess:
            # Add shapes to the graph.
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
            input_data = convert_to_list(data)
            input_node = convert_to_list('input')
            shape_dict = {e: i.shape for e, i in zip(input_node, input_data)}
            mod1, params = relay.frontend.from_tensorflow(graph_def,
                                                          shape=shape_dict)
            print('---------- Original Graph ----------')
            mod1 = relay.transform.RemoveUnusedFunctions()(mod1)
            print(mod1.astext(show_meta_data=False))
            print("---------- Annotated Graph ----------")
            mod4 = transform.AnnotateTarget("tidl")(mod1) #Looks at annotated ops and marks them in the graph with compiler.begin and compiler.end
            print(mod4.astext(show_meta_data=False))
            print("---------- Merge Compiler Regions ----------")
            mod4 = transform.MergeCompilerRegions()(mod4) #Merge annotated regions together that use the same external target, combines marked regions for each target
            print(mod4.astext(show_meta_data=False))
            print("---------- Partioned Graph ----------")
            mod4 = transform.PartitionGraph()(mod4)
            print(mod4.astext(show_meta_data=False))

if __name__ == '__main__':
    #test_tidl_annotation()
    #test_tidl_mobilenet()
    #test_tidl_mobilenet_no_composite()
    test_tidl_yolo()