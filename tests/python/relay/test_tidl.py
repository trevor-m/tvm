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

import tvm
import tvm.relay.testing
from tvm import relay
from tvm import runtime
from tvm.relay import transform
from tvm.contrib import util
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator
from tvm.contrib import cc
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_runtime

from tidl_tools import tidl
from tidl_tools import tidl_utils


@relay.transform.function_pass(opt_level=0)
class TIDLWhiteListAnnotator:
    def __init__(self, op_list, compiler):
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod, ctx):

        op_annotations = tidl.annotation(mod)
        for node in op_annotations:
            print(f'Operator {node.op.name}: {op_annotations[node]}')

        annotator = self
        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                op_supported = op_annotations[call]
                if op_supported:
                    new_args = []
                    # TODO: skip compiler_begin if all input nodes are supported
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg),
                                             annotator.compiler)
                        new_args.append(ann)
                    # TODO: skip compiler_end if all output nodes are supported
                    new_call = relay.Call(call.op, new_args, call.attrs,
                                          call.type_args)
                    return compiler_end(new_call, annotator.compiler)
                else:
                    return super().visit_call(call)
        return Annotator().visit(func)

class WholeGraphAnnotator(ExprMutator):
    """
    An annotator that creates a compiler for an entire graph.
    """

    def __init__(self, compiler):
        super(WholeGraphAnnotator, self).__init__()
        self.compiler = compiler
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Var):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = compiler_end(new_call, self.compiler)
        return new_call

# Leverage the pass manager to write a simple white list based annotator
@relay.transform.function_pass(opt_level=0)
class WhiteListAnnotator:
    def __init__(self, op_list, compiler):
        assert isinstance(op_list, (list, tuple, set))
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod, ctx):

        annotator = self
        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                op_name = call.op.name
                if op_name in annotator.op_list:
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg),
                                             annotator.compiler)
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args, call.attrs,
                                          call.type_args)
                    return compiler_end(new_call, annotator.compiler)
                else:
                    return super().visit_call(call)
        return Annotator().visit(func)

def test_extern_tidl():
#    #============= Create a Relay graph ==============
#    mod0, params0 = relay_graph_create()
#
#    #============= Annotate the graph ==============
#    mod1, params1 = relay_graph_annotate(mod0, params0)
#    
#    #============= Partition the graph ==============
#    mod2, params2 = relay_graph_partition(mod1, params1)

#    #============= Generate subgraph boundary tensors ==============
#    input_tensors, output_tensors = subgraph_tensors_generate(mod2, params2)

    #============= Constructing a simple graph ==============
    dtype = 'float32'
    data_layout = 'NCHW'
    input_shape    = (1, 3, 224, 224) # NCHW
    tidl_input_dim = (input_shape[2],input_shape[3],input_shape[1]) # HxWxC
    input_layout= 'NCHW'
    w1_shape    = (32, 3, 3, 3)    # OIHW
    w2_shape    = (1, 32, 3, 3)    # OIHW
    mnet1_conv2d_0 = np.load('MobilenetV1_Conv2d_0_weights.npy').astype(np.float32)
    # change layout from HWIO to OIHW
    w1 = mnet1_conv2d_0.transpose(3,2,0,1)
    params_w1 = tvm.nd.array(w1)
    mnet1_conv2d_1 = np.load('MobilenetV1_Conv2d_1_weights.npy').astype(np.float32) # HWIO
    w2 = mnet1_conv2d_1.transpose(3,2,0,1)
    params_w2 = tvm.nd.array(w2)

    data = relay.var('data', shape=(input_shape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1_shape), dtype=dtype)
    conv2d_1 = relay.nn.conv2d(data,
                               weight1,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               strides=(2,2),
                               data_layout = data_layout,
                               kernel_layout = 'OIHW')
    weight2 = relay.var('weight2', shape=(w2_shape), dtype=dtype)
    conv2d_2 = relay.nn.conv2d(conv2d_1,
                               weight2,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               data_layout = data_layout,
                               kernel_layout = 'OIHW')
    clip = relay.clip(conv2d_2,0,6)
    #out = conv2d_1
    out = conv2d_2
    #out  = clip
    f1 = relay.Function([data, weight1, weight2], out)
    mod1 = tvm.IRModule.from_expr(f1)
    params0 = {'weight1':params_w1, 'weight2':params_w2} 
    print('---------- Original graph ----------')
    print(mod1.astext(show_meta_data=False))

    #============= Build the graph to run on ARM =============
    print('Build the graph to run on ARM')
    with relay.build_config(opt_level=3):
        target = "llvm -target=armv7l-linux-gnueabihf"
        graph, lib, params = relay.build_module.build(mod1, target=target, params=params0)

    artifacts_folder = "./artifacts_arm/"
    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    lib.export_library(path_lib, cc=arm_gcc)
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))
    print('Model artifacts saved to run on ARM')

    #============= Annotating the graph to run on TIDL ==============
    mod1['main'] = bind_params_by_name(mod1['main'], params0)
    # whole graph offload to TIDL
    mod2 = tvm.IRModule()
    mod2['main'] = WholeGraphAnnotator('tidl').visit(mod1['main'])
    print('---------- Whole graph annotated ----------')
    print(mod2.astext(show_meta_data=False))
    mod_whole_graph_tidl = relay.transform.PartitionGraph()(mod2)
    print('---------- Whole graph annotated and partitioned ----------')
    print(mod_whole_graph_tidl.astext(show_meta_data=False))

    # subraph offload to TIDL
    mod3 = WhiteListAnnotator(["nn.conv2d"],"tidl")(mod1)
    print('---------- Subgraph annotated ----------')
    print(mod3.astext(show_meta_data=False))
    mod_subgraph_tidl = relay.transform.PartitionGraph()(mod3)
    print('---------- Subgraph annotated and partitioned ----------')
    print(mod_subgraph_tidl.astext(show_meta_data=False))

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    #mod4 = relay.transform.InferType()(mod_whole_graph_tidl)
    mod4 = relay.transform.InferType()(mod_subgraph_tidl)
    mod4 = relay.transform.Inline()(mod4)
    my_mutator = tidl.CalibrationGraphMutator("tidl")
    mod4["main"] = my_mutator.make_calibration_graph(mod4["main"])
    print("Calibration module:", mod4)
    print("Input map:", my_mutator.name_map)

    # Build and execute calibration graph to get outputs
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod4, "llvm", params=params0)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    x = np.load('./tidl_tools/dog.npy') # (1,3,224,224)
    input_data = x/np.amax(np.abs(x))
    print("Input data shape: ")
    print(input_data.shape)
    mod.set_input('data', input_data)
    mod.set_input(**params)
    mod.run()
    #mod.run(data=input_data, weight1=params_w1, weight2=params_w2)

    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    np.savetxt('graph_output.txt', results[0].flatten(), fmt='%10.5f')

    # We now have subgraph inputs
    # {1: 'tidl_1_i0', 2: 'tidl_1_o0', 3: 'tidl_0_i0', 4: 'tidl_0_o0'}
    subgraph_tensors = {}
    for i in range(len(results)):
        if i in my_mutator.name_map:
            subgraph_tensors[my_mutator.name_map[i]]=results[i]
            #print("Subgraph input: ", my_mutator.name_map[i], " tensor: ", results[i])
            file_name = my_mutator.name_map[i] + ".txt"
            np.savetxt(file_name, results[i].flatten(), fmt='%10.5f')

    for key, value in subgraph_tensors.items():
        print("Subgraph tensor: ", key, value.shape)

    #======================== Import the graph to TIDL ========================
    mod2 = mod_subgraph_tidl
    #mod2 = mod_whole_graph_tidl
    artifacts_folder = "./artifacts/"
    if tidl.import_relay_ir(mod2, params0, subgraph_tensors, data_layout, tidl_calib_tool, artifacts_folder) == True:
        print('Heterogeneous execution with TIDL.')
        graph, lib, params = relay.build_module.build(mod2, target=target, params=params0)
    else:
        print("Full graph compilation with LLVM.")
        # Future optimization: if not all subgraphs failed with TIDL import, re-partition
        # the graph to have only TIDL subgraphs with successful TIDL import. 
        graph, lib, params = relay.build_module.build(mod1, target=target, params=params0)

    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    #lib.save(path_lib) # for whole graph execute on TIDL
    lib.export_library(path_lib, cc=arm_gcc) # for heterogeneous execute on TIDL+ARM
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))


import tensorflow as tf
from tvm.relay.testing import tf as tf_testing

def test_extern_tidl_mobilenet():
    dtype = 'float32'
    input_shape    = (1, 3, 224, 224) # NCHW
    tidl_input_dim = (input_shape[2],input_shape[3],input_shape[1]) # HxWxC
    #============= Load MobileNetV1 model ==============
#    mod, params_mod = relay.testing.mobilenet.get_workload(batch_size=1, dtype='float32')

    model      = "./mobileNet1/mobilenet_v1_1.0_224_frozen.pb"
    out_node   = 'MobilenetV1/Predictions/Reshape_1'
    #model      = "./mobileNet2/mobilenet_v2_1.0_224_frozen.pb"
    #out_node   = 'MobilenetV2/Predictions/Reshape_1'
    input_node = "input"
    model_input_shape = (224,224,3)
    data_shape_input = list(model_input_shape)
    data_shape_input.insert(0,1)
    data_shape_input = tuple(data_shape_input) # Prepend batch size
    print(data_shape_input)

    layout = None
    with tf.gfile.GFile(model, 'rb') as f:
        # Import tensorflow graph definition to relay frontend.
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        
        # Add shapes to the graph.
        with tf.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)

        shape_dict = {input_node : data_shape_input}
        print("Inut node shape dict:" + str(shape_dict))
        mod, params_mod = relay.frontend.from_tensorflow(graph_def,
                                                         layout=layout,
                                                         shape=shape_dict, 
                                                         outputs=None)
        print("Tensorflow protobuf imported to relay frontend.")
   
    print('-------- Original MobileNetV1 model --------')
    print(mod.astext(show_meta_data=False))

    #============= Build the graph to run on ARM =============
    print('Build the graph to run on ARM')
    with relay.build_config(opt_level=3):
        target = "llvm -target=armv7l-linux-gnueabihf"
        graph, lib, params = relay.build_module.build(mod, target=target, params=params_mod)

    #artifacts_folder = "./artifacts_arm/"
    #path_lib    = artifacts_folder + "deploy_lib.so"
    #path_graph  = artifacts_folder + "deploy_graph.json"
    #lib.export_library(path_lib, cc=arm_gcc)
    #path_params = artifacts_folder + "deploy_param.params"
    lib.export_library('./mnet1_arm.tar')

    # TIDL annotation pass:
    #    - mark each operator either supported (True) or unsupported (False) by TIDL
    op_annotations = tidl.annotation(mod)
    
    # Check if whole graph can offload to TIDL (no graph partitioning for now)
    full_graph_tidl = True
    for node in op_annotations:
        print(f'Operator {node.op.name}: {op_annotations[node]}')
        if op_annotations[node] == False:
            full_graph_tidl= False
            break

    if full_graph_tidl == True:
        print("Try to import this model to TIDL")
        #============= Annotating the graph to run on TIDL ==============
        mod['main'] = bind_params_by_name(mod['main'], params_mod)
        # whole graph offload to TIDL
        mod_tidl = tvm.IRModule()
        mod_tidl['main'] = WholeGraphAnnotator('tidl').visit(mod['main'])
        print('---------- Whole graph annotated ----------')
        print(mod_tidl.astext(show_meta_data=False))
        mod_tidl = relay.transform.PartitionGraph()(mod_tidl)
        print('---------- Whole graph annotated and partitioned ----------')
        print(mod_tidl.astext(show_meta_data=False))

        #if tidl.relay_ir_import_whole_graph(mod, params_mod, 0) == False:
        if tidl.relay_ir_import(mod_tidl, params_mod) == False:
            print('Importing this model to TIDL failed!')
            model_imported_to_TIDL = False
        else:
            # TIDL calibration pass:
            subgraph_id = 0
            calibration_image = './tidl_tools/airshow.jpg'
            raw_image = 'raw_calib_image.bin'
            tidl_utils.tf_image_preprocess(calibration_image, raw_image, tidl_input_dim)
            tidl_calib_status, last_node_dim = tidl.tidl_calib(tidl_calib_tool, raw_image, subgraph_id)

            if tidl_calib_status == False:
                print('TIDL calibration for this model failed!')
                model_imported_to_TIDL = False
            else:
                print('TIDL Calibration for this model succeeded!')
                model_imported_to_TIDL = True
    else:
        print("Run this model on ARM")

    # Compile the graph (with or without TIDL offload)
    with relay.build_config(opt_level=3):
        target = "llvm -target=armv7l-linux-gnueabihf"
        if full_graph_tidl == True and model_imported_to_TIDL == True:
            #graph, lib, params = relay.build_module.build(mod_tidl, target=target, params=params_mod)
            graph, lib, params = relay.build_module.build(mod_tidl, target=target)
        else:
            graph, lib, params = relay.build_module.build(mod, target=target, params=params_mod)
        print(lib)

    artifacts_folder = "./artifacts/"
    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    #lib.save(path_lib) # for whole graph execute on TIDL
    lib.export_library(path_lib, cc=arm_gcc) # for heterogeneous execute on TIDL+ARM
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))

if __name__ == '__main__':
    if os.getenv("TIDL_PLSDK") is None:
      plsdk = os.getenv('HOME') + "/ti/processor-sdk-linux-am57xx-evm-06.02.00.81-GA"
    else: 
      plsdk = os.getenv('TIDL_PLSDK')
    plsdk_devkit = plsdk + "/linux-devkit/sysroots/x86_64-arago-linux/usr/bin/"
    print("PLSDK DEVKIT path set to: " + plsdk_devkit)
    tidl_calib_tool  = plsdk_devkit + "eve_test_dl_algo_ref.out"
    arm_gcc          = plsdk_devkit + "arm-linux-gnueabihf-g++"

    #test_extern_tidl_prototype()
    #test_extern_tidl_mobilenet()
    test_extern_tidl()
