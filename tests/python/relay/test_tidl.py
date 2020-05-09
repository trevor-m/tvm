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
import subprocess

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
import tvm.relay.op.contrib.tidl

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
    target = "llvm -target=armv7l-linux-gnueabihf"

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

def create_relay_graph(model, input_node, input_shape, layout):

    if model == "MobileNetV1" or model == "MobileNetV2":
        if model == "MobileNetV1":
            model    = "./mobileNet1/mobilenet_v1_1.0_224_frozen.pb"
            out_node = 'MobilenetV1/Predictions/Softmax'
        else:
            model    = "./mobileNet2/mobilenet_v2_1.0_224_frozen.pb"
            out_node = 'MobilenetV2/Predictions/Softmax'
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
                                                         layout = None,  # default: NHWC
                                                         shape  = shape_dict, 
                                                         outputs= None)
            mod = relay.transform.RemoveUnusedFunctions()(mod)
            print("Tensorflow model imported to Relay IR.")
    
    return mod, params


def generate_subgraph_tensors(mod, params, input_node, input_data):
    """
    """

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = relay.transform.Inline()(mod_tvm)
    my_mutator = tidl.CalibrationGraphMutator("tidl")
    mod_tvm["main"] = my_mutator.make_calibration_graph(mod_tvm["main"])
    #print("Calibration module:", mod_tvm)
    print("Input map:", my_mutator.name_map)

    # Build and execute calibration graph to get outputs
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    mod.set_input(input_node, input_data)
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

    return subgraph_tensors

def test_extern_tidl_mobilenet():
    target = "llvm -target=armv7l-linux-gnueabihf"
    dtype = "float32"
    data_layout = "NHWC"
    input_shape = (1, 224, 224, 3)
    x = np.load('./tidl_tools/dog.npy')  # "NCHW"
    x = x.transpose(0,2,3,1)  # TF uses "NHWC" layout
    if x.shape != input_shape:
        sys.exit("Input data shape is not correct!")
    # Normalize input data to (-1,1)
    input_data = x/np.amax(np.abs(x))
    input_node = "input"

    #============= Create a Relay graph for MobileNet model ==============
    mod0, params0 = create_relay_graph(model="MobileNetV2",
                                       input_node  = input_node,
                                       input_shape = input_shape,
                                       layout      = data_layout)
    print('-------- Original MobileNetV1 model --------')
    print(mod0.astext(show_meta_data=False))

    #============= Annotate the graph ==============
    # Looks at annotated ops and marks them in the graph with compiler.begin 
    # and compiler.end.
    # Merges annotated regions together that use the same external target, 
    # and combines marked regions for each target
    mod0['main'] = bind_params_by_name(mod0['main'], params0)
    #Merge sequence of ops into composite functions/ops
    print("---------- Merge Composite Functions ----------")
    mod1 = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod0) 
    mod1 = transform.AnnotateTarget("tidl")(mod1)
    mod1 = transform.MergeCompilerRegions()(mod1)
    #print(mod1.astext(show_meta_data=False))

    #============= Partition the graph ==============
    mod2 = transform.PartitionGraph()(mod1)
    print("---------- Partition the graph ----------")
    print(mod2.astext(show_meta_data=False))

    mod2 = tidl.UnpackComposites(mod2, "tidl")
    print("---------- Unpack composite ops in the graph ----------")
    print(mod2.astext(show_meta_data=False))

    #============= Generate subgraph boundary tensors ==============
    subgraph_tensors = generate_subgraph_tensors(mod2, params0, input_node, input_data)

    #======================== Import the graph to TIDL ========================
    if tidl.import_relay_ir(mod2, params0, subgraph_tensors, data_layout, tidl_calib_tool, artifacts_folder) == True:
        print('Heterogeneous execution with TIDL.')
        graph, lib, params = relay.build_module.build(mod2, target=target, params=params0)
    else:
        print("Full graph compilation with LLVM.")
        # Future optimization: if not all subgraphs failed with TIDL import, re-partition
        # the graph to have only TIDL subgraphs with successful TIDL import. 
        graph, lib, params = relay.build_module.build(mod0, target=target, params=params0)

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
    if os.getenv("TIDL_ARM_GCC_PATH") is None:
      sys.exit("Environment variable TIDL_ARM_GCC_PATH not set!")
    else: 
      arm_gcc_path = os.getenv("TIDL_ARM_GCC_PATH")
    if os.getenv("TIDL_TOOLS_PATH") is None:
        sys.exit("Environment variable TIDL_TOOLS_PATH not set!")
    else:
        tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
    arm_gcc          = arm_gcc_path + "arm-linux-gnueabihf-g++"
    tidl_calib_tool  = tidl_tools_path + "eve_test_dl_algo_ref.out"

    artifacts_folder = "./artifacts/"
    if os.path.isdir(artifacts_folder):
        filelist = [ f for f in os.listdir(artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(artifacts_folder, file))
    else:
        os.mkdir(artifacts_folder)

    #test_extern_tidl()
    test_extern_tidl_mobilenet()
