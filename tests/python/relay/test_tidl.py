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
from tvm.relay.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator
from tvm.contrib import cc
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_runtime

from tidl_tools import tidl
from tidl_tools import tidlAnnotation
from tidl_tools import tidl_utils


@transform.function_pass(opt_level=0)
class TIDLWhiteListAnnotator:
    def __init__(self, op_list, compiler):
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod, ctx):

        op_annotations = tidlAnnotation.tidl_annotation(mod)
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
@transform.function_pass(opt_level=0)
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
    if os.getenv("TIDL_PLSDK") is None:
      plsdk = os.getenv('HOME') + "/ti/processor-sdk-linux-am57xx-evm-06.02.00.81-GA"
    else: 
      plsdk = os.getenv('TIDL_PLSDK')
    plsdk_devkit = plsdk + "/linux-devkit/sysroots/x86_64-arago-linux/usr/bin/"
    print("PLSDK DEVKIT path set to: " + plsdk_devkit)
    tidl_calib_tool  = plsdk_devkit + "eve_test_dl_algo_ref.out"
    arm_gcc          = plsdk_devkit + "arm-linux-gnueabihf-g++"

    #============= Constructing a simple graph ==============
    dtype = "float32"
    input_shape = (1, 3, 224, 224) # NCHW
    w1_shape    = (32, 3, 3, 3)    # OIHW
    w2_shape    = (1, 32, 3, 3)    # OIHW
    #w1 = np.random.rand(*w1_shape).astype('float32')
    #w1 = np.random.uniform(-1,1,size=data_shape).astype('float32')
    mnet1_conv2d_0 = np.load('MobilenetV1_Conv2d_0_weights.npy').astype(np.float32) # HWIO
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
                               #strides=(2,2),
                               kernel_layout = 'OIHW')

    clip_1 = relay.clip(conv2d_1, 0, 6) # relu6
    out = clip_1
    f1 = relay.Function([data, weight1], out)
    mod1 = tvm.IRModule.from_expr(f1)
    params1 = {'weight1':params_w1} 
    print('---------- Original graph ----------')
    print(mod1.astext(show_meta_data=False))

    # Build the graph to run on host (x86)
    print('Build the graph to run on host (x86)')
    with relay.build_config(opt_level=3):
        target = "llvm"
        graph, lib, params = relay.build_module.build(mod1, target=target, params=params1)

    ctx = tvm.cpu()
    rand_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    # create module
    module = graph_runtime.create(graph, lib, ctx)
    # set input and parameters
    module.set_input("data", rand_data)
    module.set_input(**params)
    # run
    module.run()
    # get output
    out_shape = (1,32,224,224)
    out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
    # Print first 10 elements of output
    print(out.flatten()[0:10])

    # Build the graph to run on ARM
    print('Build the graph to run on ARM')
    with relay.build_config(opt_level=3):
        target = "llvm -target=armv7l-linux-gnueabihf"
        graph, lib, params = relay.build_module.build(mod1, target=target, params=params1)

    artifacts_folder = "./artifacts_arm/"
    path_lib    = artifacts_folder + "deploy_lib.tar"
    path_graph  = artifacts_folder + "deploy_graph.json"
    lib.export_library(path_lib)
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))
    print('Model artifacts saved to run on ARM')

    #============= Annotating the graph ==============
    #TIDL annotation pass:
    #   Check if whole graph can offload to TIDL (no graph partitioning for now)
    #      #mark each operator either supported (True) or unsupported (False) by TIDL
    #      op_annotations = tidl.annotation(mod)
    #      tidl_offload = True
    #      for node in op_annotations:
    #          print(f'Operator {node.op.name}: {op_annotations[node]}')
    #          if op_annotations[node] == False:
    #              tidl_offload = False
    #              break
    #
    #   Custom TIDL annotation (following examples in test_pass_partition_graph.py)
    #      tidl_annotator = TIDLWhiteListAnnotator([], 'tidl')
    #      mod = tidl_annotator(mod)
    #
    #   Annotate with merge (TODO: implement and register whitelisting functions)
    #      mod = transform.AnnotateTargetWithMerge(["tidl"])(mod)
    #
    # For now, just manually annotate the graph:
    #    - whole graph offload, or
    #    - subgraph offload

    # whole graph offload to TIDL
    mod1['main'] = bind_params_by_name(mod1['main'], params1)
    #mod3 = tvm.IRModule()
    #mod3['main'] = WholeGraphAnnotator('tidl').visit(mod1['main'])
    mod3 = WhiteListAnnotator(["nn.conv2d","clip"],"tidl")(mod1)
    print('---------- Whole graph annotated ----------')
    print(mod3.astext(show_meta_data=False))
    mod3_partition = relay.transform.PartitionGraph()(mod3)
    print('---------- Whole graph annotated and partitioned ----------')
    print(mod3_partition.astext(show_meta_data=False))


    # subgraph offload to TIDL
    data = relay.var('data', shape=(input_shape), dtype=dtype)
    clip_1 = relay.clip(data, 0, 6) # relu6
    clip_2 = relay.nn.relu(clip_1) # relu
    f2   = relay.Function([data], clip_2)
    mod2 = tvm.IRModule.from_expr(f2)
    mod4 = WhiteListAnnotator(['clip', 'nn.relu'], 'tidl')(mod2)
    params4 = {}
    print('---------- Annotated graph ----------')
    print(mod4.astext(show_meta_data=False))

    #============= Partitioning the graph ==============
    mod4_partition = transform.PartitionGraph()(mod4)
    print('---------- Partitioned graph ----------')
    print(mod4_partition.astext(show_meta_data=False))


    #============= Importing subgraph(s) to TIDL ==============
    # TIDL import pass - import all TIDL subgraphs
    # invoking Relay IR import from TIDL package
    print('---------- Subgraphs import to TIDL ----------')
    if tidl.relay_ir_import(mod4_partition, params4) == False:
    #if tidl.relay_ir_import(mod3_partition, params) == False:
        print('Importing this model to TIDL failed!')
    #    assert mod['main'].attrs and mod['main'].attrs.Compiler == 'tidl'
    else:
        # TIDL calibration pass:
        ############################### Gap ############################
        # Can only calibrate the first subgraph which starts from input 
        # TODO: implement boundary tensors conversions
        ############################### Gap ############################
        subgraph_id = 0
        calibration_image = './tidl_tools/airshow.jpg'
        raw_image = 'raw_calib_image.bin'
        # TODO: how to preprocess image properly for all supported frameworks??
        tidl_utils.tf_image_preprocess(calibration_image, raw_image, input_shape)
        tidl.tidl_calib(tidl_calib_tool, raw_image, subgraph_id)
        print('Importing this model to TIDL succeeded!')

    #============= Compiling the graph (with TIDL subgraphs) ==============
    print('---------- Code generation with TIDL subgraphs ----------')
    with relay.build_config(opt_level=3):
        target = "llvm -target=armv7l-linux-gnueabihf"
        #graph, lib, params = relay.build_module.build(mod2_partition, target=target)
        graph, lib, params = relay.build_module.build(mod4_partition, target=target)
        print(lib)

    artifacts_folder = "./artifacts/"
    path_lib    = artifacts_folder + "deploy_lib.tidl"
    path_graph  = artifacts_folder + "deploy_graph.json"
    # should use compiler: arm_gcc = plsdk_devkit + "arm-linux-gnueabihf-g++"
    lib.save(path_lib) # whole graph offload and heterogeneous execute may need to use diff func. Trevor to check
    #lib.export_library(path_lib) 
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))


if __name__ == '__main__':
    test_extern_tidl()
