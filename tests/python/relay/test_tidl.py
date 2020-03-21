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

    data = relay.var('data', shape=(input_shape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1_shape), dtype=dtype)
    conv2d_1 = relay.nn.conv2d(data,
                               weight1,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               kernel_layout = 'OIHW')

    clip_1 = relay.clip(conv2d_1, 0, 6) # relu6
    out = clip_1
    f1 = relay.Function([data, weight1], out)
    mod1 = tvm.IRModule.from_expr(f1)
    print('---------- Original graph ----------')
    print(mod1.astext(show_meta_data=False))

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
    begin0 = relay.annotation.compiler_begin(data, "tidl")
    begin1 = relay.annotation.compiler_begin(weight1, "tidl")
    node0  = relay.nn.conv2d(begin0,
                             begin1,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             kernel_layout = 'OIHW')
    node1  = relay.clip(node0, 0, 6) # relu6
    # whole graph offload
    out2 = relay.annotation.compiler_end(node1, "tidl")
    f2   = relay.Function([data, weight1], out2)
    mod2 = tvm.IRModule.from_expr(f2)
    print('---------- Annotated graph ----------')
    print(mod2.astext(show_meta_data=False))

    #============= Partitioning the graph ==============
    mod2_partition = transform.PartitionGraph()(mod2)
    print('---------- Partitioned graph ----------')
    print(mod2_partition.astext(show_meta_data=False))

    w1 = np.random.rand(*w1_shape).astype('float32')
    params1 = tvm.nd.array(w1)
    #params = {"weight1":params1, "weight2":params2}
    ################### Gap ############################
    # Can't use original names for params
    # TODL: params binding
    ################### Gap ############################
    params = {"tidl_input1":params1} 

    #============= Importing subgraph(s) to TIDL ==============
    # TIDL import pass - import all TIDL subgraphs
    # invoking Relay IR import from TIDL package
    print('---------- Subgraphs import to TIDL ----------')
    if tidl.relay_ir_import(mod2_partition, params) == False:
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
        tidl_utils.tf_image_preprocess(calibration_image, raw_image, input_shape)
        tidl.tidl_calib(tidl_calib_tool, raw_image, subgraph_id)
        print('Importing this model to TIDL succeeded!')

    #============= Compiling the graph (with TIDL subgraphs) ==============
    print('---------- Code generation with TIDL subgraphs ----------')
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(mod2_partition, "llvm")
        #graph, lib, params = relay.build_module.build(mod1, "llvm")  # this works fine
        print(lib)

    artifacts_folder = "./artifacts/"
    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    path_params = artifacts_folder + "deploy_param.params"
    lib.export_library(path_lib) # whole graph offload and heterogeneous execute may need to use diff func. Trevor to check

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))


if __name__ == '__main__':
    test_extern_tidl()
