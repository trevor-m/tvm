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
"""Test mobilenet annotation for TIDL."""
import numpy as np

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.op.annotation import tidl_annotation
import tvm.relay.op.contrib.tidl

@transform.function_pass(opt_level=0)
class TIDLWhiteListAnnotator:
    def __init__(self, op_list, compiler):
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod):

        op_annotations = tidl_annotation(mod)
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
    #test_tidl_mobilenet() # Waiting on ARM to fix composite functions in mergecompilerregions pass
    test_tidl_mobilenet_no_composite()