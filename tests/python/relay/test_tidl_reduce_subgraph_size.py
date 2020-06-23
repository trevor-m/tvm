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

import numpy as np
import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.backend.contrib.tidl import ReduceSubgraphSize
from test_pass_partition_graph import set_func_attr

def test_reduce_subgraph_size_single_output():
    def create_graph():
        ishape = (1, 3, 12, 12)
        x = relay.var('tidl_i0', shape=ishape, dtype='float32')
        y = relay.nn.relu(x)
        out = relay.nn.relu(y)
        func = relay.Function([x], out)
        func = set_func_attr(func, "tidl", "tidl_0")
        gv = relay.GlobalVar("tidl_0")

        mod = tvm.IRModule()
        mod[gv] = func
        x_main = relay.var('x', shape=ishape, dtype='float32')
        main_f = relay.Function([x_main], gv(x_main))
        mod['main'] = main_f
        return mod

    def expected():
        ishape = (1, 3, 12, 12)
        x = relay.var('tidl_i0', shape=ishape, dtype='float32')
        out = relay.nn.relu(x)
        func = relay.Function([x], out)
        func = set_func_attr(func, "tidl", "tidl_0")
        gv = relay.GlobalVar("tidl_0")

        mod = tvm.IRModule()
        mod[gv] = func
        x_main = relay.var('x', shape=ishape, dtype='float32')
        call = gv(x_main)
        out = relay.nn.relu(call)
        main_f = relay.Function([x_main], out)
        mod['main'] = main_f
        return mod

    ref_mod = expected()
    reduced = ReduceSubgraphSize(create_graph(), max_num_layers=1, compiler="tidl")
    assert tvm.ir.structural_equal(reduced, ref_mod, map_free_vars=True)

def test_reduce_subgraph_size_multiple_output():
    def create_graph():
        ishape = (1, 32, 14, 14)
        w1shape = (32, 1, 3, 3)
        dtype = "float32"
        data0 = relay.var("tidl_0_i0", shape=(ishape), dtype=dtype)
        input0 = relay.var("tidl_0_i1", shape=(w1shape), dtype=dtype)
        input1 = relay.var("tidl_0_i2", shape=(w1shape), dtype=dtype)
        params = {"tidl_0_i1": np.ones(w1shape, dtype="float32"), "tidl_0_i2": np.ones(w1shape, dtype="float32")}
        depthwise_conv2d_1 = relay.nn.conv2d(data0,
                                             input0,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        depthwise_conv2d_2 = relay.nn.conv2d(depthwise_conv2d_1,
                                             input1,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)
        func = relay.Function([data0, input0, input1], out)
        func = set_func_attr(func, "tidl", "tidl_0")
        func = bind_params_by_name(func, params)
        gv = relay.GlobalVar("tidl_0")

        mod = tvm.IRModule()
        mod[gv] = func
        x_main = relay.var('x', shape=ishape, dtype='float32')
        main_f = relay.Function([x_main], gv(x_main))
        mod['main'] = main_f #bind_params_by_name(main_f, params)
        return mod
   
    def expected_1():
        ishape = (1, 32, 14, 14)
        w1shape = (32, 1, 3, 3)
        dtype = "float32"
        data0 = relay.var("tidl_0_i0", shape=(ishape), dtype=dtype)
        input0 = relay.var("tidl_0_i1", shape=(w1shape), dtype=dtype)
        input1 = relay.var("tidl_0_i2", shape=(w1shape), dtype=dtype)
        params = {"tidl_0_i1": np.ones(w1shape, dtype="float32"), "tidl_0_i2": np.ones(w1shape, dtype="float32")}
        depthwise_conv2d_1 = relay.nn.conv2d(data0,
                                             input0,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        depthwise_conv2d_2 = relay.nn.conv2d(depthwise_conv2d_1,
                                             input1,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        out = relay.Tuple([depthwise_conv2d_1, depthwise_conv2d_2])
        func = relay.Function([data0, input0, input1], out)
        func = set_func_attr(func, "tidl", "tidl_0")
        func = bind_params_by_name(func, params)
        gv = relay.GlobalVar("tidl_0")

        mod = tvm.IRModule()
        mod[gv] = func
        x_main = relay.var('x', shape=ishape, dtype='float32')
        call = gv(x_main)
        get_output_0 = relay.TupleGetItem(call, 0)
        get_output_1 = relay.TupleGetItem(call, 1)
        out = relay.add(get_output_0, get_output_1)
        main_f = relay.Function([x_main], out)
        mod['main'] = bind_params_by_name(main_f, params)
        return mod

    def expected_2():
        ishape = (1, 32, 14, 14)
        w1shape = (32, 1, 3, 3)
        dtype = "float32"
        data0 = relay.var("tidl_0_i0", shape=(ishape), dtype=dtype)
        input0 = relay.var("tidl_0_i1", shape=(w1shape), dtype=dtype)
        input1 = relay.var("tidl_0_i2", shape=(w1shape), dtype=dtype)
        params = {"tidl_0_i1": np.ones(w1shape, dtype="float32"), "tidl_0_i2": np.ones(w1shape, dtype="float32")}
        depthwise_conv2d_1 = relay.nn.conv2d(data0,
                                             input0,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        out = depthwise_conv2d_1
        func = relay.Function([data0, input0, input1], out)
        func = set_func_attr(func, "tidl", "tidl_0")
        func = bind_params_by_name(func, params)
        gv = relay.GlobalVar("tidl_0")

        mod = tvm.IRModule()
        mod[gv] = func
        x_main = relay.var('x', shape=ishape, dtype='float32')
        call = gv(x_main)
        depthwise_conv2d_2 = relay.nn.conv2d(call,
                                             input1,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        tup = relay.Tuple([call, depthwise_conv2d_2])
        get_output_0 = relay.TupleGetItem(tup, 0)
        get_output_1 = relay.TupleGetItem(tup, 1)
        out = relay.add(get_output_0, get_output_1)
        main_f = relay.Function([x_main, input1], out)
        mod['main'] = bind_params_by_name(main_f, params)
        return mod
    
    # Will remove add.
    ref_mod = expected_1()
    reduced = ReduceSubgraphSize(create_graph(), max_num_layers=2, compiler="tidl")
    assert tvm.ir.structural_equal(reduced, ref_mod, map_free_vars=True)

    # Will remove 2nd conv2d.
    ref_mod = expected_2()
    reduced = ReduceSubgraphSize(create_graph(), max_num_layers=1, compiler="tidl")
    print('reduced', reduced)
    assert tvm.ir.structural_equal(reduced, ref_mod, map_free_vars=True)

if __name__ == '__main__':
    #test_reduce_subgraph_size_single_output()
    test_reduce_subgraph_size_multiple_output()
