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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
"""
Relay TensorRT codegen.
"""
import tvm
from tvm import relay
from tvm.relay.expr import Call, Constant, Tuple
import tvm.relay.transform as transform
from tvm.relay.build_module import bind_params_by_name

from tvm.relay.transform import _ffi_api 
from .expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator

from tvm.relay import op as reg

class LegalizeLayoutTranform(ExprMutator):
    """
    Legalize Relay layout transforms to transpose ops to simplify TensorRT conversion.
    """
    def visit_call(self, expr):
        visit = super().visit_call(expr)
        if expr.op == tvm.relay.op.get("layout_transform"):
            src_layout = expr.attrs['src_layout']
            dst_layout = expr.attrs['dst_layout']
            if src_layout == "NCHW" and dst_layout == "NHWC":
                return relay.transpose(visit.args[0], axes=[0, 2, 3, 1])
            elif src_layout == "NHWC" and dst_layout == "NCHW":
                return relay.transpose(visit.args[0], axes=[0, 3, 1, 2])
            elif src_layout == "HWIO" and dst_layout == "OIHW":
                return relay.transpose(visit.args[0], axes=[3, 2, 0, 1])
            elif src_layout == "HWOI" and dst_layout == "OIHW":
                return relay.transpose(visit.args[0], axes=[2, 3, 0, 1])
            elif src_layout == "HWIO" and dst_layout == "IOHW":
                return relay.transpose(visit.args[0], axes=[2, 3, 0, 1])
        return visit

class RemoveDropout(ExprMutator):
    """
    Removes all nn.dropout from an expr.
    """
    def visit_tuple_getitem(self, expr):
        visit = super().visit_tuple_getitem(expr)
        if visit.index != 0:
            return visit
        elif isinstance(visit.tuple_value, Call) and visit.tuple_value.op.name == "nn.dropout":
            return visit.tuple_value.args[0]
        return visit

class SimplifySliceLike(ExprMutator):
    """
    Legalize Relay layout transforms to transpose ops to simplify TensorRT conversion.
    """
    def visit_call(self, expr):
        if expr.op == tvm.relay.op.get("slice_like"):
            axes = expr.attrs['axes']
            shape0 = expr.args[0].checked_type.shape
            end = [int(x) for x in shape0]
            if axes is not None:
                shape1 = expr.args[1].checked_type.shape
                for axis in axes:
                    if shape1[int(axis)] is None:
                        return visit
                    end[int(axis)] = shape1[int(axis)]
            begin = [0] * len(end)
            arg = super().visit(expr.args[0])
            x = relay.strided_slice(arg, begin=begin, end=end)
            return x
        visit = super().visit_call(expr)
        return visit

@transform.function_pass(opt_level=0)
class LegalizeLayoutTranformPass:
    def transform_function(self, func, mod, _):
        if func.attrs and func.attrs['External'] == "tensorrt":
            return LegalizeLayoutTranform().mutate(func)
        return func

@transform.function_pass(opt_level=0)
class RemoveDropoutPass:
    def transform_function(self, func, mod, _):
        #print('GOT FUNC', func)
        return RemoveDropout().visit(func)
        #return func

@transform.function_pass(opt_level=0)
class SimplifySliceLikePass:
    def transform_function(self, func, mod, _):
        return SimplifySliceLike().visit(func)

def GetTrtVersion():
    """Gets the version of TensorRT that TVM is built against.

    Returns
    -------
    ret: Tuple[int]
        TensorRT version as a tuple of major, minor, and patch number. If TVM
        is not built with TensorRT, an empty tuple is returned instead.
    """
    return tuple(map(int, _ffi_api.GetTrtVersion()))

def IsTrtRuntimeAvailable():
    if not tvm.get_global_func("relay._transform.GetTrtVersion", True):
        return False
    return GetTrtVersion() != ()

def _register_external_op_helper(op_name, supported=True):
    @reg.register(op_name, "target.tensorrt")
    def _func_wrapper(attrs, args):
        return supported
    return _func_wrapper

def _register_external_op_helper_func(op_name, func, trt_version):
    @reg.register(op_name, "target.tensorrt")
    def _func_wrapper(attrs, args):
        return func(attrs, args, op_name, trt_version)
    return _func_wrapper

def register_tensorrt_annotations(trt_version):
    # Ops which are always supported
    _register_external_op_helper("nn.relu")
    _register_external_op_helper("sigmoid")
    _register_external_op_helper("tanh")
    _register_external_op_helper("add")
    _register_external_op_helper("subtract")
    _register_external_op_helper("multiply")
    _register_external_op_helper("divide")
    _register_external_op_helper("power")
    _register_external_op_helper("exp")
    _register_external_op_helper("log")
    _register_external_op_helper("sqrt")
    _register_external_op_helper("abs")
    _register_external_op_helper("negative")
    _register_external_op_helper("nn.batch_flatten")
    _register_external_op_helper("clip")
    _register_external_op_helper("split")
    #_register_external_op_helper("slice_like")
    _register_external_op_helper("nn.upsampling")
    # TODO(trevmorr): Consider whether adaptive pool should only be supported for output size (1, 1).
    _register_external_op_helper("contrib.adaptive_max_pool2d")
    _register_external_op_helper("contrib.adaptive_avg_pool2d")

    @reg.register("nn.batch_norm", "target.tensorrt")
    def batch_norm_whitelist_fn(attrs, args):
        if int(attrs.axis) != 1 and int(attrs.axis) != 3:
            print("nn.batch_norm: axis is {} but must be 1 or 3.".format(int(attrs.axis)))
            return False
        return True

    @reg.register("nn.softmax", "target.tensorrt")
    def softmax_whitelist_fn(attrs, args):
        if int(attrs.axis) == 0:
            print("nn.softmax: can't modify batch dimension.")
            return False
        return True

    @reg.register("nn.conv2d", "target.tensorrt")
    def conv2d_whitelist_fn(attrs, args):
        if attrs.data_layout != "NCHW":
            print("nn.conv2d: data_layout is {} but must be NCHW.".format(attrs.data_layout))
            return False
        if attrs.kernel_layout != "OIHW":
            print("nn.conv2d: kernel_layout is {} but must be OIHW.".format(attrs.kernel_layout))
            return False
        if attrs.out_layout and attrs.out_layout != "NCHW":
            print("nn.conv2d: out_layout is {} but must be NCHW.".format(attrs.out_layout))
            return False
        return True

    @reg.register("nn.dense", "target.tensorrt")
    def dense_whitelist_fn(attrs, args):
        input_rank = len(args[0].checked_type.shape)
        weight_rank = len(args[1].checked_type.shape)
        if input_rank < 2 or input_rank > 4:
            print("nn.dense: input has rank {} but must be 2, 3 or 4.".format(input_rank))
            return False
        if weight_rank != 2:
            print("nn.dense: weight has rank {} but must be 2.".format(weight_rank))
            return False
        return True

    @reg.register("nn.bias_add", "target.tensorrt")
    def bias_add_whitelist_fn(attrs, args):
        # TODO(trevmorr): BiasAddSimplifier creates a pattern which cannot be
        # converted to TRT without binding params and constant folding.
        # if trt_version < (6, 0, 1):
        #     return False
        input_rank = len(args[0].checked_type.shape)
        if input_rank < 2 or input_rank > 4:
            print("nn.bias_add: input rank is {} but must be 2, 3 or 4.".format(input_rank))
            return False
        return True

    @reg.register("nn.max_pool2d", "target.tensorrt")
    def max_pool_2d_whitelist_fn(attrs, args):
        if attrs.layout != "NCHW":
            print("nn.max_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        return True
    
    @reg.register("nn.avg_pool2d", "target.tensorrt")
    def avg_pool_2d_whitelist_fn(attrs, args):
        if attrs.layout != "NCHW":
            print("nn.avg_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        if attrs.count_include_pad and len(attrs.padding) == 4:
            print("nn.avg_pool2d: inclusive-counted blended or average pooling is not supported in combination with asymmetric padding")
            return False
        if attrs.ceil_mode and trt_version < (5, 1, 5):
            print("nn.avg_pool2d: ceil_mode=True requires TensorRT 5.1.5 or greater.")
            return False
        return True

    @reg.register("nn.global_max_pool2d", "target.tensorrt")
    def global_max_pool_2d_whitelist_fn(attrs, args):
        if attrs.layout != "NCHW":
            print("nn.global_max_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        return True
    
    @reg.register("nn.global_avg_pool2d", "target.tensorrt")
    def global_avg_pool_2d_whitelist_fn(attrs, args):
        if attrs.layout != "NCHW":
            print("nn.global_avg_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        return True

    @reg.register("expand_dims", "target.tensorrt")
    def expand_dims_whitelist_fn(attrs, args):
        if args[0].checked_type.dtype != "float32":
            print("expand_dims: only fp32 inputs are supported.")
            return False
        if trt_version < (6, 0, 1) and int(attrs.axis) == 0:
            print("expand_dims: can't modify batch dimension.")
            return False
        return True

    @reg.register("squeeze", "target.tensorrt")
    def squeeze_whitelist_fn(attrs, args):
        if not attrs.axis:
            print("squeeze: must explicitly set axis.")
            return False
        if trt_version < (6, 0, 1) and any([axis == 0 for axis in map(int, attrs.axis)]):
                print("squeeze: can't modify batch dimension.")
                return False
        return True

    @reg.register("concatenate", "target.tensorrt")
    def concatenate_whitelist_fn(attrs, args):
        if trt_version >= (6, 0, 1):
            return True
        if int(attrs.axis) == 0:
            print("concatenate: can't modify batch dimension.")
            return False
        if isinstance(args[0], Tuple):
            for tuple_input in args[0].fields:
                if isinstance(tuple_input, Constant):
                    print("concatenate: can't concatenate tensors with constants.".format(op.name))
                    return False
        return True

    @reg.register("nn.conv2d_transpose", "target.tensorrt")
    def conv2d_transpose_whitelist_fn(attrs, args):
        if attrs.data_layout != "NCHW":
            print("nn.conv2d_transpose: data_layout is {} but must be NCHW.".format(attrs.data_layout))
            return False
        if attrs.kernel_layout != "OIHW":
            print("nn.conv2d_transpose: kernel_layout is {} but must be OIHW.".format(attrs.kernel_layout))
            return False
        if attrs.out_layout and attrs.out_layout != "NCHW":
            print("nn.conv2d_transpose: out_layout is {} but must be NCHW.".format(attrs.out_layout))
            return False
        if attrs.dilation and any([rate != 1 for rate in map(int, attrs.dilation)]):
            print("nn.conv2d_transpose: dilation rate must be 1.")
            return False
        return True

    @reg.register("transpose", "target.tensorrt")
    def transpose_whitelist_fn(attrs, args):
        if trt_version < (6, 0, 1) and int(attrs.axes[0]) != 0:
            print("transpose: can't modify batch dimension.")
            return False
        return True

    @reg.register("reshape", "target.tensorrt")
    def reshape_whitelist_fn(attrs, args):
        if any([x < -1 for x in map(int, attrs.newshape)]):
            print("reshape: new shape dims must be explicit.")
            return False
        return True

    @reg.register("nn.pad", "target.tensorrt")
    def pad_whitelist_fn(attrs, args):
        if attrs.pad_mode != "constant":
            print("nn.pad: pad mode is {} but must be constant.".format(attrs.pad_mode))
            return False
        if float(attrs.pad_value) != 0.0:
            print("nn.pad: pad value is {} but must be 0.0.".format(float(attrs.pad_value)))
            return False
        return True

    def reduce_whitelist_fn(attrs, args, op_name, trt_version):
        if not attrs.axis or len(attrs.axis) == 0:
            print("{}: cannot reduce to scalar.".format(op_name))
            return False
        if attrs.exclude:
            print("{}: exclude not supported.".format(op_name))
            return False
        if trt_version < (6, 0, 1) and any([x == 0 for x in map(int, attrs.axis)]):
            print("{}: can't modify batch dimension.".format(op_name))
            return False
        return True
    
    _register_external_op_helper_func("sum", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("prod", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("max", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("min", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("mean", reduce_whitelist_fn, trt_version)

    def trt_5_1_5_whitelist_fn(attrs, args, op_name, trt_version):
        if trt_version < (5, 1, 5):
            print("{}: requires TensorRT version 5.1.5 or higher.".format(op_name))
            return False
        return True
    
    _register_external_op_helper_func("nn.leaky_relu", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("sin", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("cos", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("atan", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("ceil", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("floor", trt_5_1_5_whitelist_fn, trt_version)

    @reg.register("strided_slice", "target.tensorrt")
    def strided_slice_whitelist_fn(attrs, args):
        if trt_version < (5, 1, 5):
            print("strided_slice: requires TensorRT version 5.1.5 or higher.")
            return False
        if args[0].checked_type.dtype != "float32":
            print("strided_slice: only fp32 inputs are supported.")
            return False
        if trt_version < (6, 0, 1):
            batch_dim_begin_modified = attrs.begin[0] is not None and int(attrs.begin[0]) != 0
            batch_dim_end_modified = attrs.end[0] is not None and int(attrs.end[0]) != -1 and int(attrs.end[0]) != int(args[0].checked_type.shape[0])
            if batch_dim_begin_modified or batch_dim_end_modified:
                print("strided_slice: can't modify batch dimension.")
                return False
        if any([x.defined() and x <= 0 for x in attrs.strides]):
            print("strided_slice: stride must be positive")
            return False
        return True

    @reg.register("image.resize", "target.tensorrt")
    def resize_whitelist_fn(attrs, args):
        if trt_version < (6, 0, 1):
            print("strided_slice: requires TensorRT version 6.0.1 or higher.")
            return False
        if attrs.method != "nearest_neighbor" and attrs.method != "bilinear":
            return False
        # TODO(trevmorr): coordinate transform method
        return True

    # def nms_whitelist_fn(call, trt_version):
    #     return True

def ReconstructNms(mod):
    """Fuse get_valid_count and non_max_suppression into a single composite
    function which can be partitioned and converted to TRT.
    """
    def make_nms_pattern():
        x = relay.var('x')
        ret = relay.vision.get_valid_counts(x, score_threshold=0)
        nms_out = relay.vision.non_max_suppression(ret[1], ret[0])
        return nms_out

    pattern_table = [
        ('tensorrt.nms', make_nms_pattern())
    ]
    return relay.transform.MergeComposite(pattern_table)(mod)


register_tensorrt_annotations((6, 0, 1))

def EnableTrt(mod, params=None, trt_version=None):
    """Converts the "main" function in the module into one that can be executed using
    TensorRT. If any of the operators are not supported by the TensorRT
    conversion, the unmodified program will be returned instead.

    Parameters
    ----------
    mod: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    trt_version : Optional[Tuple[int]]
        Which version of TensorRT to target for partitioning as a tuple of
        (major, minor, patch). If not specified, will attempt to get using
        GetTrtVersion.

    Returns
    -------
    mod: Module
        The modified module which will use the TensorRT runtime if compatible.
    """
    if not trt_version:
        trt_version = GetTrtVersion()
        # If TVM wasn't built against TRT, default to target TRT 6. Since the
        # actual conversion to TRT is done at runtime, building against TRT is
        # not required for compilation.
        if not trt_version:
            trt_version = (6, 0, 1)
    assert isinstance(trt_version, (list, tuple))
    assert len(trt_version) == 3

    if params:
       # Bind params so that we can use FoldConstant.
       mod['main'] = bind_params_by_name(mod['main'], params)
    # Apply passes required for TRT
    mod = transform.InferType()(mod)
    mod['main'] = SimplifySliceLike().visit(mod['main'])
    seq = relay.transform.Sequential([#transform.FoldConstant(),
                                      RemoveDropoutPass(),
                                      transform.AnnotateTarget('tensorrt'),
                                      transform.MergeCompilerRegions(),
                                      transform.PartitionGraph(),
                                    #   transform.RemoveUnusedFunctions(),
                                    #   transform.ConvertLayout('NCHW'),
                                      #LegalizeLayoutTranformPass(),
                                      transform.InferType()])
    with relay.transform.PassContext(opt_level=3):
        mod = seq(mod)
    print(mod)
    return mod
