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

from . import _transform
from .expr_functor import ExprMutator
from tvm.relay.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator

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

class RemoveMultiplyByOne(ExprMutator):
    """
    Removes multiply by 1.0f. This pass when followed by
    RemoveRedundantTranspose is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def visit_call(self, expr):
        if expr.op.name == "multiply":
            if isinstance(expr.args[1], Constant):
                data = expr.args[1].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return expr.args[0]
        return super().visit_call(expr)

class RemoveRedundantTranspose(ExprMutator):
    """
    Removes Transpose([1, 0]) followed by Transpose([1, 0]). This pass, when
    preceded by with RemoveMultiplyByOne is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def check_axes(self, axes):
        return len(axes) == 2 and int(axes[0].value) == 1 and int(axes[1].value) == 0

    def visit_call(self, expr):
        if expr.op.name == "transpose":
            if self.check_axes(expr.attrs['axes']):
                if isinstance(expr.args[0], Call) and expr.args[0].op.name == "transpose":
                    if self.check_axes(expr.args[0].attrs['axes']):
                        return expr.args[0].args[0]
        return super().visit_call(expr)

def PreprocessForTrt(mod):
    """Applies passes to prepare main function for TensorRT conversion.

    Parameters
    ----------
    mod: Module
        The original module.

    Returns
    -------
    mod: Module
        The module modified for TensorRT.
    """
    mod['main'] = LegalizeLayoutTranform().visit(mod['main'])
    mod['main'] = RemoveDropout().visit(mod['main'])
    mod['main'] = RemoveMultiplyByOne().visit(mod['main'])
    mod['main'] = RemoveRedundantTranspose().visit(mod['main'])
    return mod

def GetTrtVersion():
    """Gets the version of TensorRT that TVM is built against.

    Returns
    -------
    ret: Tuple[int]
        TensorRT version as a tuple of major, minor, and patch number. If TVM
        is not built with TensorRT, an empty tuple is returned instead.
    """
    return tuple(map(int, _transform.GetTrtVersion()))

def IsTrtRuntimeAvailable():
    if not tvm.get_global_func("relay._transform.GetTrtVersion", True):
        return False
    return GetTrtVersion() != ()

def EnableTrtLegacy(mod, params=None, trt_version=None):
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

    # Apply passes required for TRT
    seq = relay.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                      relay.transform.ConvertLayout('NCHW')])
    with relay.transform.PassContext(opt_level=3):
        mod = seq(mod)
    mod = PreprocessForTrt(mod)
    if params:
        # Bind params so that we can use FoldConstant.
        mod['main'] = _bind_params(mod['main'], params)
    mod = relay.transform.FoldConstant()(mod)
    return _transform.EnableTrt(*trt_version)(mod)

def always_whitelist_fn(call, trt_version):
    return True

def batch_norm_whitelist_fn(call, trt_version):
    if int(call.attrs.axis) != 1 and int(call.attrs.axis) != 3:
        print("nn.batch_norm: axis is {} but must be 1 or 3.".format(int(call.attrs.axis)))
        return False
    return True

def softmax_whitelist_fn(call, trt_version):
    if int(call.attrs.axis) == 0:
        print("{}: can't modify batch dimension.".format(call.op.name))
        return False
    return True

def conv2d_whitelist_fn(call, trt_version):
    if call.attrs.data_layout != "NCHW":
        print("nn.conv2d: data_layout is {} but must be NCHW.".format(call.attrs.data_layout))
        return False
    if call.attrs.kernel_layout != "OIHW":
        print("nn.conv2d: kernel_layout is {} but must be OIHW.".format(call.attrs.kernel_layout))
        return False
    if call.attrs.out_layout and attrs.out_layout != "NCHW":
        print("nn.conv2d: out_layout is {} but must be NCHW.".format(call.attrs.out_layout))
        return False
    return True

def dense_whitelist_fn(call, trt_version):
    input_rank = len(call.type_args[0].shape)
    weight_rank = len(call.type_args[1].shape)
    if input_rank < 2 or input_rank > 4:
        print("nn.dense: input has rank {} but must be 2, 3 or 4.".format(input_rank))
        return False
    if weight_rank != 2:
        print("nn.dense: weight has rank {} but must be 2.".format(weight_rank))
        return False
    return True

def bias_add_whitelist_fn(call, trt_version):
    # TODO(trevmorr): BiasAddSimplifier creates a pattern which cannot be
    # converted to TRT without binding params and constant folding.
    if trt_version < (6, 0, 1):
        return False
    input_rank = len(call.type_args[0].shape)
    if input_rank < 2 or input_rank > 4:
        print("nn.bias_add: input rank is {} but must be 2, 3 or 4.".format(input_rank))
        return False
    return True

def pool_2d_whitelist_fn(call, trt_version):
    if call.attrs.layout != "NCHW":
        print("{}: layout is {} but must be NCHW.".format(call.op.name, call.attrs.layout))
        return False
    if call.op.name == "nn.avg_pool2d":
        if call.attrs.count_include_pad and len(call.attrs.padding) == 4:
            print("nn.avg_pool2d: inclusive-counted blended or average pooling is not supported in combination with asymmetric padding")
            return False
        if call.attrs.ceil_mode and trt_version < (5, 1, 5):
            print("nn.avg_pool2d: ceil_mode=True requires TensorRT 5.1.5 or greater.")
            return False
    return True

def global_pool_2d_whitelist_fn(call, trt_version):
    if call.attrs.layout != "NCHW":
        print("{}: layout is {} but must be NCHW.".format(call.op.name, call.attrs.layout))
        return False
    return True

def expand_dims_whitelist_fn(call, trt_version):
    if trt_version < (6, 0, 1) and int(call.attrs.axis) == 0:
        print("{}: can't modify batch dimension.".format(call.op.name))
        return False
    return True

def squeeze_whitelist_fn(call, trt_version):
    if not call.attrs.axis:
        print("{}: must explicitly set axis.".format(call.op.name))
        return False
    if trt_version < (6, 0, 1) and any([axis == 0 for axis in map(int, call.attrs.axis)]):
            print("{}: can't modify batch dimension.".format(call.op.name))
            return False
    return True

def concatenate_whitelist_fn(call, trt_version):
    if trt_version >= (6, 0, 1):
        return True
    if int(call.attrs.axis) == 0:
        print("{}: can't modify batch dimension.".format(call.op.name))
        return False
    if isinstance(call.args[0], Tuple):
        for tuple_input in call.args[0].fields:
            if isinstance(tuple_input, Constant):
                print("{}: can't concatenate tensors with constants.".format(call.op.name))
                return False
    return True

def conv2d_transpose_whitelist_fn(call, trt_version):
    if call.attrs.data_layout != "NCHW":
        print("nn.conv2d_transpose: data_layout is {} but must be NCHW.".format(call.attrs.data_layout))
        return False
    if call.attrs.kernel_layout != "OIHW":
        print("nn.conv2d_transpose: kernel_layout is {} but must be OIHW.".format(call.attrs.kernel_layout))
        return False
    if call.attrs.out_layout and attrs.out_layout != "NCHW":
        print("nn.conv2d_transpose: out_layout is {} but must be NCHW.".format(call.attrs.out_layout))
        return False
    if call.attrs.dilation and any([rate != 1 for rate in map(int, call.attrs.dilation)]):
        print("nn.conv2d_transpose: dilation rate must be 1.")
        return False
    return True

def transpose_whitelist_fn(call, trt_version):
    if trt_version < (6, 0, 1) and int(call.attrs.axes[0]) != 0:
        print("{}: can't modify batch dimension.".format(call.op.name))
        return False
    return True

def reshape_whitelist_fn(call, trt_version):
    if any([x < -1 for x in map(int, call.attrs.newshape)]):
        print("{}: reshape dims must be explicit.".format(call.op.name))
        return False
    return True

def pad_whitelist_fn(call, trt_version):
    if call.attrs.pad_mode != "constant":
        print("{}: pad mode is {} but must be constant.".format(call.op.name, call.attrs.pad_mode))
        return False
    if float(call.attrs.pad_value) != 0.0:
        print("{}: pad value is {} but must be 0.0.".format(call.op.name, float(call.attrs.pad_value)))
        return False
    return True

def reduce_whitelist_fn(call, trt_version):
    if not call.attrs.axis or len(call.attrs.axis) == 0:
        print("{}: cannot reduce to scalar.".format(call.op.name))
        return False
    if call.attrs.exclude:
        print("{}: exclude not supported.".format(call.op.name))
        return False
    if trt_version < (6, 0, 1) and any([x == 0 for x in map(int, call.attrs.axis)]):
        print("{}: can't modify batch dimension.".format(call.op.name))
        return False
    return True

def adaptive_pool2d_whitelist_fn(call, trt_version):
    # TODO(trevmorr): Consider whether this should only be supported for output size (1, 1)
    return True

def get_min_trt_version_whitelist_fn(min_trt_version):
    def _whitelist_fn(call, trt_version):
        return trt_version > min_trt_version
    return _whitelist_fn

def strided_slice_whitelist_fn(call, trt_version):
    if not get_min_trt_version_whitelist_fn((5, 1, 5))(call, trt_version):
        return False
    if trt_version < (6, 0, 1):
        batch_dim_begin_modified = call.attrs.begin[0] is not None and int(call.attrs.begin[0]) != 0
        batch_dim_end_modified = call.attrs.end[0] is not None and int(call.attrs.end[0]) != -1 and int(call.attrs.end[0]) != int(call.type_args[0].shape[0])
        if batch_dim_begin_modified or batch_dim_end_modified:
            print("{}: can't modify batch dimension.".format(call.op.name))
            return False
    if any([x < 0 for x in map(int, call.attrs.begin)]) or any([x < 0 for x in map(int, call.attrs.end)]):
        print("{}: start/end values must be positive".format(call.op.name))
        return False
    return True

def resize_whitelist_fn(call, trt_version):
    if not get_min_trt_version_whitelist_fn((6, 0, 1))(call, trt_version):
        return False
    if call.attrs.method != "nearest_neighbor" and call.attrs.method != "bilinear":
        return False
    # TODO(trevmorr): coordinate transform method
    return True

def nms_whitelist_fn(call, trt_version):
    return True

@transform.function_pass(opt_level=0)
class TensorRTWhiteListAnnotator:
    def __init__(self, trt_version):
        self.compiler = "tensorrt"
        self.trt_version = trt_version
        self.op_list = {
            "nn.relu": always_whitelist_fn,
            "sigmoid": always_whitelist_fn,
            "tanh": always_whitelist_fn,
            "nn.batch_norm": batch_norm_whitelist_fn,
            "nn.softmax": softmax_whitelist_fn,
            "nn.conv2d": conv2d_whitelist_fn,
            "nn.dense": dense_whitelist_fn,
            "nn.bias_add": bias_add_whitelist_fn,
            "add": always_whitelist_fn,
            "subtract": always_whitelist_fn,
            "multiply": always_whitelist_fn,
            "divide": always_whitelist_fn,
            "power": always_whitelist_fn,
            "nn.max_pool2d": pool_2d_whitelist_fn,
            "nn.avg_pool2d": pool_2d_whitelist_fn,
            "nn.global_max_pool2d": global_pool_2d_whitelist_fn,
            "nn.global_avg_pool2d": global_pool_2d_whitelist_fn,
            "exp": always_whitelist_fn,
            "log": always_whitelist_fn,
            "sqrt": always_whitelist_fn,
            "abs": always_whitelist_fn,
            "negative": always_whitelist_fn,
            "nn.batch_flatten": always_whitelist_fn,
            "expand_dims": expand_dims_whitelist_fn,
            "squeeze": squeeze_whitelist_fn,
            "concatenate": concatenate_whitelist_fn,
            "nn.conv2d_transpose": conv2d_transpose_whitelist_fn,
            "transpose": transpose_whitelist_fn,
            "reshape": reshape_whitelist_fn,
            "nn.pad": pad_whitelist_fn,
            "sum": reduce_whitelist_fn,
            "prod": reduce_whitelist_fn,
            "max": reduce_whitelist_fn,
            "min": reduce_whitelist_fn,
            "mean": reduce_whitelist_fn,
            "contrib.adaptive_max_pool2d": adaptive_pool2d_whitelist_fn,
            "contrib.adaptive_avg_pool2d": adaptive_pool2d_whitelist_fn,
            "clip": always_whitelist_fn,
            "tensorrt.nms": nms_whitelist_fn,
            # Ops which require TRT 5.1.5+
            "nn.leaky_relu": get_min_trt_version_whitelist_fn((5, 1, 5)),
            "sin": get_min_trt_version_whitelist_fn((5, 1, 5)),
            "cos": get_min_trt_version_whitelist_fn((5, 1, 5)),
            "atan": get_min_trt_version_whitelist_fn((5, 1, 5)),
            "ceil": get_min_trt_version_whitelist_fn((5, 1, 5)),
            "floor": get_min_trt_version_whitelist_fn((5, 1, 5)),
            "strided_slice": strided_slice_whitelist_fn,
            # Ops which require TRT 6.0.1+
            "image.resize": resize_whitelist_fn,
        }

    def transform_function(self, func, mod, ctx):
        annotator = self
        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if isinstance(call.op, tvm.relay.expr.Function) and call.op.attrs['Composite'] is not None:
                    # Strip quotes from string.
                    op_name = str(call.op.attrs['Composite'])[1:-1]
                else:
                    op_name = call.op.name
                if op_name in annotator.op_list and annotator.op_list[op_name](call, annotator.trt_version):
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

from tvm.relay import op as reg

def _register_external_op_helper(op_name, supported=True):
    @reg.register(op_name, "target.tensorrt")
    def _func_wrapper(attrs, args):
        return supported
    return _func_wrapper

_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("multiply")
_register_external_op_helper("nn.bias_add")
_register_external_op_helper("nn.batch_flatten")
_register_external_op_helper("nn.max_pool2d")
#_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.global_avg_pool2d")

def EnableTrt(mod, params=None, trt_version=None):
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
    mod = relay.transform.FoldConstant()(mod)

    mod['main'] = RemoveDropout().visit(mod['main'])
    mod = ReconstructNms(mod)
    mod = TensorRTWhiteListAnnotator(trt_version)(mod)
    #mod = transform.AnnotateTargetWithMerge(['tensorrt'])(mod)
    mod = transform.PartitionGraph()(mod)
    # Bind params for all TRT functions.
    # print(mod)
    # if params:
    #     for x in mod.functions.items():
    #         func_name = x[0].name_hint
    #         mod[func_name] = _bind_params(mod[func_name], params)
    mod = transform.InferType()(mod)
    return mod
