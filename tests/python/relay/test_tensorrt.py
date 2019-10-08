import numpy as np
import time
from mxnet.gluon.model_zoo.vision import get_model

import tvm
from tvm import relay
import tvm.relay.testing
import tvm.relay.transform
from tvm.contrib import graph_runtime

from tvm.relay.annotation import subgraph_begin, subgraph_end
from test_pass_partition_graph import WholeGraphAnnotator

def test_extern_tensorrt():
    dtype = 'float32'
    xshape = (1, 32, 14, 14)
    yshape = (1, 32,  1,  1)
    zshape = (1,  1,  1,  1)
    x = relay.var('x', shape=(xshape), dtype=dtype)
    y = relay.var('y', shape=(yshape), dtype=dtype)
    z = relay.var('z', shape=(zshape), dtype=dtype)
    w = z * (x + y)
    out = relay.nn.relu(w)
    f = relay.Function([x, y, z], out)

    mod = relay.Module()
    mod['main'] = WholeGraphAnnotator('tensorrt').visit(f)
    mod = relay.transform.PartitionGraph()(mod)

    ref_mod = relay.Module()
    ref_mod['main'] = f

    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)
    y_data = np.random.uniform(-1, 1, yshape).astype(dtype)
    z_data = np.random.uniform(-1, 1, zshape).astype(dtype)

    # Test against reference.
    for kind in ["vm" , "debug", "graph"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.gpu(0), target='cuda')
        # First execution will trigger build of TRT engine(s).
        res = ex.evaluate()(x_data, y_data, z_data)
        # TRT engine is reused for second execution.
        res = ex.evaluate()(x_data, y_data, z_data)

        ref_ex = relay.create_executor(kind, mod=ref_mod, ctx=tvm.cpu(0))
        ref_res = ref_ex.evaluate()(x_data, y_data, z_data)

        tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-5)

    print('Test passed.')

def test_extern_tensorrt_maskrcnn(use_trt=True, profile=False, num_iteration=1000):
    if profile:
        import ctypes
        _cudart = ctypes.CDLL('libcudart.so')

    dtype = 'float32'
    input_shape = (1, 3, 224, 224)
    from gluoncv import model_zoo
    block = model_zoo.get_model('mask_rcnn_fpn_resnet50_v1b_coco', pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

    if use_trt:
        mod['main'] = WholeGraphAnnotator('tensorrt').visit(mod['main'])
        mod = relay.transform.PartitionGraph()(mod)
        graph, lib, params = relay.build(mod, "cuda", params=params)
    else:
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "cuda", params=params)

    i_data = np.random.uniform(0, 1, input_shape).astype(dtype)

    mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
    mod.set_input(**params)
    # Warmup
    for i in range(10):
        mod.run(data=i_data)

    # Start profiling
    if profile:
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

    # Time
    times = []
    for i in range(num_iteration):
        start_time = time.time()
        mod.run(data=i_data)
        res = mod.get_output(0)
        times.append(time.time() - start_time)
    latency = 1000.0 * np.mean(times)
    print(model, latency)
    return latency

def test_extern_tensorrt_graph_runtime_perf(model, use_trt=False, profile=False, num_iteration=1000):
    if profile:
        import ctypes
        _cudart = ctypes.CDLL('libcudart.so')

    dtype = 'float32'
    input_shape = (128, 3, 224, 224)
    block = get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

    if use_trt:
        mod['main'] = WholeGraphAnnotator('tensorrt').visit(mod['main'])
        mod = relay.transform.PartitionGraph()(mod)
        graph, lib, params = relay.build(mod, "cuda", params=params)
    else:
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "cuda", params=params)

    i_data = np.random.uniform(0, 1, input_shape).astype(dtype)

    mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
    mod.set_input(**params)
    # Warmup
    for i in range(10):
        mod.run(data=i_data)

    # Start profiling
    if profile:
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

    # Time
    times = []
    for i in range(num_iteration):
        start_time = time.time()
        mod.run(data=i_data)
        res = mod.get_output(0)
        times.append(time.time() - start_time)
    latency = 1000.0 * np.mean(times)
    print(model, latency)
    return latency


def test_extern_tensorrt_perf(model='resnet50_v1', use_trt=True, profile=False, num_iteration=1000):
    if profile:
        import ctypes
        _cudart = ctypes.CDLL('libcudart.so')

    dtype = 'float32'
    input_shape = (1, 3, 224, 224)
    block = get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

    if use_trt:
        mod['main'] = WholeGraphAnnotator('tensorrt').visit(mod['main'])
        mod = relay.transform.PartitionGraph()(mod)

    i_data = np.random.uniform(0, 1, input_shape).astype(dtype)

    if use_trt:
        with relay.build_config(opt_level=2):
            vm = tvm.relay.vm.compile(mod, 'cuda')
            vm.init(tvm.gpu(0))
            vm.load_params(params)
    else:
        with relay.build_config(opt_level=3):
            vm = tvm.relay.vm.compile(mod, 'cuda', params=params)
            vm.init(tvm.gpu(0))
            vm.load_params(params)
    
    # Warmup
    for i in range(10):
        vm.invoke('main', i_data)

    # Start profiling
    if profile:
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

    # Time
    start_time = time.time()
    for i in range(num_iteration):
        vm.invoke('main', i_data)
    end_time = time.time()
    latency = (end_time-start_time)/num_iteration*1000
    print(model, use_trt, latency)
    return latency

if __name__ == "__main__":
    test_extern_tensorrt_maskrcnn()
    exit(0)
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
        'densenet201'
        ]
    for model in models:
        latency[model] = test_extern_tensorrt_graph_runtime_perf(model=model)
    
    for model in models:
        print(model, latency[model])